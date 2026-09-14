"""First-grasp diagnostic; invoked in a fresh robocraft Python subprocess.

Does not edit control.py, train weights, or optimize actions. Uses the local
control.main initialization, then intercepts planning before it begins.
Reference clouds are reconstructed MPM observations, NOT exact material IDs.
"""
import argparse
from pathlib import Path
import os
import sys
import json
import csv
import hashlib


def chamfer(a, b):
    import numpy as np
    distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(distance.min(axis=1).mean() + distance.min(axis=0).mean())


def append_frame(history, frame):
    import numpy as np
    return np.concatenate((history[1:], frame[None]), axis=0)


def make_free_frame(prediction, observed_frame, n_particle):
    # Copy known tool/floor geometry only. Never copy observed object particles
    # into the free rollout after initialization.
    import numpy as np
    return np.concatenate((prediction, observed_frame[n_particle:]), axis=0)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--workdir', required=True)
    cli.add_argument('--result-dir', required=True)
    cli.add_argument('--output-dir', required=True)
    cli.add_argument('--iteration', type=int, default=3)
    cli.add_argument('--model-path', required=True)
    options = cli.parse_args()
    workdir = Path(options.workdir).resolve()
    result_dir = Path(options.result_dir).resolve()
    output = Path(options.output_dir).resolve()
    model_path = Path(options.model_path).resolve()
    pose_file = result_dir / ('init_pose_seq_%d.npy' % options.iteration)
    action_file = result_dir / ('act_seq_%d.npy' % options.iteration)
    for path in (workdir / 'control.py', pose_file, action_file, model_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(str(workdir))
    sys.path.insert(0, str(workdir))
    # All arguments consumed above; control's argparse sees only its own flags.
    sys.argv = [str(workdir / 'control.py'), '--stage', 'control',
                '--data_type', 'ngrip_fixed', '--model_path', str(model_path),
                '--control_algo', 'predict', '--n_grips', '5',
                '--predict_horizon', '2', '--opt_algo', 'GD',
                '--correction', '1', '--shape_type', 'alphabet',
                '--goal_shape_name', 'A', '--debug', '1',
                '--controlf', str(output / 'initialization')]
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print('[diagnostic] Loading local control.py and initializing MPM/GNN (no optimization)...', flush=True)
    import control as c

    class InitializationComplete(Exception):
        pass

    captured = {}
    def capture_planner(planner):
        captured['planner'] = planner
        raise InitializationComplete()

    original_method = c.Planner.trajectory_optimization
    c.Planner.trajectory_optimization = capture_planner
    # Avoid control.Tee redirecting stdout / its destructor closing files.
    original_tee = c.Tee
    c.Tee = lambda *args, **kwargs: None
    try:
        try:
            c.main()
        except InitializationComplete:
            pass
    finally:
        c.Planner.trajectory_optimization = original_method
        c.Tee = original_tee
    if 'planner' not in captured:
        raise RuntimeError('Could not capture initialized planner. Local control.py differs.')
    p = captured['planner']
    p.model.eval()
    p.model.requires_grad_(False)
    n, h = p.n_particle, p.args.n_his
    if (n, p.n_shape) != (300, 31):
        raise RuntimeError('Expected the published ngrip_fixed 300-particle / 31-shape setup.')

    poses = np.load(str(pose_file), allow_pickle=False)
    actions = np.load(str(action_file), allow_pickle=False)
    if poses.ndim != 3 or poses.shape[1:] != (11, 14):
        raise ValueError('Unexpected saved pose shape: %s' % (poses.shape,))
    if actions.ndim != 3 or actions.shape[2] != 12 or len(actions) != len(poses):
        raise ValueError('Unexpected saved action shape: %s' % (actions.shape,))
    if not (np.isfinite(poses).all() and np.isfinite(actions).all()):
        raise ValueError('Saved actions/poses contain nonfinite values.')
    # Only the first grasp from the EXACT iteration corresponding to anim_3.gif.
    pose = poses[0].copy()
    action = actions[0].copy()
    if not np.allclose(action[:, [3, 4, 5, 9, 10, 11]], 0):
        raise ValueError('This diagnostic is for the translational ngrip_fixed example only.')
    np.save(str(output / 'first_pose.npy'), pose)
    np.save(str(output / 'first_actions.npy'), action)
    env = p.taichi_env
    env.set_state(**p.env_init_state)
    mid = c.task_params['gripper_mid_pt']
    env.primitives.primitives[0].set_state(0, pose[mid, :7])
    env.primitives.primitives[1].set_state(0, pose[mid, 7:])

    def observe():
        # Same RGBD -> sampled cloud -> tool/floor augmentation as control.py.
        state = np.asarray(c.sample_particles(env, n), dtype=np.float32)
        if state.shape != (n + p.n_shape, 3) or not np.isfinite(state).all():
            raise ValueError('Invalid reconstructed observation: %s' % (state.shape,))
        return state.copy()

    print('\n[diagnostic] Reconstructing initial MPM observation...', flush=True)
    initial = observe()
    # A second independent observation at the SAME physical state gives a
    # rough resampling/rendering-noise reference (not a rigorous lower bound).
    initial_repeat = observe()
    sample_noise = chamfer(initial[:n], initial_repeat[:n])
    initial_history = np.repeat(initial[None], h, axis=0)
    teacher_history = initial_history.copy()
    free_history = initial_history.copy()
    model_device = next(p.model.parameters()).device
    memory = p.model.init_memory(1, n + p.n_shape)
    group = c.get_env_group(p.args, n, p.scene_params.expand(1, -1), use_gpu=p.use_gpu)

    def predict(history):
        # Mirrors a SINGLE model_rollout forward: graph from current nodes,
        # input history through time t, output at t+1. Do not pass future object
        # positions. In the original, the action advances tools AFTER forward;
        # tool motion enters the predictor through the existing tool history.
        attr, _, rr, rs, rn, _ = c.prepare_input(
            history[-1], n, p.n_shape, p.args, stdreg=p.args.stdreg)
        inputs = [attr.to(model_device).unsqueeze(0),
                  torch.from_numpy(history).to(model_device).unsqueeze(0),
                  rr.to(model_device).unsqueeze(0), rs.to(model_device).unsqueeze(0),
                  rn.to(model_device).unsqueeze(0), memory, group, None]
        with torch.no_grad():
            prediction = p.model.predict_dynamics(inputs)[0][0].detach().cpu().numpy().copy()
        if prediction.shape != (n, 3) or not np.isfinite(prediction).all():
            raise ValueError('Nonfinite or incorrectly shaped GNN prediction.')
        return prediction

    # Validate our first forward against the user's local model_rollout,
    # including their checkpoint wrapper, before running the experiment.
    p.floor_state = torch.from_numpy(initial_history[:, n:n + 9].copy())
    with torch.no_grad():
        reference = p.model_rollout(
            torch.from_numpy(initial_history[:, :n].copy()),
            pose[None, None].copy(), action[None, None, :1].copy())
    reference = reference[0, 0].detach().cpu().numpy()
    direct = predict(initial_history)
    forward_max_difference = float(np.max(np.abs(reference - direct)))
    print('[diagnostic] Direct vs local model_rollout first-forward max difference:',
          forward_max_difference, flush=True)
    if not np.allclose(reference, direct, rtol=1e-4, atol=1e-5):
        raise RuntimeError('One-step adapter does not match local model_rollout. Stop and inspect source.')

    observations = [initial]
    teacher_predictions, free_predictions, rows = [], [], []
    expected_tools = np.concatenate((pose[:, :3], pose[:, 7:10]), axis=0).astype(np.float32)
    with (output / 'errors.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['step', 'one_step_cd', 'free_rollout_cd', 'hold_last_cd'])
        writer.writeheader()
        for t, act in enumerate(action):
            one_step = predict(teacher_history)
            free = predict(free_history)
            if t == 0 and not np.allclose(one_step, free, rtol=1e-4, atol=1e-5):
                raise RuntimeError('Identical first histories did not yield matching predictions.')
            env.step(act)
            actual = observe()
            # Both branches use the same actual tool motion; verify that it
            # agrees with model_rollout's action integration, including scale.
            expected_tools[:11] += (0.02 * act[:3]).astype(np.float32)
            expected_tools[11:] += (0.02 * act[6:9]).astype(np.float32)
            if not np.allclose(actual[n + 9:], expected_tools, rtol=1e-4, atol=1e-5):
                raise RuntimeError('Executed tool positions differ from the expected 0.02 action update.')
            row = dict(step=t + 1, one_step_cd=chamfer(one_step, actual[:n]),
                       free_rollout_cd=chamfer(free, actual[:n]),
                       hold_last_cd=chamfer(teacher_history[-1, :n], actual[:n]))
            rows.append(row)
            writer.writerow(row)
            file.flush()
            observations.append(actual)
            teacher_predictions.append(one_step)
            free_predictions.append(free)
            # Teacher uses observed history; free uses only its OWN object
            # predictions, plus shared known floor/tool positions.
            teacher_history = append_frame(teacher_history, actual)
            free_history = append_frame(free_history, make_free_frame(free, actual, n))
            print('[step %02d/%02d] one-step %.6f | free %.6f | hold-last %.6f' %
                  (t + 1, len(action), row['one_step_cd'], row['free_rollout_cd'], row['hold_last_cd']), flush=True)

    observations = np.stack(observations)
    teacher_predictions = np.stack(teacher_predictions)
    free_predictions = np.stack(free_predictions)
    np.savez_compressed(str(output / 'trajectories.npz'), observations=observations,
                        teacher_predictions=teacher_predictions, free_predictions=free_predictions,
                        actions=action, initial_history=initial_history)
    steps = np.arange(1, len(action) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key, label, color in [('one_step_cd', 'One-step (observed history)', '#147d92'),
                              ('free_rollout_cd', 'Free rollout (predicted history)', '#d04b36'),
                              ('hold_last_cd', 'Hold-last observation baseline', '#777777')]:
        ax.plot(steps, [r[key] for r in rows], label=label, color=color, linewidth=2)
    ax.axhline(sample_noise, color='#999999', linestyle=':', label='Same-state resampling (one pair)')
    if len(action) > c.task_params['len_per_grip']:
        ax.axvline(c.task_params['len_per_grip'] + 0.5, color='#bbbbbb', linestyle='--')
    ax.set(xlabel='Step in FIRST grasp (one env.step per step)',
           ylabel='Symmetric Chamfer distance (scene coordinate units)',
           title='Same saved action; observed-history vs predicted-history GNN')
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output / 'errors.png'), dpi=160)
    plt.close(fig)

    selected = sorted(set([0, min(19, len(action)-1), len(action)-1]))
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.2 * len(selected)), squeeze=False)
    clouds = (observations[1:, :n], teacher_predictions, free_predictions)
    all_points = np.concatenate([cloud.reshape(-1, 3) for cloud in clouds])
    low = all_points.min(axis=0) - 0.02
    high = all_points.max(axis=0) + 0.02
    for ri, idx in enumerate(selected):
        for ci, (cloud, title) in enumerate(zip(clouds, ('MPM observed cloud', 'GNN one-step', 'GNN free rollout'))):
            ax = axes[ri, ci]
            ax.scatter(observations[idx+1, :n, 0], observations[idx+1, :n, 2],
                       s=7, c='#b5bbc0', alpha=0.45)
            ax.scatter(cloud[idx, :, 0], cloud[idx, :, 2], s=7,
                       c=('#353c46', '#147d92', '#d04b36')[ci], alpha=0.7)
            tool = observations[idx+1, n+9:]
            ax.scatter(tool[:, 0], tool[:, 2], s=15, c='#e29715')
            ax.set(xlim=(low[0], high[0]), ylim=(high[2], low[2]),
                   xlabel='x', ylabel='z', title='%s | step %d' % (title, idx+1))
            ax.set_aspect('equal')
    fig.suptitle('Top view; gray background = MPM observation; orange = tools', y=1.0)
    fig.tight_layout()
    fig.savefig(str(output / 'snapshots.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)
    metadata = {
        'source_result_dir': str(result_dir), 'source_iteration': options.iteration,
        'model_path': str(model_path), 'n_history': h, 'first_grasp_steps': len(action),
        'local_control_sha256': hashlib.sha256((workdir / 'control.py').read_bytes()).hexdigest(),
        'local_model_sha256': hashlib.sha256((workdir / 'model.py').read_bytes()).hexdigest(),
        'first_forward_max_abs_difference': forward_max_difference,
        'same_state_resampling_cd_one_pair': sample_noise,
        'numpy_version': np.__version__, 'torch_version': torch.__version__,
        'mean_one_step_cd': float(np.mean([r['one_step_cd'] for r in rows])),
        'mean_free_rollout_cd': float(np.mean([r['free_rollout_cd'] for r in rows])),
        'mean_hold_last_cd': float(np.mean([r['hold_last_cd'] for r in rows])),
        'scope': 'First saved grasp only; no planning, training, or gradient calculation.',
        'reference': 'RGBD reconstructed 300-point MPM observations; includes sampling/reconstruction noise.',
        'initialization': 'Repeated static initial observation for both branches; not the original GIF initial cloud.',
        'time_alignment': 'Predict from history through t; execute saved action[t]; compare to observed t+1.',
        'action_conditioning': 'Matches original predictor call: tool history, not explicit future action tensor.',
        'limitations': ['Not a clean full-material-state oracle or a paper benchmark reproduction.',
                        'Replaying starts from a fresh MPM initialization; sampling is stochastic.',
                        'One-step and free errors alone do not isolate model error from perception or configuration error.',
                        'No claim that checkpoint gradients equal the original; all inference is no_grad.']}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nDIAGNOSTIC COMPLETE:', output, flush=True)


if __name__ == '__main__':
    main()
