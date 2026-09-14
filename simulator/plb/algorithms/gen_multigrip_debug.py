"""Diagnostic: 1 episode x 2 sequential grips, verifies state timing / grip boundaries.

Env/action logic is copied from test_tasks.ipynb (cells 1-4), behavior unchanged:
teleport -> 30 close -> 10 retreat, yield_stress=200, E=5e3, nu=0.2, pin capsule kept.
No rendering (RGB-D not needed for timing check).

Run:   python gen_multigrip_debug.py                 # simulate + save npz + plots
       python gen_multigrip_debug.py --viz <npz>     # plots only
"""
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

N_GRIPS = 2
N_OBS = 300
MID_FRAME = 25   # visualization only (frame 15 is pre-contact)
SEED = 0
REF_MATERIAL = dict(yield_stress=200, E=5e3, nu=0.2)   # notebook cell 4 values; settle reference only

# ---- copied from notebook cell 2 ----
task_name = 'ngrip'
env_type = '_fixed'
task_params = {
    "mid_point": np.array([0.5, 0.14, 0.5, 0, 0, 0]),
    "sample_radius": 0.4,
    "len_per_grip": 30,
    "len_per_grip_back": 10,
    "gripper_rate_limits": np.array([0.14, 0.06]),
    "p_noise_scale": 0.01,
}
N_FRAMES = task_params["len_per_grip"] + task_params["len_per_grip_back"]


# ---- copied from notebook cell 3 (unchanged) ----
def set_parameters(env, yield_stress, E, nu):
    env.simulator.yield_stress.fill(yield_stress)
    _mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    env.simulator.mu.fill(_mu)
    env.simulator.lam.fill(_lam)


def update_primitive(env, prim1_list, prim2_list):
    env.primitives.primitives[0].set_state(0, prim1_list)
    env.primitives.primitives[1].set_state(0, prim2_list)


def random_rotate(mid_point, gripper1_pos, gripper2_pos, z_vec):
    from transforms3d.quaternions import mat2quat
    from transforms3d.axangles import axangle2mat
    mid_point = mid_point[:3]
    z_angle = np.random.uniform(0, np.pi)
    z_mat = axangle2mat(z_vec, z_angle, is_normalized=True)
    all_mat = z_mat
    quat = mat2quat(all_mat)
    return gripper1_pos, gripper2_pos, quat


def random_pose(task_name):
    p_noise_x = task_params["p_noise_scale"] * (np.random.randn() * 2 - 1)
    p_noise_z = task_params["p_noise_scale"] * (np.random.randn() * 2 - 1)
    if task_name == 'ngrip' or task_name == 'ngrip_3d':
        p_noise = np.clip(np.array([p_noise_x, 0, p_noise_z]), a_min=-0.1, a_max=0.1)
    else:
        raise NotImplementedError

    new_mid_point = task_params["mid_point"][:3] + p_noise

    rot_noise = np.random.uniform(0, np.pi)

    x1 = new_mid_point[0] - task_params["sample_radius"] * np.cos(rot_noise)
    z1 = new_mid_point[2] + task_params["sample_radius"] * np.sin(rot_noise)
    x2 = new_mid_point[0] + task_params["sample_radius"] * np.cos(rot_noise)
    z2 = new_mid_point[2] - task_params["sample_radius"] * np.sin(rot_noise)
    y = new_mid_point[1]
    z_vec = np.array([np.cos(rot_noise), 0, np.sin(rot_noise)])
    if task_name == 'ngrip':
        gripper1_pos = np.array([x1, y, z1])
        gripper2_pos = np.array([x2, y, z2])
        quat = np.array([1, 0, 0, 0])
    elif task_name == 'ngrip_3d':
        gripper1_pos, gripper2_pos, quat = random_rotate(new_mid_point, np.array([x1, y, z1]), np.array([x2, y, z2]), z_vec)
    else:
        raise NotImplementedError
    return np.concatenate([gripper1_pos, quat]), np.concatenate([gripper2_pos, quat]), rot_noise


def get_obs(env, n_particles, t=0):
    x = env.simulator.get_x(t)
    v = env.simulator.get_v(t)
    step_size = len(x) // n_particles
    return x[::step_size], v[::step_size]


def select_tool(env, width):
    env.primitives.primitives[0].r[None] = width
    env.primitives.primitives[1].r[None] = width


# ---- small readers (same expressions as notebook cell 4) ----
def read_object_cloud(env):
    # notebook: obs = get_obs(env, 300); x = obs[0][:300]
    # get_obs returns (x[::stride], v[::stride]); stride = 20000 // 300 = 66 -> 304 rows, cut to 300
    return get_obs(env, N_OBS)[0][:N_OBS].copy()


def read_tool_pose(env):
    # movable fingers = primitives[0], primitives[1]; primitives[2] is the fixed pin (action dim 0)
    return np.stack([env.primitives.primitives[0].get_state(0),
                     env.primitives.primitives[1].get_state(0)])


def make_grip_actions(prim1, prim2, rate):
    # copied from notebook cell 4
    zero_pad = np.array([0, 0, 0])
    actions = []
    mid_point = (prim1[:3] + prim2[:3]) / 2
    prim1_direction = mid_point - prim1[:3]
    prim1_direction = prim1_direction / np.linalg.norm(prim1_direction)
    for _ in range(task_params["len_per_grip"]):
        prim1_action = rate * prim1_direction
        actions.append(np.concatenate([prim1_action / 0.02, zero_pad, -prim1_action / 0.02, zero_pad]))
    for _ in range(task_params["len_per_grip_back"]):
        prim1_action = -rate * prim1_direction
        actions.append(np.concatenate([prim1_action / 0.02, zero_pad, -prim1_action / 0.02, zero_pad]))
    return np.stack(actions)


def make_env():
    """notebook cell 1 + cell 4 episode reset. Returns env (reset, reference material) and base_state."""
    os.chdir(HERE)  # plb/envs/env.py resolves robocraft/ via os.getcwd(); notebook runs from here
    from plb.engine.taichi_env import TaichiEnv
    from plb.config import load
    import taichi as ti
    ti.init(arch=ti.gpu)

    # ---- notebook cell 1 ----
    cfg = load(f"../envs/gripper{env_type}.yml")
    env = TaichiEnv(cfg, nn=False, loss=False)
    env.initialize()
    state = env.get_state()   # dict(state=[x,v,F,C,prim0,prim1,prim2], softness, is_copy)
    env.set_state(**state)
    update_primitive(env, [0.3, 0.4, 0.5, 1, 0, 0, 0], [0.7, 0.4, 0.5, 1, 0, 0, 0])
    assert env.primitives.action_dim == 12, env.primitives.action_dim

    # ---- notebook cell 4: episode reset ----
    env.set_state(**state)
    set_parameters(env, **REF_MATERIAL)
    update_primitive(env, [0.3, 0.4, 0.5, 1, 0, 0, 0], [0.7, 0.4, 0.5, 1, 0, 0, 0])
    return env, state


def capsule_clearance(env, x):
    # Capsule SDF (primitives.py): segment along local y in [-h/2, h/2], radius r. All rotations are identity here.
    out = []
    for p in env.primitives.primitives:
        pose = p.get_state(0)
        assert np.allclose(pose[3:], [1, 0, 0, 0]), pose
        d = x - pose[:3]
        h = p.h[None]
        d[:, 1] -= np.clip(d[:, 1], -h / 2, h / 2)
        out.append(float(np.min(np.linalg.norm(d, axis=1)) - p.r[None]))
    return out  # >0: no particle inside capsule


def settle(env, n_steps):
    """Zero-action env.step n_steps times. Returns per-step metrics on ALL particles (simulator.get_x/get_v)."""
    zero_action = np.zeros(env.primitives.action_dim)
    x_prev = env.simulator.get_x(0)
    rows = []
    for t in range(n_steps):
        env.step(zero_action)   # raises ValueError on NaN
        x = env.simulator.get_x(0)
        v = env.simulator.get_v(0)
        dx = np.linalg.norm(x - x_prev, axis=1)
        speed = np.linalg.norm(v, axis=1)
        rows.append(dict(step=t, max_dx=dx.max(), mean_dx=dx.mean(), max_v=speed.max(), mean_v=speed.mean(),
                         min_y=x[:, 1].min(), max_y=x[:, 1].max(),
                         n_inf=int(np.isinf(x).sum() + np.isinf(v).sum())))
        x_prev = x
    return rows


def settle_diagnostic(out_dir, n_steps=30):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    env, _ = make_env()
    x0 = env.simulator.get_x(0)
    clearance = capsule_clearance(env, x0)
    print(f"n_particles: {len(x0)}  initial min_y {x0[:, 1].min():.5f} max_y {x0[:, 1].max():.5f}")
    print(f"initial capsule clearance [finger0, finger1, pin] (<0 = particles inside): {clearance}")

    rows = settle(env, n_steps)
    x = env.simulator.get_x(0)
    print(f"after {n_steps} steps: bbox min {x.min(0)} max {x.max(0)}  "
          f"NaN {int(np.isnan(x).sum())}  Inf {int(np.isinf(x).sum())}  "
          f"total max disp from initial {np.linalg.norm(x - x0, axis=1).max():.5f}")
    print(f"capsule clearance after settle: {capsule_clearance(env, x)}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'settle_diagnostic.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'step':>4} | {'max_dx':>10} | {'mean_dx':>10} | {'max_v':>10} | {'mean_v':>10} | {'min_y':>8} | {'max_y':>8}")
    for r in rows:
        print(f"{r['step']:4d} | {r['max_dx']:10.3e} | {r['mean_dx']:10.3e} | {r['max_v']:10.3e} | "
              f"{r['mean_v']:10.3e} | {r['min_y']:8.5f} | {r['max_y']:8.5f}")

    steps = [r['step'] for r in rows]
    for key, label in [('dx', 'particle displacement per step'), ('v', 'particle speed')]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(steps, [r[f'max_{key}'] for r in rows], 'o-', label='max')
        ax.semilogy(steps, [r[f'mean_{key}'] for r in rows], 's--', label='mean')
        ax.set_xlabel('settle step (after env.step #)'); ax.set_ylabel(label); ax.grid(True, which='both', alpha=.3)
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'settle_max_{key}.png'), dpi=100)
        plt.close(fig)
    print(f"csv + plots: {out_dir}")


def make_settled_state(out_dir, n_steps):
    """Reference material -> zero-action settle n_steps -> save env.get_state() (material saved separately)."""
    import json
    import pickle
    env, _ = make_env()
    clamp_y = 3 * env.simulator.dx   # g2p clamps particle x to [3dx, 1-3dx]
    x0 = env.simulator.get_x(0)
    settle(env, n_steps)
    settled_state = env.get_state()
    x = settled_state['state'][0]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'settled_initial_state.pkl'), 'wb') as f:
        pickle.dump(settled_state, f)
    with open(os.path.join(out_dir, 'settled_state_metadata.json'), 'w') as f:
        json.dump(dict(reference_E=REF_MATERIAL['E'], reference_sigma_y=REF_MATERIAL['yield_stress'],
                       reference_nu=REF_MATERIAL['nu'], settle_steps=n_steps, seed=SEED,
                       floor_clamp=dict(enabled=True, clamp_y=clamp_y,
                                        affected_particles=int((x0[:, 1] < clamp_y).sum()),
                                        particles_on_clamp_plane_after_settle=int((np.abs(x[:, 1] - clamp_y) < 1e-4).sum()),
                                        total_particles=len(x0)),
                       datetime=datetime.now().isoformat(), source_script=os.path.abspath(__file__)), f, indent=2)
    print(f"settled state: {out_dir}")


def simulate(out_dir):
    env, state = make_env()
    np.random.seed(SEED)

    # before any env.step(); primitive teleport does not touch particles
    episode_initial_object_cloud = read_object_cloud(env)
    assert episode_initial_object_cloud.shape == (N_OBS, 3), episode_initial_object_cloud.shape

    before_object_cloud, after_object_cloud = [], []
    before_tool_pose, grip_start_tool_pose = [], []
    object_rollout, tool_rollout, primitive_action_sequence = [], [], []
    grip_rate, grip_angle = [], []

    tool_size = 0.045
    for k in range(N_GRIPS):
        # previous grip finished (or episode reset) -> read object BEFORE teleport
        before_object_cloud.append(read_object_cloud(env))
        before_tool_pose.append(read_tool_pose(env))

        prim1, prim2, cur_angle = random_pose(task_name)
        update_primitive(env, prim1, prim2)   # teleport
        select_tool(env, tool_size)
        # after teleport, before first env.step() of this grip
        grip_start_tool_pose.append(read_tool_pose(env))

        gripper_rate_limit = [(task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][0] + 2 * tool_size)) / (2 * task_params['len_per_grip']),
                              (task_params['sample_radius'] * 2 - (task_params['gripper_rate_limits'][1] + 2 * tool_size)) / (2 * task_params['len_per_grip'])]
        rate = np.random.uniform(*gripper_rate_limit)
        actions = make_grip_actions(prim1, prim2, rate)
        assert actions.shape == (N_FRAMES, 12), actions.shape

        # Timing:
        #   before_object_cloud[k]  (state s_0, no step yet)
        #   -> action[0] -> env.step() -> object_rollout[k, 0]   (frame_in_grip=0, is_grip_start)
        #   -> ...
        #   -> action[39] -> env.step() -> object_rollout[k, 39] (frame_in_grip=39, is_grip_end)
        #   == after_object_cloud[k] == before_object_cloud[k+1]
        objs, tools = [], []
        for act in actions:
            env.step(act)
            objs.append(read_object_cloud(env))
            tools.append(read_tool_pose(env))

        # separate simulator read (not a copy of objs[-1]) so the equality check is meaningful
        after_object_cloud.append(read_object_cloud(env))

        object_rollout.append(np.stack(objs))
        tool_rollout.append(np.stack(tools))
        primitive_action_sequence.append(actions)
        grip_rate.append(rate)
        grip_angle.append(cur_angle)

    frame_in_grip = np.tile(np.arange(N_FRAMES), (N_GRIPS, 1))
    data = dict(
        episode_initial_object_cloud=episode_initial_object_cloud,
        before_object_cloud=np.stack(before_object_cloud),
        after_object_cloud=np.stack(after_object_cloud),
        before_tool_pose=np.stack(before_tool_pose),
        grip_start_tool_pose=np.stack(grip_start_tool_pose),
        object_rollout=np.stack(object_rollout),
        tool_rollout=np.stack(tool_rollout),
        primitive_action_sequence=np.stack(primitive_action_sequence),
        grip_id=np.arange(N_GRIPS),
        frame_in_grip=frame_in_grip,
        is_grip_start=frame_in_grip == 0,
        is_grip_end=frame_in_grip == N_FRAMES - 1,
        grip_rate=np.array(grip_rate),
        grip_angle=np.array(grip_angle),
        seed=np.array(SEED),
    )

    expected = dict(
        episode_initial_object_cloud=(N_OBS, 3),
        before_object_cloud=(N_GRIPS, N_OBS, 3),
        after_object_cloud=(N_GRIPS, N_OBS, 3),
        grip_start_tool_pose=(N_GRIPS, 2, 7),
        object_rollout=(N_GRIPS, N_FRAMES, N_OBS, 3),
        tool_rollout=(N_GRIPS, N_FRAMES, 2, 7),
        primitive_action_sequence=(N_GRIPS, N_FRAMES, 12),
        grip_id=(N_GRIPS,),
        frame_in_grip=(N_GRIPS, N_FRAMES),
        is_grip_start=(N_GRIPS, N_FRAMES),
        is_grip_end=(N_GRIPS, N_FRAMES),
    )
    for key, shape in expected.items():
        assert data[key].shape == shape, (key, data[key].shape, shape)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'debug_episode_000.npz')
    np.savez_compressed(path, **data)
    return path


def report(data):
    maxerr = lambda a, b: float(np.max(np.abs(a - b)))
    print(f"episode initial cloud shape: {data['episode_initial_object_cloud'].shape}")
    for k in range(len(data['grip_id'])):
        print(f"Grip {k}: before {data['before_object_cloud'][k].shape}  "
              f"rollout {data['object_rollout'][k].shape}  after {data['after_object_cloud'][k].shape}  "
              f"rate {data['grip_rate'][k]:.5f}  angle {data['grip_angle'][k]:.4f}")
    print(f"initial vs grip0 before:      {maxerr(data['episode_initial_object_cloud'], data['before_object_cloud'][0]):.3e}")
    for k in range(len(data['grip_id'])):
        print(f"grip{k} rollout[-1] vs after:   {maxerr(data['object_rollout'][k, -1], data['after_object_cloud'][k]):.3e}")
        if k + 1 < len(data['grip_id']):
            print(f"grip{k} after vs grip{k+1} before: {maxerr(data['after_object_cloud'][k], data['before_object_cloud'][k+1]):.3e}")
    # sanity: the grip actually moves the object
    print(f"grip0 before vs rollout[0] (1 step): {maxerr(data['before_object_cloud'][0], data['object_rollout'][0, 0]):.3e}")
    print(f"grip0 before vs after:              {maxerr(data['before_object_cloud'][0], data['after_object_cloud'][0]):.3e}")
    floats = [v for v in data.values() if np.issubdtype(v.dtype, np.floating)]
    print(f"NaN count: {sum(int(np.isnan(v).sum()) for v in floats)}  Inf count: {sum(int(np.isinf(v).sum()) for v in floats)}")


def visualize(data, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # (name, object cloud, finger pose (2,7))
    states = [('episode_initial', data['episode_initial_object_cloud'], data['before_tool_pose'][0])]
    for k in range(len(data['grip_id'])):
        states += [
            (f'grip{k}_before', data['before_object_cloud'][k], data['before_tool_pose'][k]),
            (f'grip{k}_start_teleported', data['before_object_cloud'][k], data['grip_start_tool_pose'][k]),
            (f'grip{k}_middle_f{MID_FRAME}', data['object_rollout'][k, MID_FRAME], data['tool_rollout'][k, MID_FRAME]),
            (f'grip{k}_after', data['after_object_cloud'][k], data['tool_rollout'][k, -1]),
        ]

    for idx, (name, cloud, tools) in enumerate(states):
        fig = plt.figure(figsize=(10, 5))
        # sim is y-up: plot (x, z) horizontal, y vertical
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(cloud[:, 0], cloud[:, 2], cloud[:, 1], s=4, c='tab:blue')
        ax.scatter(tools[:, 0], tools[:, 2], tools[:, 1], s=120, c=['tab:red', 'tab:orange'], marker='s')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 0.5)
        ax.set_xlabel('x'); ax.set_ylabel('z'); ax.set_zlabel('y (up)')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(cloud[:, 0], cloud[:, 2], s=4, c='tab:blue')
        ax2.scatter(tools[:, 0], tools[:, 2], s=120, c=['tab:red', 'tab:orange'], marker='s')
        ax2.add_patch(plt.Circle(tuple(tools[0, [0, 2]]), 0.045, fill=False, color='tab:red'))
        ax2.add_patch(plt.Circle(tuple(tools[1, [0, 2]]), 0.045, fill=False, color='tab:orange'))
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect('equal')
        ax2.set_xlabel('x'); ax2.set_ylabel('z'); ax2.set_title('top view')
        gap = np.linalg.norm(tools[0, :3] - tools[1, :3])
        fig.suptitle(f'{name}   finger center gap={gap:.3f}')
        fig.savefig(os.path.join(out_dir, f'{idx:02d}_{name}.png'), dpi=100)
        plt.close(fig)
    print(f"plots: {out_dir}/*.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', metavar='NPZ')
    parser.add_argument('--settle-diagnostic', action='store_true')
    parser.add_argument('--make-settled-state', type=int, metavar='N_STEPS')
    args = parser.parse_args()
    stamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S")
    dataset_dir = os.path.join(HERE, '..', '..', 'dataset')

    if args.settle_diagnostic:
        settle_diagnostic(os.path.join(dataset_dir, f'settle_diagnostic_{stamp}'))
        sys.exit()
    if args.make_settled_state is not None:
        make_settled_state(os.path.join(dataset_dir, f'settled_state_N{args.make_settled_state}_{stamp}'),
                           args.make_settled_state)
        sys.exit()

    if args.viz:
        npz_path = os.path.abspath(args.viz)
    else:
        stamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S")
        npz_path = simulate(os.path.join(HERE, '..', '..', 'dataset', f'debug_multigrip_{stamp}'))
    data = dict(np.load(npz_path))
    for key, v in data.items():
        print(f"{key:30s} {str(v.shape):20s} {v.dtype}")
    report(data)
    visualize(data, os.path.dirname(npz_path))
