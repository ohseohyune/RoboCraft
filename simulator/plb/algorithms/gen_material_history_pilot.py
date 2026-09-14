"""TEMPORARY PAIRED PILOT: same 5-grip action sequence x 4 materials, no reset between grips.

Question: with identical actions, do object point-cloud state sequences diverge by material?
E=5000, nu=0.2 fixed; sigma_y in {30, 200, 500, 900}; 3 fixed-seed sequences x 5 grips (30 close + 10 retreat).
Reuses gen_multigrip_debug (env/actions), material_sensitivity_debug (restore/disp/state_error),
render_material_grid (camera/render/encode), stress_test_grip (gap -> rate, pin radius). Simulator/yaml untouched.

Run: python gen_material_history_pilot.py --settled-state <dir>/settled_initial_state.pkl
"""
import argparse
import itertools
import json
import os
import pickle
import shutil
from datetime import datetime

import cv2
import numpy as np

import gen_multigrip_debug as g
import material_sensitivity_debug as msd
import render_material_grid as rmg
import stress_test_grip as st

E = 5000.0
SIGMAS = [30, 200, 500, 900]
N_SEQ, N_GRIPS = 3, 5
SEQ_SEED = 2026
GAP_RANGE = sorted(g.task_params['gripper_rate_limits'].tolist())   # notebook rate limits expressed as surface gap [0.06, 0.14]
PAIRS = [(30, 200), (200, 500), (500, 900), (30, 500), (30, 900), (200, 900)]
COLLAPSE_HEIGHT_RATIO = 0.5   # after max_y < ratio * initial max_y
NF = g.N_FRAMES               # 40


def make_sequence(s, path):
    np.random.seed(SEQ_SEED + s)
    poses, mids, angles, gaps, rates, actions = [], [], [], [], [], []
    for _ in range(N_GRIPS):
        prim1, prim2, angle = g.random_pose(g.task_name)
        gap = np.random.uniform(*GAP_RANGE)
        rate = st.gap_to_rate(gap)
        poses.append(np.stack([prim1, prim2]).astype(np.float64))
        mids.append((prim1[:3] + prim2[:3]) / 2)
        angles.append(angle); gaps.append(gap); rates.append(rate)
        actions.append(g.make_grip_actions(prim1, prim2, rate))
    np.savez_compressed(path, midpoint=np.stack(mids), angle=np.array(angles), target_gap=np.array(gaps),
                        gripper_rate=np.array(rates), grip_start_tool_pose=np.stack(poses),
                        primitive_action_sequence=np.stack(actions), seed=np.array(SEQ_SEED + s))


def run_episode(env, settled, sigma, seq, frame_dir=None):
    mat = dict(E=E, yield_stress=sigma)
    msd.restore(env, settled, mat)
    init_err = msd.state_error(env.get_state(), settled)
    assert all(v == 0 for v in init_err.values()), init_err
    material = msd.applied_material(env, mat)
    assert material['match'], material
    render = frame_dir is not None
    imgs = 0

    def save_img():
        nonlocal imgs
        # jpg: 12 x 201 png frames (~1.3GB) filled the disk
        cv2.imwrite(os.path.join(frame_dir, f'{imgs:03d}.jpg'), rmg.render(env)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        imgs += 1

    initial_cloud = g.read_object_cloud(env)
    prev_cloud, prev_state = initial_cloud, env.get_state()
    pin_xz = env.primitives.primitives[2].get_state(0)[[0, 2]]
    if render:
        save_img()
    keys = ['before', 'after', 'finger_before', 'finger_start', 'finger_after', 'object_rollout', 'tool_rollout',
            'finger_clearance', 'max_step_disp', 'finger_pin_gap']
    r = {k: [] for k in keys}
    boundary = []
    for k in range(N_GRIPS):
        # separate simulator reads (after the previous grip's render) so continuity checks are meaningful
        state_before, before = env.get_state(), g.read_object_cloud(env)
        boundary.append(dict(object=float(np.abs(before - prev_cloud).max()), **msd.state_error(state_before, prev_state)))
        r['before'].append(before); r['finger_before'].append(g.read_tool_pose(env))
        pose = seq['grip_start_tool_pose'][k]
        g.update_primitive(env, pose[0], pose[1])
        g.select_tool(env, msd.TOOL_SIZE)
        assert np.array_equal(g.read_tool_pose(env), pose)
        r['finger_start'].append(g.read_tool_pose(env))
        objs, tools, clears, steps = [], [], [], []
        x_prev = env.simulator.get_x(0)
        # before[k] -> action[0] -> env.step -> rollout[k, 0] ... rollout[k, 39] == after[k] == before[k+1]
        for i, act in enumerate(seq['primitive_action_sequence'][k]):
            env.step(act)   # ValueError on NaN
            x = env.simulator.get_x(0)
            steps.append(float(np.linalg.norm(x - x_prev, axis=1).max())); x_prev = x
            objs.append(g.read_object_cloud(env)); tools.append(g.read_tool_pose(env))
            clears.append(g.capsule_clearance(env, x)[:2])
            if render and i < NF - 1:
                save_img()
        prev_cloud, prev_state = g.read_object_cloud(env), env.get_state()
        if render:
            save_img()   # image index 40*(k+1) = P_{k+1}
        tools = np.stack(tools)
        r['after'].append(prev_cloud); r['finger_after'].append(g.read_tool_pose(env))
        r['object_rollout'].append(np.stack(objs)); r['tool_rollout'].append(tools)
        r['finger_clearance'].append(np.array(clears)); r['max_step_disp'].append(np.array(steps))
        # horizontal finger surface to pin surface distance (vertical capsules)
        r['finger_pin_gap'].append(np.linalg.norm(tools[:, :, [0, 2]] - pin_xz, axis=-1) - msd.TOOL_SIZE - st.PIN_R)
    d = {k: np.stack(v) for k, v in r.items()}
    t29 = d['tool_rollout'][:, 29]
    data = dict(
        episode_initial_object_cloud=initial_cloud,
        before_object_cloud=d['before'], primitive_action_sequence=seq['primitive_action_sequence'], after_object_cloud=d['after'],
        frame25_object_cloud=d['object_rollout'][:, 25], frame29_object_cloud=d['object_rollout'][:, 29],
        frame39_object_cloud=d['object_rollout'][:, 39],
        object_rollout=d['object_rollout'], tool_rollout=d['tool_rollout'],
        finger_pose_before=d['finger_before'], finger_pose_start=d['finger_start'], finger_pose_after=d['finger_after'],
        finger_clearance=d['finger_clearance'], max_step_disp_all_particles=d['max_step_disp'], finger_pin_gap=d['finger_pin_gap'],
        grip_id=np.arange(N_GRIPS), sequence_id=np.array(int(seq['sequence_id'])), material_id=np.array(SIGMAS.index(sigma)),
        sigma_y=np.array(float(sigma)), E=np.array(E), nu=np.array(msd.NU),
        midpoint=seq['midpoint'], angle=seq['angle'], target_gap=seq['target_gap'], gripper_rate=seq['gripper_rate'],
        actual_gap_f29=np.linalg.norm(t29[:, 0, :3] - t29[:, 1, :3], axis=-1) - 2 * msd.TOOL_SIZE,
        # env.step count: before grip k = NF*k; frame f of grip k = NF*k + f + 1
        episode_step_before=NF * np.arange(N_GRIPS), diag_frame_in_grip=np.array([25, 29, 39]),
        episode_step_after=NF * np.arange(N_GRIPS) + NF,
    )
    return data, dict(initial_state_error=init_err, material=material, boundary=boundary)


def grip_metrics(d):
    out = []
    init_h = d['episode_initial_object_cloud'][:, 1].max()
    for k in range(N_GRIPS):
        b = d['before_object_cloud'][k]
        comp, res = msd.disp(d['frame29_object_cloud'][k], b), msd.disp(d['frame39_object_cloud'][k], b)
        roll = d['object_rollout'][k]
        nan = int(np.isnan(roll).sum() + np.isnan(d['tool_rollout'][k]).sum())
        inf = int(np.isinf(roll).sum() + np.isinf(d['tool_rollout'][k]).sum())
        max_step = float(d['max_step_disp_all_particles'][k].max())
        out.append(dict(compression=comp, residual=res, normalized_residual=res['rms'] / comp['rms'],
                        recovery_ratio=1 - res['rms'] / comp['rms'], actual_gap_f29=float(d['actual_gap_f29'][k]),
                        max_step_disp=max_step, min_finger_clearance=float(d['finger_clearance'][k].min()),
                        min_finger_pin_gap=float(d['finger_pin_gap'][k].min()),
                        bbox_min=roll.min(axis=(0, 1)).tolist(), bbox_max=roll.max(axis=(0, 1)).tolist(),
                        after_max_y=float(d['after_object_cloud'][k][:, 1].max()), nan=nan, inf=inf,
                        explosion=bool(max_step > msd.EXPLOSION_STEP_DISP or roll.min() < 0 or roll.max() > 1),
                        collapse=bool(d['after_object_cloud'][k][:, 1].max() < COLLAPSE_HEIGHT_RATIO * init_h)))
    return out


def label(img, lines):
    for j, text in enumerate(lines):
        for color, thick in [((0, 0, 0), 5), ((255, 255, 255), 2)]:
            cv2.putText(img, text, (12, 34 + 34 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)
    return img


def visualize(out_dir, s, pair_rms):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vis = os.path.join(out_dir, 'visualizations')
    grid_dir = os.path.join(vis, f'sequence_{s:02d}_grid_frames')
    os.makedirs(grid_dir, exist_ok=True)
    src = lambda sy, i: cv2.imread(os.path.join(out_dir, f'sequence_{s:02d}', f'sigma_{sy}', 'frames', f'{i:03d}.jpg'))
    for i in range(N_GRIPS * NF + 1):
        k, f = (i - 1) // NF, (i - 1) % NF
        stage = 'P0 (initial)' if i == 0 else f'grip{k + 1} frame {f}'
        cv2.imwrite(os.path.join(grid_dir, f'{i:03d}.png'),
                    np.hstack([label(src(sy, i), [f'seq {s}  sigma_y={sy}', stage]) for sy in SIGMAS]))
    rmg.encode(grid_dir, os.path.join(vis, f'sequence_{s:02d}_grid.mp4'))
    shutil.rmtree(grid_dir)   # ~450MB of png per sequence
    rows = [np.hstack([cv2.resize(label(src(sy, NF * p), [f'sigma_y={sy}', f'P{p}']), (320, 320)) for p in range(N_GRIPS + 1)])
            for sy in SIGMAS]
    cv2.imwrite(os.path.join(vis, f'sequence_{s:02d}_state_grid.png'), np.vstack(rows))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, N_GRIPS + 1)
    for (a, b), ls in zip(PAIRS, ['-'] * 3 + ['--'] * 3):
        ax.plot(x, pair_rms[f'{a}vs{b}'], ls, marker='o', label=f'{a} vs {b}')
    ax.set_xticks(x); ax.set_xlabel('grip index (P_i after grip i)'); ax.set_ylabel('object cloud RMS difference (after / frame39)')
    ax.set_title(f'sequence {s:02d}: pairwise material separation'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(vis, f'sequence_{s:02d}_pairwise_rms.png'), dpi=110); plt.close(fig)


def write_report(summary, out_dir):
    L = []
    c = summary['conditions']
    L += ['[Conditions]', '', f"E: {c['E']:g}", f"nu: {c['nu']}", f"sigma_y: {c['sigma_y']}", f"num_sequences: {c['num_sequences']}",
          f"grips_per_episode: {c['grips_per_episode']}", f"total_episodes: {c['total_episodes']}", f"total_grips: {c['total_grips']}",
          f"settled_state: {c['settled_state']}", f"target_gap sampled uniform in {GAP_RANGE}, sequence seed {SEQ_SEED}+s", '']
    L += ['-' * 32, '[Sequence definitions]', '']
    for s, sd in summary['sequences'].items():
        L.append(f'sequence {s}:')
        for k, gr in enumerate(sd['grips']):
            L.append(f"  grip{k} midpoint {np.round(gr['midpoint'], 4).tolist()} angle {gr['angle']:.4f} rad "
                     f"({np.degrees(gr['angle']):.1f} deg) target_gap {gr['target_gap']:.4f} rate {gr['gripper_rate']:.6f} "
                     f"start f0 {np.round(gr['start_pose'][0][:3], 4).tolist()} f1 {np.round(gr['start_pose'][1][:3], 4).tolist()}")
        L.append('')
    ch = summary['checks']
    L += ['-' * 32, '[Identity checks]', '', f"action max error across materials (per sequence): {ch['action_max_error']}",
          f"start pose max error across materials: {ch['start_pose_max_error']}",
          f"initial 300-cloud max error across all episodes: {ch['initial_cloud_max_error']}",
          f"initial full state vs settled (x,v,F,C,finger0,finger1,pin) max: {ch['initial_state_error_max']}",
          f"determinism repeat (seq0 sigma200, no render) after-cloud max error: {ch['determinism_after_max_error']}", '']
    L += ['-' * 32, '[Boundary validation]', '']
    for name, ep in summary['episodes'].items():
        if 'failed' in ep:
            L.append(f'{name}: FAILED {ep["failed"]}'); continue
        b = ep['boundary']
        tags = ['initial == grip0_before'] + [f'grip{k - 1}_after == grip{k}_before' for k in range(1, N_GRIPS)]
        L.append(f'{name}: ' + ' | '.join(f"{t}: obj {x['object']:.1e} full {max(v for kk, v in x.items() if kk != 'object'):.1e}"
                                          for t, x in zip(tags, b)))
    L += [f"max error (object / full state): {ch['boundary_max_object']:.1e} / {ch['boundary_max_full_state']:.1e}", '']
    L += ['-' * 32, '[Per-grip material response]', '']
    for s in range(N_SEQ):
        L.append(f'sequence {s}:')
        L.append(f"{'grip':>4} | {'sigma_y':>7} | {'compression_RMS':>15} | {'residual_RMS':>12} | {'norm_residual':>13} | {'recovery':>8} | {'gap_f29':>8}")
        for k in range(N_GRIPS):
            for sy in SIGMAS:
                ep = summary['episodes'].get(f'seq{s}_sigma{sy}', {})
                if 'grips' not in ep:
                    continue
                m = ep['grips'][k]
                L.append(f"{k + 1:>4} | {sy:>7} | {m['compression']['rms']:>15.4e} | {m['residual']['rms']:>12.4e} | "
                         f"{m['normalized_residual']:>13.3f} | {m['recovery_ratio']:>+8.3f} | {m['actual_gap_f29']:>+8.4f}")
        L.append('')
    pn = [f'{a}vs{b}' for a, b in PAIRS]
    L += ['-' * 32, '[Pairwise material separation]  after(frame39) cloud RMS (mean / max in summary.json)', '']
    for s in range(N_SEQ):
        L.append(f'sequence {s}:')
        L.append(f"{'grip':>4} | " + ' | '.join(f'{p:>10}' for p in pn))
        for k in range(N_GRIPS):
            L.append(f"{k + 1:>4} | " + ' | '.join(f"{summary['pairwise'][str(s)][p][k]['rms']:>10.3e}" for p in pn))
        L.append('')
    L += ['-' * 32, '[History trend]  RMS g1 -> g5  (ratio g5/g1)', '']
    for p in pn:
        L.append(f'{p}:')
        for s in range(N_SEQ):
            v = [x['rms'] for x in summary['pairwise'][str(s)][p]]
            L.append(f'  sequence {s}: ' + ' -> '.join(f'{x:.2e}' for x in v) + f'   (x{v[-1] / v[0]:.2f})')
        L.append('')
    L += ['-' * 32, '[Visual paths]', '']
    for key in ('grid.mp4', 'state_grid.png', 'pairwise_rms.png'):
        L += [os.path.join(out_dir, 'visualizations', f'sequence_{s:02d}_{key}') for s in range(N_SEQ)]
    L.append('')
    stab = summary['stability']
    L += ['-' * 32, '[Stability]', '', f"NaN: {stab['nan']}", f"Inf: {stab['inf']}", f"ValueError episodes: {stab['failed']}",
          f"explosion (max per-step particle disp > {msd.EXPLOSION_STEP_DISP} or out of [0,1]): {stab['explosion']}  max per-step disp {stab['max_step_disp']:.4e}",
          f"collapse (after max_y < {COLLAPSE_HEIGHT_RATIO} x initial): {stab['collapse']}",
          f"clearance: min finger clearance {stab['min_finger_clearance']:+.4f}",
          f"finger-pin: min surface distance {stab['min_finger_pin_gap']:+.4f}",
          f"bbox: {np.round(stab['bbox_min'], 4).tolist()} .. {np.round(stab['bbox_max'], 4).tolist()}", '']
    L += ['-' * 32, '[Interpretation]', '', '(to be written after review of the numbers above)', '',
          '-' * 32, '[Recommendation]', '', '(to be written after review of the numbers above)', '']
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write('\n'.join(L))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settled-state', required=True)
    args = ap.parse_args()
    args.settled_state = os.path.abspath(args.settled_state)   # make_env() chdirs
    with open(args.settled_state, 'rb') as f:
        settled = pickle.load(f)
    out_dir = os.path.abspath(os.path.join(g.HERE, '..', '..', 'dataset',
                                           'material_history_pilot_' + datetime.now().strftime("%d-%b-%Y-%H:%M:%S")))
    os.makedirs(os.path.join(out_dir, 'sequences'), exist_ok=True)

    seqs = []
    for s in range(N_SEQ):
        path = os.path.join(out_dir, 'sequences', f'paired_sequence_{s:02d}.npz')
        make_sequence(s, path)
        seqs.append(dict(np.load(path), sequence_id=s))   # all episodes read the saved file, never regenerate

    env, _ = g.make_env()
    rmg.update_camera(env)
    env.number_of_cams = 1

    summary = dict(note='TEMPORARY PAIRED PILOT', episodes={}, sequences={}, pairwise={},
                   conditions=dict(E=E, nu=msd.NU, sigma_y=SIGMAS, num_sequences=N_SEQ, grips_per_episode=N_GRIPS,
                                   total_episodes=N_SEQ * len(SIGMAS), total_grips=N_SEQ * len(SIGMAS) * N_GRIPS,
                                   settled_state=os.path.abspath(args.settled_state), seq_seed=SEQ_SEED, gap_range=GAP_RANGE))
    datas = {}
    for s, seq in enumerate(seqs):
        summary['sequences'][s] = dict(grips=[dict(midpoint=seq['midpoint'][k].tolist(), angle=float(seq['angle'][k]),
                                                   target_gap=float(seq['target_gap'][k]), gripper_rate=float(seq['gripper_rate'][k]),
                                                   start_pose=seq['grip_start_tool_pose'][k].tolist()) for k in range(N_GRIPS)])
        for sy in SIGMAS:
            name = f'seq{s}_sigma{sy}'
            ep_dir = os.path.join(out_dir, f'sequence_{s:02d}', f'sigma_{sy}')
            os.makedirs(os.path.join(ep_dir, 'frames'), exist_ok=True)
            print(name, flush=True)
            try:
                data, info = run_episode(env, settled, sy, seq, os.path.join(ep_dir, 'frames'))
            except ValueError as e:
                summary['episodes'][name] = dict(failed=f'ValueError: {e}')
                continue
            np.savez_compressed(os.path.join(ep_dir, 'episode.npz'), **data)
            datas[(s, sy)] = data
            summary['episodes'][name] = dict(**info, grips=grip_metrics(data))
            # boundary mismatch aborts the pilot
            assert all(v == 0 for b in info['boundary'] for v in b.values()), (name, info['boundary'])

    # ---- identity / continuity checks ----
    ok = list(datas)
    act_err = {s: max(float(np.abs(datas[k]['primitive_action_sequence'] - seqs[s]['primitive_action_sequence']).max())
                      for k in ok if k[0] == s) for s in range(N_SEQ)}
    assert all(v == 0 for v in act_err.values()), act_err
    pose_err = max(float(np.abs(datas[k]['finger_pose_start'] - seqs[k[0]]['grip_start_tool_pose']).max()) for k in ok)
    assert pose_err == 0, pose_err
    init_err = max(float(np.abs(datas[k]['episode_initial_object_cloud'] - datas[ok[0]]['episode_initial_object_cloud']).max()) for k in ok)
    assert init_err == 0, init_err
    eps = [e for e in summary['episodes'].values() if 'failed' not in e]
    repeat, _ = run_episode(env, settled, 200, seqs[0])
    summary['checks'] = dict(
        action_max_error=act_err, start_pose_max_error=pose_err, initial_cloud_max_error=init_err,
        initial_state_error_max={n: max(e['initial_state_error'][n] for e in eps) for n in eps[0]['initial_state_error']},
        determinism_after_max_error=float(np.abs(repeat['after_object_cloud'] - datas[(0, 200)]['after_object_cloud']).max()),
        boundary_max_object=max(b['object'] for e in eps for b in e['boundary']),
        boundary_max_full_state=max(v for e in eps for b in e['boundary'] for kk, v in b.items() if kk != 'object'))

    # ---- pairwise material separation, same particle index ----
    for s in range(N_SEQ):
        summary['pairwise'][str(s)] = {f'{a}vs{b}': [msd.disp(datas[(s, a)]['after_object_cloud'][k], datas[(s, b)]['after_object_cloud'][k])
                                                     for k in range(N_GRIPS)]
                                       for a, b in PAIRS if (s, a) in datas and (s, b) in datas}
        if all((s, sy) in datas for sy in SIGMAS):
            visualize(out_dir, s, {p: [x['rms'] for x in v] for p, v in summary['pairwise'][str(s)].items()})

    gm = [m for e in eps for m in e['grips']]
    summary['stability'] = dict(nan=sum(m['nan'] for m in gm), inf=sum(m['inf'] for m in gm),
                                failed=[n for n, e in summary['episodes'].items() if 'failed' in e],
                                explosion=any(m['explosion'] for m in gm), collapse=any(m['collapse'] for m in gm),
                                max_step_disp=max(m['max_step_disp'] for m in gm),
                                min_finger_clearance=min(m['min_finger_clearance'] for m in gm),
                                min_finger_pin_gap=min(m['min_finger_pin_gap'] for m in gm),
                                bbox_min=np.min([m['bbox_min'] for m in gm], axis=0).tolist(),
                                bbox_max=np.max([m['bbox_max'] for m in gm], axis=0).tolist())
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(write_report(summary, out_dir))
    print(f'outputs: {out_dir}')


if __name__ == '__main__':
    main()
