"""STRESS-TEST DIAGNOSTIC (not main dataset config): Grip 0 only, deeper compression.

Only material and target surface gap change. Midpoint / angle / finger start pose come from the saved
material_debug_action_sequence.npz grip 0; 30 close + 10 retreat unchanged; camera/rendering = render_material_grid.

target_surface_gap -> gripper_rate uses the notebook's own gripper_rate_limits formula:
    rate = (2 * sample_radius - (gap + 2 * tool_size)) / (2 * len_per_grip)

Run: python stress_test_grip.py --settled-state <pkl> --action-seq <npz> [--E-values ..] [--yield-values ..] [--gaps ..]
Grid video: rows = E, columns = (sigma_y, gap) combinations.
"""
import argparse
import json
import os
import pickle
from datetime import datetime

import cv2
import numpy as np

import gen_multigrip_debug as g
import material_sensitivity_debug as msd
import render_material_grid as rmg

RELAX_STEPS = 50
PIN_R = 0.025   # gripper_fixed.yml pin capsule radius


def gap_to_rate(gap):
    tp, r = g.task_params, msd.TOOL_SIZE
    return (tp['sample_radius'] * 2 - (gap + 2 * r)) / (2 * tp['len_per_grip'])


def run(env, settled, mat, pose, gap):
    rate = gap_to_rate(gap)
    actions = g.make_grip_actions(pose[0], pose[1], rate)
    msd.restore(env, settled, mat)
    assert all(v == 0 for v in msd.state_error(env.get_state(), settled).values())
    applied = msd.applied_material(env, mat)
    before = g.read_object_cloud(env)
    imgs = [rmg.render(env)]
    g.update_primitive(env, pose[0], pose[1])
    g.select_tool(env, msd.TOOL_SIZE)
    start_pose = g.read_tool_pose(env)
    assert np.array_equal(start_pose, pose)
    pin_xz = env.primitives.primitives[2].get_state(0)[[0, 2]]
    objs, tools, clears = [], [], []
    for act in actions:
        env.step(act)
        objs.append(g.read_object_cloud(env))
        tools.append(g.read_tool_pose(env))
        clears.append(g.capsule_clearance(env, env.simulator.get_x(0))[:2])
        imgs.append(rmg.render(env))
    relax = msd.relax_branch(env, RELAX_STEPS)   # diagnostic only, after command end
    tools = np.stack(tools)
    data = dict(E=np.array(mat['E']), yield_stress=np.array(mat['yield_stress']), nu=np.array(msd.NU),
                target_surface_gap=np.array(gap), gripper_rate=np.array(rate),
                grip_start_tool_pose=start_pose, primitive_action_sequence=actions,
                before_object_cloud=before, object_rollout=np.stack(objs), tool_rollout=tools,
                finger_clearance=np.array(clears),
                frame25=objs[25], frame29=objs[29], frame39_after_command=objs[39],
                # horizontal finger surface to pin surface distance (both capsules vertical, y ranges overlap)
                finger_pin_gap=np.linalg.norm(tools[:, :, [0, 2]] - pin_xz, axis=-1) - msd.TOOL_SIZE - PIN_R,
                **relax)
    return data, imgs, applied


def metrics(d):
    b, roll = d['before_object_cloud'], d['object_rollout']
    dfm = {k: msd.disp(roll[f], b) for k, f in (('f25', 25), ('f29', 29), ('f39', 39))}
    dfm['relax50'] = msd.disp(d['relax_object_cloud'][-1], b)
    seq = np.concatenate([b[None], roll])
    t29 = d['tool_rollout'][29]
    return dict(deformation=dfm,
                pre_contact_f15=msd.disp(roll[msd.PRE_CONTACT_FRAME], b),
                normalized_residual=dfm['f39']['rms'] / dfm['f29']['rms'],
                recovery=dfm['f29']['rms'] - dfm['f39']['rms'],
                recovery_ratio=(dfm['f29']['rms'] - dfm['f39']['rms']) / dfm['f29']['rms'],
                actual_surface_gap_f29=float(np.linalg.norm(t29[0, :3] - t29[1, :3]) - 2 * msd.TOOL_SIZE),
                min_finger_clearance=float(d['finger_clearance'].min()),
                min_finger_pin_gap=float(d['finger_pin_gap'].min()),
                max_frame_step_disp=float(max(np.linalg.norm(np.diff(seq, axis=0), axis=-1).max(), d['relax_stats'][:, 0].max())),
                nan=int(sum(np.isnan(v).sum() for v in d.values() if v.dtype.kind == 'f')),
                inf=int(sum(np.isinf(v).sum() for v in d.values() if v.dtype.kind == 'f')),
                bbox_min=roll.min(axis=(0, 1)).tolist(), bbox_max=roll.max(axis=(0, 1)).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settled-state', required=True)
    ap.add_argument('--action-seq', required=True)
    ap.add_argument('--E-values', type=float, nargs='+', default=[2500, 5000])
    ap.add_argument('--yield-values', type=float, nargs='+', default=[3200])
    ap.add_argument('--gaps', type=float, nargs='+', default=[0.06, 0.03, 0.00])
    args = ap.parse_args()
    cols = [(sy, gap) for sy in args.yield_values for gap in args.gaps]
    with open(args.settled_state, 'rb') as f:
        settled = pickle.load(f)
    pose = dict(np.load(args.action_seq))['grip_start_tool_pose'][0]

    out_dir = os.path.abspath(os.path.join(g.HERE, '..', '..', 'dataset', 'stress_test_' + datetime.now().strftime("%d-%b-%Y-%H:%M:%S")))
    os.makedirs(out_dir, exist_ok=True)
    env, _ = g.make_env()
    rmg.update_camera(env)
    env.number_of_cams = 1

    frames, summary = {}, dict(note='STRESS-TEST DIAGNOSTIC', settled_state=args.settled_state, action_seq=args.action_seq, runs={})
    for E in args.E_values:
        for sy, gap in cols:
            mat = dict(E=E, yield_stress=sy)
            name = f"E{E:g}_sy{sy:g}_gap{gap:.2f}"
            print(name, flush=True)
            try:
                data, imgs, applied = run(env, settled, mat, pose, gap)
            except ValueError:
                summary['runs'][name] = dict(failed='ValueError (NaN in env.step)')
                continue
            np.savez_compressed(os.path.join(out_dir, f'{name}.npz'), **data)
            frame_dir = os.path.join(out_dir, name, 'frames')
            os.makedirs(frame_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(frame_dir, f'{i:03d}.png'), img[..., ::-1])
            rmg.encode(frame_dir, os.path.join(out_dir, f'{name}.mp4'))
            frames[(E, sy, gap)] = imgs
            summary['runs'][name] = dict(gripper_rate=float(data['gripper_rate']), target_surface_gap=gap,
                                         material=applied, **metrics(data))

    # identical action except rate: start pose shared, and action direction identical
    names = [n for n in summary['runs'] if 'failed' not in summary['runs'][n]]
    acts = {n: np.load(os.path.join(out_dir, f'{n}.npz'))['primitive_action_sequence'] for n in names}
    summary['check'] = dict(
        same_gap_action_diff_across_materials=max(float(np.abs(acts[a] - acts[b]).max())
                                                   for a in names for b in names if a.split('_gap')[1] == b.split('_gap')[1]),
        action_direction_diff=max(float(np.abs(acts[n] / np.abs(acts[n]).max() - acts[names[0]] / np.abs(acts[names[0]]).max()).max()) for n in names))

    # one grid per gap: rows = E, columns = sigma_y
    n_frames = len(next(iter(frames.values())))
    blank = np.zeros_like(next(iter(frames.values()))[0])
    for gap in args.gaps:
        grid_dir = os.path.join(out_dir, f'grid_frames_gap{gap:.2f}')
        os.makedirs(grid_dir, exist_ok=True)
        for i in range(n_frames):
            rows = []
            for E in args.E_values:
                tiles = []
                for sy in args.yield_values:
                    tile = np.ascontiguousarray((frames[(E, sy, gap)][i] if (E, sy, gap) in frames else blank)[..., ::-1])
                    for j, text in enumerate([f"E={E:g}", f"sigma_y={sy:g}", f"gap={gap:.2f}"]):
                        for color, thick in [((0, 0, 0), 5), ((255, 255, 255), 2)]:
                            cv2.putText(tile, text, (12, 34 + 34 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)
                    tiles.append(tile)
                rows.append(np.hstack(tiles))
            cv2.imwrite(os.path.join(grid_dir, f'{i:03d}.png'), np.vstack(rows))
        grid_name = f'grid_{len(args.E_values)}x{len(args.yield_values)}_gap{gap:.2f}'
        rmg.encode(grid_dir, os.path.join(out_dir, f'{grid_name}.mp4'))
        # last video frame = state after frame 39 (after_command)
        cv2.imwrite(os.path.join(out_dir, f'{grid_name}_frame39.png'), cv2.imread(os.path.join(grid_dir, f'{n_frames - 1:03d}.png')))

    # per-E, per-gap: frame39 cross-sigma_y differences + final-shape overlays
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary['cross_f39'] = {}
    final = ('frame39', lambda d: (d['frame39_after_command'], d['tool_rollout'][39]))
    for E in args.E_values:
        for gap in args.gaps:
            ns = [f"E{E:g}_sy{sy:g}_gap{gap:.2f}" for sy in args.yield_values]
            ns = [n for n in ns if n in names]
            ds = {n: dict(np.load(os.path.join(out_dir, f'{n}.npz'))) for n in ns}
            for i, a in enumerate(ns):
                for b in ns[i + 1:]:
                    summary['cross_f39'][f'{a} vs {b}'] = msd.disp(ds[a]['frame39_after_command'], ds[b]['frame39_after_command'])
            if len(ns) > 1:
                msd.overlay(plt, ns, [ds[n] for n in ns], [final], os.path.join(out_dir, f'final_overlay_E{E:g}_gap{gap:.2f}.png'))
    for sy in args.yield_values:
        for gap in args.gaps:
            ns = [n for n in (f"E{E:g}_sy{sy:g}_gap{gap:.2f}" for E in args.E_values) if n in names]
            ds = {n: dict(np.load(os.path.join(out_dir, f'{n}.npz'))) for n in ns}
            for i, a in enumerate(ns):
                for b in ns[i + 1:]:
                    summary['cross_f39'][f'{a} vs {b}'] = msd.disp(ds[a]['frame39_after_command'], ds[b]['frame39_after_command'])
            if len(ns) > 1:
                msd.overlay(plt, ns, [ds[n] for n in ns], [final], os.path.join(out_dir, f'final_overlay_sy{sy:g}_gap{gap:.2f}.png'))

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary['check']))
    for n, r in summary['runs'].items():
        if 'failed' in r:
            print(n, r); continue
        d = r['deformation']
        mt = r['material']
        print(f"{n}: mu {mt['applied']['mu']} lam {mt['applied']['lam']} sy {mt['applied']['yield_stress']} match {mt['match']} "
              f"yield_strain {mt['expected']['yield_stress'] / (2 * mt['expected']['mu']):.4f} | f15 {msd.fmt(r['pre_contact_f15'])} | "
              f"normalized_residual {r['normalized_residual']:.3f}")
        print(f"{n}: rate {r['gripper_rate']:.6f} actual_gap_f29 {r['actual_surface_gap_f29']:+.4f} | "
              + " | ".join(f"{k} {v['mean']:.3e}/{v['rms']:.3e}/{v['max']:.3e}" for k, v in d.items())
              + f" | recovery {r['recovery']:+.3e} ratio {r['recovery_ratio']:+.3f} | NaN {r['nan']} Inf {r['inf']} "
              f"maxstep {r['max_frame_step_disp']:.3e} minclear {r['min_finger_clearance']:+.4f} finger-pin {r['min_finger_pin_gap']:+.4f} "
              f"bbox {np.round(r['bbox_min'], 3).tolist()}..{np.round(r['bbox_max'], 3).tolist()}")
    for pair, v in summary.get('cross_f39', {}).items():
        print(f"cross f39 {pair}: {msd.fmt(v)}")
    print(f'outputs: {out_dir}')


if __name__ == '__main__':
    main()
