"""ACTION-DISTRIBUTION PILOT: 50 independent single grips from the shared settled state, one material.

E=5000, nu=0.2, sigma_y=500 fixed. 30 close + 10 retreat, no settle. Simulator/yaml untouched.
Candidates: Latin hypercube over (midpoint offset x, midpoint offset z, angle, target gap), wider than the
current sampler so bounds can be located.

Two stages (thresholds are chosen after looking at the metric distributions):
  python action_distribution_pilot.py simulate --settled-state <pkl>
  python action_distribution_pilot.py analyze <out_dir> [--thresholds ...] [--rec-* ...]
"""
import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
from datetime import datetime

import cv2
import numpy as np

import gen_multigrip_debug as g
import material_sensitivity_debug as msd
import render_material_grid as rmg
import stress_test_grip as st

MATERIAL = dict(E=5000.0, yield_stress=500.0)
N_CAND = 50
SEED = 7
MID = g.task_params['mid_point'][:3]           # (0.5, 0.14, 0.5)
OFFSET_RANGE = (-0.1, 0.1)                     # np.clip bound in random_pose / env.py stepm action space
ANGLE_RANGE = (0.0, np.pi)                     # random_pose rot_noise
GAP_RANGE = (0.05, 0.20)                       # current [0.06, 0.14] widened both ways
OBJ_LO, OBJ_HI = 0.375, 0.625                  # gripper_fixed.yml box x/z extent
CLASSES = ['GOOD', 'TOO_WEAK', 'TOO_AGGRESSIVE', 'PIN_COLLISION', 'INVALID_GEOMETRY']
COLORS = dict(GOOD='tab:green', TOO_WEAK='tab:gray', TOO_AGGRESSIVE='tab:red', PIN_COLLISION='tab:purple', INVALID_GEOMETRY='k')
MARKERS = dict(GOOD='o', TOO_WEAK='v', TOO_AGGRESSIVE='^', PIN_COLLISION='X', INVALID_GEOMETRY='s')


def pose_from(mid, angle):
    # same expressions as random_pose (ngrip), deterministic midpoint/angle
    r = g.task_params['sample_radius']
    quat = np.array([1, 0, 0, 0])
    p1 = np.array([mid[0] - r * np.cos(angle), mid[1], mid[2] + r * np.sin(angle)])
    p2 = np.array([mid[0] + r * np.cos(angle), mid[1], mid[2] - r * np.sin(angle)])
    return np.concatenate([p1, quat]), np.concatenate([p2, quat])


def check_pose_from():
    np.random.seed(123)
    ref1, ref2, ref_angle = g.random_pose(g.task_name)
    np.random.seed(123)
    s = g.task_params['p_noise_scale']
    nx, nz = s * (np.random.randn() * 2 - 1), s * (np.random.randn() * 2 - 1)
    angle = np.random.uniform(0, np.pi)
    p1, p2 = pose_from(MID + np.clip([nx, 0, nz], -0.1, 0.1), angle)
    assert angle == ref_angle and np.allclose(p1, ref1, atol=1e-15) and np.allclose(p2, ref2, atol=1e-15)


def latin_hypercube(n, ranges, rng):
    # one sample per stratum in every dimension, strata paired by independent permutations
    u = (np.stack([rng.permutation(n) for _ in ranges], 1) + rng.random((n, len(ranges)))) / n
    lo, hi = np.array(ranges).T
    return lo + u * (hi - lo)


def make_candidates(path):
    s = latin_hypercube(N_CAND, [OFFSET_RANGE, OFFSET_RANGE, ANGLE_RANGE, GAP_RANGE], np.random.default_rng(SEED))
    mids = MID + np.stack([s[:, 0], np.zeros(N_CAND), s[:, 1]], 1)
    poses, actions = [], []
    for m, a, gap in zip(mids, s[:, 2], s[:, 3]):
        p1, p2 = pose_from(m, a)
        poses.append(np.stack([p1, p2]))
        actions.append(g.make_grip_actions(p1, p2, st.gap_to_rate(gap)))
    np.savez_compressed(path, action_id=np.arange(N_CAND), midpoint=mids, angle=s[:, 2], target_gap=s[:, 3],
                        gripper_rate=np.array([st.gap_to_rate(x) for x in s[:, 3]]),
                        finger_start_pose=np.stack(poses), primitive_action_sequence=np.stack(actions), seed=np.array(SEED))


def capsule_dist(pose, h, r, x):
    # per-particle signed distance to a vertical capsule (same SDF as capsule_clearance)
    d = x - pose[:3]
    d[:, 1] -= np.clip(d[:, 1], -h / 2, h / 2)
    return np.linalg.norm(d, axis=1) - r


def run_action(env, settled, cand, i):
    msd.restore(env, settled, MATERIAL)
    err = msd.state_error(env.get_state(), settled)
    assert all(v == 0 for v in err.values()), err
    return run_grip(env, cand, i)


def run_grip(env, cand, i, on_frame=None):
    # one 30 close + 10 retreat grip from the CURRENT simulator state (no restore)
    x0 = env.simulator.get_x(0)
    before = g.read_object_cloud(env)
    pose = cand['finger_start_pose'][i]
    g.update_primitive(env, pose[0], pose[1])
    g.select_tool(env, msd.TOOL_SIZE)
    assert np.array_equal(g.read_tool_pose(env), pose)
    fingers = env.primitives.primitives[:2]
    h = [p.h[None] for p in fingers]
    pin = env.primitives.primitives[2].get_state(0)
    start_clear = [float(capsule_dist(pose[j], h[j], msd.TOOL_SIZE, x0).min()) for j in range(2)]
    objs, tools, contact, steps, full = [], [], [], [], {}
    x_prev = x0
    for f, act in enumerate(cand['primitive_action_sequence'][i]):
        env.step(act)
        x = env.simulator.get_x(0)
        steps.append(float(np.linalg.norm(x - x_prev, axis=1).max())); x_prev = x
        objs.append(g.read_object_cloud(env)); tools.append(g.read_tool_pose(env))
        dist = [capsule_dist(tools[-1][j], h[j], msd.TOOL_SIZE, x) for j in range(2)]
        contact.append([[float(dd.min()), int((dd < 0).sum())] for dd in dist])   # per finger: min clearance, particles inside
        if f in (25, 29, 39):
            full[f] = x
        if on_frame:
            on_frame(f)
    tools, contact = np.stack(tools), np.array(contact)                            # contact (40, 2, 2)
    t29 = tools[29]
    return dict(
        action_id=np.array(i), midpoint=cand['midpoint'][i], angle=cand['angle'][i], target_gap=cand['target_gap'][i],
        gripper_rate=cand['gripper_rate'][i], finger_start_pose=pose, primitive_action_sequence=cand['primitive_action_sequence'][i],
        E=np.array(MATERIAL['E']), sigma_y=np.array(MATERIAL['yield_stress']), nu=np.array(msd.NU),
        before_object_cloud=before, frame25_object_cloud=objs[25], frame29_object_cloud=objs[29], frame39_object_cloud=objs[39],
        object_rollout=np.stack(objs), tool_rollout=tools,
        actual_gap_f29=np.array(np.linalg.norm(t29[0, :3] - t29[1, :3]) - 2 * msd.TOOL_SIZE),
        finger_min_clearance=contact[:, :, 0], finger_contact_count=contact[:, :, 1], max_step_disp_all_particles=np.array(steps),
        finger_pin_gap=np.linalg.norm(tools[:, :, [0, 2]] - pin[[0, 2]], axis=-1) - msd.TOOL_SIZE - st.PIN_R,
        start_clearance=np.array(start_clear),
    ), {"x0": x0, **full}


def metrics(d, full):
    x0 = full['x0']
    dx = lambda f: np.linalg.norm(full[f] - x0, axis=1)
    rms = lambda v: float(np.sqrt((v ** 2).mean()))
    d29, d39 = dx(29), dx(39)
    cnt = d['finger_contact_count']
    first = [int(np.argmax(cnt[:, j] > 0)) if (cnt[:, j] > 0).any() else None for j in range(2)]
    u = np.array([np.cos(float(d['angle'])), 0, -np.sin(float(d['angle']))])     # closing axis (finger0 -> finger1)
    off = d['midpoint'] - MID
    bbox = lambda x: [x.min(0).tolist(), x.max(0).tolist()]
    start = d['finger_start_pose'][:, :3]
    m = dict(
        action_id=int(d['action_id']), midpoint_x=float(d['midpoint'][0]), midpoint_z=float(d['midpoint'][2]),
        offset_along_axis=float(off @ u), offset_perp_axis=float(np.linalg.norm(off - (off @ u) * u)),
        angle=float(d['angle']), angle_deg=float(np.degrees(d['angle'])), target_gap=float(d['target_gap']),
        gripper_rate=float(d['gripper_rate']), actual_gap_f29=float(d['actual_gap_f29']),
        compression_RMS=rms(d29), residual_RMS=rms(d39), normalized_residual=rms(d39) / rms(d29),
        compression_mean=float(d29.mean()), residual_mean=float(d39.mean()),
        max_disp_f29=float(d29.max()), max_disp_f39=float(d39.max()),
        frac_moved_gt_dx_f29=float((d29 > 1 / 64).mean()),
        first_contact_frame=first, both_fingers_contact=all(f is not None for f in first),
        max_contact_particles=[int(cnt[:, j].max()) for j in range(2)],
        min_finger_clearance=float(d['finger_min_clearance'].min()),
        min_finger_clearance_per_finger=d['finger_min_clearance'].min(0).tolist(),
        finger_pin_distance=float(d['finger_pin_gap'].min()),
        start_clearance=d['start_clearance'].tolist(),
        start_center_in_domain=bool(((start[:, [0, 2]] >= 0) & (start[:, [0, 2]] <= 1)).all()),
        gap_error=float(abs(d['actual_gap_f29'] - d['target_gap'])),
        max_step_disp=float(d['max_step_disp_all_particles'].max()),
        bbox_before=bbox(x0), bbox_f29=bbox(full[29]), bbox_f39=bbox(full[39]),
        height_ratio_f39=float(full[39][:, 1].max() / x0[:, 1].max()),
        min_xz_extent_f29=float(min(np.ptp(full[29][:, 0]), np.ptp(full[29][:, 2]))),
        nan=int(sum(np.isnan(v).sum() for v in d.values() if np.asarray(v).dtype.kind == 'f')),
        inf=int(sum(np.isinf(v).sum() for v in d.values() if np.asarray(v).dtype.kind == 'f')),
    )
    m['explosion'] = bool(m['max_step_disp'] > msd.EXPLOSION_STEP_DISP or min(map(min, m['bbox_f39'])) < 0 or max(map(max, m['bbox_f39'])) > 1)
    return m


def du(path):
    return subprocess.run(['du', '-sh', path], capture_output=True, text=True).stdout.split()[0]


def disk_free():
    return f"{shutil.disk_usage(g.HERE).free / 2 ** 30:.2f} GiB"


def simulate(args):
    check_pose_from()
    settled_path = os.path.abspath(args.settled_state)
    with open(settled_path, 'rb') as f:
        settled = pickle.load(f)
    out = os.path.abspath(os.path.join(g.HERE, '..', '..', 'dataset', 'action_distribution_pilot_' + datetime.now().strftime("%d-%b-%Y-%H:%M:%S")))
    os.makedirs(os.path.join(out, 'runs'), exist_ok=True)
    print(f'output size before: {du(out)}  disk free: {disk_free()}')
    make_candidates(os.path.join(out, 'candidate_actions.npz'))
    cand = dict(np.load(os.path.join(out, 'candidate_actions.npz')))
    env, _ = g.make_env()
    res = dict(settled_state=settled_path, material=dict(**MATERIAL, nu=msd.NU), seed=SEED,
               ranges=dict(offset=OFFSET_RANGE, angle=ANGLE_RANGE, gap=GAP_RANGE), disk_before=disk_free(), runs=[])
    init_cloud = None
    for i in range(N_CAND):
        try:
            d, full = run_action(env, settled, cand, i)
        except ValueError as e:
            res['runs'].append(dict(action_id=i, failed=f'ValueError: {e}')); print(i, 'ValueError'); continue
        init_cloud = d['before_object_cloud'] if init_cloud is None else init_cloud
        assert np.array_equal(d['before_object_cloud'], init_cloud)
        np.savez_compressed(os.path.join(out, 'runs', f'action_{i:03d}.npz'), **d)
        m = metrics(d, full)
        res['runs'].append(m)
        print(f"{i:3d} gap {m['target_gap']:.3f} off_ax {m['offset_along_axis']:+.3f} perp {m['offset_perp_axis']:.3f} "
              f"ang {m['angle_deg']:5.1f} | comp {m['compression_RMS']:.4f} res {m['residual_RMS']:.4f} nres {m['normalized_residual']:.2f} "
              f"| pin {m['finger_pin_distance']:+.4f} clr {m['min_finger_clearance']:+.4f} contact {m['max_contact_particles']} "
              f"first {m['first_contact_frame']} | hr {m['height_ratio_f39']:.2f} ext {m['min_xz_extent_f29']:.3f} maxstep {m['max_step_disp']:.4f}", flush=True)
    res['initial_identity'] = 'x,v,F,C,finger0,finger1,pin == settled (asserted per run); 300-cloud identical across runs (asserted)'
    with open(os.path.join(out, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print(f'output size after: {du(out)}  disk free: {disk_free()}\noutputs: {out}')


# ------------------------------------------------------------------ analyze
def classify(m, t):
    if 'failed' in m or m['nan'] or m['inf'] or m['explosion']:
        return 'TOO_AGGRESSIVE' if 'failed' not in m else 'INVALID_GEOMETRY'
    if not m['start_center_in_domain'] or min(m['start_clearance']) < 0 or m['gap_error'] > 1e-6:
        return 'INVALID_GEOMETRY'
    if m['finger_pin_distance'] < t.pin_min:
        return 'PIN_COLLISION'
    if m['height_ratio_f39'] < t.height_ratio_min or m['max_disp_f29'] > t.max_disp_max or m['min_finger_clearance'] < t.clearance_min:
        return 'TOO_AGGRESSIVE'
    if any(f is not None for f in m['first_contact_frame']) and not m['both_fingers_contact']:
        return 'INVALID_GEOMETRY'   # one finger never touches the object: one-sided push, not a grip
    if m['compression_RMS'] < t.comp_min:
        return 'TOO_WEAK'
    return 'GOOD'


def analytic_pin_distance(gap, offset_along, offset_perp):
    # closest finger-pin surface distance over the straight close: finger stops at gap/2 + r_f from midpoint
    along = np.maximum(gap / 2 + msd.TOOL_SIZE - np.abs(offset_along), 0)
    return np.sqrt(along ** 2 + offset_perp ** 2) - msd.TOOL_SIZE - st.PIN_R


def mc_pin_collision(offset_xz, angle, gap, margin=0.0):
    # analytic pin-overlap rate of a sampler (verified against simulation in analyze)
    u = np.stack([np.cos(angle), -np.sin(angle)], 1)                  # closing axis in (x, z)
    along = (offset_xz * u).sum(1)
    perp = np.abs(offset_xz[:, 0] * u[:, 1] - offset_xz[:, 1] * u[:, 0])
    return analytic_pin_distance(gap, along, perp) < margin


def mc_report(args, n=200000):
    rng = np.random.default_rng(0)
    sc = g.task_params['p_noise_scale']
    samplers = dict(
        current=(np.clip(sc * (rng.standard_normal((n, 2)) * 2 - 1), -0.1, 0.1), rng.uniform(0, np.pi, n), rng.uniform(0.06, 0.14, n)),
        recommended_box=(rng.uniform(*args.rec_offset, (n, 2)), rng.uniform(*args.rec_angle, n), rng.uniform(*args.rec_gap, n)))
    out = {}
    for name, (off, ang, gap) in samplers.items():
        col = mc_pin_collision(off, ang, gap)
        bins = {f'[{lo:.2f},{hi:.2f})': float(col[(gap >= lo) & (gap < hi)].mean()) for lo, hi in
                [(0.06, 0.07), (0.07, 0.08), (0.08, 0.10), (0.10, 0.12), (0.12, 0.14)] if ((gap >= lo) & (gap < hi)).any()}
        out[name] = dict(pin_overlap_rate=float(col.mean()),
                         rejected_with_margin=float(mc_pin_collision(off, ang, gap, args.rec_pin_margin).mean()),
                         by_gap=bins)
    return out


def render_rep(env, settled, d, path):
    tmp = path + '_frames'
    os.makedirs(tmp, exist_ok=True)
    msd.restore(env, settled, MATERIAL)
    frames = [rmg.render(env)]
    g.update_primitive(env, d['finger_start_pose'][0], d['finger_start_pose'][1])
    g.select_tool(env, msd.TOOL_SIZE)
    for act in d['primitive_action_sequence']:
        env.step(act)
        frames.append(rmg.render(env))
    assert np.allclose(g.read_object_cloud(env), d['frame39_object_cloud'], atol=1e-3)   # same run (GPU nondeterminism ~1e-4)
    for i, im in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, f'{i:03d}.png'), im[..., ::-1])
    rmg.encode(tmp, path + '.mp4')
    cv2.imwrite(path + '_f00_f29_f39.png', np.hstack([np.ascontiguousarray(frames[k][..., ::-1]) for k in (0, 30, 40)]))
    shutil.rmtree(tmp)


def analyze(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = os.path.abspath(args.out_dir)
    with open(os.path.join(out, 'metrics.json')) as f:
        res = json.load(f)
    runs = res['runs']
    for m in runs:
        m['quality_class'] = classify(m, args)
    ok = [m for m in runs if 'failed' not in m]
    A = lambda k: np.array([m[k] for m in ok])
    ana = analytic_pin_distance(A('target_gap'), A('offset_along_axis'), A('offset_perp_axis'))
    ana_err = float(np.abs(ana - A('finger_pin_distance')).max())
    assert ana_err < 2e-3 and ((ana < 0) == (A('finger_pin_distance') < 0)).all(), ana_err   # per-frame sampling of the close
    res['analytic_pin_distance_max_error'] = ana_err
    res['monte_carlo_pin'] = mc_report(args)
    cls = np.array([m['quality_class'] for m in ok])
    pdir = os.path.join(out, 'plots')
    os.makedirs(pdir, exist_ok=True)

    def scatter(ax, xk, yk, xl, yl):
        for c in CLASSES:
            s = cls == c
            if s.any():
                ax.scatter(A(xk)[s], A(yk)[s], c=COLORS[c], marker=MARKERS[c], s=55, label=f'{c} ({s.sum()})', edgecolors='w', linewidths=.5)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)

    def save(fig, name):
        fig.tight_layout(); fig.savefig(os.path.join(pdir, name), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    scatter(ax, 'target_gap', 'compression_RMS', 'target gap', 'compression RMS (f29 - before, all particles)')
    ax.axhline(args.comp_min, ls='--', c='gray'); ax.axvspan(0.06, 0.14, color='tab:blue', alpha=.07, label='current gap range')
    ax.legend(fontsize=7); save(fig, 'gap_vs_compression.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scatter(axes[0], 'target_gap', 'finger_pin_distance', 'target gap', 'min finger-pin surface distance')
    axes[0].axhline(args.pin_min, ls='--', c='purple')
    scatter(axes[1], 'target_gap', 'offset_along_axis', 'target gap', 'midpoint offset along closing axis')
    xs = np.linspace(*GAP_RANGE, 50)
    for sgn in (1, -1):   # pin overlap boundary for zero perpendicular offset: gap/2 + r_f - |off| = r_f + r_pin
        axes[1].plot(xs, sgn * (xs / 2 - st.PIN_R), 'k:', lw=1)
    axes[0].legend(fontsize=7); save(fig, 'gap_vs_quality.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    scatter(axes[0], 'midpoint_x', 'midpoint_z', 'midpoint x', 'midpoint z')
    for ax in axes[:1]:
        ax.add_patch(plt.Rectangle((OBJ_LO, OBJ_LO), OBJ_HI - OBJ_LO, OBJ_HI - OBJ_LO, fill=False, ls='--', color='orange', label='object'))
        ax.add_patch(plt.Circle((0.5, 0.5), st.PIN_R, color='k', alpha=.4))
        ax.set_xlim(0.38, 0.62); ax.set_ylim(0.38, 0.62); ax.set_aspect('equal'); ax.legend(fontsize=7)
    scatter(axes[1], 'offset_along_axis', 'offset_perp_axis', 'midpoint offset along closing axis', 'perpendicular offset')
    save(fig, 'midpoint_quality_map.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scatter(axes[0], 'angle_deg', 'compression_RMS', 'angle (deg)', 'compression RMS')
    scatter(axes[1], 'angle_deg', 'target_gap', 'angle (deg)', 'target gap')
    axes[0].legend(fontsize=7); save(fig, 'angle_vs_compression.png')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    scatter(ax, 'compression_RMS', 'normalized_residual', 'compression RMS', 'normalized residual')
    ax.legend(fontsize=7); save(fig, 'compression_vs_normalized_residual.png')

    cols = ['action_id', 'midpoint_x', 'midpoint_z', 'angle', 'angle_deg', 'target_gap', 'gripper_rate', 'actual_gap_f29',
            'offset_along_axis', 'offset_perp_axis', 'compression_RMS', 'residual_RMS', 'normalized_residual',
            'max_disp_f29', 'max_disp_f39', 'min_finger_clearance', 'finger_pin_distance', 'both_fingers_contact', 'first_contact_frame',
            'height_ratio_f39', 'quality_class']
    with open(os.path.join(out, 'action_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(runs)

    # representative cases
    vdir = os.path.join(out, 'representative_visuals')
    os.makedirs(vdir, exist_ok=True)
    reps = {}
    order = dict(GOOD=lambda m: -m['compression_RMS'], TOO_WEAK=lambda m: m['compression_RMS'],
                 TOO_AGGRESSIVE=lambda m: -m['max_disp_f29'], PIN_COLLISION=lambda m: m['finger_pin_distance'])
    for c, n in dict(GOOD=5, TOO_WEAK=3, TOO_AGGRESSIVE=3, PIN_COLLISION=3).items():
        ms = sorted([m for m in ok if m['quality_class'] == c], key=order[c])
        if c == 'GOOD' and len(ms) > n:   # spread over compression range instead of top-5 strongest
            ms = [ms[int(k)] for k in np.linspace(0, len(ms) - 1, n)]
        reps[c] = [m['action_id'] for m in ms[:n]]
    if args.render:
        with open(res['settled_state'], 'rb') as f:
            settled = pickle.load(f)
        env, _ = g.make_env()
        rmg.update_camera(env)
        env.number_of_cams = 1
        for c, ids in reps.items():
            for i in ids:
                d = dict(np.load(os.path.join(out, 'runs', f'action_{i:03d}.npz')))
                render_rep(env, settled, d, os.path.join(vdir, f'{c.lower()}_{i:03d}'))

    # recommended region + GOOD pool inside it
    rec = dict(status='CANDIDATE (not final; pending user review)',
               midpoint_x=[MID[0] + args.rec_offset[0], MID[0] + args.rec_offset[1]],
               midpoint_z=[MID[2] + args.rec_offset[0], MID[2] + args.rec_offset[1]],
               angle=args.rec_angle, target_gap=args.rec_gap, midpoint_y=float(MID[1]),
               midpoint_sampling='uniform box centered at (0.5, 0.5) (replaces biased 0.01*(randn*2-1) noise), then reject by pin_filter',
               start_pose='pose_from(midpoint, angle): fingers at midpoint -/+ 0.4*(cos a, 0, -sin a)',
               gap_to_rate='rate = (2*0.4 - (gap + 2*0.045)) / (2*30)',
               pin_filter=f'reject if analytic_pin_distance(gap, offset_along_axis, offset_perp_axis) < {args.rec_pin_margin}',
               analytic_pin_distance='sqrt(max(gap/2 + 0.045 - |offset_along|, 0)^2 + offset_perp^2) - 0.045 - 0.025; closing axis u = (cos a, 0, -sin a)',
               thresholds=dict(pin_min=args.pin_min, comp_min=args.comp_min, height_ratio_min=args.height_ratio_min,
                               max_disp_max=args.max_disp_max, clearance_min=args.clearance_min))
    inside = lambda m: (rec['midpoint_x'][0] <= m['midpoint_x'] <= rec['midpoint_x'][1] and rec['midpoint_z'][0] <= m['midpoint_z'] <= rec['midpoint_z'][1]
                        and args.rec_angle[0] <= m['angle'] <= args.rec_angle[1] and args.rec_gap[0] <= m['target_gap'] <= args.rec_gap[1]
                        and analytic_pin_distance(m['target_gap'], m['offset_along_axis'], m['offset_perp_axis']) >= args.rec_pin_margin)
    in_region = [m for m in ok if inside(m)]
    rec['pilot_candidates_in_region'] = dict(total=len(in_region), **{c: sum(m['quality_class'] == c for m in in_region) for c in CLASSES})
    with open(os.path.join(out, 'recommended_action_region.json'), 'w') as f:
        json.dump(rec, f, indent=2)
    pool = [m['action_id'] for m in in_region if m['quality_class'] == 'GOOD']
    cand = dict(np.load(os.path.join(out, 'candidate_actions.npz')))
    np.savez_compressed(os.path.join(out, 'good_action_pool.npz'), **{k: v[pool] for k, v in cand.items() if k != 'seed'},
                        seed=cand['seed'], all_good_action_id=np.array([m['action_id'] for m in ok if m['quality_class'] == 'GOOD']))

    counts = {c: int((cls == c).sum()) for c in CLASSES}
    counts['INVALID_GEOMETRY'] += sum('failed' in m for m in runs)
    summary = dict(res, runs=runs, counts=counts, representative=reps, recommended=rec,
                   thresholds=rec['thresholds'], output_size=du(out), disk_free=disk_free())
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_report(summary, ok, out)


def write_report(s, ok, out):
    L = ['[Current action space]', '',
         'midpoint x: 0.5 + p_noise_x, p_noise_x = 0.01*(randn*2-1) (mean -0.01, std 0.02), clipped to [-0.1, 0.1] -> effective ~[0.43, 0.55] (3 sigma)',
         'midpoint z: same as x (independent draw)',
         'midpoint y: 0.14 fixed',
         'angle: uniform [0, pi)  (quat always identity for ngrip; angle only sets closing axis)',
         'gap: gripper_rate uniform over rate(gap=0.14)..rate(gap=0.06), linear -> target gap uniform [0.06, 0.14]',
         'finger start: finger0 = mid + 0.4*(-cos a, 0, +sin a), finger1 = mid + 0.4*(cos a, 0, -sin a) (center distance 0.8)',
         'gap -> rate: rate = (2*sample_radius - (gap + 2*tool_size)) / (2*len_per_grip) = (0.8 - gap - 0.09)/60',
         'object bbox usage: none (midpoint fixed at 0.5,0.5 + noise; object box x,z in [0.375, 0.625])',
         'pin constraint: none (pin capsule r=0.025 at x=z=0.5)',
         'env.py stepm (learned-model action space): offset x,z in [-0.1,0.1], angle [0,pi], same rate limits; p_noise_bound=0.03 defined but unused',
         'sampling method (current): i.i.d. random; (this pilot): Latin hypercube, seed 7', '',
         '-' * 32, '[Execution]', '',
         f'num candidates: {len(s["runs"])}', f'material: E={s["material"]["E"]:g} sigma_y={s["material"]["yield_stress"]:g} nu={s["material"]["nu"]}',
         f'seed: {s["seed"]}',
         f'candidate ranges: midpoint offset {s["ranges"]["offset"]}, angle {[round(a, 4) for a in s["ranges"]["angle"]]}, gap {s["ranges"]["gap"]}',
         f'initial identity: {s["initial_identity"]}', f'thresholds (chosen after distribution review): {s["thresholds"]}', '',
         '-' * 32, '[Quality counts]', '']
    L += [f'{c}: {n} ({100 * n / len(s["runs"]):.0f}%)' for c, n in s['counts'].items()]
    L += ['', '-' * 32, '[Action table]  (sorted by target gap)', '',
          f"{'id':>3} {'gap':>6} {'mid_x':>6} {'mid_z':>6} {'ang':>6} {'off_ax':>7} {'perp':>6} {'comp':>7} {'resid':>7} {'nres':>5} "
          f"{'maxd29':>7} {'clear':>7} {'pin':>7} {'both':>5} {'h_rat':>5}  class"]
    for m in sorted(ok, key=lambda m: m['target_gap']):
        L.append(f"{m['action_id']:>3} {m['target_gap']:6.3f} {m['midpoint_x']:6.3f} {m['midpoint_z']:6.3f} {m['angle_deg']:6.1f} "
                 f"{m['offset_along_axis']:+7.3f} {m['offset_perp_axis']:6.3f} {m['compression_RMS']:7.4f} {m['residual_RMS']:7.4f} "
                 f"{m['normalized_residual']:5.2f} {m['max_disp_f29']:7.4f} {m['min_finger_clearance']:+7.4f} {m['finger_pin_distance']:+7.4f} "
                 f"{str(m['both_fingers_contact']):>5} {m['height_ratio_f39']:5.2f}  {m['quality_class']}")
    L += ['', '-' * 32, '[Gap bins]', '', f"{'bin':>13} {'n':>3} {'pin<0':>5} {'GOOD':>4} {'comp mean':>9} {'comp min':>8}"]
    for lo, hi in [(0.05, 0.065), (0.065, 0.075), (0.075, 0.09), (0.09, 0.10), (0.10, 0.14), (0.14, 0.17), (0.17, 0.20)]:
        b = [m for m in ok if lo <= m['target_gap'] < hi + (1e-9 if hi == 0.20 else 0)]
        if b:
            L.append(f"[{lo:.3f},{hi:.3f}) {len(b):>3} {sum(m['finger_pin_distance'] < 0 for m in b):>5} {sum(m['quality_class'] == 'GOOD' for m in b):>4} "
                     f"{np.mean([m['compression_RMS'] for m in b]):9.4f} {min(m['compression_RMS'] for m in b):8.4f}")
    L += ['', '-' * 32, f"[Monte-Carlo pin overlap, analytic, max error vs sim {s['analytic_pin_distance_max_error']:.1e}]", '']
    L += [f'{k}: {json.dumps(v)}' for k, v in s['monte_carlo_pin'].items()]
    L += ['', '-' * 32, '[Representative actions]', ''] + [f'{c}: {ids}' for c, ids in s['representative'].items()]
    L += ['', '-' * 32, '[Recommended candidate distribution]  CANDIDATE, not final', '']
    r = s['recommended']
    L += [f"midpoint x: {np.round(r['midpoint_x'], 4).tolist()}", f"midpoint z: {np.round(r['midpoint_z'], 4).tolist()}",
          f"angle: {np.round(r['angle'], 4).tolist()}", f"target gap: {r['target_gap']}",
          f"pilot candidates inside region: {r['pilot_candidates_in_region']}", '']
    L += ['-' * 32, '[Visual paths]', '', f"plots: {os.path.join(out, 'plots')}/", f"representative videos/images: {os.path.join(out, 'representative_visuals')}/", '']
    L += ['-' * 32, '[Disk usage]', '', f"output folder size: {s['output_size']}", f"remaining disk: {s['disk_free']}", '']
    L += ['-' * 32, '[Analysis / Recommendation]', '', '(written after review)', '']
    with open(os.path.join(out, 'report.txt'), 'w') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))


def validate(args):
    """20 single grips sampled from recommended_action_region.json: LHS 60 -> analytic pin filter -> 20 spread over gap."""
    import types
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pilot = os.path.abspath(args.pilot_dir)
    with open(os.path.join(pilot, 'recommended_action_region.json')) as f:
        rec = json.load(f)
    with open(os.path.join(pilot, 'metrics.json')) as f:
        settled_path = json.load(f)['settled_state']
    out = os.path.join(pilot, 'validation_region')
    os.makedirs(os.path.join(out, 'runs'), exist_ok=True)
    print(f'output size before: {du(out)}  disk free: {disk_free()}')

    rng = np.random.default_rng(args.seed)
    s = latin_hypercube(3 * args.n, [rec['midpoint_x'], rec['midpoint_z'], rec['angle'], rec['target_gap']], rng)
    mids = np.stack([s[:, 0], np.full(len(s), rec['midpoint_y']), s[:, 1]], 1)
    off = mids[:, [0, 2]] - MID[[0, 2]]
    keep = np.where(~mc_pin_collision(off, s[:, 2], s[:, 3], args.pin_margin))[0]
    assert len(keep) >= args.n, len(keep)
    keep = keep[np.argsort(s[keep, 3])][np.linspace(0, len(keep) - 1, args.n).round().astype(int)]   # spread over gap
    poses, actions = [], []
    for i in keep:
        p1, p2 = pose_from(mids[i], s[i, 2])
        poses.append(np.stack([p1, p2]))
        actions.append(g.make_grip_actions(p1, p2, st.gap_to_rate(s[i, 3])))
    cand = dict(action_id=np.arange(args.n), midpoint=mids[keep], angle=s[keep, 2], target_gap=s[keep, 3],
                gripper_rate=np.array([st.gap_to_rate(x) for x in s[keep, 3]]), finger_start_pose=np.stack(poses),
                primitive_action_sequence=np.stack(actions), seed=np.array(args.seed),
                lhs_total=np.array(len(s)), lhs_accepted=np.array(int((~mc_pin_collision(off, s[:, 2], s[:, 3], args.pin_margin)).sum())))
    np.savez_compressed(os.path.join(out, 'candidate_actions.npz'), **cand)

    with open(settled_path, 'rb') as f:
        settled = pickle.load(f)
    env, _ = g.make_env()
    t = types.SimpleNamespace(**rec['thresholds'])
    runs, init_cloud = [], None
    for i in range(args.n):
        try:
            d, full = run_action(env, settled, cand, i)
        except ValueError as e:
            runs.append(dict(action_id=i, failed=f'ValueError: {e}', quality_class='INVALID_GEOMETRY')); continue
        init_cloud = d['before_object_cloud'] if init_cloud is None else init_cloud
        assert np.array_equal(d['before_object_cloud'], init_cloud)
        np.savez_compressed(os.path.join(out, 'runs', f'action_{i:03d}.npz'), **d)
        m = metrics(d, full)
        m['analytic_pin_distance'] = float(analytic_pin_distance(m['target_gap'], m['offset_along_axis'], m['offset_perp_axis']))
        m['quality_class'] = classify(m, t)
        runs.append(m)
        print(f"{i:3d} gap {m['target_gap']:.3f} mid ({m['midpoint_x']:.3f},{m['midpoint_z']:.3f}) ang {m['angle_deg']:5.1f} "
              f"| comp {m['compression_RMS']:.4f} nres {m['normalized_residual']:.2f} maxd {m['max_disp_f29']:.3f} "
              f"| pin {m['finger_pin_distance']:+.4f} (analytic {m['analytic_pin_distance']:+.4f}) clr {m['min_finger_clearance']:+.4f} "
              f"first {m['first_contact_frame']} | {m['quality_class']}", flush=True)

    ok = [m for m in runs if 'failed' not in m]
    counts = {c: sum(m['quality_class'] == c for m in runs) for c in CLASSES}
    good = [m['action_id'] for m in ok if m['quality_class'] == 'GOOD']
    np.savez_compressed(os.path.join(out, 'good_action_pool.npz'),
                        **{k: v[good] for k, v in cand.items() if v.ndim and len(v) == args.n}, seed=cand['seed'])
    cols = ['action_id', 'midpoint_x', 'midpoint_z', 'angle', 'angle_deg', 'target_gap', 'gripper_rate', 'actual_gap_f29',
            'offset_along_axis', 'offset_perp_axis', 'compression_RMS', 'residual_RMS', 'normalized_residual', 'max_disp_f29',
            'min_finger_clearance', 'finger_pin_distance', 'analytic_pin_distance', 'both_fingers_contact', 'first_contact_frame',
            'height_ratio_f39', 'quality_class']
    with open(os.path.join(out, 'action_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(runs)

    # pilot (grey) vs validation (class colored)
    with open(os.path.join(pilot, 'summary.json')) as f:
        prior = [m for m in json.load(f)['runs'] if 'failed' not in m]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (xk, yk) in zip(axes, [('target_gap', 'compression_RMS'), ('target_gap', 'finger_pin_distance')]):
        ax.scatter([m[xk] for m in prior], [m[yk] for m in prior], c='lightgray', s=25, label='50 wide pilot')
        for c in CLASSES:
            ms = [m for m in ok if m['quality_class'] == c]
            if ms:
                ax.scatter([m[xk] for m in ms], [m[yk] for m in ms], c=COLORS[c], marker=MARKERS[c], s=60, label=f'region {c} ({len(ms)})')
        ax.axvspan(*rec['target_gap'], color='tab:blue', alpha=.07); ax.set_xlabel(xk); ax.set_ylabel(yk); ax.grid(alpha=.3)
    axes[0].axhline(t.comp_min, ls='--', c='gray'); axes[1].axhline(0, ls='--', c='purple'); axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'validation_gap_plots.png'), dpi=110); plt.close(fig)

    A = lambda k: np.array([m[k] for m in ok])
    summary = dict(region=rec, pin_margin=args.pin_margin, seed=args.seed, n=args.n,
                   lhs_total=int(cand['lhs_total']), lhs_accepted=int(cand['lhs_accepted']), counts=counts, good_action_id=good,
                   analytic_vs_sim_pin_max_error=float(np.abs(A('analytic_pin_distance') - A('finger_pin_distance')).max()),
                   ranges=dict(**{k: [float(A(k).min()), float(A(k).max())] for k in
                                  ('compression_RMS', 'residual_RMS', 'normalized_residual', 'max_disp_f29', 'min_finger_clearance',
                                   'finger_pin_distance', 'height_ratio_f39', 'max_step_disp', 'target_gap', 'angle_deg')}),
                   nan=int(A('nan').sum()), inf=int(A('inf').sum()), explosion=bool(A('explosion').any()),
                   runs=runs, output_size=du(out), disk_free=disk_free())
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'region')}, indent=1))
    print(f'output size after: {du(out)}  disk free: {disk_free()}\noutputs: {out}')


GAP_BINS = [(0.07, 0.08), (0.08, 0.09), (0.09, 0.10), (0.10, 0.11), (0.11, 0.12), (0.12, 0.13)]


SAME_AXIS_DEG = 20.0


def delta_axis(a, b):
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def same_axis_run(angles):
    # length of the consecutive same-axis run ending at the last angle
    n = 1
    while n < len(angles) and delta_axis(angles[-n], angles[-n - 1]) < np.radians(SAME_AXIS_DEG):
        n += 1
    return n


def check_same_axis():
    assert np.isclose(delta_axis(0.01, np.pi - 0.01), 0.02) and np.isclose(delta_axis(0.3, 0.1), 0.2)
    r = np.radians
    assert same_axis_run([r(2), r(175), r(10)]) == 3 and same_axis_run([r(2), r(90), r(100)]) == 2 and same_axis_run([r(45)]) == 1


def make_sequences(rec, n_seq, n_grip, seed, margin, max_same_axis=None):
    """Gaps stratified over GAP_BINS (extra samples go to the last bin), sorted into n_grip strength quantiles;
    sequence s puts quantile q at grip (q + s) % n_grip (Latin square). midpoint/angle rejection-sampled with the pin filter."""
    rng = np.random.default_rng(seed)
    total = n_seq * n_grip
    counts = [total // len(GAP_BINS)] * len(GAP_BINS)
    counts[-1] += total - sum(counts)
    gaps = np.sort(np.concatenate([rng.uniform(lo, hi, c) for (lo, hi), c in zip(GAP_BINS, counts)]))
    grid = np.zeros((n_seq, n_grip), int)
    for q in range(n_grip):
        members = rng.permutation(np.arange(q * n_seq, (q + 1) * n_seq))
        for s_ in range(n_seq):
            grid[s_, (q + s_) % n_grip] = members[s_]
    out = {k: [] for k in ('midpoint', 'angle', 'target_gap', 'gripper_rate', 'finger_start_pose', 'primitive_action_sequence',
                           'gap_quantile', 'analytic_pin_distance', 'rejected_draws', 'axis_resampled_draws')}
    for s_ in range(n_seq):
        row = {k: [] for k in out}
        for k in range(n_grip):
            gap, rejected = gaps[grid[s_, k]], 0
            while True:
                off = np.array([rng.uniform(*rec['midpoint_x']), rng.uniform(*rec['midpoint_z'])]) - MID[[0, 2]]
                ang = rng.uniform(*rec['angle'])
                if not mc_pin_collision(off[None], np.array([ang]), np.array([gap]), margin)[0]:
                    break
                rejected += 1
            # history-only rule: at most max_same_axis consecutive grips on the same closing axis; resample angle only
            axis_resampled = 0
            while max_same_axis and same_axis_run(row['angle'] + [ang]) > max_same_axis:
                ang = rng.uniform(*rec['angle'])
                axis_resampled += 1
                if mc_pin_collision(off[None], np.array([ang]), np.array([gap]), margin)[0]:
                    ang = row['angle'][-1]   # keep violating -> loop again (pin filter still applies to the new angle)
                assert axis_resampled < 10000
            mid = np.array([MID[0] + off[0], rec['midpoint_y'], MID[2] + off[1]])
            p1, p2 = pose_from(mid, ang)
            u = np.array([np.cos(ang), -np.sin(ang)])
            row['midpoint'].append(mid); row['angle'].append(ang); row['target_gap'].append(gap)
            row['gripper_rate'].append(st.gap_to_rate(gap)); row['finger_start_pose'].append(np.stack([p1, p2]))
            row['primitive_action_sequence'].append(g.make_grip_actions(p1, p2, st.gap_to_rate(gap)))
            row['gap_quantile'].append(grid[s_, k] // n_seq)
            row['analytic_pin_distance'].append(float(analytic_pin_distance(gap, off @ u, abs(off[0] * u[1] - off[1] * u[0]))))
            row['rejected_draws'].append(rejected)
            row['axis_resampled_draws'].append(axis_resampled)
        for k_ in out:
            out[k_].append(np.array(row[k_]))
    out = {k: np.stack(v) for k, v in out.items()}
    out['gap_bin_counts'] = np.array([[((out['target_gap'] >= lo) & (out['target_gap'] < hi)).sum() for lo, hi in GAP_BINS]])
    assert (out['analytic_pin_distance'] >= margin).all()
    assert (np.sort(out['gap_quantile'], axis=1) == np.arange(n_grip)).all()   # each sequence: one gap per quantile
    if max_same_axis:
        assert all(same_axis_run(list(out['angle'][s_, :k + 1])) <= max_same_axis for s_ in range(n_seq) for k in range(n_grip))
    return out


def sequential(args):
    import types
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pilot = os.path.abspath(args.pilot_dir)
    with open(os.path.join(pilot, 'recommended_action_region.json')) as f:
        rec = json.load(f)
    assert rec['midpoint_x'] == [0.47, 0.53] and rec['midpoint_z'] == [0.47, 0.53] and rec['target_gap'] == [0.07, 0.13], rec
    with open(os.path.join(pilot, 'metrics.json')) as f:
        settled_path = json.load(f)['settled_state']
    out = os.path.join(pilot, 'sequential_5x5')
    os.makedirs(os.path.join(out, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(out, 'visuals'), exist_ok=True)
    print(f'output size before: {du(out)}  disk free: {disk_free()}')

    seqs = make_sequences(rec, args.n_seq, args.n_grip, args.seed, args.pin_margin)
    np.savez_compressed(os.path.join(out, 'sequences.npz'), seed=np.array(args.seed), gap_bins=np.array(GAP_BINS), **seqs)
    seqs = dict(np.load(os.path.join(out, 'sequences.npz')))   # run from the saved file
    print('gap bin counts', seqs['gap_bin_counts'][0].tolist())

    with open(settled_path, 'rb') as f:
        settled = pickle.load(f)
    env, _ = g.make_env()
    rmg.update_camera(env)
    env.number_of_cams = 1
    t = types.SimpleNamespace(**rec['thresholds'])
    rows, episodes = [], []
    for s_ in range(args.n_seq):
        msd.restore(env, settled, MATERIAL)
        err = msd.state_error(env.get_state(), settled)
        assert all(v == 0 for v in err.values()), err
        cand = {k: seqs[k][s_] for k in ('midpoint', 'angle', 'target_gap', 'gripper_rate', 'finger_start_pose', 'primitive_action_sequence')}
        frame_dir = os.path.join(out, 'visuals', f'seq_{s_:02d}_frames')
        os.makedirs(frame_dir, exist_ok=True)
        n_img = [0]

        def save_img(_=None):
            cv2.imwrite(os.path.join(frame_dir, f'{n_img[0]:03d}.png'), rmg.render(env)[..., ::-1]); n_img[0] += 1

        save_img()
        init_cloud = g.read_object_cloud(env)
        init_max_y = env.simulator.get_x(0)[:, 1].max()
        prev_cloud, prev_state, boundary = init_cloud, env.get_state(), []
        failed = None
        for k in range(args.n_grip):
            state_before, before = env.get_state(), g.read_object_cloud(env)
            boundary.append(max([float(np.abs(before - prev_cloud).max())] + list(msd.state_error(state_before, prev_state).values())))
            try:
                d, full = run_grip(env, cand, k, on_frame=save_img)
            except ValueError as e:
                failed = f'grip {k}: ValueError {e}'; break
            prev_cloud, prev_state = g.read_object_cloud(env), env.get_state()   # separate reads after render
            assert np.array_equal(prev_cloud, d['frame39_object_cloud'])
            m = metrics(d, full)
            m.update(sequence_id=s_, grip_index=k, gap_quantile=int(seqs['gap_quantile'][s_, k]),
                     analytic_pin_distance=float(seqs['analytic_pin_distance'][s_, k]),
                     height_ratio_vs_episode_initial=float(full[39][:, 1].max() / init_max_y))
            m['quality_class'] = classify(m, t)
            d.update(sequence_id=np.array(s_), grip_index=np.array(k))
            np.savez_compressed(os.path.join(out, 'runs', f'seq{s_:02d}_grip{k}.npz'), **d)
            rows.append(m)
            print(f"seq{s_} g{k} gap {m['target_gap']:.3f} q{m['gap_quantile']} mid ({m['midpoint_x']:.3f},{m['midpoint_z']:.3f}) "
                  f"ang {m['angle_deg']:5.1f} | comp {m['compression_RMS']:.4f} nres {m['normalized_residual']:.2f} maxd {m['max_disp_f29']:.3f} "
                  f"| pin {m['finger_pin_distance']:+.4f} (analytic {m['analytic_pin_distance']:+.4f}) clr {m['min_finger_clearance']:+.4f} "
                  f"contact {m['max_contact_particles']} first {m['first_contact_frame']} | {m['quality_class']}", flush=True)
        rmg.encode(frame_dir, os.path.join(out, 'visuals', f'seq_{s_:02d}.mp4'))
        strip = [cv2.resize(label_img(cv2.imread(os.path.join(frame_dir, f'{40 * p:03d}.png')), f'seq{s_} P{p}'), (320, 320))
                 for p in range(args.n_grip + 1) if os.path.exists(os.path.join(frame_dir, f'{40 * p:03d}.png'))]
        cv2.imwrite(os.path.join(out, 'visuals', f'seq_{s_:02d}_states.png'), np.hstack(strip))
        shutil.rmtree(frame_dir)
        episodes.append(dict(sequence_id=s_, failed=failed, initial_state_error=err, boundary_max_error=boundary))
        assert all(b == 0 for b in boundary), boundary

    # single-grip validation on the pristine object, for comparison
    with open(os.path.join(pilot, 'validation_region', 'summary.json')) as f:
        single = [m for m in json.load(f)['runs'] if 'failed' not in m]
    A = lambda k, ms=rows: np.array([m[k] for m in ms])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(A('target_gap', single), A('compression_RMS', single), c='lightgray', s=40, label='single grip (P0), 20')
    sc = axes[0].scatter(A('target_gap'), A('compression_RMS'), c=A('grip_index'), cmap='viridis', s=60,
                         marker='o', edgecolors=['r' if m['quality_class'] != 'GOOD' else 'w' for m in rows], label='sequential')
    axes[0].axhline(t.comp_min, ls='--', c='gray'); axes[0].set_xlabel('target gap'); axes[0].set_ylabel('compression RMS (vs grip before)')
    plt.colorbar(sc, ax=axes[0], label='grip index (0 = pristine P0)'); axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
    for s_ in range(args.n_seq):
        ms = [m for m in rows if m['sequence_id'] == s_]
        axes[1].plot(A('grip_index', ms), A('compression_RMS', ms), 'o-', label=f'seq {s_}')
    axes[1].axhline(t.comp_min, ls='--', c='gray'); axes[1].set_xlabel('grip index'); axes[1].set_ylabel('compression RMS')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'sequential_compression.png'), dpi=110); plt.close(fig)

    cols = ['sequence_id', 'grip_index', 'gap_quantile', 'midpoint_x', 'midpoint_z', 'angle_deg', 'target_gap', 'actual_gap_f29',
            'compression_RMS', 'residual_RMS', 'normalized_residual', 'max_disp_f29', 'min_finger_clearance', 'finger_pin_distance',
            'analytic_pin_distance', 'both_fingers_contact', 'first_contact_frame', 'max_contact_particles', 'height_ratio_f39',
            'height_ratio_vs_episode_initial', 'max_step_disp', 'quality_class']
    with open(os.path.join(out, 'grip_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    def stats(ms):
        return dict(n=len(ms), **{c: sum(m['quality_class'] == c for m in ms) for c in CLASSES},
                    two_finger_contact=sum(m['both_fingers_contact'] for m in ms),
                    comp_min=float(min(A('compression_RMS', ms))), comp_mean=float(A('compression_RMS', ms).mean()),
                    comp_max=float(max(A('compression_RMS', ms))))
    summary = dict(
        region=rec, material=dict(**MATERIAL, nu=msd.NU), seed=args.seed, pin_margin=args.pin_margin,
        gap_bin_counts=dict(zip([f'[{lo},{hi})' for lo, hi in GAP_BINS], seqs['gap_bin_counts'][0].tolist())),
        rejected_draws_total=int(seqs['rejected_draws'].sum()), episodes=episodes, overall=stats(rows),
        by_grip_index={k: stats([m for m in rows if m['grip_index'] == k]) for k in range(args.n_grip)},
        by_gap_bin={f'[{lo},{hi})': stats([m for m in rows if lo <= m['target_gap'] < hi]) for lo, hi in GAP_BINS},
        gap_012_013_by_grip_index=[dict(sequence_id=m['sequence_id'], grip_index=m['grip_index'], gap=m['target_gap'],
                                        compression_RMS=m['compression_RMS'], both_fingers_contact=m['both_fingers_contact'],
                                        first_contact_frame=m['first_contact_frame'], quality_class=m['quality_class'])
                                   for m in rows if m['target_gap'] >= 0.12],
        stability=dict(nan=int(A('nan').sum()), inf=int(A('inf').sum()), explosion=bool(A('explosion').any()),
                       max_step_disp=float(A('max_step_disp').max()), min_finger_pin_distance=float(A('finger_pin_distance').min()),
                       min_finger_clearance=float(A('min_finger_clearance').min()),
                       height_ratio_vs_episode_initial=[float(A('height_ratio_vs_episode_initial').min()), float(A('height_ratio_vs_episode_initial').max())],
                       analytic_vs_sim_pin_max_error=float(np.abs(A('analytic_pin_distance') - A('finger_pin_distance')).max()),
                       boundary_max_error=max(max(e['boundary_max_error']) for e in episodes)),
        single_grip_reference=stats(single), rows=rows, output_size=du(out), disk_free=disk_free())
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'region', 'episodes')}, indent=1))
    print(f'output size after: {du(out)}  disk free: {disk_free()}\noutputs: {out}')


SEQ_KEYS = ('midpoint', 'angle', 'target_gap', 'gripper_rate', 'finger_start_pose', 'primitive_action_sequence')


def width_margin(before, mid, angle, gap):
    # extent of the pre-grip cloud along the closing axis inside the finger-radius slab, minus target gap
    u = np.array([np.cos(angle), 0, -np.sin(angle)])
    rel = before - mid
    along = rel @ u
    slab = np.linalg.norm(rel - along[:, None] * u, axis=1) < msd.TOOL_SIZE
    return float(np.ptp(along[slab]) - gap) if slab.sum() > 2 else -gap


def run_episode(env, settled, material, cand, t, frame_dir):
    """restore -> n grips without reset; frames (jpg) written to frame_dir, index 40*p = P_p."""
    msd.restore(env, settled, material)
    err = msd.state_error(env.get_state(), settled)
    assert all(v == 0 for v in err.values()), err
    n_img = [0]

    def save_img(_=None):
        cv2.imwrite(os.path.join(frame_dir, f'{n_img[0]:03d}.jpg'), rmg.render(env)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_img[0] += 1

    save_img()
    init_cloud = g.read_object_cloud(env)
    init_max_y = env.simulator.get_x(0)[:, 1].max()
    prev_cloud, prev_state = init_cloud, env.get_state()
    rows, datas, boundary, failed = [], [], [], None
    for k in range(len(cand['angle'])):
        state_before, before = env.get_state(), g.read_object_cloud(env)
        boundary.append(dict(object=float(np.abs(before - prev_cloud).max()), **msd.state_error(state_before, prev_state)))
        try:
            d, full = run_grip(env, cand, k, on_frame=save_img)
        except ValueError as e:
            failed = f'grip {k}: ValueError {e}'; break
        prev_cloud, prev_state = g.read_object_cloud(env), env.get_state()   # separate reads after the render
        assert np.array_equal(prev_cloud, d['frame39_object_cloud'])
        m = metrics(d, full)
        m.update(grip_index=k, sigma_y=float(material['yield_stress']), recovery_ratio=1 - m['normalized_residual'],
                 analytic_pin_distance=float(analytic_pin_distance(m['target_gap'], m['offset_along_axis'], m['offset_perp_axis'])),
                 width_margin=width_margin(before, d['midpoint'], float(d['angle']), float(d['target_gap'])),
                 weak_finger_max_contact=int(min(m['max_contact_particles'])),
                 height_ratio_vs_episode_initial=float(full[39][:, 1].max() / init_max_y))
        m['one_finger_contact'] = bool(any(f is not None for f in m['first_contact_frame']) and not m['both_fingers_contact'])
        m['quality_class'] = classify(m, t)
        d.update(E=np.array(material['E']), sigma_y=np.array(material['yield_stress']), grip_id=np.array(k))
        rows.append(m); datas.append(d)
    return rows, datas, dict(initial_state_error=err, boundary=boundary, failed=failed, n_frames=n_img[0])


def final_check(args):
    import types
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check_same_axis()
    pilot = os.path.abspath(args.pilot_dir)
    with open(os.path.join(pilot, 'recommended_action_region.json')) as f:
        rec = json.load(f)
    assert rec['midpoint_x'] == [0.47, 0.53] and rec['midpoint_z'] == [0.47, 0.53] and rec['target_gap'] == [0.07, 0.13], rec
    with open(os.path.join(pilot, 'metrics.json')) as f:
        settled_path = json.load(f)['settled_state']
    out = os.path.join(pilot, 'final_sequence_check')
    vis, tmp = os.path.join(out, 'visuals'), os.path.join(out, 'visuals', '_frames')
    for sub_ in ['sequences', 'visuals'] + [f'sigma{sy:g}' for sy in args.sigmas]:
        os.makedirs(os.path.join(out, sub_), exist_ok=True)
    size_before = du(out)
    print(f'output size before: {size_before}  disk free: {disk_free()}')

    seqs = make_sequences(rec, args.n_seq, args.n_grip, args.seed, args.pin_margin, max_same_axis=2)
    for s_ in range(args.n_seq):
        np.savez_compressed(os.path.join(out, 'sequences', f'paired_sequence_{s_:02d}.npz'), sequence_id=np.array(s_),
                            seed=np.array(args.seed), **{k: v[s_] for k, v in seqs.items() if k != 'gap_bin_counts'})
    loaded = [dict(np.load(os.path.join(out, 'sequences', f'paired_sequence_{s_:02d}.npz'))) for s_ in range(args.n_seq)]
    print('gap bin counts', seqs['gap_bin_counts'][0].tolist(), 'axis resampled draws', seqs['axis_resampled_draws'].tolist())

    with open(settled_path, 'rb') as f:
        settled = pickle.load(f)
    env, _ = g.make_env()
    rmg.update_camera(env)
    env.number_of_cams = 1
    t = types.SimpleNamespace(**rec['thresholds'])
    rows, episodes, init_clouds = [], {}, []
    for sy in args.sigmas:
        material = dict(E=MATERIAL['E'], yield_stress=float(sy))
        for s_ in range(args.n_seq):
            frame_dir = os.path.join(tmp, f'sigma{sy:g}_seq{s_:02d}')
            os.makedirs(frame_dir, exist_ok=True)
            cand = {k: loaded[s_][k] for k in SEQ_KEYS}
            ms, ds, info = run_episode(env, settled, material, cand, t, frame_dir)
            for m in ms:
                m['sequence_id'] = s_
                print(f"sigma{sy:g} seq{s_} g{m['grip_index']} gap {m['target_gap']:.3f} ang {m['angle_deg']:5.1f} | comp {m['compression_RMS']:.4f} "
                      f"res {m['residual_RMS']:.4f} nres {m['normalized_residual']:.2f} | pin {m['finger_pin_distance']:+.4f} "
                      f"clr {m['min_finger_clearance']:+.4f} contact {m['max_contact_particles']} margin {m['width_margin']:+.3f} "
                      f"| {m['quality_class']}{' ONE-FINGER' if m['one_finger_contact'] else ''}", flush=True)
            rows += ms
            if ds:
                ep = {k: np.stack([d[k] for d in ds]) for k in ds[0]}
                ep.update(sequence_id=np.array(s_), sigma_y=np.array(float(sy)), E=np.array(MATERIAL['E']), nu=np.array(msd.NU))
                assert np.array_equal(ep['primitive_action_sequence'], loaded[s_]['primitive_action_sequence'][:len(ds)])
                init_clouds.append(ds[0]['before_object_cloud'])
                np.savez_compressed(os.path.join(out, f'sigma{sy:g}', f'seq_{s_:02d}_episode.npz'), **ep)
            episodes[f'sigma{sy:g}_seq{s_:02d}'] = info
            b = info['boundary']
            assert all(v == 0 for x in b for v in x.values()), (sy, s_, b)
    assert all(np.array_equal(c, init_clouds[0]) for c in init_clouds)

    # ---- visuals: 2 x 5 synchronized grid video, per-sequence 2 x 6 state grid, 2 x 5 final-state image ----
    n_frames = args.n_grip * g.N_FRAMES + 1

    def frame(sy, s_, i):
        d = os.path.join(tmp, f'sigma{sy:g}_seq{s_:02d}')
        i = min(i, episodes[f'sigma{sy:g}_seq{s_:02d}']['n_frames'] - 1)   # a failed episode freezes on its last frame
        return cv2.imread(os.path.join(d, f'{i:03d}.jpg'))

    def lab(img, lines):
        for j, text in enumerate(lines):
            for color, thick in [((0, 0, 0), 5), ((255, 255, 255), 2)]:
                cv2.putText(img, text, (12, 34 + 34 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thick, cv2.LINE_AA)
        return img

    grid_dir = os.path.join(tmp, 'grid')
    os.makedirs(grid_dir, exist_ok=True)
    for i in range(n_frames):
        stage = 'P0 (initial)' if i == 0 else f'grip{(i - 1) // g.N_FRAMES} frame {(i - 1) % g.N_FRAMES}'
        cv2.imwrite(os.path.join(grid_dir, f'{i:03d}.png'),
                    np.vstack([np.hstack([lab(frame(sy, s_, i), [f'sigma_y = {sy:g}', f'seq = {s_:02d}', stage])
                                          for s_ in range(args.n_seq)]) for sy in args.sigmas]))
    rmg.encode(grid_dir, os.path.join(vis, 'grid_2x5_sigma30_500_sequences.mp4'))
    for s_ in range(args.n_seq):
        cv2.imwrite(os.path.join(vis, f'seq_{s_:02d}_state_grid_2x6.png'),
                    np.vstack([np.hstack([cv2.resize(lab(frame(sy, s_, g.N_FRAMES * p), [f'sigma_y={sy:g}', f'seq {s_:02d} P{p}']), (320, 320))
                                          for p in range(args.n_grip + 1)]) for sy in args.sigmas]))
    cv2.imwrite(os.path.join(vis, 'final_P5_2x5.png'),
                np.vstack([np.hstack([cv2.resize(lab(frame(sy, s_, n_frames - 1), [f'sigma_y={sy:g}', f'seq {s_:02d} P5']), (384, 384))
                                      for s_ in range(args.n_seq)]) for sy in args.sigmas]))
    shutil.rmtree(tmp)

    # ---- compression plot ----
    fig, axes = plt.subplots(1, len(args.sigmas), figsize=(6 * len(args.sigmas), 4.5), sharey=True)
    for ax, sy in zip(np.atleast_1d(axes), args.sigmas):
        for s_ in range(args.n_seq):
            ms = [m for m in rows if m['sigma_y'] == sy and m['sequence_id'] == s_]
            ax.plot([m['grip_index'] for m in ms], [m['compression_RMS'] for m in ms], 'o-', label=f'seq {s_:02d}')
            for m in ms:
                if m['quality_class'] != 'GOOD':
                    ax.scatter(m['grip_index'], m['compression_RMS'], s=160, facecolors='none', edgecolors='r')
        ax.axhline(t.comp_min, ls='--', c='gray'); ax.set_title(f'sigma_y = {sy:g}'); ax.set_xlabel('grip id'); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel('compression RMS (vs grip before)'); np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(vis, 'compression_by_grip.png'), dpi=110); plt.close(fig)

    cols = ['sigma_y', 'sequence_id', 'grip_index', 'midpoint_x', 'midpoint_z', 'angle_deg', 'target_gap', 'gripper_rate', 'actual_gap_f29',
            'compression_RMS', 'residual_RMS', 'normalized_residual', 'recovery_ratio', 'max_disp_f29', 'max_step_disp',
            'min_finger_clearance', 'finger_pin_distance', 'analytic_pin_distance', 'both_fingers_contact', 'one_finger_contact',
            'first_contact_frame', 'max_contact_particles', 'width_margin', 'height_ratio_f39', 'height_ratio_vs_episode_initial', 'quality_class']
    with open(os.path.join(out, 'grip_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

    def stats(ms):
        c = np.array([m['compression_RMS'] for m in ms])
        return dict(n=len(ms), **{k: sum(m['quality_class'] == k for m in ms) for k in CLASSES},
                    one_finger_contact=sum(m['one_finger_contact'] for m in ms), two_finger_contact=sum(m['both_fingers_contact'] for m in ms),
                    marginal_contact_lt50=sum(m['weak_finger_max_contact'] < 50 for m in ms),
                    comp_min=float(c.min()), comp_mean=float(c.mean()), comp_max=float(c.max()),
                    nan=sum(m['nan'] for m in ms), inf=sum(m['inf'] for m in ms), explosion=sum(m['explosion'] for m in ms),
                    min_pin=float(min(m['finger_pin_distance'] for m in ms)), min_clearance=float(min(m['min_finger_clearance'] for m in ms)),
                    max_step_disp=float(max(m['max_step_disp'] for m in ms)),
                    height_ratio_vs_initial=[float(min(m['height_ratio_vs_episode_initial'] for m in ms)), float(max(m['height_ratio_vs_episode_initial'] for m in ms))])
    old = dict(np.load(os.path.join(pilot, 'sequential_5x5', 'sequences.npz')))
    summary = dict(
        conditions=dict(E=MATERIAL['E'], nu=msd.NU, sigma_y=args.sigmas, num_sequences=args.n_seq, grips_per_sequence=args.n_grip,
                        total_episodes=args.n_seq * len(args.sigmas), total_grips=args.n_seq * len(args.sigmas) * args.n_grip,
                        seed=args.seed, pin_margin=args.pin_margin, same_axis_deg=SAME_AXIS_DEG, max_same_axis=2, settled_state=settled_path),
        sequences=[dict(sequence_id=s_, grips=[dict(midpoint=loaded[s_]['midpoint'][k].tolist(), angle_deg=float(np.degrees(loaded[s_]['angle'][k])),
                                                    target_gap=float(loaded[s_]['target_gap'][k]), gripper_rate=float(loaded[s_]['gripper_rate'][k]),
                                                    analytic_pin_distance=float(loaded[s_]['analytic_pin_distance'][k]),
                                                    axis_resampled_draws=int(loaded[s_]['axis_resampled_draws'][k]),
                                                    delta_axis_prev_deg=None if k == 0 else float(np.degrees(delta_axis(loaded[s_]['angle'][k], loaded[s_]['angle'][k - 1]))))
                                               for k in range(args.n_grip)]) for s_ in range(args.n_seq)],
        gap_bin_counts=dict(zip([f'[{lo},{hi})' for lo, hi in GAP_BINS], seqs['gap_bin_counts'][0].tolist())),
        same_axis=dict(new_max_run=int(max(same_axis_run(list(seqs['angle'][s_, :k + 1])) for s_ in range(args.n_seq) for k in range(args.n_grip))),
                       new_axis_resampled_grips=int((seqs['axis_resampled_draws'] > 0).sum()),
                       old_sequential_5x5_max_run=int(max(same_axis_run(list(old['angle'][s_, :k + 1])) for s_ in range(len(old['angle'])) for k in range(old['angle'].shape[1]))),
                       old_sequential_5x5_grips_violating=[(s_, k) for s_ in range(len(old['angle'])) for k in range(old['angle'].shape[1])
                                                           if same_axis_run(list(old['angle'][s_, :k + 1])) > 2]),
        action_identity='primitive actions asserted equal to saved sequence for every episode (both sigma_y)',
        initial_identity='x,v,F,C,finger0,finger1,pin == settled (asserted per episode); 300-cloud identical across all episodes (asserted)',
        boundary_max_error=dict(object=max(x['object'] for e in episodes.values() for x in e['boundary']),
                                full_state=max(v for e in episodes.values() for x in e['boundary'] for kk, v in x.items() if kk != 'object')),
        failed_episodes={k: e['failed'] for k, e in episodes.items() if e['failed']},
        by_sigma={f'{sy:g}': stats([m for m in rows if m['sigma_y'] == sy]) for sy in args.sigmas},
        by_sigma_sequence={f'{sy:g}_seq{s_:02d}': stats([m for m in rows if m['sigma_y'] == sy and m['sequence_id'] == s_])
                           for sy in args.sigmas for s_ in range(args.n_seq)},
        by_sigma_grip={f'{sy:g}_g{k}': stats([m for m in rows if m['sigma_y'] == sy and m['grip_index'] == k])
                       for sy in args.sigmas for k in range(args.n_grip)},
        corr_width_margin_compression={f'{sy:g}': float(np.corrcoef([m['width_margin'] for m in rows if m['sigma_y'] == sy],
                                                                   [m['compression_RMS'] for m in rows if m['sigma_y'] == sy])[0, 1]) for sy in args.sigmas},
        episodes=episodes, rows=rows, output_size_before=size_before, output_size=du(out), disk_free=disk_free())
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    write_final_report(summary, out)


def write_final_report(s, out):
    c = s['conditions']
    L = ['[Conditions]', '', f"E: {c['E']:g}", f"nu: {c['nu']}", f"sigma_y: {c['sigma_y']}", f"num_sequences: {c['num_sequences']}",
         f"grips_per_sequence: {c['grips_per_sequence']}", f"total_episodes: {c['total_episodes']}", f"total_grips: {c['total_grips']}",
         'action rule: midpoint x,z ~ U(0.47,0.53), y 0.14; angle ~ U(0,pi); gap ~ U(0.07,0.13) stratified over 6 bins (Latin square of strength quintiles);',
         f"             reject if analytic pin clearance < {c['pin_margin']}; same axis (delta_axis < {c['same_axis_deg']:g} deg) at most {c['max_same_axis']} consecutive, 3rd -> resample angle only",
         f"seed: {c['seed']}", f"gap bin counts: {s['gap_bin_counts']}", f"initial identity: {s['initial_identity']}", f"action identity: {s['action_identity']}", '',
         '-' * 32, '[Sequence definitions]', '']
    for sq in s['sequences']:
        L.append(f"sequence_{sq['sequence_id']:02d}:")
        for k, gr in enumerate(sq['grips']):
            dprev = '   -  ' if gr['delta_axis_prev_deg'] is None else f"{gr['delta_axis_prev_deg']:5.1f}"
            L.append(f"  grip{k} midpoint ({gr['midpoint'][0]:.4f}, {gr['midpoint'][2]:.4f}) angle {gr['angle_deg']:6.1f} deg (d_axis prev {dprev}) "
                     f"gap {gr['target_gap']:.4f} rate {gr['gripper_rate']:.6f} pin_analytic {gr['analytic_pin_distance']:+.4f} axis_resampled {gr['axis_resampled_draws']}")
        L.append('')
    sa = s['same_axis']
    L += [f"same-axis: new max consecutive run {sa['new_max_run']}, grips whose angle was resampled {sa['new_axis_resampled_grips']}",
          f"           old sequential_5x5 max run {sa['old_sequential_5x5_max_run']}, grips that would violate the rule {sa['old_sequential_5x5_grips_violating']}", '',
          '-' * 32, '[Validation]', '']
    for sy, st_ in s['by_sigma'].items():
        L += [f"sigma_y={sy}: pin collision {st_['PIN_COLLISION']} (min {st_['min_pin']:+.4f}) | invalid geometry {st_['INVALID_GEOMETRY']} | "
              f"weak {st_['TOO_WEAK']} | aggressive {st_['TOO_AGGRESSIVE']} | GOOD {st_['GOOD']}/{st_['n']}",
              f"           one-finger contact {st_['one_finger_contact']} | two-finger {st_['two_finger_contact']}/{st_['n']} | weaker finger < 50 particles {st_['marginal_contact_lt50']}",
              f"           NaN {st_['nan']} Inf {st_['inf']} explosion {st_['explosion']} | max per-step disp {st_['max_step_disp']:.4f} | min clearance {st_['min_clearance']:+.4f} | "
              f"height ratio vs initial {np.round(st_['height_ratio_vs_initial'], 3).tolist()} | comp {st_['comp_min']:.4f}-{st_['comp_max']:.4f} mean {st_['comp_mean']:.4f}"]
    L += [f"ValueError episodes: {s['failed_episodes'] or 'none'}",
          f"boundary continuity (initial==g0 before, g_k after==g_k+1 before, all episodes): object max {s['boundary_max_error']['object']:.1e}, full state max {s['boundary_max_error']['full_state']:.1e}",
          f"corr(pre-grip width margin, compression): {s['corr_width_margin_compression']}", '']
    L += ['per sequence:'] + [f"  sigma{k}: GOOD {v['GOOD']} weak {v['TOO_WEAK']} invalid {v['INVALID_GEOMETRY']} one-finger {v['one_finger_contact']} comp min {v['comp_min']:.4f}"
                             for k, v in s['by_sigma_sequence'].items()]
    L += ['', '-' * 32, '[Per-grip summary]', '',
          f"{'sequence':>8} | {'sigma_y':>7} | {'grip':>4} | {'gap':>5} | {'angle':>6} | {'comp_RMS':>8} | {'resid_RMS':>9} | {'nres':>5} | {'recov':>6} | {'margin':>6} | {'contact':>11} | class"]
    for m in sorted(s['rows'], key=lambda m: (m['sequence_id'], -m['sigma_y'], m['grip_index'])):
        L.append(f"{m['sequence_id']:>8} | {m['sigma_y']:>7g} | {m['grip_index']:>4} | {m['target_gap']:.3f} | {m['angle_deg']:6.1f} | {m['compression_RMS']:8.4f} | "
                 f"{m['residual_RMS']:9.4f} | {m['normalized_residual']:5.2f} | {m['recovery_ratio']:+6.2f} | {m['width_margin']:+6.3f} | "
                 f"{str(m['max_contact_particles']):>11} | {m['quality_class']}{' (one-finger)' if m['one_finger_contact'] else ''}"
                 f"{' (boundary: comp >= 0.015)' if m['quality_class'] == 'TOO_WEAK' and m['compression_RMS'] >= 0.015 else ''}")
    L += ['', '-' * 32, '[Visual paths]', '', f"2x5 grid video: {os.path.join(out, 'visuals', 'grid_2x5_sigma30_500_sequences.mp4')}",
          f"state-grid images: {os.path.join(out, 'visuals')}/seq_0{{0..4}}_state_grid_2x6.png",
          f"final P5 comparison: {os.path.join(out, 'visuals', 'final_P5_2x5.png')}",
          f"compression plot: {os.path.join(out, 'visuals', 'compression_by_grip.png')}", '',
          '-' * 32, '[Disk usage]', '', f"output folder size: {s['output_size']} (before {s['output_size_before']})", f"remaining disk: {s['disk_free']}", '',
          '-' * 32, '[Interpretation]', '', '(written after review)', '', '-' * 32, '[Recommendation]', '', '(written after review)', '']
    with open(os.path.join(out, 'report.txt'), 'w') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))


def label_img(img, text):
    for color, thick in [((0, 0, 0), 5), ((255, 255, 255), 2)]:
        cv2.putText(img, text, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, thick, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('simulate'); a.add_argument('--settled-state', required=True)
    b = sub.add_parser('analyze'); b.add_argument('out_dir')
    b.add_argument('--pin-min', type=float, default=0.0)
    b.add_argument('--comp-min', type=float, default=0.01)
    b.add_argument('--height-ratio-min', type=float, default=0.5)
    b.add_argument('--max-disp-max', type=float, default=0.25)
    b.add_argument('--clearance-min', type=float, default=-0.03)
    b.add_argument('--rec-pin-margin', type=float, default=0.005)
    b.add_argument('--rec-offset', type=float, nargs=2, default=list(OFFSET_RANGE))
    b.add_argument('--rec-angle', type=float, nargs=2, default=list(ANGLE_RANGE))
    b.add_argument('--rec-gap', type=float, nargs=2, default=list(GAP_RANGE))
    b.add_argument('--render', action='store_true')
    v = sub.add_parser('validate'); v.add_argument('pilot_dir')
    v.add_argument('--n', type=int, default=20)
    v.add_argument('--seed', type=int, default=11)
    v.add_argument('--pin-margin', type=float, default=0.005)
    q = sub.add_parser('sequential'); q.add_argument('pilot_dir')
    q.add_argument('--n-seq', type=int, default=5)
    q.add_argument('--n-grip', type=int, default=5)
    q.add_argument('--seed', type=int, default=21)
    q.add_argument('--pin-margin', type=float, default=0.005)
    fc = sub.add_parser('final_check'); fc.add_argument('pilot_dir')
    fc.add_argument('--n-seq', type=int, default=5)
    fc.add_argument('--n-grip', type=int, default=5)
    fc.add_argument('--seed', type=int, default=31)
    fc.add_argument('--pin-margin', type=float, default=0.005)
    fc.add_argument('--sigmas', type=float, nargs='+', default=[500.0, 30.0])
    args = ap.parse_args()
    dict(simulate=simulate, analyze=analyze, validate=validate, sequential=sequential, final_check=final_check)[args.cmd](args)


if __name__ == '__main__':
    main()
