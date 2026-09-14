"""B-3B diagnostic: same settled state + same exact grip actions, only E / yield_stress change.

TEMPORARY EXPLORATORY CANDIDATES only -- not the final low/mid/high material range.

Run (from anywhere):
  python material_sensitivity_debug.py --settled-state <dir>/settled_initial_state.pkl \
      [--E-values 2500 5000 10000] [--yield-values 100 200 400] [--action-seq <npz>]

Reuses env/action helpers from gen_multigrip_debug.py (notebook logic). Simulator core untouched.
"""
import argparse
import itertools
import json
import os
import pickle
from datetime import datetime

import numpy as np

import gen_multigrip_debug as g

NU = 0.2
PRE_CONTACT_FRAME = 15
VIZ_FRAME = 25
MAX_CLOSE_FRAME = g.task_params["len_per_grip"] - 1   # 29
ZERO_DRIFT_STEPS = (1, 5, 10, 15, 20)
PRE_CONTACT_FRAMES = (5, 10, 15)
TOOL_SIZE = 0.045
EXPLOSION_STEP_DISP = 0.05   # per-frame particle displacement; fingers move ~0.01/frame
RELAX_CHECKPOINTS = (0, 1, 5, 10, 20, 50, 100)
STAT_NAMES = ['dx_max', 'dx_mean', 'dx_rms', 'v_max', 'v_mean', 'v_rms', 'cum_max', 'cum_mean', 'cum_rms']


def disp(a, b):
    d = np.linalg.norm(a - b, axis=-1)
    return dict(mean=float(d.mean()), rms=float(np.sqrt((d ** 2).mean())), max=float(d.max()))


def fmt(m):
    return f"mean {m['mean']:.3e} / RMS {m['rms']:.3e} / max {m['max']:.3e}"


def make_action_sequence(path):
    """Grip poses/rates/actions generated exactly once (same RNG calls as gen_multigrip_debug.simulate)."""
    np.random.seed(g.SEED)
    poses, rates, angles, actions = [], [], [], []
    for _ in range(g.N_GRIPS):
        prim1, prim2, angle = g.random_pose(g.task_name)
        limit = [(g.task_params['sample_radius'] * 2 - (g.task_params['gripper_rate_limits'][0] + 2 * TOOL_SIZE)) / (2 * g.task_params['len_per_grip']),
                 (g.task_params['sample_radius'] * 2 - (g.task_params['gripper_rate_limits'][1] + 2 * TOOL_SIZE)) / (2 * g.task_params['len_per_grip'])]
        rate = np.random.uniform(*limit)
        poses.append(np.stack([prim1, prim2]).astype(np.float64))
        rates.append(rate)
        angles.append(angle)
        actions.append(g.make_grip_actions(prim1, prim2, rate))
    np.savez_compressed(path, grip_start_tool_pose=np.stack(poses), grip_rate=np.array(rates),
                        grip_angle=np.array(angles), primitive_action_sequence=np.stack(actions), seed=np.array(g.SEED))


def restore(env, settled, mat):
    # every test starts from the ONE shared settled state; material applied afterwards (not part of get_state)
    env.set_state(**settled)
    g.set_parameters(env, yield_stress=mat['yield_stress'], E=mat['E'], nu=NU)


def applied_material(env, mat):
    E, nu = mat['E'], NU
    out = dict(requested=dict(E=E, sigma_y=mat['yield_stress'], nu=nu),
               expected=dict(mu=E / (2 * (1 + nu)), lam=E * nu / ((1 + nu) * (1 - 2 * nu)), yield_stress=mat['yield_stress']))
    applied = {}
    for name, field in [('mu', env.simulator.mu), ('lam', env.simulator.lam), ('yield_stress', env.simulator.yield_stress)]:
        a = field.to_numpy()
        applied[name] = dict(min=float(a.min()), max=float(a.max()), uniform=bool(a.min() == a.max()))
    out['applied'] = applied
    out['match'] = all(np.isclose(applied[k]['min'], out['expected'][k]) and applied[k]['uniform'] for k in applied)
    return out


def state_error(a, b):
    names = ['x', 'v', 'F', 'C', 'prim0', 'prim1', 'pin']
    return {n: float(np.max(np.abs(np.asarray(p) - np.asarray(q)))) for n, p, q in zip(names, a['state'], b['state'])}


def zero_action_drift(env, settled, mat):
    """Diagnostic branch only: material change + zero action. Never used as grip initial state."""
    restore(env, settled, mat)
    x0 = env.simulator.get_x(0)
    out = {}
    for t in range(1, max(ZERO_DRIFT_STEPS) + 1):
        env.step(np.zeros(12))
        if t in ZERO_DRIFT_STEPS:
            speed = np.linalg.norm(env.simulator.get_v(0), axis=1)
            out[t] = dict(disp(env.simulator.get_x(0), x0),
                          v_mean=float(speed.mean()), v_rms=float(np.sqrt((speed ** 2).mean())), v_max=float(speed.max()))
    return out


def stats3(d):
    return [d.max(), d.mean(), np.sqrt((d ** 2).mean())]


def relax_branch(env, n_steps):
    """Diagnostic branch at command end (frame 39): zero action n_steps with fingers parked,
    then restore the frame-39 state so the next grip is exactly the unmodified protocol."""
    snap = env.get_state()
    x0 = x_prev = env.simulator.get_x(0)
    clouds, vels = [g.read_object_cloud(env)], [g.get_obs(env, g.N_OBS)[1][:g.N_OBS].copy()]
    tools, clears, stats = [g.read_tool_pose(env)], [g.capsule_clearance(env, x0)[:2]], []
    for _ in range(n_steps):
        env.step(np.zeros(12))
        x, v = env.simulator.get_x(0), env.simulator.get_v(0)
        stats.append(stats3(np.linalg.norm(x - x_prev, axis=1)) + stats3(np.linalg.norm(v, axis=1))
                     + stats3(np.linalg.norm(x - x0, axis=1)))   # all particles
        clouds.append(g.read_object_cloud(env)); vels.append(g.get_obs(env, g.N_OBS)[1][:g.N_OBS].copy())
        tools.append(g.read_tool_pose(env)); clears.append(g.capsule_clearance(env, x)[:2])
        x_prev = x
    env.set_state(**snap)
    assert np.array_equal(g.read_object_cloud(env), clouds[0])
    # index t of relax_* arrays = t zero-action steps after frame 39 (t=0 is frame 39 itself)
    return dict(relax_object_cloud=np.stack(clouds), relax_object_velocity=np.stack(vels), relax_tool_pose=np.stack(tools),
                relax_clearance=np.array(clears), relax_stats=np.array(stats))


def run_grips(env, settled, mat, seq, relax_steps=0):
    restore(env, settled, mat)
    res = dict(state_error_vs_settled=state_error(env.get_state(), settled),
               material=applied_material(env, mat))
    episode_initial_object_cloud = g.read_object_cloud(env)
    initial_tool_pose = g.read_tool_pose(env)

    before, after, start_pose, obj_roll, tool_roll, clear_roll, relax, obj_vel = [], [], [], [], [], [], [], []
    for k in range(g.N_GRIPS):
        before.append(g.read_object_cloud(env))
        pose = seq['grip_start_tool_pose'][k]
        g.update_primitive(env, pose[0], pose[1])   # exact saved teleport pose
        g.select_tool(env, TOOL_SIZE)
        start_pose.append(g.read_tool_pose(env))
        objs, tools, clears, vels = [], [], [], []
        # before[k] -> action[0] -> env.step -> rollout[k, 0] ... rollout[k, 39] == after[k]
        for act in seq['primitive_action_sequence'][k]:
            env.step(act)
            objs.append(g.read_object_cloud(env))
            vels.append(g.get_obs(env, g.N_OBS)[1][:g.N_OBS].copy())
            tools.append(g.read_tool_pose(env))
            clears.append(g.capsule_clearance(env, env.simulator.get_x(0))[:2])   # full particles, fingers only
        after.append(g.read_object_cloud(env))
        if relax_steps:
            relax.append(relax_branch(env, relax_steps))
        obj_vel.append(np.stack(vels)); obj_roll.append(np.stack(objs)); tool_roll.append(np.stack(tools)); clear_roll.append(np.array(clears))

    res['data'] = dict(
        episode_initial_object_cloud=episode_initial_object_cloud, initial_tool_pose=initial_tool_pose,
        before_object_cloud=np.stack(before), after_object_cloud=np.stack(after),
        grip_start_tool_pose=np.stack(start_pose), object_rollout=np.stack(obj_roll), object_velocity_rollout=np.stack(obj_vel), tool_rollout=np.stack(tool_roll),
        finger_clearance=np.stack(clear_roll), primitive_action_sequence=seq['primitive_action_sequence'],
        grip_rate=seq['grip_rate'], grip_angle=seq['grip_angle'], seed=seq['seed'],
        E=np.array(mat['E']), yield_stress=np.array(mat['yield_stress']), nu=np.array(NU),
        applied_mu=np.array(res['material']['applied']['mu']['min']),
        applied_lam=np.array(res['material']['applied']['lam']['min']),
        applied_yield_stress=np.array(res['material']['applied']['yield_stress']['min']),
    )
    for key in (relax[0] if relax else {}):
        res['data'][key] = np.stack([r[key] for r in relax])
    return res


def analyze(res, dx):
    d = res['data']
    roll = d['object_rollout']
    m = {}
    m['pre_contact_f15'] = disp(roll[0, PRE_CONTACT_FRAME], d['episode_initial_object_cloud'])
    m['pre_contact'] = {f: disp(roll[0, f], d['episode_initial_object_cloud']) for f in PRE_CONTACT_FRAMES}
    m['bbox'] = dict(min=roll.min(axis=(0, 1, 2)).tolist(), max=roll.max(axis=(0, 1, 2)).tolist())
    minc = d['finger_clearance'].min(axis=2)   # (grips, frames)
    m['first_contact_frame_tol0'] = [int(np.argmax(c <= 0)) if (c <= 0).any() else None for c in minc]
    m['first_contact_frame_tol_dx'] = [int(np.argmax(c <= dx)) if (c <= dx).any() else None for c in minc]
    m['min_finger_clearance'] = [float(c.min()) for c in minc]
    m['deformation'] = [dict(f25=disp(roll[k, VIZ_FRAME], d['before_object_cloud'][k]),
                             f29=disp(roll[k, MAX_CLOSE_FRAME], d['before_object_cloud'][k]),
                             after=disp(d['after_object_cloud'][k], d['before_object_cloud'][k]))
                        for k in range(g.N_GRIPS)]
    for dfm in m['deformation']:
        dfm['recovery_rms'] = dfm['f29']['rms'] - dfm['after']['rms']   # >0 recovered after release, <0 kept flowing
        dfm['recovery_ratio'] = dfm['recovery_rms'] / dfm['f29']['rms'] if dfm['f29']['rms'] > 1e-9 else None
    seq = np.concatenate([d['before_object_cloud'][:, None], roll], axis=1)
    step = np.linalg.norm(np.diff(seq, axis=1), axis=-1)
    m['max_frame_step_disp'] = float(step.max())
    if 'relax_stats' in d:
        m['max_relax_step_disp_all_particles'] = float(d['relax_stats'][..., 0].max())
        m['max_frame_step_disp'] = max(m['max_frame_step_disp'], m['max_relax_step_disp_all_particles'])
    m['nan'] = int(sum(np.isnan(v).sum() for v in d.values() if np.issubdtype(np.asarray(v).dtype, np.floating)))
    m['inf'] = int(sum(np.isinf(v).sum() for v in d.values() if np.issubdtype(np.asarray(v).dtype, np.floating)))
    m['explosion'] = bool(m['max_frame_step_disp'] > EXPLOSION_STEP_DISP or roll.min() < 0 or roll.max() > 1)
    m['height'] = dict(initial_max_y=float(d['episode_initial_object_cloud'][:, 1].max()),
                       after_max_y=[float(a[:, 1].max()) for a in d['after_object_cloud']])
    return m


VIZ_STATES = [('initial', lambda d: (d['episode_initial_object_cloud'], d['initial_tool_pose'])),
              ('grip0_f25', lambda d: (d['object_rollout'][0, VIZ_FRAME], d['tool_rollout'][0, VIZ_FRAME])),
              ('grip0_f29', lambda d: (d['object_rollout'][0, MAX_CLOSE_FRAME], d['tool_rollout'][0, MAX_CLOSE_FRAME])),
              ('grip0_after', lambda d: (d['after_object_cloud'][0], d['tool_rollout'][0, -1])),
              ('grip1_f25', lambda d: (d['object_rollout'][1, VIZ_FRAME], d['tool_rollout'][1, VIZ_FRAME])),
              ('grip1_f29', lambda d: (d['object_rollout'][1, MAX_CLOSE_FRAME], d['tool_rollout'][1, MAX_CLOSE_FRAME])),
              ('grip1_after', lambda d: (d['after_object_cloud'][1], d['tool_rollout'][1, -1]))]
E_STATES = [VIZ_STATES[0],
            ('grip0_f15', lambda d: (d['object_rollout'][0, PRE_CONTACT_FRAME], d['tool_rollout'][0, PRE_CONTACT_FRAME]))] + \
           [s for s in VIZ_STATES[1:] if s[0] != 'grip0_f15']
TOP_LIM = dict(x=(0.2, 0.8), z=(0.2, 0.8))
SIDE_LIM = dict(x=(0.2, 0.8), y=(0.0, 0.45))


def visualize_sweep(name, labels, datas, out_dir, states=VIZ_STATES, final_tag='after', overlays=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # side-by-side grids: rows = states, cols = materials; fixed limits everywhere
    for view, (a, b), lim in [('top', (0, 2), TOP_LIM), ('side', (0, 1), SIDE_LIM)]:
        fig, axes = plt.subplots(len(states), len(datas), figsize=(3.2 * len(datas), 3.0 * len(states)), squeeze=False)
        for r, (sname, get) in enumerate(states):
            for c, (label, d) in enumerate(zip(labels, datas)):
                cloud, tools = get(d)
                ax = axes[r, c]
                ax.scatter(cloud[:, a], cloud[:, b], s=3, c='tab:blue')
                ax.scatter(tools[:, a], tools[:, b], s=60, c=['tab:red', 'tab:orange'], marker='s')
                ax.set_xlim(*lim[list(lim)[0]]); ax.set_ylim(*lim[list(lim)[1]]); ax.set_aspect('equal')
                ax.set_title(f'{label}\n{sname}', fontsize=8); ax.tick_params(labelsize=6)
                ax.set_xlabel(list(lim)[0], fontsize=7); ax.set_ylabel(list(lim)[1], fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{name}_grid_{view}.png'), dpi=110)
        plt.close(fig)

    if overlays:
        overlay(plt, labels, datas, states, os.path.join(out_dir, f'{name}_overlay.png'))
        overlay(plt, labels, datas, [s for s in states if s[0].endswith(final_tag)], os.path.join(out_dir, f'{name}_{final_tag}_overlay.png'))


def overlay(plt, labels, datas, states, path):
    # all materials in one axis per state
    colors = ['tab:blue', 'tab:green', 'tab:purple', 'tab:red', 'tab:brown', 'tab:pink']
    fig, axes = plt.subplots(len(states), 2, figsize=(9, 3.6 * len(states)), squeeze=False)
    for r, (sname, get) in enumerate(states):
        for c, ((a, b), lim, view) in enumerate([((0, 2), TOP_LIM, 'top'), ((0, 1), SIDE_LIM, 'side')]):
            ax = axes[r, c]
            for i, (label, d) in enumerate(zip(labels, datas)):
                cloud, tools = get(d)
                ax.scatter(cloud[:, a], cloud[:, b], s=4, c=colors[i], alpha=0.5, label=label)
            ax.scatter(tools[:, a], tools[:, b], s=60, c='k', marker='s')
            ax.set_xlim(*lim[list(lim)[0]]); ax.set_ylim(*lim[list(lim)[1]]); ax.set_aspect('equal')
            ax.set_title(f'{sname} ({view})', fontsize=9)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--settled-state', required=True)
    ap.add_argument('--E-values', type=float, nargs='*', default=[2500, 5000, 10000])
    ap.add_argument('--yield-values', type=float, nargs='*', default=[100, 200, 400])
    ap.add_argument('--action-seq')
    ap.add_argument('--E-sweep-yield', type=float, help='sigma_y fixed during E sweep (default: settled reference)')
    ap.add_argument('--scale-yield-with-E', action='store_true',
                    help='OPTIONAL / INTERPRETATION ONLY: sigma_y = E_sweep_yield * E / reference_E (constant yield strain)')
    ap.add_argument('--relax-steps', type=int, default=0, help='zero-action branch after each grip command end')
    args = ap.parse_args()

    out_dir = os.path.join(g.HERE, '..', '..', 'dataset', 'material_sensitivity_' + datetime.now().strftime("%d-%b-%Y-%H:%M:%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.settled_state, 'rb') as f:
        settled = pickle.load(f)
    meta_path = os.path.join(os.path.dirname(args.settled_state), 'settled_state_metadata.json')
    with open(meta_path) as f:
        settled_meta = json.load(f)
    ref_E, ref_sy = settled_meta['reference_E'], settled_meta['reference_sigma_y']

    seq_path = args.action_seq or os.path.join(out_dir, 'material_debug_action_sequence.npz')
    if not args.action_seq:
        make_action_sequence(seq_path)
    seq = dict(np.load(seq_path))

    e_sy = args.E_sweep_yield if args.E_sweep_yield is not None else ref_sy
    e_sy_of = (lambda E: e_sy * E / ref_E) if args.scale_yield_with_E else (lambda E: e_sy)
    sweeps = dict(E_sweep=[dict(E=E, yield_stress=e_sy_of(E)) for E in args.E_values],
                  yield_sweep=[dict(E=ref_E, yield_stress=s) for s in args.yield_values])
    key = lambda m: f"E{m['E']:g}_sy{m['yield_stress']:g}"
    materials = {key(m): m for ms in sweeps.values() for m in ms}   # reference shared by both sweeps, run once

    env, _ = g.make_env()
    dx = env.simulator.dx
    results, summary = {}, dict(settled_state=args.settled_state, settled_meta=settled_meta, action_seq=seq_path,
                                note='TEMPORARY EXPLORATORY CANDIDATES', materials={})
    for name, mat in materials.items():
        print(f"===== {name} =====")
        try:
            drift = zero_action_drift(env, settled, mat)
            res = run_grips(env, settled, mat, seq, args.relax_steps)
        except ValueError:
            print(f"{name}: FAILED (NaN in env.step)")
            summary['materials'][name] = dict(failed=True)
            continue
        res['zero_drift'] = drift
        res['metrics'] = analyze(res, dx)
        results[name] = res
        np.savez_compressed(os.path.join(out_dir, f'{name}.npz'), **res['data'])
        summary['materials'][name] = dict(material=res['material'], state_error_vs_settled=res['state_error_vs_settled'],
                                          zero_drift=drift, metrics=res['metrics'])

    # ---- identity checks across materials ----
    names = list(results)
    ref = results[names[0]]['data']
    identity = dict(
        initial_cloud=max(float(np.abs(results[n]['data']['episode_initial_object_cloud'] - ref['episode_initial_object_cloud']).max()) for n in names),
        grip_start_tool_pose=max(float(np.abs(results[n]['data']['grip_start_tool_pose'] - ref['grip_start_tool_pose']).max()) for n in names),
        grip_start_tool_pose_vs_saved=max(float(np.abs(results[n]['data']['grip_start_tool_pose'] - seq['grip_start_tool_pose']).max()) for n in names),
        primitive_action=max(float(np.abs(results[n]['data']['primitive_action_sequence'] - ref['primitive_action_sequence']).max()) for n in names),
        full_state_vs_settled={k: max(results[n]['state_error_vs_settled'][k] for n in names) for k in ref_state_keys(results)},
    )
    summary['identity'] = identity

    # ---- cross-material differences (same particle index) ----
    frames = dict(g0_f15=(0, PRE_CONTACT_FRAME), g0_f25=(0, VIZ_FRAME), g0_f29=(0, MAX_CLOSE_FRAME), g0_after=(0, -1),
                  g1_f25=(1, VIZ_FRAME), g1_f29=(1, MAX_CLOSE_FRAME), g1_after=(1, -1))
    summary['cross'] = {}
    for sweep, ms in sweeps.items():
        ks = [key(m) for m in ms if key(m) in results]
        if not ks:
            continue
        summary['cross'][sweep] = {f'{a} vs {b}': {f: disp(results[a]['data']['object_rollout'][gi, fi], results[b]['data']['object_rollout'][gi, fi])
                                                   for f, (gi, fi) in frames.items()}
                                   for a, b in itertools.combinations(ks, 2)}
        ds = [results[k]['data'] for k in ks]
        if sweep == 'E_sweep':
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            visualize_sweep(sweep, ks, ds, out_dir, E_STATES, overlays=False)
            pick = lambda *names: [s for s in E_STATES if s[0] in names]
            overlay(plt, ks, ds, pick('grip0_f15'), os.path.join(out_dir, f'{sweep}_f15_overlay.png'))
            overlay(plt, ks, ds, pick('grip0_after', 'grip1_after'), os.path.join(out_dir, f'{sweep}_f39_overlay.png'))
        else:
            visualize_sweep(sweep, ks, ds, out_dir)

    if args.relax_steps:
        ks = [k for ms in sweeps.values() for k in dict.fromkeys(key(m) for m in ms) if k in results]
        summary['relax'] = relax_analysis({k: results[k]['data'] for k in dict.fromkeys(ks)}, out_dir)

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print_report(summary, seq, out_dir)
    if args.relax_steps:
        print_relax_report(summary['relax'])


def _state(name, cloud_fn, tool_fn):
    return (name, lambda d: (cloud_fn(d), tool_fn(d)))


def _roll(k, f):
    return _state(f'grip{k}_f{f}', lambda d: d['object_rollout'][k, f], lambda d: d['tool_rollout'][k, f])


def _relax(k, t):
    name = f'grip{k}_after_command' if t == 0 else f'grip{k}_relax{t}'
    return _state(name, lambda d: d['relax_object_cloud'][k, t], lambda d: d['relax_tool_pose'][k, t])


def relax_analysis(datas, out_dir):
    out = dict(materials={}, cross={})
    for name, d in datas.items():
        grips = []
        for k in range(g.N_GRIPS):
            b = d['before_object_cloud'][k]
            c = d['relax_clearance'][k].min(axis=1)          # (steps+1,), min over two fingers
            contact_steps = np.where(c <= 0)[0]
            dfm = dict(f29=disp(d['object_rollout'][k, MAX_CLOSE_FRAME], b),
                       **{('after_command' if t == 0 else f'relax{t}'): disp(d['relax_object_cloud'][k, t], b)
                          for t in (0, 10, 20, 50, 100) if t < len(c)})
            last = f'relax{len(c) - 1}'
            comp, cmd, rel = dfm['f29']['rms'], dfm['after_command']['rms'], dfm[last]['rms']
            grips.append(dict(
                frame39_clearance=[float(x) for x in d['relax_clearance'][k, 0]],
                contact_during_relax=bool(len(contact_steps)),
                contact_end_relax_step=(int(contact_steps.max()) + 1) if len(contact_steps) else 0,
                relaxation='constrained_relaxation' if len(contact_steps) else 'free_relaxation',
                deformation=dfm, retreat_recovery=comp - cmd, post_recovery=cmd - rel, total_recovery=comp - rel,
                total_recovery_ratio=(comp - rel) / comp if comp > 1e-9 else None,
                convergence={t: dict(zip(STAT_NAMES, map(float, d['relax_stats'][k, t - 1])))
                             for t in RELAX_CHECKPOINTS if 0 < t <= len(d['relax_stats'][k])},
            ))
        out['materials'][name] = grips

    names = list(datas)
    for a, b in itertools.combinations(names, 2):
        out['cross'][f'{a} vs {b}'] = {
            f'g{k}_{tag}': disp(get(datas[a], k), get(datas[b], k))
            for k in range(g.N_GRIPS)
            for tag, get in [('f29', lambda d, k: d['object_rollout'][k, MAX_CLOSE_FRAME]),
                             ('f39', lambda d, k: d['relax_object_cloud'][k, 0]),
                             ('relax20', lambda d, k: d['relax_object_cloud'][k, 20]),
                             ('relax50', lambda d, k: d['relax_object_cloud'][k, 50]),
                             ('relax100', lambda d, k: d['relax_object_cloud'][k, -1])]}

    labels, ds = names, [datas[n] for n in names]
    grid_states = [VIZ_STATES[0]] + [s for k in range(g.N_GRIPS)
                                     for s in (_roll(k, MAX_CLOSE_FRAME), _relax(k, 0), _relax(k, 20), _relax(k, 100))]
    visualize_sweep('relax_grid', labels, ds, out_dir, grid_states, final_tag='relax100')
    traj = [_roll(0, MAX_CLOSE_FRAME)] + [_relax(0, t) for t in (0, 10, 20, 50, 100)]
    visualize_sweep('grip0_relax_trajectory', labels, ds, out_dir, traj, overlays=False)
    return out


def print_relax_report(r):
    print("\n==================== RELAXATION (branch after command end; next grip unchanged) ====================")
    for name, grips in r['materials'].items():
        for k, gr in enumerate(grips):
            dfm = gr['deformation']
            print(f"\n## {name} grip{k}: frame39 clearance {np.round(gr['frame39_clearance'], 4).tolist()}  "
                  f"{gr['relaxation']}  contact_end_relax_step {gr['contact_end_relax_step']}")
            print("  RMS vs before: " + "  ".join(f"{t} {v['rms']:.4e}" for t, v in dfm.items()))
            print(f"  retreat_recovery {gr['retreat_recovery']:+.4e}  post_recovery {gr['post_recovery']:+.4e}  "
                  f"total_recovery {gr['total_recovery']:+.4e}  ratio {gr['total_recovery_ratio']:+.3f}")
            for t, st in gr['convergence'].items():
                print(f"  relax step {t:3d}: dx max/mean/rms {st['dx_max']:.2e}/{st['dx_mean']:.2e}/{st['dx_rms']:.2e}  "
                      f"v {st['v_max']:.2e}/{st['v_mean']:.2e}/{st['v_rms']:.2e}  cum-from-f39 rms {st['cum_rms']:.2e} max {st['cum_max']:.2e}")
    print("\n[relax cross]")
    for pair, fr in r['cross'].items():
        print(f"  {pair}: " + "  ".join(f"{f} {v['rms']:.2e}/{v['max']:.2e}" for f, v in fr.items()))


def ref_state_keys(results):
    return list(next(iter(results.values()))['state_error_vs_settled'])


def print_report(s, seq, out_dir):
    fc = s['settled_meta']['floor_clamp']
    print(f"\n[shared settled state] {s['settled_state']}  steps {s['settled_meta']['settle_steps']}  "
          f"clamp_y {fc['clamp_y']} affected {fc['affected_particles']}/{fc['total_particles']}")
    for k in range(len(seq['grip_rate'])):
        print(f"[action] grip{k} start pose {np.round(seq['grip_start_tool_pose'][k][:, :3], 4).tolist()} "
              f"rate {seq['grip_rate'][k]:.6f} angle {seq['grip_angle'][k]:.6f}")
    print(f"[identity] {json.dumps(s['identity'])}")
    for name, r in s['materials'].items():
        if r.get('failed'):
            print(f"\n## {name}: FAILED"); continue
        mt, m = r['material'], r['metrics']
        print(f"\n## {name}  (TEMPORARY EXPLORATORY)")
        print(f"  yield_strain_approx sigma_y/(2 mu) = {mt['expected']['yield_stress'] / (2 * mt['expected']['mu']):.4f}")
        print(f"  expected mu {mt['expected']['mu']:.4f} lam {mt['expected']['lam']:.4f} sy {mt['expected']['yield_stress']}")
        print(f"  applied  mu {mt['applied']['mu']} lam {mt['applied']['lam']} sy {mt['applied']['yield_stress']}  match {mt['match']}")
        for t, v in r['zero_drift'].items():
            print(f"  zero-action drift step {t}: {fmt(v)}  v mean/RMS/max {v['v_mean']:.3e}/{v['v_rms']:.3e}/{v['v_max']:.3e}")
        for f, v in m['pre_contact'].items():
            print(f"  grip0 pre-contact f{f}: {fmt(v)}")
        print(f"  rollout bbox min {np.round(m['bbox']['min'], 4).tolist()} max {np.round(m['bbox']['max'], 4).tolist()}")
        print(f"  first contact frame (clearance<=0): {m['first_contact_frame_tol0']}  (<=dx): {m['first_contact_frame_tol_dx']}  "
              f"min finger clearance {np.round(m['min_finger_clearance'], 4).tolist()}")
        for k, dfm in enumerate(m['deformation']):
            print(f"  grip{k} deform " + "  ".join(f"{t} {fmt(dfm[t])}" for t in ('f25', 'f29', 'after'))
                  + f"  recovery_rms {dfm['recovery_rms']:+.4e}  ratio {dfm['recovery_ratio']:+.3f}")
        print(f"  NaN {m['nan']} Inf {m['inf']} explosion {m['explosion']} max frame step disp {m['max_frame_step_disp']:.4e}  height {m['height']}")
    for sweep, pairs in s['cross'].items():
        print(f"\n[cross] {sweep}")
        for pair, fr in pairs.items():
            print(f"  {pair}: " + "  ".join(f"{f} rms {v['rms']:.2e}/max {v['max']:.2e}" for f, v in fr.items()))
    ref = [k for k in s['materials'] if k.startswith('E5000_')]
    for sweep, pairs in s['cross'].items():
        if sweep != 'E_sweep' or not ref:
            continue
        print("\n[drift fraction vs E5000] precontact g0_f15 RMS / g0_after RMS")
        for pair, fr in pairs.items():
            if ref[0] in pair.split(' vs '):
                print(f"  {pair}: precontact {fr['g0_f15']['rms']:.3e}  after diff {fr['g0_after']['rms']:.3e}  "
                      f"fraction {fr['g0_f15']['rms'] / fr['g0_after']['rms']:.3f}")
    print(f"\noutputs: {out_dir}")


if __name__ == '__main__':
    main()
