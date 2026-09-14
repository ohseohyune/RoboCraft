"""Multi-material sequential-grip RAW dataset generator (20k-particle rollouts).

One episode = one material (E=5000, nu=0.2, sigma_y in SIGMAS), 5 sequential grips (30 close + 10 retreat) from the
shared settled state, no reset between grips. Action sequences are generated first, saved to sequence_bank/, and every
episode is simulated from the saved file (paired sequences are shared bit-identically across materials).

Reused (not re-derived): action_distribution_pilot (pose_from, analytic pin filter, same-axis rule, run_grip, metrics,
classify), material_sensitivity_debug (restore, applied_material, state_error), stress_test_grip (gap_to_rate),
gen_multigrip_debug (env, primitive actions, 300-cloud), gen_material_history_pilot (collapse ratio).
Simulator / yaml / existing scripts untouched.

  python gen_full_dataset.py --mode dry-run        # 1 paired x 4 materials + 1 random (sigma 500) = 5 episodes
  python gen_full_dataset.py --mode full --confirm-full   # 6 paired x 4 + 14 random x 4 = 80 episodes (needs approval)
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import types
import zipfile
from datetime import datetime

import numpy as np

import action_distribution_pilot as adp
import gen_material_history_pilot as mhp
import gen_multigrip_debug as g
import material_sensitivity_debug as msd
import stress_test_grip as st

DATASET_VERSION = 'multimaterial_seqgrip_raw_v1'
E = 5000.0
SIGMAS = [30.0, 200.0, 500.0, 900.0]
N_GRIP = 5
NF = g.N_FRAMES                                   # 40 = 30 close + 10 retreat
N_PARTICLES = 20000
MASTER_SEED = 20260914
PIN_MARGIN = 0.005
MAX_SAME_AXIS = 2
GAP_RANGE = (0.07, 0.13)
MID_RANGE = (0.47, 0.53)
DATASET_DIR = os.path.abspath(os.path.join(g.HERE, '..', '..', 'dataset'))
SETTLED = os.path.join(DATASET_DIR, 'settled_state_N100_13-Sep-2026-20:13:14', 'settled_initial_state.pkl')
PILOT = os.path.join(DATASET_DIR, 'action_distribution_pilot_14-Sep-2026-12:55:44')
SEQ_KEYS = ('midpoint', 'angle', 'target_gap', 'gripper_rate', 'finger_start_pose', 'primitive_action_sequence')
MODE_CODE = {'dry-run': 0, 'full': 1}
KIND_CODE = {'paired': 0, 'random': 1}


# ------------------------------------------------------------------ plan
def plan(mode):
    """sequences: [(sequence_id, kind, index, sigma or None)], episodes: [(episode_id, sequence_id, sigma)]"""
    if mode == 'dry-run':
        seqs = [('paired_dryrun_000', 'paired', 0, None), ('random_sigma500_dryrun_000', 'random', 0, 500.0)]
    else:
        seqs = [(f'paired_{i:03d}', 'paired', i, None) for i in range(6)]
        seqs += [(f'sigma{sy:g}/random_{i:03d}', 'random', i, sy) for sy in SIGMAS for i in range(14)]
    eps = []
    for sid, kind, _, sy in seqs:
        for s in (SIGMAS if kind == 'paired' else [sy]):
            eps.append((f'{sid}_sigma{s:g}' if kind == 'paired' else sid.replace('/', '_'), sid, s))
    return seqs, eps


def sequence_seed(mode, kind, index, sigma):
    return int(np.random.SeedSequence([MASTER_SEED, MODE_CODE[mode], KIND_CODE[kind], int(sigma or 0), index]).generate_state(1)[0])


# ------------------------------------------------------------------ actions
def make_sequence(seed):
    """i.i.d. per grip: gap ~ U(0.07,0.13); (midpoint x,z, angle) rejection-sampled with the analytic pin filter;
    same-axis rule resamples the angle only. Inner loop copied from adp.make_sequences (its gap stratification is not
    used: frozen spec is i.i.d. uniform, and with 1 sequence it would put all 5 gaps in the last bin)."""
    rng = np.random.default_rng(seed)
    MID = adp.MID
    row = {k: [] for k in SEQ_KEYS + ('analytic_pin_distance', 'rejected_draws', 'axis_resampled_draws')}
    for k in range(N_GRIP):
        gap, rejected = rng.uniform(*GAP_RANGE), 0
        while True:
            off = np.array([rng.uniform(*MID_RANGE), rng.uniform(*MID_RANGE)]) - MID[[0, 2]]
            ang = rng.uniform(*adp.ANGLE_RANGE)
            if not adp.mc_pin_collision(off[None], np.array([ang]), np.array([gap]), PIN_MARGIN)[0]:
                break
            rejected += 1
        axis_resampled = 0
        while adp.same_axis_run(row['angle'] + [ang]) > MAX_SAME_AXIS:
            ang = rng.uniform(*adp.ANGLE_RANGE)
            axis_resampled += 1
            if adp.mc_pin_collision(off[None], np.array([ang]), np.array([gap]), PIN_MARGIN)[0]:
                ang = row['angle'][-1]   # keep violating -> loop again (pin filter still applies to the new angle)
            assert axis_resampled < 10000
        mid = np.array([MID[0] + off[0], MID[1], MID[2] + off[1]])
        p1, p2 = adp.pose_from(mid, ang)
        u = np.array([np.cos(ang), -np.sin(ang)])
        rate = st.gap_to_rate(gap)
        row['midpoint'].append(mid); row['angle'].append(ang); row['target_gap'].append(gap); row['gripper_rate'].append(rate)
        row['finger_start_pose'].append(np.stack([p1, p2]))
        row['primitive_action_sequence'].append(g.make_grip_actions(p1, p2, rate))
        row['analytic_pin_distance'].append(float(adp.analytic_pin_distance(gap, off @ u, abs(off[0] * u[1] - off[1] * u[0]))))
        row['rejected_draws'].append(rejected); row['axis_resampled_draws'].append(axis_resampled)
    out = {k: np.array(v) for k, v in row.items()}
    out['high_level_action'] = np.concatenate([out['midpoint'], out['angle'][:, None], out['target_gap'][:, None]], 1)   # [x,y,z,angle,gap]
    assert (out['analytic_pin_distance'] >= PIN_MARGIN).all()
    assert all(adp.same_axis_run(list(out['angle'][:k + 1])) <= MAX_SAME_AXIS for k in range(N_GRIP))
    assert out['primitive_action_sequence'].shape == (N_GRIP, NF, 12)
    return out


def action_hash(d):
    h = hashlib.sha256()
    for k in ('high_level_action', 'gripper_rate', 'finger_start_pose', 'primitive_action_sequence'):
        h.update(np.ascontiguousarray(d[k], dtype=np.float64).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ simulation
def run_episode(env, settled, sigma, seq, t):
    mat = dict(E=E, yield_stress=float(sigma))
    msd.restore(env, settled, mat)
    init_err = msd.state_error(env.get_state(), settled)
    material = msd.applied_material(env, mat)
    x_init = env.simulator.get_x(0)
    assert x_init.shape == (N_PARTICLES, 3), x_init.shape
    init_max_y = x_init[:, 1].max()
    rollout = np.zeros((N_GRIP, NF, N_PARTICLES, 3), np.float32)
    bvel, bcloud = [env.simulator.get_v(0).astype(np.float32)], [g.read_object_cloud(env)]
    prev_cloud, prev_state = bcloud[0], env.get_state()
    boundary, f39_vs_state, rows, ds, failed = [], [], [], [], None
    for k in range(N_GRIP):
        state_before, before = env.get_state(), g.read_object_cloud(env)
        boundary.append(dict(object=float(np.abs(before - prev_cloud).max()), **msd.state_error(state_before, prev_state)))
        done = [0]

        def capture(f, k=k):
            rollout[k, f] = env.simulator.get_x(0); done[0] = f + 1

        try:
            d, full = adp.run_grip(env, seq, k, on_frame=capture)
        except ValueError as e:
            failed = f'grip {k} after {done[0]} frames: ValueError {e}'; break
        prev_cloud, prev_state = g.read_object_cloud(env), env.get_state()
        f39_vs_state.append(float(np.abs(rollout[k, NF - 1] - prev_state['state'][0].astype(np.float32)).max()))
        bvel.append(env.simulator.get_v(0).astype(np.float32)); bcloud.append(prev_cloud)
        m = adp.metrics(d, full)
        m.update(grip_index=k, sigma_y=float(sigma), recovery_ratio=1 - m['normalized_residual'],
                 analytic_pin_distance=float(seq['analytic_pin_distance'][k]),
                 weak_finger_max_contact=int(min(m['max_contact_particles'])),
                 height_ratio_vs_episode_initial=float(full[39][:, 1].max() / init_max_y))
        m['one_finger_contact'] = bool(any(f is not None for f in m['first_contact_frame']) and not m['both_fingers_contact'])
        m['collapse'] = m['height_ratio_vs_episode_initial'] < mhp.COLLAPSE_HEIGHT_RATIO
        m['pin_collision'] = m['finger_pin_distance'] < 0
        m['weak'] = m['compression_RMS'] < t.comp_min
        m['quality_class'] = adp.classify(m, t)   # informational only; weak / one-finger are kept
        rows.append(m); ds.append(d)
    n = len(ds)
    stack = lambda key: np.stack([d[key] for d in ds]) if n else np.zeros((0,))
    data = dict(
        initial_object_pos=x_init.astype(np.float32),
        rollout_object_pos=rollout,
        boundary_object_vel=np.stack(bvel),
        boundary_cloud_300=np.stack(bcloud).astype(np.float32),
        tool_pose_rollout=stack('tool_rollout'),
        tool_start_pose=seq['finger_start_pose'][:n].copy(),
        primitive_action=seq['primitive_action_sequence'][:n].copy(),
        high_level_action=seq['high_level_action'][:n].copy(),
        midpoint=seq['midpoint'][:n].copy(), angle=seq['angle'][:n].copy(), target_gap=seq['target_gap'][:n].copy(),
        actual_gap_f29=stack('actual_gap_f29'), gripper_rate=seq['gripper_rate'][:n].copy(),
        finger_min_clearance=stack('finger_min_clearance'), finger_contact_count=stack('finger_contact_count'),
        finger_pin_gap=stack('finger_pin_gap'), max_step_disp=stack('max_step_disp_all_particles'),
    )
    info = dict(initial_state_error=init_err, material=material, boundary=boundary, frame39_vs_full_state_x_f32=f39_vs_state,
                failed=failed, completed_grips=n)
    return data, rows, info


EXPECTED = dict(
    initial_object_pos=((N_PARTICLES, 3), np.float32), rollout_object_pos=((N_GRIP, NF, N_PARTICLES, 3), np.float32),
    boundary_object_vel=((N_GRIP + 1, N_PARTICLES, 3), np.float32), boundary_cloud_300=((N_GRIP + 1, g.N_OBS, 3), np.float32),
    tool_pose_rollout=((N_GRIP, NF, 2, 7), np.float64), tool_start_pose=((N_GRIP, 2, 7), np.float64),
    primitive_action=((N_GRIP, NF, 12), np.float64), high_level_action=((N_GRIP, 5), np.float64),
    midpoint=((N_GRIP, 3), np.float64), angle=((N_GRIP,), np.float64), target_gap=((N_GRIP,), np.float64),
    actual_gap_f29=((N_GRIP,), np.float64), gripper_rate=((N_GRIP,), np.float64),
    finger_min_clearance=((N_GRIP, NF, 2), np.float64), finger_contact_count=((N_GRIP, NF, 2), np.float64),
    finger_pin_gap=((N_GRIP, NF, 2), np.float64), max_step_disp=((N_GRIP, NF), np.float64))


def verify_reload(path, data, seq, settled):
    r = dict(np.load(path))
    out = dict(missing=[k for k in EXPECTED if k not in r], shape_dtype_mismatch=[], reload_max_error={}, nan=0, inf=0)
    for k, (shape, dt) in EXPECTED.items():
        if k not in r:
            continue
        if r[k].shape != shape or r[k].dtype != dt:
            out['shape_dtype_mismatch'].append([k, list(r[k].shape), str(r[k].dtype)])
        out['reload_max_error'][k] = float(np.abs(r[k].astype(np.float64) - data[k].astype(np.float64)).max()) if r[k].shape == data[k].shape else None
        out['nan'] += int(np.isnan(r[k]).sum()); out['inf'] += int(np.isinf(r[k]).sum())
    out['initial_vs_settled_f32'] = float(np.abs(r['initial_object_pos'] - settled['state'][0].astype(np.float32)).max())
    out['initial_f32_quantization_vs_f64'] = float(np.abs(r['initial_object_pos'].astype(np.float64) - settled['state'][0]).max())
    out['frame39_vs_boundary_cloud'] = float(max(np.abs(r['rollout_object_pos'][k, NF - 1, ::N_PARTICLES // g.N_OBS][:g.N_OBS]
                                                      - r['boundary_cloud_300'][k + 1]).max() for k in range(N_GRIP)))
    out['initial_vs_boundary_cloud'] = float(np.abs(r['initial_object_pos'][::N_PARTICLES // g.N_OBS][:g.N_OBS] - r['boundary_cloud_300'][0]).max())
    out['actions_vs_sequence_bank'] = max(float(np.abs(r[a] - seq[b]).max()) for a, b in
                                          [('primitive_action', 'primitive_action_sequence'), ('tool_start_pose', 'finger_start_pose'),
                                           ('high_level_action', 'high_level_action'), ('gripper_rate', 'gripper_rate')])
    out['action_hash'] = action_hash(dict(high_level_action=r['high_level_action'], gripper_rate=r['gripper_rate'],
                                          finger_start_pose=r['tool_start_pose'], primitive_action_sequence=r['primitive_action']))
    with zipfile.ZipFile(path) as z:
        out['member_bytes'] = {i.filename[:-4]: dict(stored=i.compress_size, raw=i.file_size) for i in z.infolist()}
    out['ok'] = (not out['missing'] and not out['shape_dtype_mismatch'] and out['nan'] == 0 and out['inf'] == 0
                 and all(v == 0 for v in out['reload_max_error'].values()) and out['initial_vs_settled_f32'] == 0
                 and out['frame39_vs_boundary_cloud'] < 1e-6 and out['initial_vs_boundary_cloud'] < 1e-6 and out['actions_vs_sequence_bank'] == 0)
    return out


# ------------------------------------------------------------------ utils
def git(*a):
    return subprocess.run(['git', '-C', g.HERE, *a], capture_output=True, text=True).stdout.strip()


def gib(b):
    return b / 2 ** 30


def dir_bytes(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def jdump(obj, path):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(path + '.tmp', path)


def check_frozen_region():
    with open(os.path.join(PILOT, 'recommended_action_region.json')) as f:
        rec = json.load(f)
    assert rec['midpoint_x'] == list(MID_RANGE) and rec['midpoint_z'] == list(MID_RANGE) and rec['target_gap'] == list(GAP_RANGE), rec
    assert rec['midpoint_y'] == adp.MID[1] == 0.14 and f'< {PIN_MARGIN}' in rec['pin_filter'], rec
    assert adp.ANGLE_RANGE == (0.0, np.pi) and adp.SAME_AXIS_DEG == 20.0
    return rec


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['dry-run', 'full'], required=True)
    ap.add_argument('--confirm-full', action='store_true', help='required for --mode full (user approval)')
    ap.add_argument('--settled-state', default=SETTLED)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.mode == 'full' and not args.confirm_full:
        sys.exit('--mode full requires --confirm-full (explicit user approval after dry-run review)')
    adp.check_pose_from(); adp.check_same_axis()
    rec = check_frozen_region()
    t = types.SimpleNamespace(**rec['thresholds'])
    settled_path = os.path.abspath(args.settled_state)
    with open(settled_path, 'rb') as f:
        settled = pickle.load(f)
    assert settled['state'][0].shape == (N_PARTICLES, 3)

    stamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S")
    root = os.path.abspath(args.out or os.path.join(DATASET_DIR, f"full_dataset_{args.mode.replace('-', '')}_{stamp}"))
    assert not os.path.exists(root), root
    free_before = shutil.disk_usage(DATASET_DIR).free
    git_status_before = git('status', '--short')
    os.makedirs(os.path.join(root, 'sequence_bank')); os.makedirs(os.path.join(root, 'episodes'))
    commit = git('rev-parse', 'HEAD')
    with open(os.path.join(root, 'git_commit.txt'), 'w') as f:
        f.write(f'{commit}\n\ngit status --short at generation start:\n{git_status_before}\n')

    seq_plan, ep_plan = plan(args.mode)
    config = dict(dataset_version=DATASET_VERSION, mode=args.mode, generator_script=os.path.abspath(__file__), git_commit=commit,
                  created=datetime.now().isoformat(), shared_settled_state=settled_path, E=E, nu=msd.NU, sigma_y=SIGMAS,
                  action_distribution=dict(midpoint_x=MID_RANGE, midpoint_z=MID_RANGE, midpoint_y=float(adp.MID[1]),
                                           angle=adp.ANGLE_RANGE, target_gap=GAP_RANGE, sampling='i.i.d. per grip; gap first, then (x,z,angle) rejection by pin filter',
                                           start_pose=rec['start_pose'], gap_to_rate=rec['gap_to_rate'], frozen_region_file=os.path.join(PILOT, 'recommended_action_region.json')),
                  pin_filter=dict(threshold=PIN_MARGIN, formula=rec['analytic_pin_distance']),
                  same_axis=dict(threshold_deg=adp.SAME_AXIS_DEG, max_consecutive=MAX_SAME_AXIS, on_violation='resample angle only'),
                  n_grips=N_GRIP, close_steps=g.task_params['len_per_grip'], retreat_steps=g.task_params['len_per_grip_back'],
                  simulator_particle_count=N_PARTICLES, simulator_dtype='float64', raw_particle_dtype='float32',
                  action_pose_dtype='float64 (exact simulator inputs; bit-identity across paired materials)',
                  master_seed=MASTER_SEED, seed_rule='SeedSequence([master_seed, mode(dry-run 0/full 1), kind(paired 0/random 1), sigma or 0, index]).generate_state(1)[0]',
                  metric_thresholds=rec['thresholds'], collapse_height_ratio=mhp.COLLAPSE_HEIGHT_RATIO, explosion_step_disp=msd.EXPLOSION_STEP_DISP,
                  file_format='npz (np.savez_compressed), written to a .partial dir and atomically renamed after reload verification',
                  frame_semantics='rollout_object_pos[k, f] = simulator x after env.step(primitive_action[k, f]); P0=initial_object_pos, P_{k+1}=rollout_object_pos[k, 39]',
                  boundary_vel_semantics='boundary_object_vel[0]=P0, [k+1]=after grip k (frame 39)',
                  tool_semantics='tool_start_pose[k] = finger poses (2x7: xyz + quat) teleported before grip k; tool_pose_rollout[k, f] after env.step',
                  split_policy='dry-run: split="dryrun"; full: split_manifest.json later, paired groups kept in one split')
    jdump(config, os.path.join(root, 'dataset_config.json'))

    # ---- 1. sequence bank (generated once, then reloaded) ----
    manifest, seqs = [], {}
    for sid, kind, idx, sy in seq_plan:
        seed = sequence_seed(args.mode, kind, idx, sy)
        s = make_sequence(seed)
        path = os.path.join(root, 'sequence_bank', sid + '.npz')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, sequence_id=np.array(sid), seed=np.array(seed), **s)
        loaded = dict(np.load(path))
        regen = make_sequence(seed)
        repro = max(float(np.abs(regen[k] - loaded[k]).max()) for k in s)
        seqs[sid] = loaded
        manifest.append(dict(sequence_id=sid, kind=kind, index=idx, sigma_y=sy, seed=seed, path=os.path.relpath(path, root),
                             action_hash=action_hash(loaded), regenerate_from_seed_max_error=repro,
                             materials=SIGMAS if kind == 'paired' else [sy],
                             grips=[dict(midpoint=loaded['midpoint'][k].tolist(), angle_deg=float(np.degrees(loaded['angle'][k])),
                                         target_gap=float(loaded['target_gap'][k]), gripper_rate=float(loaded['gripper_rate'][k]),
                                         analytic_pin_distance=float(loaded['analytic_pin_distance'][k]),
                                         rejected_draws=int(loaded['rejected_draws'][k]), axis_resampled_draws=int(loaded['axis_resampled_draws'][k]),
                                         delta_axis_prev_deg=None if k == 0 else float(np.degrees(adp.delta_axis(loaded['angle'][k], loaded['angle'][k - 1]))))
                                    for k in range(N_GRIP)]))
        print(f'sequence {sid} seed {seed} hash {manifest[-1]["action_hash"][:16]} regen err {repro}', flush=True)
    jdump(manifest, os.path.join(root, 'sequence_manifest.json'))
    hashes = {m['sequence_id']: m['action_hash'] for m in manifest}

    # ---- 2. episodes ----
    env, _ = g.make_env()
    index, results = [], {}
    for eid, sid, sy in ep_plan:
        seq = seqs[sid]
        print(f'episode {eid} (sequence {sid}, sigma_y {sy:g})', flush=True)
        data, rows, info = run_episode(env, settled, sy, seq, t)
        reasons = []
        if info['failed']:
            reasons.append(info['failed'])
        if any(v != 0 for v in info['initial_state_error'].values()):
            reasons.append('initial state != settled')
        if not info['material']['match']:
            reasons.append('material mismatch')
        if any(v != 0 for b in info['boundary'] for v in b.values()):
            reasons.append('boundary discontinuity')
        if any(v != 0 for v in info['frame39_vs_full_state_x_f32']):
            reasons.append('frame39 != boundary state')
        for m in rows:
            for key in ('nan', 'inf', 'explosion', 'collapse', 'pin_collision'):
                if m[key]:
                    reasons.append(f"grip {m['grip_index']}: {key}")
        if hashes[sid] != action_hash(dict(high_level_action=data['high_level_action'], gripper_rate=data['gripper_rate'],
                                           finger_start_pose=data['tool_start_pose'], primitive_action_sequence=data['primitive_action'])):
            reasons.append('action hash != sequence bank')
        partial = os.path.join(root, 'episodes', f'.{eid}.partial')
        os.makedirs(partial)
        npz = os.path.join(partial, 'episode.npz')
        np.savez_compressed(npz, **data)
        reload = verify_reload(npz, data, seq, settled) if not info['failed'] else dict(ok=False, note='partial episode, reload check skipped')
        if not reload['ok']:
            reasons.append('reload verification failed')
        status = 'ok' if not reasons else 'INVALID'
        meta = dict(episode_id=eid, sequence_id=sid, sequence_kind='paired' if sid.startswith('paired') else 'random',
                    split='dryrun' if args.mode == 'dry-run' else None, status=status, invalid_reasons=reasons,
                    E=E, nu=msd.NU, sigma_y=sy, seed=int(seq['seed']), action_hash=hashes[sid],
                    material_applied=info['material'], initial_state_error=info['initial_state_error'], boundary_continuity=info['boundary'],
                    frame39_vs_full_state_x_f32=info['frame39_vs_full_state_x_f32'], completed_grips=info['completed_grips'],
                    grips=rows, reload=reload, arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in data.items()})
        jdump(meta, os.path.join(partial, 'metadata.json'))
        final = os.path.join(root, 'episodes', eid if status == 'ok' else eid + '__INVALID')
        os.replace(partial, final)
        results[eid] = dict(meta=meta, path=final, data={k: v for k, v in data.items() if k not in ('rollout_object_pos', 'boundary_object_vel')})   # keep RAM bounded in full mode
        index.append(dict(episode_id=eid, sequence_id=sid, sequence_kind=meta['sequence_kind'], sigma_y=sy, E=E, nu=msd.NU,
                          split=meta['split'], status=status, action_hash=hashes[sid], path=os.path.relpath(final, root),
                          npz_bytes=os.path.getsize(os.path.join(final, 'episode.npz')),
                          metadata_bytes=os.path.getsize(os.path.join(final, 'metadata.json'))))
        print(f"  status {status} {reasons}  npz {index[-1]['npz_bytes'] / 2 ** 20:.1f} MiB  "
              + ' | '.join(f"g{m['grip_index']} comp {m['compression_RMS']:.4f} nres {m['normalized_residual']:.2f} pin {m['finger_pin_distance']:+.4f}"
                           f"{' WEAK' if m['weak'] else ''}{' ONE-FINGER' if m['one_finger_contact'] else ''}" for m in rows), flush=True)
    with open(os.path.join(root, 'episode_index.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader(); w.writerows(index)

    # ---- 3. cross-episode checks ----
    ok = {e: r for e, r in results.items() if r['meta']['completed_grips'] == N_GRIP}
    first = next(iter(ok.values()))['data']
    cross_init = {e: float(np.abs(r['data']['initial_object_pos'] - first['initial_object_pos']).max()) for e, r in ok.items()}
    cross_init_cloud = {e: float(np.abs(r['data']['boundary_cloud_300'][0] - first['boundary_cloud_300'][0]).max()) for e, r in ok.items()}
    paired = {}
    for sid, kind, _, _ in seq_plan:
        eps = [e for e, s_, _ in ep_plan if s_ == sid and e in ok]
        if kind != 'paired' or len(eps) < 2:
            continue
        ref = ok[eps[0]]['data']
        paired[sid] = dict(episodes=eps, **{k: max(float(np.abs(ok[e]['data'][k] - ref[k]).max()) for e in eps) for k in
                                            ('high_level_action', 'midpoint', 'angle', 'target_gap', 'gripper_rate', 'tool_start_pose', 'primitive_action')},
                           hash_identical=len({ok[e]['meta']['reload']['action_hash'] for e in eps} | {hashes[sid]}) == 1,
                           initial_object_pos=max(float(np.abs(ok[e]['data']['initial_object_pos'] - ref['initial_object_pos']).max()) for e in eps))
    grips = [m for r in results.values() for m in r['meta']['grips']]

    # ---- 4. disk ----
    free_after = shutil.disk_usage(DATASET_DIR).free
    npz_sizes = {r['episode_id']: r['npz_bytes'] for r in index}
    ep_sizes = {r['episode_id']: r['npz_bytes'] + r['metadata_bytes'] for r in index}
    mean_ep = float(np.mean(list(ep_sizes.values())))
    members = [r['meta']['reload']['member_bytes'] for r in ok.values() if 'member_bytes' in r['meta']['reload']]
    rollout_share = float(np.mean([m['rollout_object_pos']['stored'] / sum(v['stored'] for v in m.values()) for m in members]))
    raw_ep = float(np.mean([sum(v['raw'] for v in m.values()) for m in members]))
    bank_bytes = dir_bytes(os.path.join(root, 'sequence_bank')) / len(seq_plan)
    total = dir_bytes(root)
    n_full_eps, n_full_seqs = len(plan('full')[1]), len(plan('full')[0])
    projected = mean_ep * n_full_eps + bank_bytes * n_full_seqs + (total - sum(ep_sizes.values()) - bank_bytes * len(seq_plan))
    git_status_after = git('status', '--short')

    summary = dict(root=root, n_episodes=len(ep_plan), n_grips=len(ep_plan) * N_GRIP, statuses={e: r['meta']['status'] for e, r in results.items()},
                   invalid={e: r['meta']['invalid_reasons'] for e, r in results.items() if r['meta']['invalid_reasons']},
                   cross_initial_object_pos=cross_init, cross_initial_cloud=cross_init_cloud, paired_identity=paired,
                   sequence_regen_max_error={m['sequence_id']: m['regenerate_from_seed_max_error'] for m in manifest},
                   stability=dict(nan=sum(m['nan'] for m in grips), inf=sum(m['inf'] for m in grips),
                                  value_error=[e for e, r in results.items() if 'ValueError' in str(r['meta']['invalid_reasons'])],
                                  explosion=sum(m['explosion'] for m in grips), collapse=sum(m['collapse'] for m in grips),
                                  pin_collision=sum(m['pin_collision'] for m in grips), weak=sum(m['weak'] for m in grips),
                                  one_finger=sum(m['one_finger_contact'] for m in grips),
                                  min_finger_pin_distance=min(m['finger_pin_distance'] for m in grips),
                                  max_step_disp=max(m['max_step_disp'] for m in grips),
                                  height_ratio_vs_initial_min=min(m['height_ratio_vs_episode_initial'] for m in grips)),
                   disk=dict(free_before_gib=gib(free_before), free_after_gib=gib(free_after), total_gib=gib(total),
                             npz_mib={e: b / 2 ** 20 for e, b in npz_sizes.items()}, episode_mib={e: b / 2 ** 20 for e, b in ep_sizes.items()},
                             mean_episode_mib=mean_ep / 2 ** 20, uncompressed_episode_mib=raw_ep / 2 ** 20, rollout_share_of_npz=rollout_share,
                             sequence_bank_kib_per_sequence=bank_bytes / 2 ** 10,
                             projected_full_gib=gib(projected), projected_free_after_full_gib=gib(free_after - projected)),
                   git_status_before=git_status_before, git_status_after=git_status_after)
    jdump(summary, os.path.join(root, 'dryrun_summary.json' if args.mode == 'dry-run' else 'generation_summary.json'))
    report = write_report(summary, results, manifest, config)
    with open(os.path.join(root, 'dryrun_report.txt' if args.mode == 'dry-run' else 'generation_report.txt'), 'w') as f:
        f.write(report)
    print(report)


def write_report(s, results, manifest, c):
    L = ['[Dry-run configuration]', '', f"materials: E={c['E']:g} nu={c['nu']} sigma_y={c['sigma_y']}",
         f"sequences: {[m['sequence_id'] for m in manifest]}", f"episodes: {s['n_episodes']}  grips: {s['n_grips']}",
         f"master seed: {c['master_seed']}  ({c['seed_rule']})"]
    for m in manifest:
        L.append(f"  {m['sequence_id']}: seed {m['seed']} hash {m['action_hash']} regen-from-seed max error {m['regenerate_from_seed_max_error']}")
        for k, gr in enumerate(m['grips']):
            dp = '  -  ' if gr['delta_axis_prev_deg'] is None else f"{gr['delta_axis_prev_deg']:5.1f}"
            L.append(f"    grip{k} mid ({gr['midpoint'][0]:.4f}, {gr['midpoint'][1]:.2f}, {gr['midpoint'][2]:.4f}) angle {gr['angle_deg']:6.1f} (d_axis {dp}) "
                     f"gap {gr['target_gap']:.4f} rate {gr['gripper_rate']:.6f} pin_analytic {gr['analytic_pin_distance']:+.4f} "
                     f"pin_rejects {gr['rejected_draws']} axis_resamples {gr['axis_resampled_draws']}")
    L += ['', '-' * 32, '[Raw schema]', '', f"{'array':22s} | {'shape':22s} | dtype"]
    any_ep = next(iter(results.values()))['meta']['arrays']
    L += [f"{k:22s} | {str(tuple(v['shape'])):22s} | {v['dtype']}" for k, v in any_ep.items()]
    L += ['', '-' * 32, '[Initial-state verification]', '']
    for e, r in results.items():
        L.append(f"{e}: full state vs settled {r['meta']['initial_state_error']}")
    L += [f"cross-episode initial_object_pos max error: {s['cross_initial_object_pos']}", f"cross-episode initial 300-cloud max error: {s['cross_initial_cloud']}",
          '', '-' * 32, '[Paired action identity]', '']
    for sid, p in s['paired_identity'].items():
        L.append(f"{sid}: " + ', '.join(f'{k} {v}' for k, v in p.items()))
    L += ['', '-' * 32, '[Material verification]', '']
    for e, r in results.items():
        m = r['meta']['material_applied']
        L.append(f"{e}: requested {m['requested']} expected {m['expected']} applied {m['applied']} match {m['match']}")
    L += ['', '-' * 32, '[Boundary continuity]', '']
    for e, r in results.items():
        b = r['meta']['boundary_continuity']
        L.append(f"{e}: object max {max(x['object'] for x in b):.1e} full state max {max(v for x in b for k, v in x.items() if k != 'object'):.1e} "
                 f"| frame39(float32) vs boundary full-state x {max(r['meta']['frame39_vs_full_state_x_f32'] or [np.nan]):.1e}")
    L += ['', '-' * 32, '[Stability]', ''] + [f'{k}: {v}' for k, v in s['stability'].items()]
    L += ['', f"{'episode':32s} {'g':>1} {'gap':>5} {'ang':>6} {'comp':>7} {'resid':>7} {'nres':>5} {'pin':>7} {'clr':>7} {'contact':>11} {'h_ratio':>7} class"]
    for e, r in results.items():
        for m in r['meta']['grips']:
            L.append(f"{e:32s} {m['grip_index']:>1} {m['target_gap']:.3f} {m['angle_deg']:6.1f} {m['compression_RMS']:7.4f} {m['residual_RMS']:7.4f} "
                     f"{m['normalized_residual']:5.2f} {m['finger_pin_distance']:+7.4f} {m['min_finger_clearance']:+7.4f} {str(m['max_contact_particles']):>11} "
                     f"{m['height_ratio_vs_episode_initial']:7.3f} {m['quality_class']}{' WEAK' if m['weak'] else ''}{' ONE-FINGER' if m['one_finger_contact'] else ''}")
    L += ['', '-' * 32, '[Disk reload integrity]', '']
    for e, r in results.items():
        rl = r['meta']['reload']
        L.append(f"{e}: ok {rl['ok']} missing {rl.get('missing')} shape/dtype mismatch {rl.get('shape_dtype_mismatch')} "
                 f"max reload error {max((v for v in rl.get('reload_max_error', {}).values() if v is not None), default=None)} "
                 f"nan {rl.get('nan')} inf {rl.get('inf')} | initial vs settled(f32) {rl.get('initial_vs_settled_f32')} "
                 f"(f32 quantization {rl.get('initial_f32_quantization_vs_f64')}) | frame39 vs boundary cloud {rl.get('frame39_vs_boundary_cloud')} "
                 f"| actions vs bank {rl.get('actions_vs_sequence_bank')}")
    d = s['disk']
    L += ['', '-' * 32, '[Disk usage]', ''] + [f"{e}: npz {v:.2f} MiB" for e, v in d['npz_mib'].items()]
    L += [f"mean episode size (npz+metadata): {d['mean_episode_mib']:.2f} MiB  (uncompressed arrays {d['uncompressed_episode_mib']:.2f} MiB)",
          f"rollout_object_pos share of npz: {100 * d['rollout_share_of_npz']:.1f}%", f"sequence bank per sequence: {d['sequence_bank_kib_per_sequence']:.1f} KiB",
          f"dry-run total: {d['total_gib'] * 1024:.1f} MiB", f"free disk before: {d['free_before_gib']:.2f} GiB  after: {d['free_after_gib']:.2f} GiB",
          '', '-' * 32, '[Projected full dataset]', '', f"80 episode estimated size: {d['projected_full_gib']:.2f} GiB",
          f"expected remaining disk: {d['projected_free_after_full_gib']:.2f} GiB",
          '', '-' * 32, '[Status]', '', f"episodes: {s['statuses']}", f"invalid: {s['invalid'] or 'none'}",
          '', '-' * 32, '[git status]', '', 'before:', s['git_status_before'], 'after:', s['git_status_after'], '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
