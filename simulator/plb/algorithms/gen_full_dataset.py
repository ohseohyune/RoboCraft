"""Multi-material sequential-grip RAW dataset generator (20k-particle rollouts).

One episode = one material (E=5000, nu=0.2, sigma_y in SIGMAS), 5 sequential grips (30 close + 10 retreat) from the
shared settled state, no reset between grips. Action sequences are generated first, saved to sequence_bank/, and every
episode is simulated from the saved file (paired sequences are shared bit-identically across materials).

Reused (not re-derived): action_distribution_pilot (pose_from, analytic pin filter, same-axis rule, run_grip, metrics,
classify), material_sensitivity_debug (restore, applied_material, state_error), stress_test_grip (gap_to_rate),
gen_multigrip_debug (env, primitive actions, 300-cloud), gen_material_history_pilot (collapse ratio).
Simulator / yaml / existing scripts untouched.

  python gen_full_dataset.py --mode smoke          # 1 random episode, sigma 500, no resample on invalid (PASS/FAIL)
  python gen_full_dataset.py --mode dry-run        # 1 paired x 4 materials + 1 random (sigma 500) = 5 episodes
  python gen_full_dataset.py --mode full --confirm-full   # 6 paired x 4 + 14 random x 4 = 80 episodes (needs approval)

Layout: sequence_bank/{paired_000, sigma30_random_000}.npz, episodes/sigma30/{paired_000, random_000}/{episode.npz, metadata.json}.
Full mode only: an invalid episode replaces its whole sequence group (paired: all 4 materials) with a new seed;
rejected sequences/episodes are kept under rejected/ and logged in sequence_manifest.json.
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
import time
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
MODE_CODE = {'dry-run': 0, 'full': 1, 'smoke': 2}
KIND_CODE = {'paired': 0, 'random': 1}
N_PAIRED, N_RANDOM = 6, 14
SPLITS = dict(paired=['train'] * 4 + ['val', 'test'], random=['train'] * 10 + ['val'] * 2 + ['test'] * 2)   # by sequence index
MAX_REGEN = 3
BANK_NAMES = dict(finger_start_pose='tool_start_pose', primitive_action_sequence='primitive_action')   # pilot key -> bank key
PAIRS = [(30.0, 200.0), (200.0, 500.0), (500.0, 900.0)]


# ------------------------------------------------------------------ plan
def plan(mode):
    """sequences: [(sequence_id, kind, index, sigma or None)], episodes: [(episode_id, sequence_id, sigma, rel_dir)]"""
    if mode == 'smoke':
        seqs = [('sigma500_smoke_000', 'random', 0, 500.0)]
    elif mode == 'dry-run':
        seqs = [('paired_dryrun_000', 'paired', 0, None), ('sigma500_random_dryrun_000', 'random', 0, 500.0)]
    else:
        seqs = [(f'paired_{i:03d}', 'paired', i, None) for i in range(N_PAIRED)]
        seqs += [(f'sigma{sy:g}_random_{i:03d}', 'random', i, sy) for sy in SIGMAS for i in range(N_RANDOM)]
    eps = []
    for sid, kind, _, sy in seqs:
        for s in (SIGMAS if kind == 'paired' else [sy]):
            name = sid if kind == 'paired' else sid.split('_', 1)[1]                  # paired_000 / random_000
            eps.append((f'sigma{s:g}_{name}', sid, s, f'sigma{s:g}/{name}'))
    return seqs, eps


def split_of(mode, kind, index):
    return SPLITS[kind][index] if mode == 'full' else mode.replace('-', '')


def sequence_seed(mode, kind, index, sigma, attempt=0):
    entropy = [MASTER_SEED, MODE_CODE[mode], KIND_CODE[kind], int(sigma or 0), index] + ([attempt] if attempt else [])
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


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
        finger_min_clearance=stack('finger_min_clearance'), finger_contact_count=stack('finger_contact_count').astype(np.int32),
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
    finger_min_clearance=((N_GRIP, NF, 2), np.float64), finger_contact_count=((N_GRIP, NF, 2), np.int32),
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


# ------------------------------------------------------------------ orchestration
def save_sequence(root, sid, seed):
    """generate -> save -> reload; simulation only ever uses the reloaded arrays"""
    s = make_sequence(seed)
    path = os.path.join(root, 'sequence_bank', sid + '.npz')
    np.savez_compressed(path, sequence_id=np.array(sid), seed=np.array(seed), action_hash=np.array(action_hash(s)),
                        **{BANK_NAMES.get(k, k): v for k, v in s.items()})
    loaded = dict(np.load(path))
    for pilot_key, bank_key in BANK_NAMES.items():
        loaded[pilot_key] = loaded[bank_key]          # run_grip / verify_reload use the pilot key names
    assert str(loaded['action_hash']) == action_hash(loaded)
    regen = make_sequence(seed)
    return loaded, max(float(np.abs(regen[k] - loaded[k]).max()) for k in regen)


def sequence_entry(sid, kind, idx, sy, seed, attempt, seq, repro):
    return dict(sequence_id=sid, kind=kind, index=idx, sigma_y=sy, seed=seed, attempt=attempt, path=f'sequence_bank/{sid}.npz',
                action_hash=str(seq['action_hash']), regenerate_from_seed_max_error=repro,
                materials=SIGMAS if kind == 'paired' else [sy],
                grips=[dict(midpoint=seq['midpoint'][k].tolist(), angle_deg=float(np.degrees(seq['angle'][k])),
                            target_gap=float(seq['target_gap'][k]), gripper_rate=float(seq['gripper_rate'][k]),
                            analytic_pin_distance=float(seq['analytic_pin_distance'][k]),
                            rejected_draws=int(seq['rejected_draws'][k]), axis_resampled_draws=int(seq['axis_resampled_draws'][k]),
                            delta_axis_prev_deg=None if k == 0 else float(np.degrees(adp.delta_axis(seq['angle'][k], seq['angle'][k - 1]))))
                       for k in range(N_GRIP)], regenerations=[])


def simulate_and_save(env, settled, t, root, eid, rel, sid, kind, sy, seq, split):
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
    ep_hash = action_hash(dict(high_level_action=data['high_level_action'], gripper_rate=data['gripper_rate'],
                               finger_start_pose=data['tool_start_pose'], primitive_action_sequence=data['primitive_action']))
    if ep_hash != str(seq['action_hash']):
        reasons.append('action hash != sequence bank')
    final = os.path.join(root, 'episodes', rel)
    partial = os.path.join(os.path.dirname(final), f'.{os.path.basename(final)}.partial')
    os.makedirs(partial)
    npz = os.path.join(partial, 'episode.npz')
    np.savez_compressed(npz, **data)
    reload = verify_reload(npz, data, seq, settled) if not info['failed'] else dict(ok=False, note='partial episode, reload check skipped')
    if not reload['ok']:
        reasons.append('reload verification failed')
    status = 'ok' if not reasons else 'INVALID'
    meta = dict(episode_id=eid, sequence_id=sid, sequence_kind=kind, split=split, status=status, invalid_reasons=reasons,
                E=E, nu=msd.NU, sigma_y=sy, seed=int(seq['seed']), action_hash=str(seq['action_hash']),
                material_applied=info['material'], initial_state_error=info['initial_state_error'], boundary_continuity=info['boundary'],
                frame39_vs_full_state_x_f32=info['frame39_vs_full_state_x_f32'], completed_grips=info['completed_grips'],
                grips=rows, reload=reload, arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in data.items()})
    jdump(meta, os.path.join(partial, 'metadata.json'))
    if status != 'ok':
        final += '__INVALID'
    os.replace(partial, final)
    row = dict(episode_id=eid, sequence_id=sid, sequence_kind=kind, sigma_y=sy, E=E, nu=msd.NU, split=split, status=status,
               action_hash=meta['action_hash'], path=os.path.relpath(final, root),
               npz_bytes=os.path.getsize(os.path.join(final, 'episode.npz')), metadata_bytes=os.path.getsize(os.path.join(final, 'metadata.json')))
    return meta, row


def validate(root, index, mode, seq_plan, settled):
    """dataset-level checks, re-read from disk (episode.npz + metadata.json)"""
    metas, arrays = {}, {}
    x_settled = settled['state'][0].astype(np.float32)
    small = ('initial_object_pos', 'boundary_cloud_300', 'high_level_action', 'midpoint', 'angle', 'target_gap', 'gripper_rate',
             'tool_start_pose', 'primitive_action')
    schema_bad, init_err, cross_init = [], 0.0, 0.0
    ref_init = None
    for r in index:
        d = os.path.join(root, r['path'])
        with open(os.path.join(d, 'metadata.json')) as f:
            metas[r['episode_id']] = json.load(f)
        with np.load(os.path.join(d, 'episode.npz')) as z:
            full = {k: z[k] for k in z.files}
        for k, (shape, dt) in EXPECTED.items():
            if k not in full or full[k].shape != shape or full[k].dtype != dt:
                schema_bad.append((r['episode_id'], k))
        a = {k: full[k] for k in small}
        a['frame39'] = full['rollout_object_pos'][:, NF - 1].copy()
        del full
        init_err = max(init_err, float(np.abs(a['initial_object_pos'] - x_settled).max()))
        ref_init = a['initial_object_pos'] if ref_init is None else ref_init
        cross_init = max(cross_init, float(np.abs(a['initial_object_pos'] - ref_init).max()))
        arrays[r['episode_id']] = a
    M = list(metas.values())
    grips = [m for e in M for m in e['grips']]
    by_mat = {sy: [e for e in M if e['sigma_y'] == sy] for sy in sorted({e['sigma_y'] for e in M})}

    paired, pairwise = {}, {f'{a:g}vs{b:g}': [] for a, b in PAIRS}
    for sid, kind, _, _ in seq_plan:
        eps = {e['sigma_y']: e['episode_id'] for e in M if e['sequence_id'] == sid}
        if kind != 'paired' or not eps:
            continue
        ref = arrays[next(iter(eps.values()))]
        paired[sid] = dict(n_materials=len(eps), splits=sorted({metas[e]['split'] for e in eps.values()}),
                           hashes=len({metas[e]['action_hash'] for e in eps.values()}),
                           max_error=max(float(np.abs(arrays[e][k] - ref[k]).max()) for e in eps.values()
                                         for k in ('high_level_action', 'midpoint', 'angle', 'target_gap', 'gripper_rate', 'tool_start_pose', 'primitive_action')))
        for a, b in PAIRS:
            if a in eps and b in eps:
                pairwise[f'{a:g}vs{b:g}'] += [msd.disp(arrays[eps[a]]['frame39'][k], arrays[eps[b]]['frame39'][k])['rms'] for k in range(N_GRIP)]

    def stat(v):
        v = np.array(v, float)
        return dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max())) if len(v) else None

    splits = {s: [e['episode_id'] for e in M if e['split'] == s] for s in sorted({e['split'] for e in M})}
    out = dict(
        n_episodes=len(M), n_grips=len(grips), n_materials=len(by_mat),
        per_material={f'{sy:g}': dict(episodes=len(es), paired=sum(e['sequence_kind'] == 'paired' for e in es),
                                      random=sum(e['sequence_kind'] == 'random' for e in es), grips=sum(len(e['grips']) for e in es),
                                      splits={s: sum(e['split'] == s for e in es) for s in splits},
                                      GOOD=sum(m['quality_class'] == 'GOOD' for e in es for m in e['grips']),
                                      TOO_WEAK=sum(m['weak'] for e in es for m in e['grips']),
                                      one_finger=sum(m['one_finger_contact'] for e in es for m in e['grips']),
                                      compression_RMS=stat([m['compression_RMS'] for e in es for m in e['grips']]),
                                      normalized_residual=stat([m['normalized_residual'] for e in es for m in e['grips']]))
                      for sy, es in by_mat.items()},
        split_counts={s: len(v) for s, v in splits.items()},
        paired_groups=len(paired), paired_split_leakage=sum(len(p['splits']) != 1 for p in paired.values()),
        paired_action_max_error=max([p['max_error'] for p in paired.values()], default=None),
        paired_hash_mismatch=sum(p['hashes'] != 1 for p in paired.values()), paired_incomplete=sum(p['n_materials'] != len(SIGMAS) for p in paired.values()),
        initial_state_max_error_disk=init_err, cross_episode_initial_max_error=cross_init,
        initial_state_max_error_full_state=max(v for e in M for v in e['initial_state_error'].values()),
        continuity_max_error=max((v for e in M for b in e['boundary_continuity'] for v in b.values()), default=0.0),
        frame39_vs_state_max_error=max((v for e in M for v in e['frame39_vs_full_state_x_f32']), default=0.0),
        material_mismatch=sum(not e['material_applied']['match'] for e in M),
        reload_failures=sum(not e['reload']['ok'] for e in M), schema_mismatch=schema_bad,
        finger_contact_count_dtypes=sorted({e['arrays']['finger_contact_count']['dtype'] for e in M}),
        nan=sum(m['nan'] for m in grips), inf=sum(m['inf'] for m in grips), explosion=sum(m['explosion'] for m in grips),
        collapse=sum(m['collapse'] for m in grips), pin_collision=sum(m['pin_collision'] for m in grips),
        weak=sum(m['weak'] for m in grips), one_finger=sum(m['one_finger_contact'] for m in grips),
        min_finger_pin_distance=min((m['finger_pin_distance'] for m in grips), default=None),
        max_step_disp=max((m['max_step_disp'] for m in grips), default=None),
        pairwise_frame39_rms={k: stat(v) for k, v in pairwise.items()}, paired_detail=paired)
    if mode == 'full':
        exp = dict(n_episodes=80, n_grips=400, n_materials=4, paired_groups=N_PAIRED)
        out['expected_counts_ok'] = (all(out[k] == v for k, v in exp.items())
                                     and all(p['episodes'] == 20 and p['paired'] == N_PAIRED and p['random'] == N_RANDOM for p in out['per_material'].values())
                                     and out['split_counts'] == dict(test=12, train=56, val=12)
                                     and all(p['splits'] == dict(test=3, train=14, val=3) for p in out['per_material'].values()))
    return out, metas


def write_report(root, config, s, manifest, metas, timing):
    L = [f"[{config['mode']}]  {root}", '', f"git commit: {config['git_commit']}  (tree clean at start: {config['git_clean_at_start']})",
         f"materials: E={E:g} nu={msd.NU} sigma_y={SIGMAS}   master seed {MASTER_SEED}",
         f"episodes {s['n_episodes']}  grips {s['n_grips']}  materials {s['n_materials']}  paired groups {s['paired_groups']}",
         f"expected full counts ok: {s.get('expected_counts_ok', 'n/a')}", '', '-' * 32, '[Per material]', '']
    for sy, p in s['per_material'].items():
        c, n = p['compression_RMS'], p['normalized_residual']
        L.append(f"sigma {sy}: episodes {p['episodes']} (paired {p['paired']} random {p['random']}) grips {p['grips']} splits {p['splits']} "
                 f"| GOOD {p['GOOD']} TOO_WEAK {p['TOO_WEAK']} one-finger {p['one_finger']}")
        L.append(f"          compression_RMS mean {c['mean']:.4f} std {c['std']:.4f} min {c['min']:.4f} max {c['max']:.4f} | "
                 f"normalized_residual mean {n['mean']:.3f} std {n['std']:.3f} min {n['min']:.3f} max {n['max']:.3f}")
    L += ['', '-' * 32, '[Integrity]', '',
          f"split counts: {s['split_counts']}   paired split leakage: {s['paired_split_leakage']}",
          f"paired action max error: {s['paired_action_max_error']}  hash mismatch groups: {s['paired_hash_mismatch']}  incomplete groups: {s['paired_incomplete']}",
          f"initial state max error: full state {s['initial_state_max_error_full_state']}  disk float32 vs settled {s['initial_state_max_error_disk']}  cross-episode {s['cross_episode_initial_max_error']}",
          f"continuity max error (object + full state): {s['continuity_max_error']}  frame39 vs boundary state: {s['frame39_vs_state_max_error']}",
          f"material mismatch: {s['material_mismatch']}  reload failures: {s['reload_failures']}  schema mismatch: {s['schema_mismatch'] or 'none'}",
          f"finger_contact_count dtype: {s['finger_contact_count_dtypes']}", '', '-' * 32, '[Stability]', '',
          f"NaN {s['nan']}  Inf {s['inf']}  explosion {s['explosion']}  collapse {s['collapse']}  pin collision {s['pin_collision']} (min finger-pin {s['min_finger_pin_distance']})",
          f"weak grips {s['weak']}  one-finger {s['one_finger']}  max per-step disp {s['max_step_disp']}", '',
          '-' * 32, '[Paired material response sanity]  frame39 20k-particle RMS, same particle index, per (sequence, grip)', '']
    L += [f"{k}: " + (f"mean {v['mean']:.4f} min {v['min']:.4f} max {v['max']:.4f} (n={len(s['paired_detail']) * N_GRIP})" if v else 'n/a')
          for k, v in s['pairwise_frame39_rms'].items()]
    regen = [dict(sequence_id=m['sequence_id'], **r) for m in manifest for r in m['regenerations']]
    L += ['', '-' * 32, '[Regenerations]', ''] + ([json.dumps(r) for r in regen] or ['none'])
    d = timing['disk']
    L += ['', '-' * 32, '[Disk]', '', f"dataset size: {d['total_gib']:.3f} GiB   mean episode {d['mean_episode_mib']:.2f} MiB (min {d['min_episode_mib']:.2f} max {d['max_episode_mib']:.2f})",
          f"free disk before {d['free_before_gib']:.2f} GiB  after {d['free_after_gib']:.2f} GiB",
          '', '-' * 32, '[Runtime]', '', f"generation {timing['generation_s'] / 60:.1f} min  validation {timing['validation_s'] / 60:.1f} min  total {timing['total_s'] / 60:.1f} min",
          '', '-' * 32, '[Per grip]', '',
          f"{'episode':26s} {'split':5s} g {'gap':>5} {'ang':>6} {'comp':>7} {'resid':>7} {'nres':>5} {'pin':>7} {'contact':>11} class"]
    for e in metas.values():
        for m in e['grips']:
            L.append(f"{e['episode_id']:26s} {e['split']:5s} {m['grip_index']} {m['target_gap']:.3f} {m['angle_deg']:6.1f} {m['compression_RMS']:7.4f} "
                     f"{m['residual_RMS']:7.4f} {m['normalized_residual']:5.2f} {m['finger_pin_distance']:+7.4f} {str(m['max_contact_particles']):>11} "
                     f"{m['quality_class']}{' WEAK' if m['weak'] else ''}{' ONE-FINGER' if m['one_finger_contact'] else ''}")
    L.append('')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'dry-run', 'full'], required=True)
    ap.add_argument('--confirm-full', action='store_true', help='required for --mode full (user approval)')
    ap.add_argument('--settled-state', default=SETTLED)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    t_start = time.time()
    if args.mode == 'full' and not args.confirm_full:
        sys.exit('--mode full requires --confirm-full (explicit user approval after dry-run review)')
    git_status_before = git('status', '--short')
    if args.mode == 'full' and git_status_before:
        sys.exit(f'--mode full requires a clean git tree (provenance):\n{git_status_before}')
    adp.check_pose_from(); adp.check_same_axis()
    rec = check_frozen_region()
    t = types.SimpleNamespace(**rec['thresholds'])
    settled_path = os.path.abspath(args.settled_state)
    with open(settled_path, 'rb') as f:
        settled = pickle.load(f)
    assert settled['state'][0].shape == (N_PARTICLES, 3)

    stamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S")
    root = os.path.abspath(args.out or os.path.join(DATASET_DIR, f"full_dataset{'' if args.mode == 'full' else '_' + args.mode.replace('-', '')}_{stamp}"))
    assert not os.path.exists(root), root
    free_before = shutil.disk_usage(DATASET_DIR).free
    os.makedirs(os.path.join(root, 'sequence_bank')); os.makedirs(os.path.join(root, 'episodes'))
    commit = git('rev-parse', 'HEAD')
    with open(os.path.join(root, 'git_commit.txt'), 'w') as f:
        f.write(f'{commit}\n\ngit status --short at generation start:\n{git_status_before or "(clean)"}\n')

    seq_plan, ep_plan = plan(args.mode)
    config = dict(dataset_version=DATASET_VERSION, mode=args.mode, generator_script=os.path.abspath(__file__), git_commit=commit,
                  git_clean_at_start=not git_status_before,
                  created=datetime.now().isoformat(), shared_settled_state=settled_path, E=E, nu=msd.NU, sigma_y=SIGMAS,
                  n_episodes=len(ep_plan), n_sequences=len(seq_plan), paired_per_material=N_PAIRED if args.mode == 'full' else None,
                  random_per_material=N_RANDOM if args.mode == 'full' else None,
                  action_distribution=dict(midpoint_x=MID_RANGE, midpoint_z=MID_RANGE, midpoint_y=float(adp.MID[1]),
                                           angle=adp.ANGLE_RANGE, target_gap=GAP_RANGE, sampling='i.i.d. per grip; gap first, then (x,z,angle) rejection by pin filter',
                                           start_pose=rec['start_pose'], gap_to_rate=rec['gap_to_rate'], frozen_region_file=os.path.join(PILOT, 'recommended_action_region.json')),
                  pin_filter=dict(threshold=PIN_MARGIN, formula=rec['analytic_pin_distance']),
                  same_axis=dict(threshold_deg=adp.SAME_AXIS_DEG, max_consecutive=MAX_SAME_AXIS, on_violation='resample angle only'),
                  n_grips=N_GRIP, close_steps=g.task_params['len_per_grip'], retreat_steps=g.task_params['len_per_grip_back'],
                  simulator_particle_count=N_PARTICLES, simulator_dtype='float64', raw_particle_dtype='float32',
                  action_pose_dtype='float64 (exact simulator inputs; bit-identity across paired materials)', finger_contact_count_dtype='int32',
                  master_seed=MASTER_SEED, seed_rule='SeedSequence([master_seed, mode(dry-run 0/full 1/smoke 2), kind(paired 0/random 1), sigma or 0, index] + [attempt if attempt > 0]).generate_state(1)[0]',
                  metric_thresholds=rec['thresholds'], collapse_height_ratio=mhp.COLLAPSE_HEIGHT_RATIO, explosion_step_disp=msd.EXPLOSION_STEP_DISP,
                  invalid_definition='ValueError, NaN, Inf, explosion, collapse, pin collision, continuity/initial-state/material/action-hash/reload mismatch; TOO_WEAK and one-finger are flags only',
                  regeneration=f'full mode only: invalid episode -> whole sequence group (paired: 4 materials) re-sampled with attempt-indexed seed, max {MAX_REGEN}; smoke/dry-run: never',
                  file_format='npz (np.savez_compressed), written to a .partial dir and atomically renamed after reload verification',
                  frame_semantics='rollout_object_pos[k, f] = simulator x after env.step(primitive_action[k, f]); P0=initial_object_pos, P_{k+1}=rollout_object_pos[k, 39]',
                  boundary_vel_semantics='boundary_object_vel[0]=P0, [k+1]=after grip k (frame 39)',
                  tool_semantics='tool_start_pose[k] = finger poses (2x7: xyz + quat) teleported before grip k; tool_pose_rollout[k, f] after env.step',
                  split_policy=dict(per_material=dict(train='paired 0-3 + random 0-9', val='paired 4 + random 10-11', test='paired 5 + random 12-13'),
                                    note='assigned by sequence index; all materials of a paired sequence share its split') if args.mode == 'full' else args.mode)
    jdump(config, os.path.join(root, 'dataset_config.json'))

    env, _ = g.make_env()
    manifest, index, invalid, aborted = [], [], {}, None
    n_done = 0
    for sid, kind, idx, sy in seq_plan:
        group = [e for e in ep_plan if e[1] == sid]
        split = split_of(args.mode, kind, idx)
        attempt = 0
        while True:
            seed = sequence_seed(args.mode, kind, idx, sy, attempt)
            seq, repro = save_sequence(root, sid, seed)
            entry = sequence_entry(sid, kind, idx, sy, seed, attempt, seq, repro) if attempt == 0 else entry
            if attempt:
                entry.update(seed=seed, attempt=attempt, action_hash=str(seq['action_hash']), regenerate_from_seed_max_error=repro,
                             grips=sequence_entry(sid, kind, idx, sy, seed, attempt, seq, repro)['grips'])
                entry['regenerations'][-1].update(replacement_seed=seed, replacement_action_hash=str(seq['action_hash']))
            rows, bad = [], {}
            for eid, _, s, rel in group:
                t0 = time.time()
                meta, row = simulate_and_save(env, settled, t, root, eid, rel, sid, kind, s, seq, split)
                rows.append(row)
                if meta['status'] != 'ok':
                    bad[eid] = meta['invalid_reasons']
                print(f"[{n_done + len(rows)}/{len(ep_plan)}] sigma={s:g} type={kind} id={rel.split('/')[1]} split={split} attempt={attempt} "
                      f"status={meta['status'].upper()} size={row['npz_bytes'] / 2 ** 20:.1f} MiB episode={time.time() - t0:.0f}s "
                      f"elapsed={(time.time() - t_start) / 60:.1f}min weak={sum(m['weak'] for m in meta['grips'])} "
                      f"one_finger={sum(m['one_finger_contact'] for m in meta['grips'])}{' ' + str(meta['invalid_reasons']) if bad.get(eid) else ''}", flush=True)
            if not bad or args.mode != 'full':
                break
            # full mode: replace the whole group, keep every rejected artifact
            rej = os.path.join(root, 'rejected', f'{sid}_attempt{attempt}')
            os.makedirs(rej)
            os.replace(os.path.join(root, 'sequence_bank', sid + '.npz'), os.path.join(rej, sid + '.npz'))
            for r in rows:
                dst = os.path.join(rej, r['path'])                                       # rejected/<sid>_attemptN/episodes/sigmaX/<name>
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(os.path.join(root, r['path']), dst)
            entry['regenerations'].append(dict(rejected_attempt=attempt, rejected_seed=seed, rejected_action_hash=str(seq['action_hash']),
                                               rejection_reasons=bad, rejected_dir=os.path.relpath(rej, root)))
            print(f'  group {sid} attempt {attempt} rejected -> regenerate ({bad})', flush=True)
            attempt += 1
            if attempt > MAX_REGEN:
                aborted = f'{sid}: {MAX_REGEN} regenerations exhausted'
                break
        manifest.append(entry)
        n_done += len(group)
        index += [r for r in rows if r['status'] == 'ok']
        invalid.update(bad)
        if aborted:
            print('ABORT', aborted, flush=True)
            break
    jdump(manifest, os.path.join(root, 'sequence_manifest.json'))
    with open(os.path.join(root, 'episode_index.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['episode_id', 'sequence_id', 'sequence_kind', 'sigma_y', 'E', 'nu', 'split', 'status',
                                          'action_hash', 'path', 'npz_bytes', 'metadata_bytes'])
        w.writeheader(); w.writerows(index)
    t_gen = time.time()

    s, metas = validate(root, index, args.mode, seq_plan, settled) if index else (None, {})
    splits = {}
    for r in index:
        splits.setdefault(r['split'], []).append(r['episode_id'])
    jdump(dict(policy=config['split_policy'], counts={k: len(v) for k, v in splits.items()}, splits=splits,
               paired_groups={sid: sorted({r['split'] for r in index if r['sequence_id'] == sid}) for sid, kind, _, _ in seq_plan if kind == 'paired'},
               paired_split_leakage=s and s['paired_split_leakage']), os.path.join(root, 'split_manifest.json'))
    sizes = [r['npz_bytes'] + r['metadata_bytes'] for r in index] or [0]
    timing = dict(generation_s=t_gen - t_start, validation_s=time.time() - t_gen, total_s=time.time() - t_start,
                  disk=dict(free_before_gib=gib(free_before), free_after_gib=gib(shutil.disk_usage(DATASET_DIR).free), total_gib=gib(dir_bytes(root)),
                            mean_episode_mib=float(np.mean(sizes)) / 2 ** 20, min_episode_mib=min(sizes) / 2 ** 20, max_episode_mib=max(sizes) / 2 ** 20))
    checks = None
    if s:
        checks = dict(finger_contact_count_int32=s['finger_contact_count_dtypes'] == ['int32'], schema=not s['schema_mismatch'],
                      initial_state=s['initial_state_max_error_full_state'] == 0 and s['initial_state_max_error_disk'] == 0,
                      continuity=s['continuity_max_error'] == 0 and s['frame39_vs_state_max_error'] == 0,
                      nan_inf=s['nan'] == 0 and s['inf'] == 0, explosion_collapse=s['explosion'] == 0 and s['collapse'] == 0,
                      pin_collision=s['pin_collision'] == 0, material=s['material_mismatch'] == 0, reload=s['reload_failures'] == 0,
                      no_invalid_episodes=not invalid, not_aborted=not aborted)
        if args.mode == 'full':
            checks.update(counts=s['expected_counts_ok'], paired_identity=s['paired_action_max_error'] == 0 and s['paired_hash_mismatch'] == 0,
                          split_leakage=s['paired_split_leakage'] == 0)
    verdict = 'PASS' if checks and all(checks.values()) else 'FAIL'
    summary = dict(verdict=verdict, checks=checks, invalid_episodes=invalid, aborted=aborted, validation=s, timing=timing,
                   git_commit=commit, git_status_before=git_status_before, git_status_after=git('status', '--short'))
    jdump(summary, os.path.join(root, 'generation_summary.json'))
    report = (f"VERDICT: {verdict}\nchecks: {json.dumps(checks)}\ninvalid episodes: {invalid or 'none'}\naborted: {aborted}\n\n"
              + (write_report(root, config, s, manifest, metas, timing) if s else '')
              + f"\ngit status after:\n{summary['git_status_after'] or '(clean)'}\n")
    with open(os.path.join(root, 'generation_report.txt'), 'w') as f:
        f.write(report)
    print(report)
    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()
