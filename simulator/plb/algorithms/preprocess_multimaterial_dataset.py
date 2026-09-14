"""Raw 20k-particle multi-material episodes -> deterministic 300-point processed dataset ("RoboCraft simulator-GT setting").

Object subset = particle indices np.arange(0, 20000, 66)[:300], the same stride as the notebook's get_obs(env, 300) that
produced RoboCraft's shape_gt_*.h5 (NOT the perception path shape_*.h5). Same indices for every material/episode/grip/frame.

Processed files keep PHYSICAL time: object and tool arrays are aligned to the same simulator step. The legacy
object(t) + tool(t+1) shift is applied by robocraft/multimaterial_dataset.py at sample time, never baked in here.

  python preprocess_multimaterial_dataset.py --mode smoke   # sigma500/paired_000 only
  python preprocess_multimaterial_dataset.py --mode full    # 80 episodes + train-only normalization stats
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(HERE, '..', '..', 'dataset'))
RAW_ROOT = os.path.join(DATASET_DIR, 'full_dataset_14-Sep-2026-15:07:13')
RAW_PROVENANCE = 'c1a4aef2f591dad7f0a45d1f78f2f4d4c2691086'
N_RAW, STRIDE, N_POINTS = 20000, 66, 300
PARTICLE_INDICES_300 = np.arange(0, N_RAW, STRIDE)[:N_POINTS]
N_GRIP, NF = 5, 40
N_SHAPE, N_HIS, SEQUENCE_LENGTH = 31, 4, 6
STD_EPS = 1e-8
SMOKE_EPISODE = 'sigma500_paired_000'

EXPECTED = dict(
    initial_object_pos_300=((N_POINTS, 3), np.float32), rollout_object_pos_300=((N_GRIP, NF, N_POINTS, 3), np.float32),
    boundary_object_pos_300=((N_GRIP + 1, N_POINTS, 3), np.float32), particle_indices_300=((N_POINTS,), np.int64),
    tool_start_pose=((N_GRIP, 2, 7), np.float64), tool_pose_rollout=((N_GRIP, NF, 2, 7), np.float64),
    high_level_action=((N_GRIP, 5), np.float64), primitive_action=((N_GRIP, NF, 12), np.float64),
    midpoint=((N_GRIP, 3), np.float64), angle=((N_GRIP,), np.float64), target_gap=((N_GRIP,), np.float64),
    gripper_rate=((N_GRIP,), np.float64), actual_gap_f29=((N_GRIP,), np.float64),
    compression_RMS=((N_GRIP,), np.float64), normalized_residual=((N_GRIP,), np.float64),
    too_weak=((N_GRIP,), np.bool_), one_finger=((N_GRIP,), np.bool_), sigma_y=((), np.float64))
COPY_KEYS = ('tool_start_pose', 'tool_pose_rollout', 'high_level_action', 'primitive_action', 'midpoint', 'angle',
             'target_gap', 'gripper_rate', 'actual_gap_f29')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    return subprocess.run(['git', '-C', HERE, *a], capture_output=True, text=True).stdout.strip()


def jdump(obj, path):
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(path + '.tmp', path)


def dir_bytes(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs)


def object_timeline(p):
    """unique physical object states of an episode: P0 + 5*40 rollout states -> [201, 300, 3] (no duplicated boundary)"""
    return np.concatenate([p['initial_object_pos_300'][None], p['rollout_object_pos_300'].reshape(-1, N_POINTS, 3)])


def process_episode(raw_dir, out_dir, row, split):
    raw = np.load(os.path.join(raw_dir, 'episode.npz'))
    with open(os.path.join(raw_dir, 'metadata.json')) as f:
        meta = json.load(f)
    grips = meta['grips']
    assert len(grips) == N_GRIP and [gr['grip_index'] for gr in grips] == list(range(N_GRIP))
    init = raw['initial_object_pos'][PARTICLE_INDICES_300]
    roll = raw['rollout_object_pos'][:, :, PARTICLE_INDICES_300]
    data = dict(
        initial_object_pos_300=init, rollout_object_pos_300=roll,
        boundary_object_pos_300=np.concatenate([init[None], roll[:, NF - 1]]),
        particle_indices_300=PARTICLE_INDICES_300.astype(np.int64),
        **{k: raw[k] for k in COPY_KEYS},
        compression_RMS=np.array([gr['compression_RMS'] for gr in grips]), normalized_residual=np.array([gr['normalized_residual'] for gr in grips]),
        too_weak=np.array([gr['weak'] for gr in grips]), one_finger=np.array([gr['one_finger_contact'] for gr in grips]),
        sigma_y=np.array(float(meta['sigma_y'])))
    partial = os.path.join(os.path.dirname(out_dir), f'.{os.path.basename(out_dir)}.partial')
    os.makedirs(partial)
    np.savez_compressed(os.path.join(partial, 'processed.npz'), **data)
    pmeta = dict(episode_id=row['episode_id'], sequence_id=meta['sequence_id'], sequence_kind=meta['sequence_kind'], split=split,
                 E=meta['E'], nu=meta['nu'], sigma_y=meta['sigma_y'], seed=meta['seed'], action_hash=meta['action_hash'],
                 raw_episode_dir=os.path.relpath(raw_dir, RAW_ROOT), raw_episode_npz_sha256=sha256(os.path.join(raw_dir, 'episode.npz')),
                 sampling_method='simulator_stride', stride=STRIDE, n_points=N_POINTS,
                 grips=[dict(grip_index=gr['grip_index'], quality_class=gr['quality_class'], too_weak=gr['weak'],
                             one_finger=gr['one_finger_contact'], compression_RMS=gr['compression_RMS'],
                             residual_RMS=gr['residual_RMS'], normalized_residual=gr['normalized_residual'],
                             max_contact_particles=gr['max_contact_particles'], finger_pin_distance=gr['finger_pin_distance'])
                        for gr in grips],
                 arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in data.items()})
    jdump(pmeta, os.path.join(partial, 'metadata.json'))
    os.replace(partial, out_dir)


def verify_episode(raw_dir, out_dir, row, split):
    """reload processed from disk; compare with the raw subset on every frame"""
    out = dict(episode_id=row['episode_id'], missing=[], dtype_shape_mismatch=[], nan_inf=0)
    with np.load(os.path.join(out_dir, 'processed.npz')) as z:
        p = {k: z[k] for k in z.files}
    with open(os.path.join(out_dir, 'metadata.json')) as f:
        pmeta = json.load(f)
    with open(os.path.join(raw_dir, 'metadata.json')) as f:
        rmeta = json.load(f)
    for k, (shape, dt) in EXPECTED.items():
        if k not in p:
            out['missing'].append(k); continue
        if p[k].shape != shape or p[k].dtype != dt:
            out['dtype_shape_mismatch'].append([k, list(p[k].shape), str(p[k].dtype)])
        if p[k].dtype.kind == 'f':
            out['nan_inf'] += int((~np.isfinite(p[k])).sum())
    if out['missing'] or out['dtype_shape_mismatch']:
        out['ok'] = False
        return out, p
    raw = np.load(os.path.join(raw_dir, 'episode.npz'))
    r_init, r_roll = raw['initial_object_pos'], raw['rollout_object_pos']
    out['indices_ok'] = bool(np.array_equal(p['particle_indices_300'], PARTICLE_INDICES_300))
    out['raw_subset_max_error'] = max(float(np.abs(p['initial_object_pos_300'] - r_init[PARTICLE_INDICES_300]).max()),
                                      float(np.abs(p['rollout_object_pos_300'] - r_roll[:, :, PARTICLE_INDICES_300]).max()))
    out['raw_subset_bit_identical'] = bool(np.array_equal(p['initial_object_pos_300'], r_init[PARTICLE_INDICES_300])
                                           and np.array_equal(p['rollout_object_pos_300'], r_roll[:, :, PARTICLE_INDICES_300]))
    out['raw_boundary_cloud_300_max_error'] = float(np.abs(p['boundary_object_pos_300'] - raw['boundary_cloud_300']).max())
    out['boundary_max_error'] = max(float(np.abs(p['boundary_object_pos_300'][0] - p['initial_object_pos_300']).max()),
                                    float(np.abs(p['boundary_object_pos_300'][1:] - p['rollout_object_pos_300'][:, NF - 1]).max()))
    out['action_max_error'] = max(float(np.abs(p[k] - raw[k]).max()) for k in COPY_KEYS)
    del r_init, r_roll, raw
    g = rmeta['grips']
    out['flags_ok'] = bool(np.array_equal(p['too_weak'], [x['weak'] for x in g]) and np.array_equal(p['one_finger'], [x['one_finger_contact'] for x in g])
                           and np.array_equal(p['compression_RMS'], [x['compression_RMS'] for x in g])
                           and [x['quality_class'] for x in pmeta['grips']] == [x['quality_class'] for x in g])
    out['material_ok'] = bool(pmeta['sigma_y'] == rmeta['sigma_y'] == float(p['sigma_y']) == row['sigma_y'] and pmeta['E'] == rmeta['E'] and pmeta['nu'] == rmeta['nu'])
    out['split_ok'] = bool(pmeta['split'] == split == rmeta['split'])
    out['action_hash_ok'] = pmeta['action_hash'] == rmeta['action_hash']
    out['ok'] = (out['nan_inf'] == 0 and out['indices_ok'] and out['raw_subset_bit_identical'] and out['raw_boundary_cloud_300_max_error'] == 0
                 and out['boundary_max_error'] == 0 and out['action_max_error'] == 0 and out['flags_ok'] and out['material_ok']
                 and out['split_ok'] and out['action_hash_ok'])
    return out, p


def normalization_stats(processed, train_ids):
    """mean/std per axis over TRAIN episodes only.
    positions: every unique physical object state (P0 + 200 rollout states) x 300 points
    displacements: every consecutive physical transition (200 per episode, grip boundaries are continuous) x 300 points"""
    pos_sum, pos_sq, n_pos = np.zeros(3), np.zeros(3), 0
    d_sum, d_sq, n_d = np.zeros(3), np.zeros(3), 0
    for eid in train_ids:
        x = object_timeline(processed[eid]).astype(np.float64)            # [201, 300, 3]
        d = np.diff(x, axis=0)                                            # [200, 300, 3]
        pos_sum += x.sum((0, 1)); pos_sq += (x ** 2).sum((0, 1)); n_pos += x.shape[0] * x.shape[1]
        d_sum += d.sum((0, 1)); d_sq += (d ** 2).sum((0, 1)); n_d += d.shape[0] * d.shape[1]
    mean_p, mean_d = pos_sum / n_pos, d_sum / n_d
    std_p_raw, std_d_raw = np.sqrt(pos_sq / n_pos - mean_p ** 2), np.sqrt(d_sq / n_d - mean_d ** 2)
    return dict(mean_p=mean_p, std_p=np.maximum(std_p_raw, STD_EPS), mean_d=mean_d, std_d=np.maximum(std_d_raw, STD_EPS),
                std_p_raw=std_p_raw, std_d_raw=std_d_raw, std_eps=np.array(STD_EPS),
                n_episodes=np.array(len(train_ids)), n_states=np.array(n_pos // N_POINTS), n_position_samples=np.array(n_pos),
                n_transitions=np.array(n_d // N_POINTS), n_displacement_samples=np.array(n_d),
                train_episode_ids=np.array(sorted(train_ids)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke', 'full'], required=True)
    ap.add_argument('--raw-root', default=RAW_ROOT)
    args = ap.parse_args()
    t0 = time.time()
    status = git('status', '--short')
    if status:
        sys.exit(f'preprocessing requires a clean git tree (provenance):\n{status}')
    commit = git('rev-parse', 'HEAD')
    raw_root = os.path.abspath(args.raw_root)
    with open(os.path.join(raw_root, 'git_commit.txt')) as f:
        assert f.readline().strip() == RAW_PROVENANCE
    split_path = os.path.join(raw_root, 'split_manifest.json')
    with open(split_path) as f:
        raw_split = json.load(f)
    with open(os.path.join(raw_root, 'episode_index.csv')) as f:
        rows = [dict(r, sigma_y=float(r['sigma_y'])) for r in csv.DictReader(f)]
    split_of = {e: s for s, ids in raw_split['splits'].items() for e in ids}
    assert len(rows) == 80 and len(split_of) == 80 and all(r['split'] == split_of[r['episode_id']] for r in rows)
    if args.mode == 'smoke':
        rows = [r for r in rows if r['episode_id'] == SMOKE_EPISODE]

    stamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S")
    root = os.path.join(DATASET_DIR, f"processed_multimaterial_v1{'_smoke' if args.mode == 'smoke' else ''}_{stamp}")
    assert not os.path.exists(root)
    os.makedirs(os.path.join(root, 'episodes'))
    free_before = shutil.disk_usage(DATASET_DIR).free
    with open(os.path.join(root, 'git_commit.txt'), 'w') as f:
        f.write(f'{commit}\n\npreprocessing git status at start: (clean)\nraw dataset provenance: {RAW_PROVENANCE}\n')

    processed, verify, index = {}, [], []
    for i, r in enumerate(rows):
        raw_dir = os.path.join(raw_root, r['path'])
        out_dir = os.path.join(root, 'episodes', os.path.relpath(raw_dir, os.path.join(raw_root, 'episodes')))
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        process_episode(raw_dir, out_dir, r, split_of[r['episode_id']])
        v, p = verify_episode(raw_dir, out_dir, r, split_of[r['episode_id']])
        processed[r['episode_id']] = {k: p[k] for k in ('initial_object_pos_300', 'rollout_object_pos_300')}
        verify.append(v)
        index.append(dict(episode_id=r['episode_id'], sequence_id=r['sequence_id'], sequence_kind=r['sequence_kind'], sigma_y=r['sigma_y'],
                          E=r['E'], nu=r['nu'], split=split_of[r['episode_id']], path=os.path.relpath(out_dir, root),
                          npz_bytes=os.path.getsize(os.path.join(out_dir, 'processed.npz')), raw_path=r['path']))
        print(f"[{i + 1}/{len(rows)}] {r['episode_id']} split={index[-1]['split']} ok={v['ok']} raw_subset_err={v.get('raw_subset_max_error')} "
              f"size={index[-1]['npz_bytes'] / 2 ** 20:.2f} MiB", flush=True)

    with open(os.path.join(root, 'episode_index.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(index[0]))
        w.writeheader(); w.writerows(index)
    splits = {s: [r['episode_id'] for r in index if r['split'] == s] for s in ('train', 'val', 'test')}
    paired_groups = {}
    for r in index:
        if r['sequence_kind'] == 'paired':
            paired_groups.setdefault(r['sequence_id'], set()).add(r['split'])
    split_manifest = dict(source_split_manifest=split_path, source_split_manifest_sha256=sha256(split_path), policy=raw_split['policy'],
                          counts={s: len(v) for s, v in splits.items()}, splits=splits,
                          per_material={f'{sy:g}': {s: sum(1 for r in index if r['split'] == s and r['sigma_y'] == sy) for s in splits}
                                        for sy in sorted({r['sigma_y'] for r in index})},
                          paired_groups={k: sorted(v) for k, v in paired_groups.items()},
                          paired_split_leakage=sum(len(v) != 1 for v in paired_groups.values()))
    jdump(split_manifest, os.path.join(root, 'split_manifest.json'))

    train_ids = splits['train']
    stats = normalization_stats(processed, train_ids)
    assert not set(stats['train_episode_ids'].tolist()) & (set(split_of) - set(raw_split['splits']['train']))
    np.savez(os.path.join(root, 'normalization_stats.npz'), **stats)

    windows_per_grip = (NF + 1) - SEQUENCE_LENGTH + 1   # 36: object states S[0..40]; see multimaterial_dataset.py
    config = dict(
        dataset_version='processed_multimaterial_v1', setting='RoboCraft simulator-GT setting (shape_gt_* stride subset); not the RGB-D perception path',
        mode=args.mode, created=datetime.now().isoformat(), raw_dataset_root=raw_root, raw_dataset_provenance_hash=RAW_PROVENANCE,
        preprocessing_code_git_hash=commit, preprocessing_script=os.path.abspath(__file__), dataset_class='robocraft/multimaterial_dataset.py:MultiMaterialDynamicsDataset',
        sampling_method='simulator_stride', stride=STRIDE, n_points=N_POINTS, particle_indices_300=PARTICLE_INDICES_300.tolist(),
        particle_index_rule='np.arange(0, 20000, 66)[:300], identical for every material/episode/grip/frame, order never shuffled',
        n_particle=N_POINTS, n_shape=N_SHAPE, node_layout=dict(object=[0, 300], floor=[300, 309], finger0=[309, 320], finger1=[320, 331]),
        n_his=N_HIS, sequence_length=SEQUENCE_LENGTH, n_grips=N_GRIP, frames_per_grip=NF,
        split_counts=split_manifest['counts'], source_split_manifest_sha256=split_manifest['source_split_manifest_sha256'],
        physical_timeline=('grip g: object S_g[0]=boundary_object_pos_300[g], S_g[k+1]=rollout_object_pos_300[g,k] (41 states); '
                           'tool T_g[0]=tool_start_pose[g], T_g[k+1]=tool_pose_rollout[g,k] (41 states)'),
        window_policy=f'windows never cross grips; start u in [0, {windows_per_grip - 1}] -> {windows_per_grip} windows/grip, {windows_per_grip * N_GRIP}/episode',
        tool_alignment_policy=('legacy RoboCraft: node frame i of a window starting at u = object S_g[u+i] + fingers from T_g[u+i+1], applied in __getitem__; '
                               'the 6th frame of the last window (u=35) would need T_g[41], which does not exist: T_g[40] is used as placeholder '
                               '(train.py uses particles[:, 5] shape nodes only in the final pred_pos concat, line 215, never as model input)'),
        normalization_formula=dict(
            mean_p='mean over train episodes, physical object states P0 + 200 rollout states (201/episode, boundary not duplicated), 300 points, per axis',
            std_p='population std, same samples as mean_p, max(std, std_eps)',
            mean_d='mean over train episodes, 200 consecutive physical transitions x[t+1]-x[t] per episode (continuous across grips), 300 corresponding points, per axis',
            std_d='population std, same samples as mean_d, max(std, std_eps)',
            model_usage='model.py:174 (mean_d/std_d on all-node history differences), :176 (mean_p/std_p on all-node current position), :362 (mean_d/std_d de-normalize predicted object motion)'),
        std_eps=STD_EPS, filtering='none (TOO_WEAK and one-finger grips kept, flags in metadata/npz)')
    jdump(config, os.path.join(root, 'processed_config.json'))

    raw_size = dir_bytes(raw_root) if args.mode == 'full' else sum(os.path.getsize(os.path.join(raw_root, r['path'], 'episode.npz')) for r in rows)
    proc_size = dir_bytes(root)
    summary = dict(
        verdict='PASS' if all(v['ok'] for v in verify) and (args.mode == 'smoke' or (len(index) == 80 and split_manifest['counts'] == dict(train=56, val=12, test=12)
                                                                                     and split_manifest['paired_split_leakage'] == 0)) else 'FAIL',
        episodes=len(index), per_material_split=split_manifest['per_material'], split_counts=split_manifest['counts'],
        paired_split_leakage=split_manifest['paired_split_leakage'],
        integrity=dict(failed=[v['episode_id'] for v in verify if not v['ok']],
                       raw_subset_max_error=max(v['raw_subset_max_error'] for v in verify), raw_subset_bit_identical=all(v['raw_subset_bit_identical'] for v in verify),
                       raw_boundary_cloud_300_max_error=max(v['raw_boundary_cloud_300_max_error'] for v in verify),
                       boundary_max_error=max(v['boundary_max_error'] for v in verify), action_max_error=max(v['action_max_error'] for v in verify),
                       nan_inf=sum(v['nan_inf'] for v in verify), missing=sum(len(v['missing']) for v in verify),
                       dtype_shape_mismatch=sum(len(v['dtype_shape_mismatch']) for v in verify),
                       flags_mismatch=sum(not v['flags_ok'] for v in verify), material_mismatch=sum(not v['material_ok'] for v in verify),
                       split_mismatch=sum(not v['split_ok'] for v in verify), action_hash_mismatch=sum(not v['action_hash_ok'] for v in verify)),
        flags=dict(too_weak=int(sum(processed_flag(root, r, 'too_weak') for r in index)), one_finger=int(sum(processed_flag(root, r, 'one_finger') for r in index)),
                   one_finger_grips=[(r['episode_id'], int(k)) for r in index for k in np.nonzero(load_flag(root, r, 'one_finger'))[0]], filtered=0),
        normalization={k: v.tolist() for k, v in stats.items() if k != 'train_episode_ids'},
        disk=dict(raw_bytes=raw_size, processed_bytes=proc_size, mean_processed_episode_bytes=float(np.mean([r['npz_bytes'] for r in index])),
                  compression_ratio_raw_over_processed=raw_size / proc_size, free_before_gib=free_before / 2 ** 30,
                  free_after_gib=shutil.disk_usage(DATASET_DIR).free / 2 ** 30),
        runtime_s=time.time() - t0, verify=verify, git_status_after=git('status', '--short'))
    jdump(summary, os.path.join(root, 'preprocessing_summary.json'))
    L = [f"VERDICT: {summary['verdict']}", f'processed root: {root}', f'raw root: {raw_root} (provenance {RAW_PROVENANCE})',
         f'preprocessing git hash: {commit}', f"episodes: {len(index)}  split counts: {split_manifest['counts']}  per material: {split_manifest['per_material']}",
         f"paired split leakage: {split_manifest['paired_split_leakage']}", f"integrity: {json.dumps(summary['integrity'])}",
         f"flags: {json.dumps(summary['flags'])}", 'normalization (train only):'] + \
        [f'  {k}: {v}' for k, v in summary['normalization'].items()] + \
        [f"disk: raw {raw_size / 2 ** 30:.3f} GiB  processed {proc_size / 2 ** 20:.1f} MiB  mean episode {summary['disk']['mean_processed_episode_bytes'] / 2 ** 20:.2f} MiB  "
         f"ratio {summary['disk']['compression_ratio_raw_over_processed']:.1f}x  free after {summary['disk']['free_after_gib']:.2f} GiB",
         f"runtime: {summary['runtime_s']:.1f} s", f"git status after: {summary['git_status_after'] or '(clean)'}", '']
    with open(os.path.join(root, 'preprocessing_report.txt'), 'w') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))
    sys.exit(0 if summary['verdict'] == 'PASS' else 1)


def load_flag(root, r, key):
    with np.load(os.path.join(root, r['path'], 'processed.npz')) as z:
        return z[key]


def processed_flag(root, r, key):
    return load_flag(root, r, key).sum()


if __name__ == '__main__':
    main()
