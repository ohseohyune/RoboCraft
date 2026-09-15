"""Manifest-based dataset over processed_multimaterial_v1 (RoboCraft simulator-GT setting), baseline mode only.

Returns the same 9-tuple as utils.PhysicsFleXDataset (attr, particles, n_particle, n_shape, scene_params, Rr, Rs, Rn,
cluster_onehot) and works with utils.my_collate. Nodes: object 0:300, floor 300:309, finger0 309:320, finger1 320:331.

Physical timeline per grip g (41 states): object S_g[k], tool T_g[k] at the same simulator step
(S_g[0] = boundary before grip g, T_g[0] = teleported start pose). Legacy RoboCraft alignment is applied here:
node frame i of a window starting at u = object S_g[u+i] + finger nodes from T_g[u+i+1].
Windows never cross grips: u in [0, 35]. The 6th frame of u=35 needs T_g[41] (does not exist); T_g[40] is used as a
placeholder because train.py only concatenates particles[:, 5] shape nodes into the final pred_pos (line 215).

Self-check: python multimaterial_dataset.py --root <processed root>
"""
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import prepare_input

N_PARTICLE, N_SHAPE, N_FLOOR, N_FINGER_NODES = 300, 31, 9, 11
N_STATES = 41                                            # per grip: boundary + 40 rollout states
SCENE_PARAMS = np.array([1., 1., 0.])                    # sample_data.py: [n_instance, gravity, draw_mesh]
FLOOR = np.array([[0.25, 0., 0.25], [0.25, 0., 0.5], [0.25, 0., 0.75],
                  [0.5, 0., 0.25], [0.5, 0., 0.5], [0.5, 0., 0.75],
                  [0.75, 0., 0.25], [0.75, 0., 0.5], [0.75, 0., 0.75]])   # sample_data.shape_aug


def shape_nodes(finger0_xyz, finger1_xyz):
    """floor 9 + 11 nodes per finger along y (spacing 0.018), same arithmetic as sample_data.shape_aug -> [31, 3]"""
    fingers = [np.stack([np.array([p[0], p[1] + 0.018 * (j - 5), p[2]]) for j in range(N_FINGER_NODES)]) for p in (finger0_xyz, finger1_xyz)]
    return np.concatenate([FLOOR, fingers[0], fingers[1]])


class MultiMaterialDynamicsDataset(Dataset):

    def __init__(self, args, split, root, augment=True, episode_ids=None, contact_only=False, sample_keys=None):
        assert split in ('train', 'val', 'test')
        self.args, self.split, self.root, self.augment = args, split, root, augment
        with open(os.path.join(root, 'split_manifest.json')) as f:
            ids = set(json.load(f)['splits'][split])
        if episode_ids is not None:                        # subset of this split only (e.g. tiny overfit)
            assert set(episode_ids) <= ids, (split, sorted(set(episode_ids) - ids))
            ids = set(episode_ids)
        with open(os.path.join(root, 'episode_index.csv')) as f:
            rows = [r for r in csv.DictReader(f) if r['episode_id'] in ids]
        assert len(rows) == len(ids), (split, len(rows), len(ids))
        self.episodes, self.S, self.T = [], [], []
        for r in rows:
            with np.load(os.path.join(root, r['path'], 'processed.npz')) as z:
                init, roll, start, tool = z['initial_object_pos_300'], z['rollout_object_pos_300'], z['tool_start_pose'], z['tool_pose_rollout']
            boundary = np.concatenate([init[None], roll[:, -1]])
            self.S.append(np.concatenate([boundary[:-1, None], roll], 1))          # [5, 41, 300, 3]
            self.T.append(np.concatenate([start[:, None], tool], 1))               # [5, 41, 2, 7]
            self.episodes.append(dict(episode_id=r['episode_id'], material=float(r['sigma_y']), split=r['split']))
        self.windows_per_grip = N_STATES - args.sequence_length + 1
        self.index = [dict(episode=e, episode_id=ep['episode_id'], material=ep['material'], split=ep['split'], grip_id=g, start_step=u)
                      for e, ep in enumerate(self.episodes) for g in range(self.S[e].shape[0]) for u in range(self.windows_per_grip)]
        if contact_only:   # keep windows whose last input frame has a finger relation (DynamicsPredictor stationary-gate test at j=0)
            self.index = [r for r in self.index if self.in_contact(r)]
        if sample_keys:    # fixed (grip_id, start_step) windows, e.g. tiny N-sample overfit
            keep = [r for r in self.index if (r['grip_id'], r['start_step']) in sample_keys]
            assert len(keep) == len(sample_keys) * len(self.episodes), (sample_keys, len(keep))
            self.index = keep

    def in_contact(self, rec):
        Rs = prepare_input(self.node_frame(rec['episode'], rec['grip_id'], rec['start_step'] + self.args.n_his - 1),
                           N_PARTICLE, N_SHAPE, self.args, stdreg=self.args.stdreg)[3]
        return torch.count_nonzero(Rs[:, N_PARTICLE + N_FLOOR:]).item() > 0

    def __len__(self):
        return len(self.index)

    def node_frame(self, e, g, step):
        """legacy baseline node frame at physical step: object S_g[step] + fingers at T_g[step + 1] (T_g[40] placeholder past the end)"""
        tool = self.T[e][g, min(step + 1, N_STATES - 1), :, :3]
        return np.concatenate([self.S[e][g, step].astype(np.float64), shape_nodes(tool[0], tool[1])])

    def __getitem__(self, idx):
        args, rec = self.args, self.index[idx]
        e, g, u = rec['episode'], rec['grip_id'], rec['start_step']
        attrs, particles, Rrs, Rss, Rns, cluster_onehots = [], [], [], [], [], []
        max_n_rel = 0
        for t in range(u, u + args.sequence_length):
            attr, particle, Rr, Rs, Rn, cluster_onehot = prepare_input(self.node_frame(e, g, t), N_PARTICLE, N_SHAPE, args, stdreg=args.stdreg)
            max_n_rel = max(max_n_rel, Rr.size(0))
            attrs.append(attr); particles.append(particle.numpy())
            Rrs.append(Rr); Rss.append(Rs); Rns.append(Rn)
            if cluster_onehot is not None:
                cluster_onehots.append(cluster_onehot)

        # observation noise on the n_his input frames, as PhysicsFleXDataset.__getitem__
        if self.augment:
            for t in range(args.n_his):
                particles[t][:N_PARTICLE] += np.random.randn(N_PARTICLE, 3) * args.std_d * args.augment_ratio

        for i in range(len(Rrs)):
            pad = lambda R: torch.cat([R, torch.zeros(max_n_rel - R.size(0), N_PARTICLE + N_SHAPE)], 0)
            Rrs[i], Rss[i], Rns[i] = pad(Rrs[i]), pad(Rss[i]), pad(Rns[i])
        cluster_onehot = torch.FloatTensor(np.stack(cluster_onehots)) if cluster_onehots else None
        return (torch.FloatTensor(attrs[0]), torch.FloatTensor(np.stack(particles)), N_PARTICLE, N_SHAPE,
                torch.FloatTensor(SCENE_PARAMS), torch.FloatTensor(np.stack(Rrs)), torch.FloatTensor(np.stack(Rss)),
                torch.FloatTensor(np.stack(Rns)), cluster_onehot)


# ------------------------------------------------------------------ self-check
def self_check(root, out_json=None):
    import sys
    sys.argv = sys.argv[:1]
    from config import gen_args
    from utils import my_collate
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'simulator', 'plb', 'algorithms'))
    import sample_data
    args = gen_args()
    report = dict(root=root, args=dict(n_his=args.n_his, sequence_length=args.sequence_length, neighbor_radius=args.neighbor_radius,
                                       gripper_extra_neighbor_radius=args.gripper_extra_neighbor_radius, shape_aug=args.shape_aug, stdreg=args.stdreg))
    with open(os.path.join(root, 'split_manifest.json')) as f:
        manifest = json.load(f)
    rng = np.random.default_rng(0)

    # shape-node parity vs sample_data.shape_aug on representative tool poses (+ random)
    dsets = {s: MultiMaterialDynamicsDataset(args, s, root, augment=False) for s in ('train', 'val', 'test') if manifest['splits'][s]}
    poses = [d.T[e][g, k, :, :3] for d in dsets.values() for e in range(min(3, len(d.S))) for g in range(5) for k in (0, 1, 20, 29, 40)]
    poses += [rng.uniform(0, 1, (2, 3)) for _ in range(50)]
    parity = 0.0
    for p in poses:
        legacy_in = np.zeros((N_PARTICLE + 3, 3)); legacy_in[:N_PARTICLE] = rng.random((N_PARTICLE, 3)); legacy_in[N_PARTICLE] = [0.5, 0, 0.5]
        legacy_in[N_PARTICLE + 1], legacy_in[N_PARTICLE + 2] = p[0], p[1]
        legacy = sample_data.shape_aug(legacy_in, N_PARTICLE)
        parity = max(parity, float(np.abs(legacy[N_PARTICLE:] - shape_nodes(p[0], p[1])).max()))
    report['shape_parity'] = dict(n_poses=len(poses), max_error=parity)

    per_split = {}
    for s, d in dsets.items():
        crossing = sum(1 for r in d.index if r['start_step'] + args.sequence_length > N_STATES)
        grips = {}
        for r in d.index:
            grips[(r['episode_id'], r['grip_id'])] = grips.get((r['episode_id'], r['grip_id']), 0) + 1
        last = [r for r in d.index if r['start_step'] == d.windows_per_grip - 1]
        # alignment on a few samples: object == S_g[u+i], finger nodes == T_g[u+i+1] (placeholder T_g[40] only at i=5, u=35)
        align = dict(object=0.0, finger_center_next_tool=0.0, finger_center_same_step_tool_diff_min=np.inf, placeholder_used=0)
        for idx in [0, 1, d.windows_per_grip - 1, len(d) // 2, len(d) - 1] + rng.integers(0, len(d), 20).tolist():
            rec = d.index[idx]
            attr, particles, n_p, n_s, scene, Rr, Rs, Rn, cl = d[idx]
            e, g, u = rec['episode'], rec['grip_id'], rec['start_step']
            for i in range(args.sequence_length):
                step = u + i
                align['object'] = max(align['object'], float(np.abs(particles[i, :N_PARTICLE].numpy() - d.S[e][g, step]).max()))
                nxt = min(step + 1, N_STATES - 1)
                align['placeholder_used'] += int(step + 1 > N_STATES - 1)
                assert step + 1 <= N_STATES - 1 or i == args.sequence_length - 1, (u, i)
                centers = particles[i, [309 + 5, 320 + 5]].numpy()
                align['finger_center_next_tool'] = max(align['finger_center_next_tool'], float(np.abs(centers - d.T[e][g, nxt, :, :3].astype(np.float32)).max()))
                if i < args.n_his and step + 1 <= 29:   # closing phase: fingers move every step, so T[step] must differ
                    align['finger_center_same_step_tool_diff_min'] = min(align['finger_center_same_step_tool_diff_min'],
                                                                         float(np.abs(centers - d.T[e][g, step, :, :3]).max()))
            nodes_ok = bool((attr[:300] == 0).all() and (attr[300:309] == torch.tensor([0., 1., 0.])).all() and (attr[309:] == torch.tensor([0., 0., 1.])).all())
            assert nodes_ok and particles.shape == (6, 331, 3) and n_p == 300 and n_s == 31 and attr.shape == (331, 3)
            assert np.allclose(particles[0, 300:309].numpy(), FLOOR)
        sample = d[0]
        batch = my_collate([d[0], d[len(d) - 1]])
        per_split[s] = dict(
            episodes=len(d.episodes), materials=sorted({ep['material'] for ep in d.episodes}),
            episodes_per_material={f'{m:g}': sum(ep['material'] == m for ep in d.episodes) for m in sorted({ep['material'] for ep in d.episodes})},
            windows_per_grip=d.windows_per_grip, samples_per_episode=d.windows_per_grip * 5, samples=len(d),
            grip_sample_counts=sorted(set(grips.values())), cross_grip_windows=crossing, last_window_start=sorted({r['start_step'] for r in last}),
            alignment=align,
            sample={name: (list(v.shape) if torch.is_tensor(v) else v) for name, v in
                    zip(['attr', 'particles', 'n_particle', 'n_shape', 'scene_params', 'Rr', 'Rs', 'Rn', 'cluster_onehot'], sample)},
            sample_dtypes={name: str(v.dtype) for name, v in zip(['attr', 'particles', 'scene_params', 'Rr', 'Rs', 'Rn'], [sample[i] for i in (0, 1, 4, 5, 6, 7)])},
            batch={name: (list(v.shape) if torch.is_tensor(v) else v) for name, v in
                   zip(['attr', 'particles', 'n_particles', 'n_shapes', 'scene_params', 'Rrs', 'Rss', 'Rns', 'cluster_onehots'], batch)})
        assert crossing == 0 and len(d) == len(d.episodes) * 5 * d.windows_per_grip
    report['splits'] = per_split
    # augmentation path (default on) leaves shapes/tools untouched
    aug = MultiMaterialDynamicsDataset(args, next(iter(dsets)), root, augment=True)
    a, b = aug[0][1], next(iter(dsets.values()))[0][1]
    report['augment_check'] = dict(object_noise_max=float((a[:4, :300] - b[:4, :300]).abs().max()), shapes_equal=bool(torch.equal(a[:, 300:], b[:, 300:])),
                                   target_frames_equal=bool(torch.equal(a[4:], b[4:])))
    ok = (parity == 0 and all(p['cross_grip_windows'] == 0 and p['alignment']['object'] == 0 and p['alignment']['finger_center_next_tool'] == 0
                              and p['alignment']['finger_center_same_step_tool_diff_min'] > 0 for p in per_split.values())
          and report['augment_check']['shapes_equal'] and report['augment_check']['target_frames_equal'])
    report['verdict'] = 'PASS' if ok else 'FAIL'
    print(json.dumps(report, indent=1, default=str))
    if out_json:
        with open(out_json, 'w') as f:
            json.dump(report, f, indent=1, default=str)
    return ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out-json', default=None)
    a = ap.parse_args()
    raise SystemExit(0 if self_check(os.path.abspath(a.root), a.out_json) else 1)
