"""Evaluate a multimaterial run checkpoint on one split with the train.py 2-step rollout (no grad, fixed order).
Per-sample step RMSE (same particle indices), zero-motion RMSE, 0.9 EMD + 0.1 Chamfer loss; grouped by all / material / grip / gate contact.
RMSE of a group = sqrt(mean over its samples of per-sample mean squared point error).

usage (from robocraft/): python eval_multimaterial.py <run_dir> --split test --ckpt net_best.pth
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
ap.add_argument('--ckpt', default='net_best.pth')
ap.add_argument('--out', default=None)
cli = ap.parse_args()
hp = json.load(open(os.path.join(cli.run_dir, 'run_config.json')))['hyperparameters']
sys.argv = [sys.argv[0], '--stage', 'dy', '--data_type', hp['data_type'], '--dataset_type', 'multimaterial', '--processed_root', hp['processed_root'],
            '--normalization_source', hp['normalization_source'], '--random_seed', str(hp['random_seed']), '--stationary_gate', str(hp['stationary_gate'])]

import numpy as np
import torch
from config import gen_args
from model import Model, ChamferLoss, EarthMoverLoss
from multimaterial_dataset import MultiMaterialDynamicsDataset
from utils import prepare_input, get_env_group, set_seed, my_collate

args = gen_args()
set_seed(args.random_seed)
ds = MultiMaterialDynamicsDataset(args, cli.split, args.processed_root, augment=False)
model = Model(args, True).cuda()
model.load_state_dict(torch.load(os.path.join(cli.run_dir, cli.ckpt)))
model.eval()
emd, chamfer = EarthMoverLoss(), ChamferLoss()
n_p, n_s, B, T = 300, 31, args.batch_size, args.sequence_length - args.n_his


def relations(pred_pos):
    R = [prepare_input(p.cpu().numpy(), n_p, n_s, args, stdreg=args.stdreg)[2:5] for p in pred_pos]
    m = max(r[0].size(0) for r in R)
    pad = lambda x: torch.cat([x, torch.zeros(m - x.size(0), n_p + n_s)], 0)
    return [torch.stack([pad(r[k]) for r in R]).cuda() for k in range(3)]


recs = []
with torch.no_grad():
    for s in range(0, len(ds), B):
        ids = list(range(s, min(s + B, len(ds))))
        attrs, particles, _, _, scene, Rrs, Rss, Rns, _ = my_collate([ds[i] for i in ids])
        attrs, particles = attrs.cuda(), particles.cuda()
        groups = get_env_group(args, n_p, scene, use_gpu=True)
        mem = model.init_memory(len(ids), n_p + n_s)
        per = [dict(split=cli.split, **{k: ds.index[i][k] for k in ('episode_id', 'material', 'grip_id', 'start_step')}) for i in ids]
        for j in range(T):
            if j == 0:
                state = particles[:, :args.n_his]
                Rr, Rs, Rn = Rrs[:, args.n_his - 1].cuda(), Rss[:, args.n_his - 1].cuda(), Rns[:, args.n_his - 1].cuda()
            else:
                Rr, Rs, Rn = relations(pred_pos)
                state = torch.cat([state[:, -3:], pred_pos.unsqueeze(1)], 1)
            pred_p, _, _ = model.predict_dynamics([attrs, state, Rr, Rs, Rn, mem, groups, None], j)
            gt = particles[:, args.n_his + j]
            pred_pos = torch.cat([pred_p, gt[:, n_p:]], 1)
            contact = torch.count_nonzero(Rs[:, :, 309:331], dim=(1, 2)) > 0
            for k in range(len(ids)):
                p, g, cur = pred_p[k:k + 1], gt[k:k + 1, :n_p], state[k:k + 1, -1, :n_p]
                per[k][f'se{j}'] = ((p - g) ** 2).sum(-1).mean().item()
                per[k][f'zse{j}'] = ((cur - g) ** 2).sum(-1).mean().item()
                per[k][f'loss{j}'] = (args.emd_weight * emd(p, g) + args.chamfer_weight * chamfer(p, g)).item()
                per[k][f'contact{j}'] = bool(contact[k])
                per[k][f'finite{j}'] = bool(torch.isfinite(p).all())
        recs += per
        if s % (100 * B) == 0:
            print(f'{s + len(ids)}/{len(ds)}', flush=True)


def summary(rs):
    out = dict(n=len(rs), loss=float(np.mean([r['loss0'] + r['loss1'] for r in rs])),
               loss_train_meter=float(np.mean([(2 * r['loss0'] + r['loss1']) / 2 for r in rs])),
               nonfinite=sum(not (r['finite0'] and r['finite1']) for r in rs))
    for j in range(T):
        se, zse = np.mean([r[f'se{j}'] for r in rs]), np.mean([r[f'zse{j}'] for r in rs])
        out[f'step{j + 1}_rmse'], out[f'step{j + 1}_zero_rmse'], out[f'step{j + 1}_ratio'] = float(np.sqrt(se)), float(np.sqrt(zse)), float(np.sqrt(se / zse))
    return out


by = lambda key: {f'{key}={v:g}' if isinstance(v, float) else f'{key}={v}': summary([r for r in recs if r[key] == v]) for v in sorted({r[key] for r in recs})}
report = dict(run_dir=os.path.abspath(cli.run_dir), split=cli.split, ckpt=cli.ckpt, samples=len(recs),
              episodes=len({r['episode_id'] for r in recs}), loss_note='loss = sum over 2 steps of 0.9 EMD + 0.1 Chamfer (the optimized loss); loss_train_meter = (L1 + (L1+L2))/2, what train.py logs as epoch loss (meter updated inside the step loop)',
              all=summary(recs), by_material=by('material'), by_grip=by('grip_id'),
              by_step1_contact={'contact': summary([r for r in recs if r['contact0']]), 'stationary': summary([r for r in recs if not r['contact0']])})
out = cli.out or os.path.join(cli.run_dir, f'eval_{cli.split}_{os.path.splitext(cli.ckpt)[0]}.json')
json.dump(dict(report, per_sample=recs), open(out, 'w'), indent=1)
print(json.dumps(report, indent=1))
print('saved', out)
