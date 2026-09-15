import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import gen_args
from tqdm import tqdm
from utils import PhysicsFleXDataset
from utils import prepare_input, get_scene_info, get_env_group
from model import Model, ChamferLoss, EarthMoverLoss, HausdorffLoss
from utils import set_seed, AverageMeter, get_lr, Tee, count_parameters, my_collate, matched_motion

from eval import evaluate

args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)

tee = Tee(os.path.join(args.outf, 'train.log'), 'w')


def write_run_config(datasets):
    import json
    import subprocess
    with open(os.path.join(args.processed_root, 'processed_config.json')) as f:
        processed = json.load(f)
    with open(os.path.join(args.processed_root, 'split_manifest.json')) as f:
        manifest = json.load(f)
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
    cfg = dict(git_commit=git('rev-parse', 'HEAD'), git_status=git('status', '--short'),
               processed_root=os.path.abspath(args.processed_root),
               processed_preprocessing_git_hash=processed['preprocessing_code_git_hash'],
               raw_dataset_provenance_hash=processed['raw_dataset_provenance_hash'],
               phases={phase: dict(split=d.split, episodes=[ep['episode_id'] for ep in d.episodes], samples=len(d), augment=d.augment)
                       for phase, d in datasets.items()},
               seed=args.random_seed, normalization_source=args.normalization_source, stationary_gate=bool(args.stationary_gate),
               contact_only=bool(args.contact_only), selected_samples={phase: len(d) for phase, d in datasets.items()},
               tiny_four_samples=bool(args.sample_keys) and len(datasets['train']) == 4,
               selected_sample_ids=[dict(episode_id=r['episode_id'], grip_id=r['grip_id'], start_step=r['start_step'],
                                         episode_sample_index=r['grip_id'] * datasets['train'].windows_per_grip + r['start_step'])
                                    for r in datasets['train'].index] if args.sample_keys else None,
               resume_from=os.path.abspath(args.resume_from) if args.resume_from else None,
               lr=args.lr, batch_size=args.batch_size, epochs=args.n_epoch, num_workers=args.num_workers,
               num_updates=args.n_epoch * -(-len(datasets['train']) // args.batch_size),
               split_samples={s: len(ids) * 5 * datasets['train'].windows_per_grip for s, ids in manifest['splits'].items()},
               split_manifest_sha256=processed['source_split_manifest_sha256'],
               normalization={k: np.asarray(getattr(args, k)).tolist() for k in ('mean_p', 'std_p', 'mean_d', 'std_d')},
               augment_ratio=args.augment_ratio, hyperparameters={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(args).items()})
    with open(os.path.join(args.outf, 'run_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def main():
    ### training

    # load training data

    phases = {0: ['train'], 1: ['valid'], 2: ['train', 'valid']}[args.valid]
    if args.dataset_type == 'multimaterial':
        from multimaterial_dataset import MultiMaterialDynamicsDataset
        split_of = {'train': 'train', 'valid': 'val'}
        sample_keys = [tuple(map(int, k.split(':'))) for k in args.sample_keys.split(',')] if args.sample_keys else None
        datasets = {phase: MultiMaterialDynamicsDataset(
            args, split_of[phase], args.processed_root, augment=args.augment_ratio > 0,
            episode_ids=[args.tiny_episode] if args.tiny_episode and phase == 'train' else None,
            contact_only=bool(args.contact_only) and phase == 'train',
            sample_keys=sample_keys if phase == 'train' else None) for phase in phases}
        write_run_config(datasets)
    else:
        datasets = {phase: PhysicsFleXDataset(args, phase) for phase in phases}

        for phase in phases:
            if args.gen_data:
                datasets[phase].gen_data(args.env)
            else:
                datasets[phase].load_data(args.env)

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=my_collate) for phase in phases}

    # create model and train
    use_gpu = torch.cuda.is_available()
    model = Model(args, use_gpu)

    print("model #params: %d" % count_parameters(model))


    # checkpoint to reload model from
    model_path = None

    # resume training of a saved model (if given)
    if args.resume == 0:
        print("Randomly initialize the model's parameters")

    elif args.resume == 1:
        model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (
            args.resume_epoch, args.resume_iter))
        print("Loading saved ckp from %s" % model_path)

        if args.stage == 'dy':
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()

            # only load parameters in dynamics_predictor
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() \
                if 'dynamics_predictor' in k and k in model_dict}
            model.load_state_dict(pretrained_dict, strict=False)


    # optimizer
    if args.stage == 'dy':
        params = model.dynamics_predictor.parameters()
    else:
        raise AssertionError("unknown stage: %s" % args.stage)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=args.lr, betas=(args.beta1, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=0.9)
    else:
        raise AssertionError("unknown optimizer: %s" % args.optimizer)

    # reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    # define loss
    chamfer_loss = ChamferLoss()
    emd_loss = EarthMoverLoss()
    h_loss = HausdorffLoss()

    if use_gpu:
        model = model.cuda()

    # log args
    print(vars(args))

    # start training
    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = np.inf
    best_epoch = None

    training_stats = {'args':vars(args), 'loss':[], 'loss_raw':[], 'iters': [], 'loss_emd': [], 'loss_motion': []}

    if args.resume_from:   # continue a finished multimaterial run in this new outf: weights, epoch log, best, scheduler state
        import json
        import shutil
        with open(os.path.join(args.resume_from, 'epoch_log.json')) as f:
            training_stats['epochs'] = json.load(f)
        model.load_state_dict(torch.load(os.path.join(args.resume_from, 'net_final.pth'), map_location=None if use_gpu else 'cpu'))
        valid = [e for e in training_stats['epochs'] if e['phase'] == 'valid']
        for e in valid:   # replay val losses: same ReduceLROnPlateau best / num_bad_epochs / lr as the parent
            scheduler.step(e['loss'])
            best_valid_loss, best_epoch = (e['loss'], e['epoch']) if e['loss'] < best_valid_loss else (best_valid_loss, best_epoch)
        optim_path = os.path.join(args.resume_from, 'optim_final.pth')
        if os.path.exists(optim_path):
            optimizer.load_state_dict(torch.load(optim_path, map_location=None if use_gpu else 'cpu'))
        else:   # ponytail: parent run saved no optimizer state, Adam moments restart (scheduler state is replayed exactly above)
            print('resume_from: no optimizer state in parent, Adam moments reinitialized')
        assert abs(get_lr(optimizer) - valid[-1]['lr_after']) < 1e-12, (get_lr(optimizer), valid[-1]['lr_after'])
        st_epoch = valid[-1]['epoch'] + 1
        shutil.copy(os.path.join(args.resume_from, 'net_best.pth'), os.path.join(args.outf, 'net_best.pth'))
        print('resumed from %s at epoch %d, best epoch %d (%.6f), lr %g' % (args.resume_from, st_epoch, best_epoch, best_valid_loss, get_lr(optimizer)))

    rollout_epoch = -1
    rollout_iter = -1
    for epoch in range(st_epoch, args.n_epoch):

        for phase in phases:

            model.train(phase == 'train')

            meter_loss = AverageMeter()
            meter_loss_raw = AverageMeter()

            meter_loss_ref = AverageMeter()
            meter_loss_nxt = AverageMeter()

            meter_loss_param = AverageMeter()

            # object next-state error with the same particle indices (= per-point motion error), per predicted step
            meter_motion = [AverageMeter() for _ in range(args.sequence_length - args.n_his)]
            meter_zero_motion = AverageMeter()
            n_nonfinite = 0

            for i, data in enumerate(tqdm(dataloaders[phase], desc=f'Epoch {epoch}/{args.n_epoch}')):
                # each "data" is a trajectory of sequence_length time steps

                if args.stage == 'dy':
                    # attrs: B x (n_p + n_s) x attr_dim
                    # particles: B x seq_length x (n_p + n_s) x state_dim
                    # n_particles: B
                    # n_shapes: B
                    # scene_params: B x param_dim
                    # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                    attrs, particles, n_particles, n_shapes, scene_params, Rrs, Rss, Rns, cluster_onehots = data

                    if use_gpu:
                        attrs = attrs.cuda()
                        particles = particles.cuda()
                        # sdf_list = sdf_list.cuda()
                        Rrs, Rss, Rns = Rrs.cuda(), Rss.cuda(), Rns.cuda()
                        if cluster_onehots is not None:
                            cluster_onehots = cluster_onehots.cuda()

                    # statistics
                    B = attrs.size(0)
                    n_particle = n_particles[0].item()
                    n_shape = n_shapes[0].item()

                    # p_rigid: B x n_instance
                    # p_instance: B x n_particle x n_instance
                    # physics_param: B x n_particle
                    groups_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

                    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                    # for now, only used as a placeholder
                    memory_init = model.init_memory(B, n_particle + n_shape)
                    loss = 0
                    for j in range(args.sequence_length - args.n_his):
                        with torch.set_grad_enabled(phase == 'train'):
                            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                            if j == 0:
                                state_cur = particles[:, :args.n_his]
                                # Rrs_cur, Rss_cur: B x n_rel x (n_p + n_s)
                                Rr_cur = Rrs[:, args.n_his - 1]
                                Rs_cur = Rss[:, args.n_his - 1]
                                Rn_cur = Rns[:, args.n_his - 1]
                            else: # elif pred_pos.size(0) >= args.batch_size:
                                Rr_cur = []
                                Rs_cur = []
                                Rn_cur = []
                                max_n_rel = 0
                                for k in range(pred_pos.size(0)):
                                    _, _, Rr_cur_k, Rs_cur_k, Rn_cur_k, _ = prepare_input(pred_pos[k].detach().cpu().numpy(), n_particle, n_shape, args, stdreg=args.stdreg)
                                    Rr_cur.append(Rr_cur_k)
                                    Rs_cur.append(Rs_cur_k)
                                    Rn_cur.append(Rn_cur_k)
                                    max_n_rel = max(max_n_rel, Rr_cur_k.size(0))
                                for w in range(pred_pos.size(0)):
                                    Rr_cur_k, Rs_cur_k, Rn_cur_k = Rr_cur[w], Rs_cur[w], Rn_cur[w]
                                    Rr_cur_k = torch.cat([Rr_cur_k, torch.zeros(max_n_rel - Rr_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rs_cur_k = torch.cat([Rs_cur_k, torch.zeros(max_n_rel - Rs_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rn_cur_k = torch.cat([Rn_cur_k, torch.zeros(max_n_rel - Rn_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rr_cur[w], Rs_cur[w], Rn_cur[w] = Rr_cur_k, Rs_cur_k, Rn_cur_k
                                Rr_cur = torch.FloatTensor(np.stack(Rr_cur))
                                Rs_cur = torch.FloatTensor(np.stack(Rs_cur))
                                Rn_cur = torch.FloatTensor(np.stack(Rn_cur))
                                if use_gpu:
                                    Rr_cur = Rr_cur.cuda()
                                    Rs_cur = Rs_cur.cuda()
                                    Rn_cur = Rn_cur.cuda()
                                state_cur = torch.cat([state_cur[:,-3:], pred_pos.detach().unsqueeze(1)], dim=1)


                            if cluster_onehots is not None:
                                cluster_onehot = cluster_onehots[:, args.n_his - 1]
                            else:
                                cluster_onehot = None
                            # predict the velocity at the next time step
                            inputs = [attrs, state_cur, Rr_cur, Rs_cur, Rn_cur, memory_init, groups_gt, cluster_onehot]

                            # pred_pos (unnormalized): B x n_p x state_dim
                            # pred_motion_norm (normalized): B x n_p x state_dim
                            pred_pos_p, pred_motion_norm, std_cluster = model.predict_dynamics(inputs, j)

                            # concatenate the state of the shapes
                            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                            gt_pos = particles[:, args.n_his + j]
                            gt_pos_p = gt_pos[:, :n_particle]
                            # gt_sdf = sdf_list[:, args.n_his]
                            pred_pos = torch.cat([pred_pos_p, gt_pos[:, n_particle:]], 1)

                            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            # gt_motion_norm should match then calculate if matched_motion enabled
                            if args.matched_motion:
                                gt_motion = matched_motion(particles[:, args.n_his], particles[:, args.n_his - 1], n_particles=n_particle)
                            else:
                                gt_motion = particles[:, args.n_his] - particles[:, args.n_his - 1]

                            mean_d, std_d = model.stat[2:]
                            gt_motion_norm = (gt_motion - mean_d) / std_d
                            pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)
                            if args.loss_type == 'emd_chamfer_h':
                                if args.emd_weight > 0:
                                    emd_l = args.emd_weight * emd_loss(pred_pos_p, gt_pos_p)
                                    loss += emd_l
                                if args.chamfer_weight > 0:
                                    chamfer_l = args.chamfer_weight * chamfer_loss(pred_pos_p, gt_pos_p)
                                    loss += chamfer_l
                                if args.h_weight > 0:
                                    h_l = args.h_weight * h_loss(pred_pos_p, gt_pos_p)
                                    loss += h_l
                                # print(f"EMD: {emd_l.item()}; Chamfer: {chamfer_l.item()}; H: {h_l.item()}")
                            else:
                                raise NotImplementedError

                            if args.stdreg:
                                loss += args.stdreg_weight * std_cluster
                            loss_raw = F.l1_loss(pred_pos_p, gt_pos_p)

                            meter_loss.update(loss.item(), B)
                            meter_loss_raw.update(loss_raw.item(), B)

                            with torch.no_grad():
                                meter_motion[j].update(torch.sqrt(((pred_pos_p - gt_pos_p) ** 2).sum(-1).mean()).item(), B)
                                if j == 0:
                                    meter_zero_motion.update(torch.sqrt(((state_cur[:, -1, :n_particle] - gt_pos_p) ** 2).sum(-1).mean()).item(), B)
                                n_nonfinite += int(not torch.isfinite(loss).item()) + int(not torch.isfinite(pred_pos_p).all().item())

                if i % args.log_per_iter == 0:
                    print()
                    print('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                        phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                        loss.item(), meter_loss.avg, loss_raw.item(), meter_loss_raw.avg))
                    print('std_cluster', std_cluster)
                    if phase == 'train':
                        training_stats['loss'].append(loss.item())
                        training_stats['loss_raw'].append(loss_raw.item())
                        training_stats['iters'].append(epoch * len(dataloaders[phase]) + i)
                    # with open(args.outf + '/train.npy', 'wb') as f:
                    #     np.save(f, training_stats)

                # update model parameters
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if phase == 'train' and i > 0 and ((epoch * len(dataloaders[phase])) + i) % args.ckp_per_iter == 0:
                    model_path = '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                    torch.save(model.state_dict(), model_path)
                    rollout_epoch = epoch
                    rollout_iter = i

            print('%s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss))

            with open(args.outf + '/train.npy','wb') as f:
                np.save(f, training_stats)

            lr_epoch = get_lr(optimizer)   # lr used during this phase, before any scheduler step
            if phase == 'valid' and not args.eval:
                scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg
                    best_epoch = epoch
                    torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))

            if args.dataset_type == 'multimaterial':
                import json
                if phase == phases[-1] and 'train' in phases:   # last completed epoch, overwritten every epoch (saved before the log line)
                    torch.save(model.state_dict(), '%s/net_final.pth' % args.outf)
                    torch.save(optimizer.state_dict(), '%s/optim_final.pth' % args.outf)
                training_stats.setdefault('epochs', []).append(dict(
                    epoch=epoch, phase=phase, loss=meter_loss.avg, loss_raw_l1=meter_loss_raw.avg,
                    motion_rmse=[m.avg for m in meter_motion], zero_motion_rmse=meter_zero_motion.avg,
                    nonfinite=n_nonfinite, lr=lr_epoch, lr_after=get_lr(optimizer),
                    best_valid_loss=best_valid_loss, best_epoch=best_epoch))
                print('epoch summary', training_stats['epochs'][-1])
                with open(os.path.join(args.outf, 'epoch_log.json'), 'w') as f:
                    json.dump(training_stats['epochs'], f, indent=1)
    
    if args.dataset_type == 'multimaterial':
        torch.save(model.state_dict(), '%s/net_final.pth' % args.outf)

    if args.eval and model_path is not None:
        args.model_path = model_path
        evaluate(args)

if __name__ == '__main__':
    main()