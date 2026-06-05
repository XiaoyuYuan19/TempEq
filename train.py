"""TempEq self-supervised pretraining on UCF101.

Supports DDP via SLURM; single-process mode also works for smoke tests.
Three SSL losses available via --loss {infonce, pf2, vicreg}.

Typical launch (LUMI 2-GPU):
    srun python train.py --video-root /tmp/UCF-101 \
        --split-root /path/to/ucfTrainTestlist --out-dir logs/clip_pf2 \
        --loss pf2 --clip-len 4 --clip-stride 8 --sampler geometric \
        --batch-size 64 --epochs 100 --optimizer lars --lr 0.2 \
        --proj-dim 8192 --proj-hidden 8192 --proj-layers 3 --proj-bn

See configs/*.env for canonical settings reproducing the paper tables.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import UCF101Pair, UCF101Clip
from dist_utils import setup_dist, is_main, cleanup_dist, all_reduce_mean
from linear_probe import run_probe
from loss import build_loss
from model import SSLModel


# ---------- optimizer ----------

class LARS(torch.optim.Optimizer):
    """LARS: per-parameter trust-ratio scaled SGD with momentum + weight decay.

    Follows the VICReg / BYOL recipe:
        trust = eta * ||w|| / (||g|| + wd * ||w||)
        v = momentum * v + lr * trust * (g + wd * w)
        w -= v
    Parameters whose group has `exclude_from_lars=True` bypass the trust
    ratio (plain SGD with momentum, no wd) -- used for bias/BN per VICReg
    convention.
    """

    def __init__(self, params, lr, momentum=0.9, weight_decay=1e-6, eta=0.001):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, exclude_from_lars=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            eta = group['eta']
            exclude = group['exclude_from_lars']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if not exclude:
                    g = g.add(p, alpha=wd)
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(g)
                    trust = torch.where(
                        (w_norm > 0) & (g_norm > 0),
                        eta * w_norm / (g_norm + 1e-8),
                        torch.ones_like(w_norm),
                    )
                    g = g.mul(trust)
                state = self.state[p]
                if 'v' not in state:
                    state['v'] = torch.zeros_like(p)
                v = state['v']
                v.mul_(mu).add_(g, alpha=lr)
                p.add_(v, alpha=-1.0)
        return loss


def _build_lars(model, lr, weight_decay, momentum=0.9, eta=0.001):
    """Split params: bias/BN -> SGD only, rest -> LARS."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    return LARS([
        {'params': decay, 'weight_decay': weight_decay, 'exclude_from_lars': False},
        {'params': no_decay, 'weight_decay': 0.0, 'exclude_from_lars': True},
    ], lr=lr, momentum=momentum, eta=eta)


# ---------- args ----------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video-root', required=True,
                   help='dir with per-class subfolders of .avi (e.g. /tmp/UCF-101)')
    p.add_argument('--split-root', required=True,
                   help='dir with classInd.txt + trainlist01.txt + testlist01.txt')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--split-file', default='trainlist01.txt')

    # data / sampler
    p.add_argument('--window', type=int, default=5,
                   help='temporal window for sampler=window')
    p.add_argument('--sampler', default='geometric',
                   choices=['window', 'geometric', 'adjacent', 'overlap'])
    p.add_argument('--geometric-p', type=float, default=0.008,
                   help='p for Geometric(p) gap sampler; mean gap = 1/p frames')
    p.add_argument('--min-k', type=int, default=1)
    p.add_argument('--max-k', type=int, default=150)
    p.add_argument('--pairs-per-video', type=int, default=1)
    p.add_argument('--aug-mode', default='ssl',
                   choices=['ssl', 'strong', 'none'])
    p.add_argument('--clip-len', type=int, default=1,
                   help='1 = frame mode; >1 = clip mode (per-frame encode + mean-pool)')
    p.add_argument('--clip-stride', type=int, default=1,
                   help='stride between frames in a clip; clip_len=4 stride=8 ~= 0.8s span')

    # loss
    p.add_argument('--loss', default='pf2',
                   choices=['infonce', 'pf2', 'vicreg'])
    p.add_argument('--pf2-dual-lr', type=float, default=0.01,
                   help='dual ascent step size for lam')
    p.add_argument('--pf2-lam-init', type=float, default=0.0)
    p.add_argument('--pf2-lam-max', type=float, default=0.0,
                   help='upper cap on lam; 0 = uncapped')
    p.add_argument('--vicreg-sim', type=float, default=25.0)
    p.add_argument('--vicreg-var', type=float, default=25.0)
    p.add_argument('--vicreg-cov', type=float, default=1.0)
    p.add_argument('--temperature', type=float, default=0.1)

    # model
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--proj-dim', type=int, default=128)
    p.add_argument('--proj-hidden', type=int, default=512)
    p.add_argument('--proj-layers', type=int, default=2)
    p.add_argument('--proj-bn', action='store_true')

    # optim / schedule
    p.add_argument('--batch-size', type=int, default=64,
                   help='per-GPU batch size (effective = bs * world_size)')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.2,
                   help='base LR (scaled linearly by world_size)')
    p.add_argument('--weight-decay', type=float, default=1e-6)
    p.add_argument('--optimizer', default='lars', choices=['adamw', 'lars'])
    p.add_argument('--grad-clip', type=float, default=0.0)
    p.add_argument('--warmup-epochs', type=int, default=5)

    # ckpt
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--resume', default=None)

    # inline probe
    p.add_argument('--probe-every', type=int, default=5,
                   help='run linear probe every N SSL epochs; 0 disables')
    p.add_argument('--probe-epochs', type=int, default=20)
    p.add_argument('--probe-lr', type=float, default=1e-3)
    p.add_argument('--probe-batch', type=int, default=32)
    p.add_argument('--probe-num-frames', type=int, default=8)
    p.add_argument('--train-split', default='trainlist01.txt')
    p.add_argument('--test-split', default='testlist01.txt')
    p.add_argument('--num-classes', type=int, default=101)
    return p.parse_args()


def cosine_lr(step, total_steps, base_lr, warmup_steps):
    """Linear warmup -> cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_dist()

    out_dir = Path(args.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'world_size={world_size}  device={device}')

    # ---- data ----
    dataset = UCF101Pair(
        video_root=args.video_root, split_root=args.split_root,
        split_file=args.split_file, window=args.window, size=args.size,
        sampler=args.sampler, geometric_p=args.geometric_p,
        min_k=args.min_k, max_k=args.max_k,
        pairs_per_video=args.pairs_per_video,
        aug_mode=args.aug_mode,
        clip_len=args.clip_len, clip_stride=args.clip_stride,
        verbose=is_main(),
    )
    if is_main():
        print(f'train videos: {len(dataset)}')

    sampler = (DistributedSampler(dataset, shuffle=True, drop_last=True)
               if world_size > 1 else None)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )

    probe_train_ds = probe_test_ds = None
    if args.probe_every > 0:
        probe_train_ds = UCF101Clip(
            video_root=args.video_root, split_root=args.split_root,
            split_file=args.train_split,
            num_frames=args.probe_num_frames, size=args.size,
        )
        probe_test_ds = UCF101Clip(
            video_root=args.video_root, split_root=args.split_root,
            split_file=args.test_split,
            num_frames=args.probe_num_frames, size=args.size,
        )
        if is_main():
            print(f'probe train clips: {len(probe_train_ds)}  '
                  f'test clips: {len(probe_test_ds)}')

    # ---- model / loss / optim ----
    model = SSLModel(
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
        proj_layers=args.proj_layers,
        proj_bn=args.proj_bn,
    ).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    loss_fn = build_loss(
        args.loss,
        temperature=args.temperature,
        pf2_dual_lr=args.pf2_dual_lr,
        pf2_lam_init=args.pf2_lam_init,
        pf2_lam_max=args.pf2_lam_max,
        vicreg_sim=args.vicreg_sim,
        vicreg_var=args.vicreg_var,
        vicreg_cov=args.vicreg_cov,
    ).to(device)
    if is_main():
        print(f'loss={args.loss}  loss_fn={loss_fn}')

    base_lr = args.lr * world_size
    if args.optimizer == 'lars':
        optim = _build_lars(model, base_lr, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        target = model.module if world_size > 1 else model
        target.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        if 'loss' in ckpt:
            loss_fn.load_state_dict(ckpt['loss'])
        if 'rng_cpu' in ckpt and len(ckpt['rng_cpu']) == world_size:
            torch.set_rng_state(ckpt['rng_cpu'][rank].cpu())
            torch.cuda.set_rng_state(ckpt['rng_cuda'][rank].cpu(), device)
            if is_main():
                print(f'restored per-rank RNG state ({world_size} ranks)')
        elif is_main():
            print('WARNING: ckpt has no RNG state -- aug trajectory will diverge')
        start_epoch = ckpt['epoch'] + 1
        if is_main():
            print(f'resumed from epoch {start_epoch}')

    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_metrics = {}
        for a, p in loader:
            a = a.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)

            lr = cosine_lr(global_step, total_steps, base_lr, warmup_steps)
            for g in optim.param_groups:
                g['lr'] = lr

            # One forward over the concatenated batch (avoids DDP
            # version-mismatch caused by two separate forward passes).
            B = a.size(0)
            x = torch.cat([a, p], dim=0)
            z = model(x)
            z_a, z_p = z[:B], z[B:]
            loss, metrics = loss_fn(z_a, z_p)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=args.grad_clip)
            optim.step()

            # PF2 dual ascent on lam using cross-rank mean H.
            if hasattr(loss_fn, 'dual_step') and 'H' in metrics:
                H_reduced = all_reduce_mean(metrics['H'].item(), device)
                loss_fn.dual_step(H_reduced)

            running_loss += loss.item()
            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0.0) + (
                    v.item() if torch.is_tensor(v) else float(v))
            global_step += 1

        avg_loss = all_reduce_mean(running_loss / max(1, steps_per_epoch), device)
        avg_metrics = {
            k: all_reduce_mean(v / max(1, steps_per_epoch), device)
            for k, v in running_metrics.items()
        }
        dt = time.time() - t0
        if is_main():
            parts = [f'loss={avg_loss:.4f}']
            for k, v in avg_metrics.items():
                if k == 'acc':
                    parts.append(f'acc={v:.1%}')
                else:
                    parts.append(f'{k}={v:.4f}')
            parts.append(f'lr={lr:.2e}')
            parts.append(f'time={dt:.1f}s')
            print(f'epoch {epoch:03d}: ' + ' '.join(parts), flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            # Gather per-rank RNG state so resume restores each rank's
            # aug/data stream and avoids trajectory divergence.
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)
            if world_size > 1:
                gathered_cpu = [None] * world_size
                gathered_cuda = [None] * world_size
                dist.all_gather_object(gathered_cpu, cpu_rng)
                dist.all_gather_object(gathered_cuda, cuda_rng)
            else:
                gathered_cpu = [cpu_rng]
                gathered_cuda = [cuda_rng]

            if is_main():
                state = (model.module if world_size > 1 else model).state_dict()
                ckpt = {
                    'model': state,
                    'optim': optim.state_dict(),
                    'loss': loss_fn.state_dict(),
                    'epoch': epoch,
                    'args': vars(args),
                    'rng_cpu': gathered_cpu,
                    'rng_cuda': gathered_cuda,
                }
                path = out_dir / f'ssl_ep{epoch:03d}.pt'
                torch.save(ckpt, path)
                torch.save(ckpt, out_dir / 'ssl_latest.pt')
                print(f'  saved {path}', flush=True)

        # ---- inline linear probe ----
        if args.probe_every > 0 and ((epoch + 1) % args.probe_every == 0
                                     or epoch == args.epochs - 1):
            target = model.module if world_size > 1 else model
            encoder = target.encoder
            probe_metrics = run_probe(
                encoder, probe_train_ds, probe_test_ds,
                device, world_size, local_rank,
                extract_batch=args.probe_batch, workers=args.workers,
                num_classes=args.num_classes,
                probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
            )
            model.train()
            if is_main():
                print(f'[probe @ epoch {epoch:03d}] '
                      f'train={probe_metrics["train_acc"]:.1%} '
                      f'test={probe_metrics["test_acc"]:.1%}  '
                      f'(n_train={probe_metrics["n_train"]} '
                      f'n_test={probe_metrics["n_test"]}  '
                      f'extract={probe_metrics["extract_sec"]:.1f}s '
                      f'linear={probe_metrics["linear_sec"]:.1f}s)',
                      flush=True)

    cleanup_dist()


if __name__ == '__main__':
    main()
