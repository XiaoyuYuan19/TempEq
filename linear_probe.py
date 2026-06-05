"""Linear probe with feature caching.

Two phases:
    1. Extract pooled features (one forward per video) using the frozen
       SSL encoder. Features are gathered across DDP ranks.
    2. Train a linear classifier on the cached features. Fast (~seconds
       per epoch) since it operates on D-dim feature vectors only.

Can be used standalone (`python linear_probe.py ...`) or inline from
train.py via `run_probe(encoder, ...)`.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from data import UCF101Clip
from dist_utils import setup_dist, is_main, cleanup_dist
from model import SSLModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video-root', required=True)
    p.add_argument('--split-root', required=True)
    p.add_argument('--ssl-ckpt', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--train-split', default='trainlist01.txt')
    p.add_argument('--test-split', default='testlist01.txt')
    p.add_argument('--num-frames', type=int, default=8)
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--extract-batch', type=int, default=32)
    p.add_argument('--linear-batch', type=int, default=512)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--num-classes', type=int, default=101)
    return p.parse_args()


@torch.no_grad()
def extract_features(encoder, dataset, device, world_size, batch_size, workers):
    """Run encoder on every video, mean-pool frame features.

    Returns (features (N, D), labels (N,)) -- same on all ranks after gather.
    """
    sampler = (DistributedSampler(dataset, shuffle=False, drop_last=False)
               if world_size > 1 else None)
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=workers, pin_memory=True,
    )
    encoder.eval()

    feats_local = []
    labels_local = []
    for clip, label in loader:
        clip = clip.to(device, non_blocking=True)
        B, N = clip.shape[:2]
        h = encoder(clip.flatten(0, 1))
        h = h.view(B, N, -1).mean(dim=1)
        feats_local.append(h.cpu())
        labels_local.append(label)
    feats_local = torch.cat(feats_local, dim=0)
    labels_local = torch.cat(labels_local, dim=0)

    if world_size <= 1:
        return feats_local, labels_local

    gathered = [None] * world_size
    dist.all_gather_object(gathered, (feats_local, labels_local))
    feats = torch.cat([g[0] for g in gathered], dim=0)
    labels = torch.cat([g[1] for g in gathered], dim=0)
    return feats[: len(dataset)], labels[: len(dataset)]


def train_classifier(train_feats, train_labels, test_feats, test_labels,
                     device, world_size, local_rank, num_classes,
                     epochs=30, lr=1e-3, weight_decay=0.0,
                     batch=512, verbose=False):
    """Train linear classifier on cached features. Returns (best_train, best_test)."""
    train_tds = TensorDataset(train_feats, train_labels)
    test_tds = TensorDataset(test_feats, test_labels)

    train_sampler = (DistributedSampler(train_tds, shuffle=True, drop_last=True)
                     if world_size > 1 else None)
    train_loader = DataLoader(
        train_tds, batch_size=batch,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_tds, batch_size=batch, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    feature_dim = train_feats.shape[1]
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    if world_size > 1:
        classifier = DDP(classifier, device_ids=[local_rank],
                         output_device=local_rank)

    base_lr = lr * world_size
    optim = torch.optim.AdamW(classifier.parameters(), lr=base_lr,
                              weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_test = 0.0
    best_train = 0.0
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        classifier.train()
        correct = total = 0
        for feats, label in train_loader:
            feats = feats.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = classifier(feats)
            loss = loss_fn(logits, label)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.numel()
        sched.step()
        train_acc = correct / max(1, total)

        classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, label in test_loader:
                feats = feats.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                pred = classifier(feats).argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.numel()
        test_acc = correct / max(1, total)

        if test_acc > best_test:
            best_test = test_acc
            best_train = train_acc

        if verbose and is_main():
            print(f'  probe epoch {epoch:02d}: '
                  f'train={train_acc:.4f} test={test_acc:.4f}', flush=True)

    return best_train, best_test


def run_probe(encoder, train_ds, test_ds, device, world_size, local_rank,
              extract_batch=32, workers=4, num_classes=101,
              probe_epochs=20, probe_lr=1e-3, probe_weight_decay=0.0):
    """Full probe: extract features + train linear classifier."""
    t_extract = time.time()
    train_f, train_l = extract_features(encoder, train_ds, device, world_size,
                                        extract_batch, workers)
    test_f, test_l = extract_features(encoder, test_ds, device, world_size,
                                      extract_batch, workers)
    t_extract = time.time() - t_extract

    t_lin = time.time()
    train_acc, test_acc = train_classifier(
        train_f, train_l, test_f, test_l,
        device, world_size, local_rank, num_classes,
        epochs=probe_epochs, lr=probe_lr, weight_decay=probe_weight_decay,
    )
    t_lin = time.time() - t_lin
    return {
        'n_train': int(train_f.shape[0]),
        'n_test': int(test_f.shape[0]),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'extract_sec': t_extract,
        'linear_sec': t_lin,
    }


def main():
    args = parse_args()
    rank, local_rank, world_size, device = setup_dist()

    out_dir = Path(args.out_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'world_size={world_size}  device={device}')

    ssl_model = SSLModel()
    ckpt = torch.load(args.ssl_ckpt, map_location='cpu')
    ssl_model.load_state_dict(ckpt['model'])
    encoder = ssl_model.encoder.to(device)
    if is_main():
        print(f'loaded encoder from {args.ssl_ckpt} '
              f'(epoch {ckpt.get("epoch", "?")})')

    train_ds = UCF101Clip(
        video_root=args.video_root, split_root=args.split_root,
        split_file=args.train_split,
        num_frames=args.num_frames, size=args.size,
    )
    test_ds = UCF101Clip(
        video_root=args.video_root, split_root=args.split_root,
        split_file=args.test_split,
        num_frames=args.num_frames, size=args.size,
    )
    if is_main():
        print(f'train clips: {len(train_ds)}  test clips: {len(test_ds)}')

    t0 = time.time()
    train_f, train_l = extract_features(encoder, train_ds, device, world_size,
                                        args.extract_batch, args.workers)
    test_f, test_l = extract_features(encoder, test_ds, device, world_size,
                                      args.extract_batch, args.workers)
    if is_main():
        print(f'[phase 1] done in {time.time() - t0:.1f}s. '
              f'train: {tuple(train_f.shape)}, test: {tuple(test_f.shape)}')
        torch.save({
            'train_feats': train_f, 'train_labels': train_l,
            'test_feats': test_f, 'test_labels': test_l,
            'ssl_ckpt': args.ssl_ckpt,
        }, out_dir / 'features.pt')

    del encoder, ssl_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_main():
        print('[phase 2] training linear classifier')
    train_acc, test_acc = train_classifier(
        train_f, train_l, test_f, test_l,
        device, world_size, local_rank, args.num_classes,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        batch=args.linear_batch, verbose=True,
    )
    if is_main():
        print(f'best: train={train_acc:.4f}  test={test_acc:.4f}')
        torch.save({'train_acc': train_acc, 'test_acc': test_acc},
                   out_dir / 'probe_summary.pt')

    cleanup_dist()


if __name__ == '__main__':
    main()
