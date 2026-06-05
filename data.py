"""UCF101 datasets for TempEq self-supervised learning and linear probing.

Two datasets:
    UCF101Pair      - SSL pretraining. Returns a pair of frames or clips
                      from the same video, separated by a sampled temporal
                      offset. Each output gets independent spatial augmentation.

    UCF101Clip      - Linear probe / supervised eval. Returns N frames sampled
                      uniformly across a video, plus the class label.

Both datasets take two roots: one for the videos, one for the standard
UCF101 split files (classInd.txt, trainlist01.txt, testlist01.txt).
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms


# ---------- split file helpers ----------

def _read_classind(path):
    """Return dict {ClassName: label_index_0based}."""
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split()
            mapping[name] = int(idx_str) - 1
    return mapping


def _read_split(split_file, classind):
    """Parse trainlist01.txt / testlist01.txt -> list of (relpath, label)."""
    items = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            relpath = line.split()[0]
            class_name = relpath.split('/')[0]
            items.append((relpath, classind[class_name]))
    return items


# ---------- augmentations ----------

_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])


def _ssl_transform(size=224, mode='ssl'):
    """SSL frame transform.

    mode='ssl'    -> crop+flip+colorjitter (SimCLR/PF2 recipe)
    mode='strong' -> ssl + grayscale + GaussianBlur + RandomSolarize
                     (BYOL/VICReg recipe)
    mode='none'   -> deterministic Resize+CenterCrop
    """
    if mode == 'none':
        return transforms.Compose([
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            _NORM,
        ])
    if mode not in ('ssl', 'strong'):
        raise ValueError(f'unknown aug mode: {mode}')
    base = [
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    ]
    if mode == 'strong':
        base += [
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            transforms.RandomSolarize(threshold=128, p=0.2),
        ]
    base += [transforms.ToTensor(), _NORM]
    return transforms.Compose(base)


def _eval_transform(size=224):
    """Deterministic transform for linear probe."""
    return transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        _NORM,
    ])


# ---------- SSL pair dataset ----------

class UCF101Pair(Dataset):
    """Returns (anchor, positive) frame or clip pairs from the same video.

    Samplers (set via `sampler=`):
      - 'window'    : positive in [t-w, t+w] (small fixed window)
      - 'geometric' : k ~ Geometric(p), clipped to [min_k, max_k]; anchor t
                      ~ U[0, T-1-k], positive at t+k. Default for PF2/SimCLR.
      - 'adjacent'  : two clips back-to-back (gap = clip_len * stride frames)
      - 'overlap'   : two clips with 50% overlap (gap = clip_len*stride/2)

    Clip mode (clip_len > 1): each side returns L frames at stride S as
    (L, C, H, W). The training script encodes per-frame and mean-pools.

    Frame mode (clip_len == 1, default): each side returns a single
    augmented frame as (C, H, W).
    """

    def __init__(self, video_root, split_root, split_file='trainlist01.txt',
                 window=5, size=224,
                 sampler='geometric', geometric_p=0.008,
                 min_k=1, max_k=150, pairs_per_video=1,
                 aug_mode='ssl',
                 clip_len=1, clip_stride=1,
                 verbose=True):
        self.video_root = Path(video_root)
        classind = _read_classind(Path(split_root) / 'classInd.txt')
        items = _read_split(Path(split_root) / split_file, classind)
        candidate_paths = [self.video_root / relpath for relpath, _ in items]

        # Pre-scan frame counts; drop unopenable / too-short videos.
        t0 = time.time()
        self.video_info = []
        skipped = 0
        for path in candidate_paths:
            try:
                vr = VideoReader(str(path), ctx=cpu(0))
                n = len(vr)
                if n >= 2:
                    self.video_info.append((path, n))
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if verbose:
            print(f'[UCF101Pair] scanned {len(candidate_paths)} videos in '
                  f'{time.time()-t0:.1f}s, kept {len(self.video_info)}, '
                  f'skipped {skipped}', flush=True)

        if sampler not in ('window', 'geometric', 'adjacent', 'overlap'):
            raise ValueError(f'unknown sampler: {sampler}')
        self.sampler = sampler
        self.window = window
        self.geometric_p = geometric_p
        self.min_k = min_k
        self.max_k = max_k
        self.pairs_per_video = pairs_per_video
        self.clip_len = max(1, int(clip_len))
        self.clip_stride = max(1, int(clip_stride))
        self.transform = _ssl_transform(size, mode=aug_mode)

    def __len__(self):
        return len(self.video_info) * self.pairs_per_video

    def _sample_pair(self, T):
        if self.sampler == 'geometric':
            k = int(np.random.geometric(self.geometric_p))
            k = max(self.min_k, min(self.max_k, k, T - 1))
            t = random.randint(0, T - 1 - k)
            return t, t + k
        if self.sampler in ('adjacent', 'overlap'):
            L, S = self.clip_len, self.clip_stride
            span = (L - 1) * S
            if self.sampler == 'adjacent':
                gap = L * S
            else:
                gap = max(1, (L * S) // 2)
            max_t = T - 1 - gap - span
            if max_t <= 0:
                return 0, gap
            t = random.randint(0, max_t)
            return t, t + gap
        # window
        w = min(self.window, T - 1)
        t = random.randint(0, T - 1)
        lo = max(0, t - w)
        hi = min(T - 1, t + w)
        choices = [i for i in range(lo, hi + 1) if i != t]
        return t, random.choice(choices)

    def __getitem__(self, idx):
        vid_idx = idx // self.pairs_per_video
        path, T = self.video_info[vid_idx]
        try:
            t, t_pos = self._sample_pair(T)
            vr = VideoReader(str(path), ctx=cpu(0))
            if self.clip_len == 1:
                frames = vr.get_batch([t, t_pos]).asnumpy()
                img_a = Image.fromarray(frames[0])
                img_p = Image.fromarray(frames[1])
                return self.transform(img_a), self.transform(img_p)
            L, S = self.clip_len, self.clip_stride
            span = (L - 1) * S
            t_a = max(0, min(t, T - 1 - span))
            t_p = max(0, min(t_pos, T - 1 - span))
            a_idxs = [t_a + i * S for i in range(L)]
            p_idxs = [t_p + i * S for i in range(L)]
            frames = vr.get_batch(a_idxs + p_idxs).asnumpy()
            a = torch.stack([self.transform(Image.fromarray(frames[i]))
                             for i in range(L)], dim=0)
            p = torch.stack([self.transform(Image.fromarray(frames[L + i]))
                             for i in range(L)], dim=0)
            return a, p
        except Exception:
            return self.__getitem__((idx + 1) % len(self))


# ---------- linear probe dataset ----------

class UCF101Clip(Dataset):
    """Returns (clip_tensor, label) for supervised linear probe.

    Frames are sampled uniformly across each video. Output shape is
    (N, 3, H, W). The downstream classifier mean-pools over N.
    """

    def __init__(self, video_root, split_root, split_file='trainlist01.txt',
                 num_frames=8, size=224):
        self.video_root = Path(video_root)
        classind = _read_classind(Path(split_root) / 'classInd.txt')
        self.items = _read_split(Path(split_root) / split_file, classind)
        self.num_frames = num_frames
        self.transform = _eval_transform(size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        relpath, label = self.items[idx]
        path = self.video_root / relpath
        try:
            vr = VideoReader(str(path), ctx=cpu(0))
            T = len(vr)
            indices = np.linspace(0, T - 1, self.num_frames).astype(int).tolist()
            frames = vr.get_batch(indices).asnumpy()
            tensors = [self.transform(Image.fromarray(f)) for f in frames]
            clip = torch.stack(tensors, dim=0)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.items))
        return clip, label


if __name__ == '__main__':
    import sys
    video_root = sys.argv[1] if len(sys.argv) > 1 else './data/UCF-101'
    split_root = sys.argv[2] if len(sys.argv) > 2 else './data/ucfTrainTestlist'
    print(f'video_root={video_root}  split_root={split_root}')

    ds = UCF101Pair(video_root, split_root)
    print(f'SSL dataset size: {len(ds)}')
    a, p = ds[0]
    print(f'  anchor: {a.shape}  positive: {p.shape}')

    ds2 = UCF101Clip(video_root, split_root)
    print(f'Probe dataset size: {len(ds2)}')
    clip, label = ds2[0]
    print(f'  clip: {clip.shape}  label: {label}')
