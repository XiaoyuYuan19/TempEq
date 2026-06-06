# TempEq

**Constrained Temporal Representation Learning via Smoothness–Diversity Equilibrium.**

SEI Workshop at HHAI 2026, July 2026, Brussels.

TempEq learns visual representations from unlabeled video by formulating
self-supervised learning as a **primal-dual minimax** between two terms:

- **Temporal smoothness** — paired clips drawn from the same video at a
  controlled temporal offset are pulled close in feature space.
- **Representation diversity** — a `0.5 · logdet Cov(z)` term keeps the
  features from collapsing onto a degenerate subspace.

A single dual variable `λ` adaptively balances the two, removing the need
for hand-tuned trade-off weights:

```
  min_θ  max_{λ ≥ 0}     S(θ)  −  λ · H(θ)
            S(θ) = ‖ z_a − z_p ‖²              (per-pair stability)
            H(θ) = 0.5 · logdet Cov(z) + ε·I   (global diversity)
```

The equilibrium `H(θ*) = 0` corresponds to `det Cov = 1`, i.e. features
occupy a non-degenerate unit volume in the projection space.

## Results (UCF-101 linear probe, ResNet-18 from scratch)

| Method | Frame-level | Clip-level (T=4, stride=8) |
| --- | --- | --- |
| SimCLR  | 43.7 | 41.8 |
| VICReg  | 33.0 | 38.7 |
| **TempEq (ours)** | **41.3** | **45.2** ⭐ |

*Peak linear-probe accuracy over 100 epochs.*

## Quick start

```bash
# 1. install
pip install -r requirements.txt

# 2. prepare UCF101 (downloads + repacks into a single tar)
DATA_ROOT=./data bash scripts/prepare_ucf101.sh

# 3. train (single GPU example; for multi-GPU use the LUMI sbatch)
python train.py \
    --video-root ./data/UCF-101 \
    --split-root ./data/ucfTrainTestlist \
    --out-dir runs/clip_pf2 \
    --loss pf2 --clip-len 4 --clip-stride 8 --sampler geometric \
    --proj-dim 128 --proj-hidden 8192 --proj-layers 3 --proj-bn \
    --optimizer lars --lr 0.2 --weight-decay 1e-6 \
    --batch-size 64 --epochs 100 \
    --probe-every 5
```

## Reproducing the paper tables (LUMI)

The `configs/` directory holds canonical `.env` files for each row of
the main results table. Launch with the `CONFIG=` env var:

```bash
# headline: clip-level TempEq
CONFIG=clip_pf2    sbatch scripts/lumi_2gpu.sbatch

# baselines
CONFIG=clip_simclr sbatch scripts/lumi_2gpu.sbatch
CONFIG=clip_vicreg sbatch scripts/lumi_2gpu.sbatch
CONFIG=frame_pf2    sbatch scripts/lumi_2gpu.sbatch
CONFIG=frame_simclr sbatch scripts/lumi_2gpu.sbatch
CONFIG=frame_vicreg sbatch scripts/lumi_2gpu.sbatch
```

Each run writes `logs/<RUN_NAME>/train.log` (per-epoch loss + linear-probe
accuracy) and periodic checkpoints `ssl_ep*.pt` / `ssl_latest.pt`.

## Standalone linear probe

`train.py` runs the probe inline every `--probe-every` epochs. To re-probe
a saved checkpoint:

```bash
python linear_probe.py \
    --video-root ./data/UCF-101 \
    --split-root ./data/ucfTrainTestlist \
    --ssl-ckpt runs/clip_pf2/ssl_latest.pt \
    --out-dir runs/clip_pf2_probe \
    --num-frames 8 --epochs 30
```

## Files

```
data.py          UCF101Pair + UCF101Clip datasets
model.py         ResNet-18 encoder + projection head + linear classifier
loss.py          InfoNCE, VICReg, and PF2 (ours)
train.py         DDP SSL training loop with LARS + cosine LR + inline probe
linear_probe.py  Standalone linear-probe runner (feature cache + classifier)
dist_utils.py    Minimal SLURM/DDP helpers

configs/*.env    Per-experiment env-var configs sourced by the LUMI sbatch
scripts/         LUMI sbatch launcher + UCF101 data-prep script
```

## Citation

```bibtex
@inproceedings{yuan2026tempeq,
  title  = {Constrained Temporal Representation Learning via
            Smoothness--Diversity Equilibrium},
  author = {Yuan, Xiaoyu and Wang, Chengyan and Chen, Haoyu},
  booktitle = {SEI Workshop at HHAI 2026},
  year   = {2026},
}
```

## License

MIT. See [LICENSE](LICENSE).
