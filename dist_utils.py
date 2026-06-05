"""Minimal DDP helpers for SLURM-launched training.

Usage at the top of main():
    rank, local_rank, world_size, device = setup_dist()
    if is_main(): print(...)
    ...
    cleanup_dist()
"""

import os

import torch
import torch.distributed as dist


def setup_dist():
    """Initialize torch.distributed using SLURM env vars.

    Falls back to single-process mode if SLURM_PROCID isn't set.
    Returns (rank, local_rank, world_size, device).
    """
    if 'SLURM_PROCID' in os.environ and int(os.environ.get('SLURM_NTASKS', 1)) > 1:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        # MASTER_ADDR / MASTER_PORT are exported from the sbatch script.
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        rank, local_rank, world_size = 0, 0, 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    return rank, local_rank, world_size, device


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value, device):
    """Reduce a python scalar across ranks (mean). Pass-through if single-rank."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()
