"""SSL losses for TempEq: InfoNCE, VICReg, and PF2.

PF2 (ours): a constrained minimax loss that balances temporal stability
against representation diversity via primal-dual optimization. See
README for the formulation.

Loss API:
    loss, metrics = loss_fn(z_a, z_p)

`metrics` is a dict of detached scalars used for logging. PF2 additionally
exposes a `dual_step(H_value)` method that the training loop calls after
backward, using the cross-rank mean of H.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def _gather_with_local_grad(z):
    """All-gather z across DDP ranks; keep gradient on the local rank's slot.

    dist.all_gather is not differentiable; we restore gradient flow on the
    local rank by overwriting that slot with the original (graph-attached)
    tensor before concat. Gradients on the cross-rank slots are dropped,
    which under DDP's AllReduce is equivalent to dividing the dual term by
    world_size -- absorbed into the learning-rate / dual-step scale.
    """
    if not (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1):
        return z
    ws = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.empty_like(z) for _ in range(ws)]
    dist.all_gather(gathered, z)
    gathered[rank] = z
    return torch.cat(gathered, dim=0)


# ---------- InfoNCE (SimCLR) ----------

class InfoNCE(nn.Module):
    """NT-Xent loss on L2-normalized features."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, z_a, z_p):
        B = z_a.shape[0]
        z = torch.cat([z_a, z_p], dim=0)
        z = F.normalize(z, dim=1)
        sim = z @ z.t() / self.t
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))
        targets = torch.arange(2 * B, device=z.device)
        targets = (targets + B) % (2 * B)
        loss = F.cross_entropy(sim, targets)
        with torch.no_grad():
            acc = (sim.argmax(dim=1) == targets).float().mean()
        return loss, {'acc': acc}


# ---------- PF2 (ours) ----------

def _h_mu(z, eps=1e-4, gather=True):
    """Diversity surrogate H = 0.5 * logdet(Cov(z) + eps*I).

    With gather=True (default) under DDP, z is all-gathered so the
    covariance uses the full global batch. Required when per-rank
    B < proj_dim, otherwise padding eigenvalues at eps drag H negative.
    """
    if gather:
        z = _gather_with_local_grad(z)
    B, D = z.shape
    z_c = z - z.mean(dim=0, keepdim=True)
    cov = (z_c.t() @ z_c) / (B - 1) + eps * torch.eye(D, device=z.device, dtype=z.dtype)
    eigvals = torch.linalg.eigvalsh(cov)
    return 0.5 * torch.log(eigvals.clamp(min=1e-8)).sum()


class PF2Loss(nn.Module):
    """TempEq's PF2 minimax loss:  min_theta max_{lam >= 0}  S - lam * H.

      S = ||z_a - z_p||^2        (per-pair temporal stability)
      H = 0.5 * logdet Cov(z_a)  (global diversity / anti-collapse)

    Dual ascent on lam (clamped non-negative):
        lam <- max(0, lam - dual_lr * H)

    Equilibrium: H(theta*) = 0, i.e. det(Cov) = 1 -- features occupy a
    non-degenerate unit volume in the projected space.
    """

    def __init__(self, dual_lr=0.01, lam_init=0.0, lam_max=0.0, eps=1e-4):
        super().__init__()
        self.dual_lr = dual_lr
        self.lam_max = lam_max  # 0 means uncapped
        self.eps = eps
        # buffer so DDP broadcasts once and resume picks it up via state_dict
        self.register_buffer('lam', torch.tensor(float(lam_init)))

    def forward(self, z_a, z_p):
        S = ((z_a - z_p) ** 2).sum(dim=1).mean()
        H = _h_mu(z_a, eps=self.eps)
        loss = S - self.lam * H
        return loss, {
            'stab': S.detach(),
            'H': H.detach(),
            'lam': self.lam.detach().clone(),
        }

    @torch.no_grad()
    def dual_step(self, H_value):
        """Dual ascent on lam. H_value: scalar tensor or float (already reduced)."""
        if not torch.is_tensor(H_value):
            H_value = torch.tensor(float(H_value),
                                   device=self.lam.device, dtype=self.lam.dtype)
        new_lam = (self.lam - self.dual_lr * H_value).clamp(min=0.0)
        if self.lam_max > 0:
            new_lam = new_lam.clamp(max=self.lam_max)
        self.lam.copy_(new_lam)


# ---------- VICReg ----------

class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance regularization (Bardes et al., 2022).

      sim  = ((z_a - z_p)^2).mean()             invariance
      var  = relu(1 - std_per_dim(z)).mean()    per-dim variance hinge
      cov  = sum_offdiag(Cov(z)^2) / D          off-diagonal decorrelation

    Cov is computed on the gathered global batch so DDP per-rank B < D
    is fine. Fixed weights -- no dual variable.
    """

    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0,
                 eps=1e-4, gather=True):
        super().__init__()
        self.sim_w = sim_weight
        self.var_w = var_weight
        self.cov_w = cov_weight
        self.eps = eps
        self.gather = gather

    def forward(self, z_a, z_p):
        sim_loss = ((z_a - z_p) ** 2).mean()
        if self.gather:
            z_a_g = _gather_with_local_grad(z_a)
            z_p_g = _gather_with_local_grad(z_p)
        else:
            z_a_g, z_p_g = z_a, z_p
        z = torch.cat([z_a_g, z_p_g], dim=0)
        B, D = z.shape
        std = torch.sqrt(z.var(dim=0) + self.eps)
        var_loss = torch.relu(1.0 - std).mean()
        z_c = z - z.mean(dim=0, keepdim=True)
        cov = (z_c.t() @ z_c) / (B - 1)
        off_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        cov_loss = (cov.masked_select(off_mask) ** 2).sum() / D
        loss = self.sim_w * sim_loss + self.var_w * var_loss + self.cov_w * cov_loss
        return loss, {
            'sim': sim_loss.detach(),
            'var': var_loss.detach(),
            'cov': cov_loss.detach(),
        }


# ---------- factory ----------

def build_loss(name, **kwargs):
    name = name.lower()
    if name == 'infonce':
        return InfoNCE(temperature=kwargs.get('temperature', 0.1))
    if name == 'pf2':
        return PF2Loss(
            dual_lr=kwargs.get('pf2_dual_lr', 0.01),
            lam_init=kwargs.get('pf2_lam_init', 0.0),
            lam_max=kwargs.get('pf2_lam_max', 0.0),
        )
    if name == 'vicreg':
        return VICRegLoss(
            sim_weight=kwargs.get('vicreg_sim', 25.0),
            var_weight=kwargs.get('vicreg_var', 25.0),
            cov_weight=kwargs.get('vicreg_cov', 1.0),
        )
    raise ValueError(f'unknown loss: {name}')


if __name__ == '__main__':
    z_a = torch.randn(8, 128)
    z_p = torch.randn(8, 128)
    for name in ['infonce', 'pf2', 'vicreg']:
        loss_fn = build_loss(name)
        loss, m = loss_fn(z_a, z_p)
        print(f'{name}: loss={loss.item():.4f}  '
              f'metrics={ {k: v.item() for k, v in m.items()} }')
        if hasattr(loss_fn, 'dual_step'):
            loss_fn.dual_step(m['H'])
            print(f'  after dual_step lam={loss_fn.lam.item():.4f}')
