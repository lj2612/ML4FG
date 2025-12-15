# vector field
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

from geometry import project_onto_tangent

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for full determinism (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class VectorFieldCFM(nn.Module):
    def __init__(self, d, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden),
            nn.CELU(),
            nn.Linear(hidden, hidden),
            nn.CELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x, tau):
        # x: (B, d), tau: (B,)
        if tau.ndim == 1:
            tau = tau.unsqueeze(1)
        return self.net(torch.cat([x, tau], dim=1))
    
def sample_cfm_batch(X0, X1, batch_size):
    N = X0.shape[0]
    idx = np.random.randint(0, N, size=batch_size)
    x0 = X0[idx]
    x1 = X1[idx]
    tau = np.random.rand(batch_size, 1).astype(np.float32)
    x_tau = (1.0 - tau) * x0 + tau * x1
    u = (x1 - x0).astype(np.float32)
    return idx, x_tau, tau.squeeze(1), u

def train_step(
    model,
    optimizer,
    X0,
    X1,
    tangent_pairs,
    device,
    use_tangent: bool = False,
    batch_size: int = 512,
    lambda_cos: float = 0.5,
    lambda_speed: float = 0.1,
):
    """
    One training step for CFM vector field.

    Parameters
    ----------
    model : nn.Module
        Vector field model f(x, tau).
    optimizer : torch.optim.Optimizer
    X0, X1 : (N, d) arrays
        OT-coupled point pairs.
    tangent_pairs : (N, d) array
        Tangents aligned with X0 locations.
    device : torch.device
    use_tangent : bool
        Whether to apply tangent regularization.
    batch_size : int
    lambda_cos : float
        Weight for cosine alignment loss.
    lambda_speed : float
        Weight for speed regularization.

    Returns
    -------
    loss_value : float
    stats : dict
        Optional diagnostics.
    """
    model.train()

    # ---- sample CFM batch ----
    idx, x_tau, tau, u = sample_cfm_batch(X0, X1, batch_size)

    # ---- tangent projection (numpy space) ----
    if use_tangent:
        T = tangent_pairs[idx]                  # (B, d)
        u = project_onto_tangent(u, T)           # (B, d)

    # ---- move to torch ----
    x_tau = torch.as_tensor(x_tau, dtype=torch.float32, device=device)
    tau   = torch.as_tensor(tau,   dtype=torch.float32, device=device)
    u     = torch.as_tensor(u,     dtype=torch.float32, device=device)

    optimizer.zero_grad(set_to_none=True)

    # ---- forward ----
    v_pred = model(x_tau, tau)                  # (B, d)

    # ---- base OT loss ----
    loss_mse = F.mse_loss(v_pred, u)

    # ---- tangent regularization ----
    if use_tangent:
        T_tc = torch.as_tensor(T, dtype=torch.float32, device=device)

        # cosine alignment (direction)
        cos = F.cosine_similarity(v_pred, T_tc, dim=1)
        loss_cos = 1.0 - cos.abs().mean()

        # speed regularization (avoid collapse)
        speed_pred = v_pred.norm(dim=1)
        speed_tgt  = u.norm(dim=1).detach()
        loss_speed = F.mse_loss(speed_pred, speed_tgt)

        loss = loss_mse + lambda_cos * loss_cos + lambda_speed * loss_speed
    else:
        loss = loss_mse

    # ---- backward ----
    loss.backward()
    optimizer.step()

    # ---- diagnostics ----
    stats = {
        "loss": loss.item(),
        "loss_mse": loss_mse.item(),
    }
    if use_tangent:
        stats.update({
            "loss_cos": loss_cos.item(),
            "loss_speed": loss_speed.item(),
            "mean_cos": cos.mean().item(),
            "mean_speed": speed_pred.mean().item(),
        })

    return loss.item(), stats