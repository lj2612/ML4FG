import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def off_manifold_energy(model, x_tau_tc, tau_tc, tang_tc):
    model.eval()
    with torch.no_grad():
        v = model(x_tau_tc, tau_tc).cpu().numpy()
    T = tang_tc.cpu().numpy()
    proj = np.sum(v * T, axis=1, keepdims=True) * T
    perp = v - proj
    return np.mean(np.linalg.norm(perp, axis=1)), np.mean(np.linalg.norm(v, axis=1))

def tangent_alignment(model, x_tau_tc, tau_tc, tang_tc):
    model.eval()
    with torch.no_grad():
        v = model(x_tau_tc, tau_tc).cpu().numpy()
    T = tang_tc.cpu().numpy()
    v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    T_norm = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-8)
    cos = np.sum(v_norm * T_norm, axis=1)
    return np.mean(cos)

def generate_trajectories_cfm(
    model,
    Z_init,          # (n_points, d)
    n_bins=100,
    device="cuda"
):
    model.eval()
    t_eval = torch.linspace(0., 1., n_bins).to(device)
    Z0 = torch.tensor(Z_init, dtype=torch.float32, device=device)

    with torch.no_grad():
        traj = odeint(
            lambda t, x: model(x, t.repeat(x.shape[0])),
            Z0,
            t_eval
        )

    return traj.cpu().numpy()

def trim_trajectory_to_data(path, Z, thresh=0.01):
    """
    Trim trajectory once it leaves data support.

    Parameters
    ----------
    path : (T, d) array
        Single trajectory.
    Z : (N, d) array
        Data manifold.
    thresh : float
        Max allowed distance to nearest data point.

    Returns
    -------
    trimmed_path : (T', d) array or None
    """
    nn = NearestNeighbors(n_neighbors=1).fit(Z)
    dists, _ = nn.kneighbors(path)
    dists = dists[:, 0]

    keep = dists < thresh
    if not keep.any():
        return None

    last = keep.nonzero()[0].max()
    return path[:last + 1]

def plot_streamlines(
    Z,
    t_idx,
    traj,
    figsize=(6, 5),
    point_size=3,
    point_alpha=0.15,
    line_color="black",
    line_alpha=0.4,
    line_width=1.0,
    title="Streamlines from t = 0",
):
    """
    Plot background data and streamlines.

    Parameters
    ----------
    Z : (N, 2) array
        Embedding coordinates.
    t_idx : (N,) array
        Discrete time labels for coloring points.
    traj : (T, M, 2) array
        Trajectories (T time steps, M streamlines).
    """
    plt.figure(figsize=figsize)

    plt.scatter(
        Z[:, 0], Z[:, 1],
        c=t_idx,
        s=point_size,
        alpha=point_alpha
    )

    for i in range(traj.shape[1]):
        path = traj[:, i, :]

        # ---- trim to data support ----
        path = trim_trajectory_to_data(path, Z, thresh=0.008)

        if path is None or len(path) < 2:
            continue

        plt.plot(
            path[:, 0],
            path[:, 1],
            color=line_color,
            alpha=line_alpha,
            lw=line_width
        )

    plt.title(title)
    plt.xlabel("d1")
    plt.ylabel("d2")
    plt.tight_layout()
    plt.show()