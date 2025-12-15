import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import ot

def estimate_tangents(Z, t_idx, k=20):
    """
    Estimate 1D tangent vectors at each point using local PCA
    within the same discrete timepoint.

    Parameters
    ----------
    Z : (N, d) array
        Embedding coordinates.
    t_idx : (N,) array
        Discrete time labels.
    k : int
        Number of neighbors for local PCA.

    Returns
    -------
    tangents : (N, d) array
        Unit-norm tangent vectors.
    """
    N, d = Z.shape
    tangents = np.zeros((N, d), dtype=np.float32)

    for t in np.unique(t_idx):
        mask = t_idx == t
        Z_t = Z[mask]
        idx_t = np.where(mask)[0]

        if len(Z_t) <= k:
            continue

        nn = NearestNeighbors(n_neighbors=k).fit(Z_t)
        _, nbrs = nn.kneighbors(Z_t)

        for i_local, i_global in enumerate(idx_t):
            Z_local = Z_t[nbrs[i_local]]
            Z_centered = Z_local - Z_local.mean(axis=0, keepdims=True)

            pca = PCA(n_components=1)
            pca.fit(Z_centered)

            v = pca.components_[0]
            v /= np.linalg.norm(v) + 1e-8
            tangents[i_global] = v

    return tangents


def project_onto_tangent(u, T):
    """
    Project vectors u onto tangent directions T.

    u : (B, d)
    T : (B, d)
    """
    dot = np.sum(u * T, axis=1, keepdims=True)
    return dot * T

def compute_ot_pairs(Z, t_idx, max_pairs_per_edge=4000):
    """
    Compute OT couplings between consecutive timepoints in PHATE space.
    Returns X0, X1, T0, T1.
    """
    times = np.unique(t_idx)
    X0_list, X1_list, T0_list, T1_list = [], [], [], []

    for t0, t1 in zip(times[:-1], times[1:]):
        mask0 = (t_idx == t0)
        mask1 = (t_idx == t1)
        Z0 = Z[mask0]
        Z1 = Z[mask1]
        n0, n1 = len(Z0), len(Z1)
        print(f"OT for t={t0}->{t1}, shapes {Z0.shape}, {Z1.shape}")

        a = np.ones(n0) / n0
        b = np.ones(n1) / n1
        C = ot.dist(Z0, Z1, metric="euclidean")**2

        Gamma = ot.emd(a, b, C)   # (n0, n1)
        rows, cols = np.where(Gamma > 0)
        w = Gamma[rows, cols]
        w = w / w.sum()

        m = min(max_pairs_per_edge, len(w))
        idx = np.random.choice(len(w), size=m, replace=True, p=w)

        X0_list.append(Z0[rows[idx]])
        X1_list.append(Z1[cols[idx]])
        T0_list.append(np.full(m, t0))
        T1_list.append(np.full(m, t1))

    X0 = np.vstack(X0_list)
    X1 = np.vstack(X1_list)
    T0 = np.concatenate(T0_list)
    T1 = np.concatenate(T1_list)
    return X0, X1, T0, T1