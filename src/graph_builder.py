"""Graph construction and parameter-free topological message passing (Weeks 5-6).

Edge semantics: two flows are connected if they share a src_ip or dst_ip within
the same temporal window. The dataframe is sorted by 'ts' before adjacency
construction so that ordering reflects actual communication time, not file order.
If neither 'ts' nor IP columns are present, falls back to consecutive linkage
with a warning.
"""

import warnings
import numpy as np
import pandas as pd


def build_host_adjacency(df: pd.DataFrame, window_size: int = 1000) -> np.ndarray:
    """
    Construct a flow-level adjacency matrix within a temporal window:
    - Nodes  = individual network flows (rows)
    - Edges  = flows that share a src_ip or dst_ip (same endpoint)
    - Window = first `window_size` rows after sorting by timestamp

    Graph window is bounded by window_size to keep O(n^2) complexity tractable.
    """
    # Sort chronologically to prevent label-leakage from file ordering
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)

    n = min(len(df), window_size)
    A = np.eye(n, dtype=np.float32)  # Self-loops included per GCN convention

    has_src = "src_ip" in df.columns
    has_dst = "dst_ip" in df.columns

    if has_src or has_dst:
        # Build endpoint-sharing edges: O(n^2) within window
        for i in range(n):
            for j in range(i + 1, n):
                shared = False
                if has_src and df["src_ip"].iloc[i] == df["src_ip"].iloc[j]:
                    shared = True
                if has_dst and df["dst_ip"].iloc[i] == df["dst_ip"].iloc[j]:
                    shared = True
                if has_src and has_dst:
                    # Cross-direction: src_i == dst_j or dst_i == src_j
                    if df["src_ip"].iloc[i] == df["dst_ip"].iloc[j]:
                        shared = True
                    if df["dst_ip"].iloc[i] == df["src_ip"].iloc[j]:
                        shared = True
                if shared:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
    else:
        # Fallback: consecutive temporal linkage (weak signal, logged as warning)
        warnings.warn(
            "No src_ip/dst_ip columns found; falling back to consecutive-row linkage. "
            "Graph topology will not reflect host relationships.",
            RuntimeWarning,
            stacklevel=2,
        )
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0

    return A


def propagate_features(X: np.ndarray, A: np.ndarray, hops: int = 2) -> np.ndarray:
    """
    Symmetric normalised graph propagation (Kipf & Welling, 2017):
      A_hat = D^(-1/2) · A · D^(-1/2)
      H^(k) = A_hat · H^(k-1),   H^(0) = X

    If X has more rows than A (because the full dataset exceeds window_size),
    rows beyond the window are passed through unchanged so that H and X always
    share the same length — preventing shape mismatches in downstream metrics.
    """
    n_window = len(A)           # adjacency window size
    n_total  = len(X)           # full feature matrix length

    X_window = X[:n_window]     # only propagate within the window

    # Degree matrix normalization
    degrees = np.sum(A, axis=1)
    d_inv_sqrt = np.where(degrees > 0, np.power(degrees, -0.5), 0.0)
    D_mat = np.diag(d_inv_sqrt)
    A_norm = D_mat @ A @ D_mat

    # Multi-hop propagation
    H_window = X_window.copy()
    for _ in range(hops):
        H_window = A_norm @ H_window

    # Pad rows beyond the window with original features (no graph info available)
    if n_total > n_window:
        H_tail = X[n_window:]
        H = np.vstack([H_window, H_tail])
    else:
        H = H_window

    return H
