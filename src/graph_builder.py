"""Graph construction and parameter-free topological message passing (Weeks 5-6)."""

import numpy as np
import pandas as pd


def build_host_adjacency(df: pd.DataFrame, window_size: int = 1000) -> np.ndarray:
    """
    Construct flow graph adjacency matrix:
    - Nodes = network flows
    - Edges = consecutive flows originating from or targeting the same host IP
    - Graph window bounded by window_size to control O(n^2) complexity
    """
    n = min(len(df), window_size)
    A = np.eye(n, dtype=np.float32)  # Self-loops included

    if "host_ip" in df.columns:
        hosts = df["host_ip"].iloc[:n].values
        for i in range(n - 1):
            if hosts[i] == hosts[i + 1]:
                A[i, i + 1] = 1.0
                A[i + 1, i] = 1.0
    else:
        # Fallback consecutive temporal linkage
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0

    return A


def propagate_features(X: np.ndarray, A: np.ndarray, hops: int = 2) -> np.ndarray:
    """
    Perform symmetric normalized graph propagation:
    A_hat = D^(-1/2) * A * D^(-1/2)
    X_prop = (A_hat ^ hops) * X
    """
    n = min(len(X), len(A))
    X_sub = X[:n]
    A_sub = A[:n, :n]

    # Degree matrix normalization
    degrees = np.sum(A_sub, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
    d_inv_sqrt[degrees == 0] = 0.0
    D_mat = np.diag(d_inv_sqrt)

    A_norm = D_mat @ A_sub @ D_mat

    # Multi-hop propagation
    H = X_sub.copy()
    for _ in range(hops):
        H = A_norm @ H

    return H
