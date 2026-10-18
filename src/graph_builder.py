"""Graph construction and parameter-free topological message passing."""

import numpy as np
import pandas as pd


def _host_columns(df: pd.DataFrame) -> list:
    if "src_ip" in df.columns and "dst_ip" in df.columns:
        return ["src_ip", "dst_ip"]
    if "host_ip" in df.columns:
        return ["host_ip"]
    return []


def build_host_adjacency(df: pd.DataFrame, window_size: int = 1000) -> list:
    """
    Build flow graphs window by window (bounds the O(n^2) adjacency memory).
    - Nodes = flows (df must be in chronological order)
    - Edges = consecutive flows sharing a host IP (src_ip / dst_ip, or host_ip)
    - Fallback if no host columns: link consecutive flows in time
    Returns a list of (start, end, A_norm) blocks, A_norm = D^-1/2 (A+I) D^-1/2.
    """
    cols = _host_columns(df)
    blocks = []
    for start in range(0, len(df), window_size):
        end = min(start + window_size, len(df))
        window = df.iloc[start:end]
        n = end - start
        A = np.eye(n, dtype=np.float32)  # self-loops

        if cols:
            for col in cols:
                last_seen = {}
                for i, host in enumerate(window[col].to_numpy()):
                    j = last_seen.get(host)
                    if j is not None:
                        A[i, j] = A[j, i] = 1.0
                    last_seen[host] = i
        else:
            for i in range(n - 1):
                A[i, i + 1] = A[i + 1, i] = 1.0

        d_inv_sqrt = 1.0 / np.sqrt(A.sum(axis=1))
        blocks.append((start, end, A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]))
    return blocks


def propagate_features(X: np.ndarray, blocks: list, hops: int = 2) -> np.ndarray:
    """X_prop = (A_norm ^ hops) X, applied independently inside each window."""
    H = X.astype(np.float32, copy=True)
    for _ in range(hops):
        out = np.empty_like(H)
        for start, end, A_norm in blocks:
            out[start:end] = A_norm @ H[start:end]
        H = out
    return H
