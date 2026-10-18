"""Model architectures for IoT Zero-Day Benchmark (Weeks 5-6)."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class FlatLRModel:
    """Non-graph baseline: logistic regression on raw flow features (H2 baseline)."""

    def __init__(self, seed: int = 42):
        self.name = "flat_lr"
        self.model = LogisticRegression(random_state=seed, max_iter=200)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def size_mb(self) -> float:
        n = self.model.coef_.size + self.model.intercept_.size
        return n * 4 / 1e6  # counted as FP32 for a fair comparison


class GNNFP32Model:
    """Classical GNN baseline: parameter-free propagation + FP32 linear head."""

    def __init__(self, seed: int = 42):
        self.name = "gnn_fp32"
        self.head = LogisticRegression(random_state=seed, max_iter=200)

    def fit(self, H_train, y_train):
        self.head.fit(H_train, y_train)
        return self

    def predict(self, H_test):
        return self.head.predict(H_test)

    def size_mb(self) -> float:
        n = self.head.coef_.size + self.head.intercept_.size
        return n * 4 / 1e6


class GNNINT8Model:
    """
    Quantized GNN head (H1 target), symmetric INT8:
    q = clip(round(v / scale), -127, 127), scale = max|v| / 127.
    Integer matmul is accumulated in INT32 (int8 @ int8 would overflow).
    NOTE: in Week 5-6 only the head is quantized; propagation stays FP32.
    """

    def __init__(self, seed: int = 42):
        self.name = "gnn_int8"
        self.head = LogisticRegression(random_state=seed, max_iter=200)
        self.scale_w = 1.0
        self.scale_b = 1.0
        self.quantized_coef_ = None
        self.quantized_intercept_ = None

    @staticmethod
    def _quantize(v, max_val=None):
        max_val = max(float(np.max(np.abs(v))) if max_val is None else max_val, 1e-6)
        scale = max_val / 127.0
        return np.clip(np.round(v / scale), -127, 127).astype(np.int8), scale

    def fit(self, H_train, y_train):
        self.head.fit(H_train, y_train)
        self.quantized_coef_, self.scale_w = self._quantize(self.head.coef_)
        self.quantized_intercept_, self.scale_b = self._quantize(self.head.intercept_)
        return self

    def predict(self, H_test):
        q_x, scale_x = self._quantize(H_test)
        int_logits = q_x.astype(np.int32) @ self.quantized_coef_.astype(np.int32).T  # INT32 accumulation
        logits = int_logits.astype(np.float32) * (scale_x * self.scale_w) + (
            self.quantized_intercept_.astype(np.float32) * self.scale_b
        )
        return (logits.ravel() >= 0.0).astype(int)

    def size_mb(self) -> float:
        n_bytes = self.quantized_coef_.nbytes + self.quantized_intercept_.nbytes + 8  # + two FP32 scales
        return n_bytes / 1e6


class QGNNStubModel:
    """
    Quantum-enhanced GNN placeholder. No fake predictions: fit() raises
    NotImplementedError, the benchmark records the row as 'skipped'.
    Week 7: PennyLane / Qiskit parameterized quantum circuit on a simulator.
    """

    def __init__(self, seed: int = 42):
        self.name = "qgnn"

    def fit(self, H_train, y_train):
        raise NotImplementedError("QGNN (PennyLane/Qiskit PQC) is planned for Week 7")
