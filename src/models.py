"""Model architectures for IoT Zero-Day Benchmark (Weeks 5-6)."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class FlatLRModel:
    """Non-graph baseline: Flat Logistic Regression on raw flow features (H2 baseline)."""

    def __init__(self, seed: int = 42):
        self.name = "flat_lr"
        self.model = LogisticRegression(random_state=seed, max_iter=200)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_size_mb(self) -> float:
        """Return model parameter footprint in MB (FP32 weights + intercept)."""
        w = self.model.coef_.astype(np.float32)
        b = self.model.intercept_.astype(np.float32)
        return float((w.nbytes + b.nbytes) / (1024 * 1024))


class GNNFP32Model:
    """Classical GNN baseline: Parameter-free propagation + FP32 linear head (H1/H2 baseline)."""

    def __init__(self, seed: int = 42):
        self.name = "gnn_fp32"
        self.head = LogisticRegression(random_state=seed, max_iter=200)

    def fit(self, H_train: np.ndarray, y_train: np.ndarray):
        self.head.fit(H_train, y_train)
        return self

    def predict(self, H_test: np.ndarray) -> np.ndarray:
        return self.head.predict(H_test)

    def predict_proba(self, H_test: np.ndarray) -> np.ndarray:
        return self.head.predict_proba(H_test)

    def param_size_mb(self) -> float:
        """Return FP32 parameter footprint in MB."""
        w = self.head.coef_.astype(np.float32)
        b = self.head.intercept_.astype(np.float32)
        return float((w.nbytes + b.nbytes) / (1024 * 1024))


class GNNINT8Model:
    """
    Quantized GNN (H1 target):
    Simulates INT8 quantization of weights and activations via uniform symmetric scaling:
      Q(w) = clip(round(w / scale), -128, 127)
      De-quantization: w_hat = Q(w) * scale
    Matrix multiplication is performed in INT32 accumulation to avoid overflow
    (numpy int8 @ int8 wraps to int8; explicit cast to int32 is required).
    Evaluates execution latency, model footprint reduction, and zero-day guardrail metrics.
    """

    def __init__(self, seed: int = 42):
        self.name = "gnn_int8"
        self.head = LogisticRegression(random_state=seed, max_iter=200)
        self.scale_w: float = 1.0
        self.scale_x: float = 1.0
        self.scale_b: float = 1.0
        self.quantized_coef_: np.ndarray = np.array([], dtype=np.int8)
        self.quantized_intercept_: np.ndarray = np.array([], dtype=np.int8)

    def fit(self, H_train: np.ndarray, y_train: np.ndarray):
        self.head.fit(H_train, y_train)

        # Quantize weights to symmetric INT8
        w = self.head.coef_
        b = self.head.intercept_
        max_val_w = max(float(np.max(np.abs(w))), 1e-6)
        self.scale_w = max_val_w / 127.0
        self.quantized_coef_ = np.clip(np.round(w / self.scale_w), -128, 127).astype(np.int8)

        max_val_b = max(float(np.max(np.abs(b))), 1e-6)
        self.scale_b = max_val_b / 127.0
        self.quantized_intercept_ = np.clip(np.round(b / self.scale_b), -128, 127).astype(np.int8)
        return self

    def predict(self, H_test: np.ndarray) -> np.ndarray:
        # Quantize inputs to INT8
        max_val_x = max(float(np.max(np.abs(H_test))), 1e-6)
        self.scale_x = max_val_x / 127.0
        q_x = np.clip(np.round(H_test / self.scale_x), -128, 127).astype(np.int8)

        # Cast to int32 BEFORE multiply to avoid int8 overflow (int8 @ int8 → int8 in numpy,
        # which wraps at ±127 and produces incorrect logits).
        int_logits = q_x.astype(np.int32) @ self.quantized_coef_.astype(np.int32).T

        float_logits = (int_logits.astype(np.float32) * (self.scale_x * self.scale_w)) + (
            self.quantized_intercept_.astype(np.float32) * self.scale_b
        )
        probs = 1.0 / (1.0 + np.exp(-np.clip(float_logits, -15, 15)))
        return (probs.ravel() >= 0.5).astype(int)

    def param_size_mb(self) -> float:
        """Return INT8 parameter footprint in MB (4× smaller than FP32)."""
        return float(
            (self.quantized_coef_.nbytes + self.quantized_intercept_.nbytes) / (1024 * 1024)
        )


class QGNNStubModel:
    """
    Quantum-enhanced GNN (qgnn) — exploratory stub for Weeks 5-6 skeleton.
    Does NOT produce real predictions; metrics are recorded as NaN and status as 'skipped'
    so that no fabricated numbers enter results.csv.
    Full PennyLane / Qiskit Parameterized Quantum Circuit (PQC) integration in Week 7.
    """

    def __init__(self, seed: int = 42):
        self.name = "qgnn"
        self.is_exploratory = True
        self.fitted = False

    def fit(self, H_train: np.ndarray, y_train: np.ndarray):
        # No real training in stub; Week 7 will initialise PQC parameters here.
        self.fitted = True
        return self

    def predict(self, H_test: np.ndarray) -> np.ndarray:
        # Return NaN-equivalent sentinel: caller checks is_exploratory before using predictions.
        return np.full(len(H_test), np.nan)

    def param_size_mb(self) -> float:
        return float("nan")
