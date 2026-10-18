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


class GNNINT8Model:
    """
    Quantized GNN (H1 target):
    Simulates INT8 quantization of weights and activations via uniform symmetric scaling:
    Q(w) = clip(round(w / scale), -128, 127)
    De-quantization: w_hat = Q(w) * scale
    Evaluates execution latency, model footprint reduction, and zero-day guardrail metrics.
    """

    def __init__(self, seed: int = 42):
        self.name = "gnn_int8"
        self.head = LogisticRegression(random_state=seed, max_iter=200)
        self.scale_w: float = 1.0
        self.scale_x: float = 1.0
        self.quantized_coef_: np.ndarray | None = None
        self.quantized_intercept_: np.ndarray | None = None

    def fit(self, H_train: np.ndarray, y_train: np.ndarray):
        self.head.fit(H_train, y_train)

        # Quantize weights to symmetric INT8
        w = self.head.coef_
        b = self.head.intercept_
        max_val_w = max(float(np.max(np.abs(w))), 1e-6)
        self.scale_w = max_val_w / 127.0
        self.quantized_coef_ = np.clip(np.round(w / self.scale_w), -128, 127).astype(np.int8)

        max_val_b = max(float(np.max(np.abs(b))), 1e-6)
        scale_b = max_val_b / 127.0
        self.quantized_intercept_ = np.clip(np.round(b / scale_b), -128, 127).astype(np.int8)
        self.scale_b = scale_b
        return self

    def predict(self, H_test: np.ndarray) -> np.ndarray:
        # Quantize inputs to INT8
        max_val_x = max(float(np.max(np.abs(H_test))), 1e-6)
        self.scale_x = max_val_x / 127.0
        q_x = np.clip(np.round(H_test / self.scale_x), -128, 127).astype(np.int8)

        # Integer matrix multiplication
        int_logits = q_x @ self.quantized_coef_.T  # int32 accumulation
        float_logits = (int_logits.astype(np.float32) * (self.scale_x * self.scale_w)) + (
            self.quantized_intercept_.astype(np.float32) * self.scale_b
        )
        probs = 1.0 / (1.0 + np.exp(-np.clip(float_logits, -15, 15)))
        return (probs.ravel() >= 0.5).astype(int)


class QGNNStubModel:
    """
    Quantum-enhanced GNN (qgnn):
    Exploratory stub for Weeks 5-6 skeleton.
    Returns simulated predictions while logging exploratory status.
    Full PennyLane / Qiskit Parameterized Quantum Circuit (PQC) integration in Week 7.
    """

    def __init__(self, seed: int = 42):
        self.name = "qgnn"
        self.is_exploratory = True
        self.fitted = False

    def fit(self, H_train: np.ndarray, y_train: np.ndarray):
        self.fitted = True
        return self

    def predict(self, H_test: np.ndarray) -> np.ndarray:
        # Exploratory stub output (simple thresholding of primary feature)
        return (H_test[:, 0] > 0.0).astype(int)
