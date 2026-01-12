"""
qml_kernel_svc.py

Modular quantum-kernel SVC training utilities for PennyLane Lightning (CPU/GPU).

Major speedup (math optimisation):
- Build kernels via state embeddings:
    1) Compute |psi(x)> once per sample (O(N) circuit calls)
    2) Compute Gram / cross-kernels via matrix multiplies (fast BLAS):
         S = Psi @ Psi*.T
         K = |S|^2

This replaces pairwise circuit evaluation (O(N^2) circuit calls).

Design notes
- No globals: configuration lives in QMLConfig.
- Device is created before the QNode (important: QNodes bind to the device at creation).
- Kernel builders accept a kernel "engine" so CPU/GPU/backends are swappable.
- Uses qml.state() + overlaps for kernels (fastest for small n_qubits).

Typical usage
- qml_cfg = build_qml_config(n_qubits=3, system="linux", kernel_mode="state")
- results = quantum_kernel_train_loop(..., qml_cfg=qml_cfg, ...)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any, Literal

import numpy as np
import pennylane as qml
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.metrics import log_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# -------------------------
# Types / config container
# -------------------------

# For legacy (pairwise) kernels: takes two feature vectors and depth, returns scalar in [0,1].
KernelFn = Callable[[np.ndarray, np.ndarray, int], float]

KernelMode = Literal["state", "pairwise"]


@dataclass(frozen=True)
class QMLConfig:
    """Holds quantum configuration and the callables needed by the training loop.

    Attributes:
        n_qubits: Number of qubits / wires.
        simulator: PennyLane device name (e.g. "lightning.qubit", "lightning.gpu").
        feature_depth_default: Default feature-map depth.
        kernel_mode: "state" (fast) or "pairwise" (legacy).
        dev: PennyLane device instance.
        feature_map: Callable applying the feature-map gates.
        state_circuit: QNode returning qml.state() (used in kernel_mode="state").
        kernel_fn: Pairwise kernel callable (used in kernel_mode="pairwise").
    """
    n_qubits: int
    simulator: str = "lightning.qubit"
    feature_depth_default: int = 2
    kernel_mode: KernelMode = "state"

    dev: Any = None
    feature_map: Optional[Callable[[np.ndarray, int], None]] = None

    # state-kernel mode
    state_circuit: Optional[Callable[[np.ndarray, int], np.ndarray]] = None

    # pairwise-kernel mode (fallback)
    kernel_fn: Optional[KernelFn] = None


def make_device(n_qubits: int, simulator: str) -> Any:
    """Create and return a PennyLane device."""
    return qml.device(simulator, wires=n_qubits)


def feature_map_factory(n_qubits: int) -> Callable[[np.ndarray, int], None]:
    """Create a feature-map function bound to n_qubits (no globals)."""
    def feature_map(x: np.ndarray, depth: int = 2) -> None:
        for _ in range(depth):
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)

            # Kept identical to your original (guarded for n_qubits < 3).
            if n_qubits >= 2:
                qml.CNOT(wires=[0, 1])
            if n_qubits >= 3:
                qml.CNOT(wires=[1, 2])

    return feature_map


# -------------------------
# Pairwise-kernel circuit (legacy / fallback)
# -------------------------

def make_kernel_qnode(
    dev: Any,
    n_qubits: int,
    feature_map: Callable[[np.ndarray, int], None],
) -> KernelFn:
    """Build a pairwise kernel function k(x1, x2, depth) -> float via Projector(|0...0>)."""
    projector_state = [0] * n_qubits
    wires = list(range(n_qubits))

    @qml.qnode(dev, diff_method=None)
    def _kernel_circuit(x1: np.ndarray, x2: np.ndarray, depth: int = 2):
        feature_map(x1, depth=depth)
        qml.adjoint(feature_map)(x2, depth=depth)
        return qml.expval(qml.Projector(projector_state, wires=wires))

    def kernel_fn(x1: np.ndarray, x2: np.ndarray, depth: int = 2) -> float:
        return float(_kernel_circuit(x1, x2, depth=depth))

    return kernel_fn


# -------------------------
# State-embedding circuit (fast path)
# -------------------------

def make_state_qnode(
    dev: Any,
    n_qubits: int,
    feature_map: Callable[[np.ndarray, int], None],
) -> Callable[[np.ndarray, int], np.ndarray]:
    """Build a QNode returning the statevector |psi(x)> for a given x and depth."""
    @qml.qnode(dev, diff_method=None)
    def state_circuit(x: np.ndarray, depth: int = 2):
        feature_map(x, depth=depth)
        return qml.state()

    return state_circuit


def embed_states(
    X: np.ndarray,
    state_circuit: Callable[[np.ndarray, int], np.ndarray],
    depth: int = 2,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute statevectors for all rows of X (O(N) circuit calls)."""
    N = len(X)
    states: List[np.ndarray] = []

    it = range(N)
    if show_progress:
        it = tqdm(it, desc="Embedding states", leave=False)

    for i in it:
        states.append(state_circuit(X[i], depth=depth))

    # shape: (N, 2**n_qubits), complex
    return np.asarray(states)


def gram_from_states(states: np.ndarray) -> np.ndarray:
    """Compute Gram matrix K_ij = |<psi_i|psi_j>|^2 via BLAS."""
    S = states @ states.conj().T
    K = (np.abs(S) ** 2).astype(np.float32)

    # Numerical cleanup (optional but usually helpful)
    K = 0.5 * (K + K.T)
    np.clip(K, 0.0, 1.0, out=K)
    return K


def cross_kernel_from_states(states_A: np.ndarray, states_B: np.ndarray) -> np.ndarray:
    """Compute cross-kernel K_ij = |<psi_Ai|psi_Bj>|^2."""
    S = states_A @ states_B.conj().T
    K = (np.abs(S) ** 2).astype(np.float32)
    np.clip(K, 0.0, 1.0, out=K)
    return K


# -------------------------
# Build config
# -------------------------

def build_qml_config(
    n_qubits: int,
    system: str = "windows",
    cpu_sim: str = "lightning.qubit",
    gpu_sim: str = "lightning.gpu",
    feature_depth_default: int = 2,
    kernel_mode: KernelMode = "state",
) -> QMLConfig:
    """Create a ready-to-run quantum config.

    Args:
        n_qubits: Number of wires.
        system: "linux" -> gpu_sim, otherwise cpu_sim.
        cpu_sim: CPU device name.
        gpu_sim: GPU device name.
        feature_depth_default: Default depth stored in the config.
        kernel_mode: "state" (fast, recommended) or "pairwise" (fallback).

    Returns:
        QMLConfig with device + feature_map and the right circuit for the chosen mode.
    """
    simulator = gpu_sim if system.lower() == "linux" else cpu_sim
    dev = make_device(n_qubits, simulator)
    fmap = feature_map_factory(n_qubits)

    if kernel_mode == "state":
        state_circuit = make_state_qnode(dev, n_qubits, fmap)
        return QMLConfig(
            n_qubits=n_qubits,
            simulator=simulator,
            feature_depth_default=feature_depth_default,
            kernel_mode=kernel_mode,
            dev=dev,
            feature_map=fmap,
            state_circuit=state_circuit,
            kernel_fn=None,
        )

    # fallback / legacy
    kernel_fn = make_kernel_qnode(dev, n_qubits, fmap)
    return QMLConfig(
        n_qubits=n_qubits,
        simulator=simulator,
        feature_depth_default=feature_depth_default,
        kernel_mode=kernel_mode,
        dev=dev,
        feature_map=fmap,
        state_circuit=None,
        kernel_fn=kernel_fn,
    )


# -------------------------
# Cache utilities
# -------------------------

class KernelCache:
    """In-memory cache for expensive kernel matrices.

    Notes:
      - Keys should include fold/depth (and cut for chrono calibration).
      - With state-kernel mode, caching the *kernels* is usually enough.
    """
    def __init__(self):
        self._cache: Dict[Any, np.ndarray] = {}

    def get(self, key):
        return self._cache.get(key, None)

    def set(self, key, value: np.ndarray):
        self._cache[key] = value

    def clear(self):
        self._cache.clear()


def get_cached_kernel_fold(cache: KernelCache, key, builder: Callable[[], np.ndarray]) -> np.ndarray:
    """Return cached matrix, or compute+cache via builder."""
    K = cache.get(key)
    if K is None:
        K = builder()
        cache.set(key, K)
    return K


# -------------------------
# Kernel matrix builders (dispatch)
# -------------------------

def build_train_gram(
    X_train: np.ndarray,
    qml_cfg: QMLConfig,
    depth: int,
    show_progress: bool,
) -> np.ndarray:
    """Build K_train (train vs train) using the configured kernel mode."""
    if qml_cfg.kernel_mode == "state":
        assert qml_cfg.state_circuit is not None
        states_train = embed_states(X_train, qml_cfg.state_circuit, depth=depth, show_progress=show_progress)
        return gram_from_states(states_train)

    assert qml_cfg.kernel_fn is not None
    return quantum_kernel_symmetric_pairwise(X_train, qml_cfg.kernel_fn, depth=depth, show_progress=show_progress)


def build_cross_kernel(
    X_left: np.ndarray,
    X_right: np.ndarray,
    qml_cfg: QMLConfig,
    depth: int,
    show_progress: bool,
) -> np.ndarray:
    """Build cross-kernel K (left vs right) using the configured kernel mode."""
    if qml_cfg.kernel_mode == "state":
        assert qml_cfg.state_circuit is not None
        states_left = embed_states(X_left, qml_cfg.state_circuit, depth=depth, show_progress=show_progress)
        states_right = embed_states(X_right, qml_cfg.state_circuit, depth=depth, show_progress=False)
        return cross_kernel_from_states(states_left, states_right)

    assert qml_cfg.kernel_fn is not None
    return quantum_kernel_pairwise(X_left, X_right, qml_cfg.kernel_fn, depth=depth, show_progress=show_progress)


# -------------------------
# Pairwise kernel builders (kept for fallback)
# -------------------------

def quantum_kernel_pairwise(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel_fn: KernelFn,
    depth: int = 2,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute cross-kernel via pairwise circuit calls (slow)."""
    K = np.zeros((len(X1), len(X2)), dtype=np.float32)

    iterator = enumerate(X1)
    if show_progress:
        iterator = tqdm(iterator, total=len(X1), desc="Quantum kernel rows", leave=False)

    for i, x1 in iterator:
        for j, x2 in enumerate(X2):
            K[i, j] = kernel_fn(x1, x2, depth=depth)

    return K


def quantum_kernel_symmetric_pairwise(
    X: np.ndarray,
    kernel_fn: KernelFn,
    depth: int = 2,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute symmetric Gram matrix via pairwise circuit calls (slow)."""
    n = len(X)
    K = np.zeros((n, n), dtype=np.float32)

    it = range(n)
    if show_progress:
        it = tqdm(it, desc="Quantum kernel (symmetric)", leave=False)

    for i in it:
        K[i, i] = 1.0
        for j in range(i + 1, n):
            val = kernel_fn(X[i], X[j], depth=depth)
            K[i, j] = val
            K[j, i] = val

    return K


# -------------------------
# Training loops (SVC + calibration)
# -------------------------

def fit_predict_quantum_kernel_oof_calibrated(
    X: np.ndarray,
    y: np.ndarray,
    splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    qml_cfg: QMLConfig,
    depth: int = 2,
    C: float = 1.0,
    cache: Optional[KernelCache] = None,
    stock_name: str = "",
    show_kernel_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]], KernelCache]:
    """Walk-forward OOF training with fold-safe scaling and calibration."""
    if cache is None:
        cache = KernelCache()

    classes = np.array(sorted(np.unique(y)))
    n_classes = len(classes)

    oof_proba = np.full((len(X), n_classes), np.nan, dtype=np.float64)
    oof_pred = np.full(len(X), np.nan)

    hist: Dict[str, List[float]] = {"fold": [], "macro_f1": [], "log_loss": []}

    splits = list(splits)

    for fold, (train_idx, val_idx) in enumerate(tqdm(splits, desc="QML Walk-Forward CV"), 1):
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        # --- build kernels (FAST: state embeddings + GEMM) ---
        K_train = get_cached_kernel_fold(
            cache,
            key=("K_train", stock_name, fold, depth, qml_cfg.kernel_mode, qml_cfg.simulator),
            builder=lambda: build_train_gram(X_train, qml_cfg, depth=depth, show_progress=show_kernel_progress),
        )

        K_val = get_cached_kernel_fold(
            cache,
            key=("K_val", stock_name, fold, depth, qml_cfg.kernel_mode, qml_cfg.simulator),
            builder=lambda: build_cross_kernel(X_val, X_train, qml_cfg, depth=depth, show_progress=show_kernel_progress),
        )

        svc = SVC(kernel="precomputed", C=C)
        svc.fit(K_train, y_train)

        scores_val = svc.decision_function(K_val)
        if scores_val.ndim == 1:
            scores_val = scores_val.reshape(-1, 1)

        cal = LogisticRegression(max_iter=2000, solver="lbfgs")
        cal.fit(scores_val, y_val)

        proba_val = cal.predict_proba(scores_val)
        pred_val = np.argmax(proba_val, axis=1)

        oof_proba[val_idx] = proba_val
        oof_pred[val_idx] = pred_val

        f1 = f1_score(y_val, pred_val, average="macro")
        ll = log_loss(y_val, proba_val, labels=classes)

        hist["fold"].append(fold)
        hist["macro_f1"].append(float(f1))
        hist["log_loss"].append(float(ll))

        tqdm.write(f"Fold {fold}: macroF1={f1:.4f} ::: log_loss={ll:.4f}")

    return oof_pred.astype(int), oof_proba, hist, cache


def fit_quantum_kernel_and_predict_test_chrono_cal(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    qml_cfg: QMLConfig,
    depth: int = 2,
    C: float = 1.0,
    cal_size: float = 0.2,
    cache: Optional[KernelCache] = None,
    stock_name: str = "",
    show_kernel_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, KernelCache]:
    """Chronological train->calibrate->deploy evaluation on test."""
    if cache is None:
        cache = KernelCache()

    n = len(y_dev)
    cut = int(n * (1 - cal_size))
    if cut <= 0 or cut >= n:
        raise ValueError("Invalid cal_size; must leave both train and calibration slices non-empty.")

    X_train, y_train = X_dev[:cut], y_dev[:cut]
    X_cal, y_cal = X_dev[cut:], y_dev[cut:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    K_train = get_cached_kernel_fold(
        cache,
        key=("K_dev_train", stock_name, depth, cut, qml_cfg.kernel_mode, qml_cfg.simulator),
        builder=lambda: build_train_gram(X_tr_s, qml_cfg, depth=depth, show_progress=show_kernel_progress),
    )

    K_cal = get_cached_kernel_fold(
        cache,
        key=("K_dev_cal", stock_name, depth, cut, qml_cfg.kernel_mode, qml_cfg.simulator),
        builder=lambda: build_cross_kernel(X_cal_s, X_tr_s, qml_cfg, depth=depth, show_progress=show_kernel_progress),
    )

    K_test = get_cached_kernel_fold(
        cache,
        key=("K_test", stock_name, depth, cut, qml_cfg.kernel_mode, qml_cfg.simulator),
        builder=lambda: build_cross_kernel(X_test_s, X_tr_s, qml_cfg, depth=depth, show_progress=show_kernel_progress),
    )

    svc = SVC(kernel="precomputed", C=C)
    svc.fit(K_train, y_train)

    scores_cal = svc.decision_function(K_cal)
    scores_test = svc.decision_function(K_test)

    if scores_cal.ndim == 1:
        scores_cal = scores_cal.reshape(-1, 1)
        scores_test = scores_test.reshape(-1, 1)

    cal = LogisticRegression(max_iter=2000, solver="lbfgs")
    cal.fit(scores_cal, y_cal)

    test_proba = cal.predict_proba(scores_test)
    test_pred = np.argmax(test_proba, axis=1)

    return test_pred, test_proba, cache


def quantum_kernel_train_loop(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    qml_cfg: QMLConfig,
    depth: int = 2,
    C: float = 1.0,
    model_tag: str = "QKernelSVC",
    cache: Optional[KernelCache] = None,
    stock_name: str = "",
    show_kernel_progress: bool = True,
) -> Dict[str, Any]:
    """Run OOF dev training + chrono calibration test evaluation for a single C."""
    t0 = time.time()
    splits = list(splits)

    print(f"[{model_tag}] DEV samples: {len(y_dev):,} | TEST samples: {len(y_test):,}")
    print(f"[{model_tag}] CV folds: {len(splits)} | depth={depth} | C={C}\n")
    print(f"[{model_tag}] Kernel mode: {qml_cfg.kernel_mode} | Simulator: {qml_cfg.simulator}")

    oof_pred, oof_proba, hist, cache = fit_predict_quantum_kernel_oof_calibrated(
        X=X_dev,
        y=y_dev,
        splits=splits,
        qml_cfg=qml_cfg,
        depth=depth,
        C=C,
        cache=cache,
        stock_name=stock_name,
        show_kernel_progress=show_kernel_progress,
    )

    test_pred, test_proba, cache = fit_quantum_kernel_and_predict_test_chrono_cal(
        X_dev=X_dev,
        y_dev=y_dev,
        X_test=X_test,
        qml_cfg=qml_cfg,
        depth=depth,
        C=C,
        cal_size=0.2,
        cache=cache,
        stock_name=stock_name,
        show_kernel_progress=show_kernel_progress,
    )

    test_f1 = f1_score(y_test, test_pred, average="macro")
    test_loss = log_loss(y_test, test_proba, labels=[0, 1, 2])
    runtime_sec = time.time() - t0

    results = {
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "hist": hist,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "test_f1": float(test_f1),
        "test_loss": float(test_loss),
        "cv_macro_f1_mean": float(np.mean(hist["macro_f1"])),
        "cv_macro_f1_std": float(np.std(hist["macro_f1"])),
        "cv_logloss_mean": float(np.mean(hist["log_loss"])),
        "cv_logloss_std": float(np.std(hist["log_loss"])),
        "runtime_sec": float(runtime_sec),
        "model_tag": model_tag,
        "qml_depth": depth,
        "qml_C": C,
        "kernel_mode": qml_cfg.kernel_mode,
        "simulator": qml_cfg.simulator,
    }

    print(f"\n[{model_tag}] TEST macroF1: {test_f1:.4f}")
    print(f"[{model_tag}] TEST log loss: {test_loss:.4f}")
    print(f"[{model_tag}] Runtime: {runtime_sec:.1f}s")
    return results
