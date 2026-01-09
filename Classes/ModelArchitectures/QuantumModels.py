import time
import hashlib

import numpy as np
import pennylane as qml

from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


n_qubits = 3
dev = qml.device("lightning.qubit", wires=n_qubits)
'''
Need to dual boot with linux and insta;; ;ightening-gpu:
    * pip install pennylane-lightning-gpu
'''

def feature_map(x, depth=2):
    ''' Quantum feature map that embeds the classical feature vector into a quantum state.
    Args:
        x (array-like): Classical input features (length = n_qubits)
        depth (int): Number of repeated encoding + entanglement layers
    '''
    # Repeat encoding/entanglement to increase circuit expressivity
    for _ in range(depth):
        # Encode each feature as a single-qubit rotation
        # NOTE - ANGULAR ENCODING
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

        # NOTE - Lightweight entangled encoding
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])



@qml.qnode(dev)
def kernel_circuit(x1, x2, depth=2):
    ''' Quantum circuit computing the state overlap (fidelity) between two feature-embedded quantum states.
    Args:
        x1 (array-like): First input feature vector
        x2 (array-like): Second input feature vector
        depth (int): Feature-map depth
    Returns:
        probs (np.ndarray): Measurement probabilities over all basis states (|000> probability equals kernel value)
    '''

    # Prepare |ψ(x1)>
    feature_map(x1, depth=depth)
    # Apply inverse of feature map for x2: U(x2)†
    qml.adjoint(feature_map)(x2, depth=depth)
    # Measure probability of returning to |0...0>
    return qml.probs(wires=range(n_qubits))



class KernelCache:
    ''' Simple in-memory cache for storing precomputed kernel matrices.
    Used to avoid recomputing expensive quantum kernels across:
      - multiple C values
      - repeated folds
    '''
    def __init__(self):
        self._cache = {}

    def get(self, key):
        ''' Retrieve cached object.
        Args:
            key (hashable): Unique identifier for kernel matrix
        Returns:
            Cached value or None if not found
        '''
        return self._cache.get(key, None)

    def set(self, key, value):
        ''' Store kernel matrix in cache.
        Args:
            key (hashable): Cache key
            value (np.ndarray): Kernel matrix
        '''
        self._cache[key] = value

    def clear(self):
        '''Clear all cached kernels to free memory.'''
        self._cache.clear()



def quantum_kernel(X1, X2, depth=2, show_progress=False):
    ''' Compute a (non-symmetric) quantum kernel matrix.
    Args:
        X1 (np.ndarray): Left feature matrix, shape (N1, d)
        X2 (np.ndarray): Right feature matrix, shape (N2, d)
        depth (int): Feature-map depth
        show_progress (bool): Whether to display tqdm progress bar
    Returns:
        K (np.ndarray): Kernel matrix of shape (N1, N2)
    '''

    # Allocate kernel matrix
    K = np.zeros((len(X1), len(X2)), dtype=np.float32)

    # Optional progress bar over rows
    iterator = enumerate(X1)
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=len(X1), desc="Quantum kernel rows", leave=False)

    # Compute kernel entries pairwise
    for i, x1 in iterator:
        for j, x2 in enumerate(X2):
            probs = kernel_circuit(x1, x2, depth=depth)
            K[i, j] = probs[0]  # fidelity |⟨ψ(x1)|ψ(x2)⟩|²

    return K




def quantum_kernel_symmetric(X, depth=2, show_progress=False):
    ''' Compute a symmetric quantum kernel matrix efficiently.
    Args:
        X (np.ndarray): Feature matrix, shape (N, d)
        depth (int): Feature-map depth
        show_progress (bool): Whether to display tqdm progress bar
    Returns:
        K (np.ndarray): Symmetric kernel matrix, shape (N, N)
    '''
    n = len(X)
    K = np.zeros((n, n), dtype=np.float32)

    # Optional progress bar
    it = range(n)
    if show_progress:
        from tqdm import tqdm
        it = tqdm(it, desc="Quantum kernel (symmetric)", leave=False)

    for i in it:
        K[i, i] = 1.0  # exact self-overlap
        for j in range(i + 1, n):
            probs = kernel_circuit(X[i], X[j], depth=depth)
            val = float(probs[0])
            K[i, j] = val
            K[j, i] = val  # exploit symmetry

    return K


def get_cached_kernel_fold(cache, key, builder):
    ''' Retrieve a kernel matrix from cache or compute and store it.
    Args:
        cache (KernelCache): Cache instance
        key (tuple): Unique identifier (e.g. stock, fold, depth)
        builder (callable): Zero-argument function that computes the kernel
    Returns:
        K (np.ndarray): Kernel matrix
    '''
    K = cache.get(key)
    if K is None:
        K = builder()
        cache.set(key, K)
    return K



def quantum_kernel_train_loop(X_dev, y_dev, X_test, y_test, splits, depth=2, C=1.0, model_tag="QKernelSVC", cache=None, stock_name=""):
    ''' End-to-end training loop for a quantum-kernel SVC:
        Walk-forward OOF training on dev set
        Chronologically calibrated evaluation on test set
        Aggregates economic- and ML-relevant metrics
    Args:
        X_dev (np.ndarray): Development features (time-ordered)
        y_dev (np.ndarray): Development labels
        X_test (np.ndarray): Test features (strictly future of dev)
        y_test (np.ndarray): Test classes (strictly future of dev)
        depth (int): Quantum feature map depth
        C (float): SVC regularization parameter
        cal_size (float): Fraction of dev reserved for calibration (0 < cal_size < 1)
        cache (KernelCache): Cache for storing/reusing kernel matrices
        stock_name (str): Identifier to prevent cache collisions across assets
    '''
    t0 = time.time()
    print(f"[{model_tag}] DEV samples: {len(y_dev):,} | TEST samples: {len(y_test):,}")
    print(f"[{model_tag}] CV folds: {len(splits)} | depth={depth} | C={C}\n")

    # NOTE - OOF Training
    oof_pred, oof_proba, hist, cache = fit_predict_quantum_kernel_oof_calibrated(X=X_dev, y=y_dev, splits=splits, depth=depth, C=C, cache=cache, stock_name=stock_name)
    
    # NOTE - Test
    # Final model evaluation on test set with chronological calibration (train --> calibrate --> deploy)
    test_pred, test_proba, cache = fit_quantum_kernel_and_predict_test_chrono_cal(
        X_dev=X_dev, y_dev=y_dev, X_test=X_test,
        depth=depth, C=C,
        cal_size=0.2,
        cache=cache,
        stock_name=stock_name
    )
    # Test set metrics
    test_f1 = f1_score(y_test, test_pred, average="macro")
    test_loss = log_loss(y_test, test_proba, labels=[0, 1, 2])
    runtime_sec = time.time() - t0

    # Collect results for downstream analysis / backtesting
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
    }

    print(f"\n[{model_tag}] TEST macroF1: {test_f1:.4f}")
    print(f"[{model_tag}] TEST log loss: {test_loss:.4f}")
    print(f"[{model_tag}] Runtime: {runtime_sec:.1f}s")
    return results



def fit_predict_quantum_kernel_oof_calibrated(X, y, splits, depth=2, C=1.0, cache=None, stock_name=""):
    ''' Walk-forward out-of-fold (OOF) training for a quantum-kernel SVC with fold-safe probability calibration.
    For each time-series split:
      - Compute quantum kernel Gram matrices (with caching)
      - Train an SVC on the training fold
      - Calibrate decision scores -> probabilities on the validation fold
      - Store OOF probabilities and predictions aligned to original indices

    Args:
        X (np.ndarray): Feature matrix (time-ordered)
        y (np.ndarray): Target labels
        splits (iterable): Walk-forward splits yielding (train_idx, val_idx)
        depth (int): Depth of the quantum feature map
        C (float): SVC regularization parameter
        cache (KernelCache): Cache for storing/reusing kernel matrices
        stock_name (str): Identifier to avoid cache collisions across assets

    Returns:
        oof_pred (np.ndarray): OOF class predictions, shape (T,)
        oof_proba (np.ndarray): OOF class probabilities, shape (T, n_classes)
        hist (dict): Per-fold performance metrics
        cache (KernelCache): Updated kernel cache
    '''
    # Initialize cache if not provided (allows reuse across C values)
    if cache is None:
        cache = KernelCache()

    # Identify class labels and dimensionality
    classes = np.array(sorted(np.unique(y)))
    n_classes = len(classes)

    # Preallocate OOF containers (NaNs ensure alignment sanity)
    oof_proba = np.full((len(X), n_classes), np.nan)
    oof_pred = np.full(len(X), np.nan)

    # Store fold-level metrics for diagnostics
    hist = {
        "fold": [],
        "macro_f1": [],
        "log_loss": []
    }

    # Walk-forward cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(tqdm(splits, desc="QML Walk-Forward CV", total=len(splits)), 1):
        # Split raw data (strict temporal ordering)
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        
        # Fold-safe scaling:
        #   - Fit scaler on training fold only
        #   - Apply to validation fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        
        # Quantum kernel computation with caching
        # K_train:
        #   - Symmetric Gram matrix (train vs train)
        #   - Cached per (stock, fold, depth)
        K_train = get_cached_kernel_fold(
            cache,
            key=("K_train", stock_name, fold, depth),
            builder=lambda: quantum_kernel_symmetric(
                X_train,
                depth=depth,
                show_progress=True
            )
        )
        # K_val:
        #   - Cross-kernel (val vs train)
        K_val = get_cached_kernel_fold(
            cache,
            key=("K_val", stock_name, fold, depth),
            builder=lambda: quantum_kernel(
                X_val,
                X_train,
                depth=depth,
                show_progress=True
            )
        )

        # Train base SVC on precomputed quantum kernel
        svc = SVC(kernel="precomputed", C=C)
        svc.fit(K_train, y_train)

        # Obtain raw decision scores on validation set shape handling for binary vs multiclass
        scores_val = svc.decision_function(K_val)
        if scores_val.ndim == 1:
            scores_val = scores_val.reshape(-1, 1)

        
        # Fold-safe probability calibration
        # Logistic regression is used as a multinomial calibrator
        # mapping SVC decision scores --> calibrated probabilities.
        cal = LogisticRegression(
            multi_class="multinomial",
            max_iter=2000
        )
        cal.fit(scores_val, y_val)

        # Calibrated probabilities and predictions
        proba_val = cal.predict_proba(scores_val)
        pred_val = np.argmax(proba_val, axis=1)

        
        # Store OOF predictions aligned to original indices
        oof_proba[val_idx] = proba_val
        oof_pred[val_idx] = pred_val

        # Fold evaluation metrics
        f1 = f1_score(y_val, pred_val, average="macro")
        ll = log_loss(y_val, proba_val, labels=classes)

        hist["fold"].append(fold)
        hist["macro_f1"].append(float(f1))
        hist["log_loss"].append(float(ll))

        tqdm.write(
            f"Fold {fold}: macroF1={f1:.4f} ::: log_loss={ll:.4f}"
        )

    # Return OOF predictions, probabilities, metrics, and updated cache
    return oof_pred.astype(int), oof_proba, hist, cache




def fit_quantum_kernel_and_predict_test_chrono_cal(X_dev:np.ndarray, y_dev:np.ndarray, X_test:np.ndarray, depth:int=2, C:float=1.0, cal_size:float=0.2, cache=None, stock_name:str=""):
    ''' Train a quantum-kernel SVC on the early portion of the development set,
        calibrate probabilities on a later (chronologically subsequent) slice and evaluate on the test set.
        This avoids in sample calibration optimism by enforcing the
        time ordering:
            train --> calibrate --> deploy
    Args:
        X_dev (np.ndarray): Development features (time-ordered)
        y_dev (np.ndarray): Development labels
        X_test (np.ndarray): Test features (strictly future of dev)
        depth (int): Quantum feature map depth
        C (float): SVC regularization parameter
        cal_size (float): Fraction of dev reserved for calibration (0 < cal_size < 1)
        cache (KernelCache): Cache for storing/reusing kernel matrices
        stock_name (str): Identifier to prevent cache collisions across assets
    Returns:
        test_pred (np.ndarray): Predicted test labels
        test_proba (np.ndarray): Calibrated test probabilities
        cache (KernelCache): Updated kernel cache
    '''
    # Initialize cache if not provided
    if cache is None:
        cache = KernelCache()

    # Determine chronological split point for calibration
    n = len(y_dev)
    cut = int(n * (1 - cal_size))

    # Ensure both training and calibration sets are non-empty
    if cut <= 0 or cut >= n:
        raise ValueError(
            "Invalid cal_size; must leave both train and calibration slices non-empty."
        )

    # Chronological split of development data
    # dev_train: early period (model fitting)
    # dev_cal: later period (probability calibration)
    X_train, y_train = X_dev[:cut], y_dev[:cut]
    X_cal, y_cal = X_dev[cut:], y_dev[cut:]

    
    # Fold-safe scaling
    # Scaler is fit ONLY on dev_train to prevent leakage and then applied to calibration and test sets.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    
    # Quantum kernel construction (with caching)
    # Symmetric Gram matrix for dev_train
    K_train = get_cached_kernel_fold(
        cache,
        key=("K_dev_train", stock_name, depth, cut),
        builder=lambda: quantum_kernel_symmetric(
            X_tr_s,
            depth=depth,
            show_progress=True
        )
    )
    # Cross-kernel between dev_cal and dev_train
    K_cal = get_cached_kernel_fold(
        cache,
        key=("K_dev_cal", stock_name, depth, cut),
        builder=lambda: quantum_kernel(
            X_cal_s,
            X_tr_s,
            depth=depth,
            show_progress=True
        )
    )
    # Cross-kernel between test and dev_train
    K_test = get_cached_kernel_fold(
        cache,
        key=("K_test", stock_name, depth, cut),
        builder=lambda: quantum_kernel(
            X_test_s,
            X_tr_s,
            depth=depth,
            show_progress=True
        )
    )
    
    # Train base SVC on dev_train only
    svc = SVC(kernel="precomputed", C=C)
    svc.fit(K_train, y_train)

    
    # Obtain decision scores for calibration and test sets
    scores_cal = svc.decision_function(K_cal)
    scores_test = svc.decision_function(K_test)

    # Handle binary vs multiclass output shapes
    if scores_cal.ndim == 1:
        scores_cal = scores_cal.reshape(-1, 1)
        scores_test = scores_test.reshape(-1, 1)

    
    # Chronological probability calibration
    # Logistic regression maps raw SVC scores to calibrated
    # class probabilities using ONLY dev_cal data.
    cal = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000
    )
    cal.fit(scores_cal, y_cal)

    # Calibrated test probabilities and predictions
    test_proba = cal.predict_proba(scores_test)
    test_pred = np.argmax(test_proba, axis=1)

    
    # Return calibrated test predictions and updated cache
    return test_pred, test_proba, cache




# def fit_quantum_kernel_and_predict_test(X_dev, y_dev, X_test, depth=2, C=1.0, cache=None, stock_name=""):
#     if cache is None:
#         cache = KernelCache()

#     scaler = StandardScaler()
#     X_dev_s = scaler.fit_transform(X_dev)
#     X_test_s = scaler.transform(X_test)

#     K_dev = get_cached_kernel_fold(
#         cache,
#         key=("K_dev", stock_name, depth),
#         builder=lambda: quantum_kernel_symmetric(X_dev_s, depth=depth, show_progress=True)
#     )

#     K_test = get_cached_kernel_fold(
#         cache,
#         key=("K_test", stock_name, depth),
#         builder=lambda: quantum_kernel(X_test_s, X_dev_s, depth=depth, show_progress=True)
#     )

#     svc = SVC(kernel="precomputed", C=C)
#     svc.fit(K_dev, y_dev)

#     scores_dev = svc.decision_function(K_dev)
#     scores_test = svc.decision_function(K_test)

#     if scores_dev.ndim == 1:
#         scores_dev = scores_dev.reshape(-1, 1)
#         scores_test = scores_test.reshape(-1, 1)

#     cal = LogisticRegression(multi_class="multinomial", max_iter=2000)
#     cal.fit(scores_dev, y_dev)

#     test_proba = cal.predict_proba(scores_test)
#     test_pred = np.argmax(test_proba, axis=1)

#     return test_pred, test_proba, cache
