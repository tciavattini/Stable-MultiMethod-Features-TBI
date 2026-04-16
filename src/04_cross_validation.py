import os
import math
import random
import csv
from collections import Counter
import sys
import getopt
import numpy as np
from tqdm import trange
from copy import deepcopy

# sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from scipy import stats


from data_loading import *

from pathlib import Path
# ============================================================
# Device
# ============================================================
# --- device selection ---
FORCE_DEVICE = None  # will be set by CLI, if provided

def pick_device(clf_str, force_device=None):
    if force_device is not None:
        d = force_device.lower()
        if d == "cpu":
            return torch.device("cpu")
        if d == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise ValueError(f"Requested device={force_device} not available.")

    # default behavior: use CPU for net, otherwise fastest available
    if clf_str == "net":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ============================================================
# Continuous features (names must match feature_names.csv)
# ============================================================

SENTINEL = np.nan  # your missing-value marker

from pathlib import Path

# Use absolute path based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
CONTINUOUS_FEATURES_PATH = SCRIPT_DIR / "../dataset/continuous_cols.txt"

# Check if file exists, provide helpful error
if not CONTINUOUS_FEATURES_PATH.exists():
    # Try alternative location
    CONTINUOUS_FEATURES_PATH = SCRIPT_DIR.parent / "dataset" / "continuous_cols.txt"

if not CONTINUOUS_FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"continuous_cols.txt not found. Tried:\n"
        f"  {SCRIPT_DIR / '../dataset/continuous_cols.txt'}\n"
        f"  {SCRIPT_DIR.parent / 'dataset' / 'continuous_cols.txt'}"
    )

with open(CONTINUOUS_FEATURES_PATH, "r") as f:
    CONTINUOUS_FEATURES = [
        line.strip() for line in f
        if line.strip()
    ]

# ============================================================
# Seeding / determinism (best effort)
# ============================================================
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort determinism; note: MPS may still be nondeterministic
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # cuDNN flags only matter on CUDA
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Helpers: feature names + continuous indices
# ============================================================
def load_feature_names(path=None):
    import pandas as pd
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent / "data"

    if path is None:
        path = DATA_DIR / "feature_names.csv"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"feature_names.csv not found: {path}")

    df = pd.read_csv(path)
    return df["feature_name"].astype(str).tolist()

def build_continuous_idx(feature_names, continuous_features):
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    idx, missing = [], []
    for c in continuous_features:
        if c in name_to_idx:
            idx.append(name_to_idx[c])
        else:
            missing.append(c)
    return sorted(set(idx)), missing

def remap_indices_after_column_subset(old_indices, selected_columns):
    pos = {old_idx: new_pos for new_pos, old_idx in enumerate(selected_columns)}
    return sorted([pos[i] for i in old_indices if i in pos])


# ============================================================
# Imputation (median/mode, no missingness indicators)
# ============================================================
def impute_missing(X_train, X_test, continuous_idx, sentinel=SENTINEL):
    """
    Impute missing values:
    - Continuous features: median from training set
    - Discrete features: mode from training set
    - Fit on training data only to prevent leakage
    """
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    
    cont_set = set(continuous_idx)
    
    for j in range(X_train.shape[1]):
        train_col = X_train[:, j]
        observed = train_col[~np.isnan(train_col)]
        
        if len(observed) == 0:
            # All missing - fill with 0
            fill_value = 0.0
        elif j in cont_set:
            # Continuous: use median
            fill_value = np.median(observed)
        else:
            # Discrete: use mode
            fill_value = stats.mode(observed, keepdims=True)[0][0]
        
        X_train_imp[np.isnan(X_train_imp[:, j]), j] = fill_value
        X_test_imp[np.isnan(X_test_imp[:, j]), j] = fill_value
    
    return X_train_imp, X_test_imp


# ============================================================
# Leakage-safe scaling with sentinel handling
# ============================================================
def fit_scaler_on_train(X_train, continuous_idx, sentinel=SENTINEL):
    """
    Fit per-column mean/std on TRAIN only, ignoring sentinel values.
    Returns an object with mean_ and scale_ like StandardScaler.
    """
    if not continuous_idx:
        return None

    Xc = X_train[:, continuous_idx].astype(float)
    Xc_masked = Xc.copy()
    Xc_masked[np.isnan(Xc_masked)] = np.nan  # No-op, but clear intent

    means = np.nanmean(Xc_masked, axis=0)
    stds  = np.nanstd(Xc_masked, axis=0, ddof=0)

    # If a column is all-missing or constant, avoid division-by-zero
    means = np.where(np.isnan(means), 0.0, means)
    stds  = np.where((stds == 0) | np.isnan(stds), 1.0, stds)

    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.scale_ = stds
    scaler.var_ = stds ** 2
    scaler.n_features_in_ = len(continuous_idx)
    return scaler

def apply_scaler(X, scaler, continuous_idx, sentinel=SENTINEL):
    """
    Apply scaler to continuous columns only, leaving sentinel unchanged.
    """
    if scaler is None or not continuous_idx:
        return X

    X_out = X.copy().astype(float)
    Xc = X_out[:, continuous_idx]

    for k in range(Xc.shape[1]):
        col = Xc[:, k]
        mask = ~np.isnan(col)
        if np.any(mask):
            col[mask] = (col[mask] - scaler.mean_[k]) / scaler.scale_[k]
        Xc[:, k] = col

    X_out[:, continuous_idx] = Xc
    return X_out


# ============================================================
# Entropy gain on mixed data (discretize train-only)
# ============================================================
def fit_entropy_discretizers(X_train, continuous_idx, n_bins=5, sentinel=SENTINEL):
    """
    Fit discretizers on TRAIN only:
      - continuous columns -> quantile bin edges
      - non-continuous columns -> fixed mapping of unique observed values to integers
    """
    n_features = X_train.shape[1]
    cont_set = set(continuous_idx)

    # Continuous: quantile edges
    bin_edges = {}
    for j in continuous_idx:
        col = X_train[:, j].astype(float)
        obs = col[~np.isnan(col)]
        if obs.size == 0 or np.all(obs == obs[0]):
            bin_edges[j] = None
            continue
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        e = np.unique(np.quantile(obs, qs))
        bin_edges[j] = e if e.size > 0 else None

    # Discrete: value->code mapping
    value_maps = {}
    for j in range(n_features):
        if j in cont_set:
            continue
        col = X_train[:, j]
        obs = col[~np.isnan(col)]

        # If everything missing, map is empty
        if obs.size == 0:
            value_maps[j] = {}
            continue
        # Stable ordering: sort unique by numeric value
        uniq = np.unique(obs)
        # Reserve 0 for missing; observed start at 1
        value_maps[j] = {val: (i + 1) for i, val in enumerate(uniq.tolist())}

    return bin_edges, value_maps

def apply_entropy_discretizers(X, bin_edges, value_maps, continuous_idx, sentinel=SENTINEL):
    """
    Transform X -> int-coded matrix:
      - missing (sentinel) always 0
      - continuous -> bins {1..K} (missing 0)
      - discrete -> learned mapping {1..M} (unknown/unseen -> 0 by default)
    """
    n_samples, n_features = X.shape
    cont_set = set(continuous_idx)
    Xd = np.zeros((n_samples, n_features), dtype=int)

    for j in range(n_features):
        col = X[:, j]
        miss = np.isnan(col)

        if j in cont_set:
            e = bin_edges.get(j, None)
            out = np.zeros(n_samples, dtype=int)
            if e is None:
                out[~miss] = 1
            else:
                out[~miss] = 1 + np.digitize(col[~miss].astype(float), e, right=False)
            out[miss] = 0
            Xd[:, j] = out
        else:
            mp = value_maps.get(j, {})
            out = np.zeros(n_samples, dtype=int)
            for i in np.where(~miss)[0]:
                out[i] = mp.get(col[i], 0)  # unseen values -> 0
            Xd[:, j] = out

    return Xd

def calc_entropy_gain_discrete(Xd, y):
    """
    Entropy gain for discrete Xd (int-coded), y can be {0,1}.
    """
    def entropy(arr):
        arr = arr.astype(int)
        arr = arr - arr.min()
        counts = np.bincount(arr)
        probs = counts[counts > 0] / counts.sum()
        return stats.entropy(probs)

    base = entropy(y)
    gains = np.zeros(Xd.shape[1], dtype=float)

    for i in range(Xd.shape[1]):
        new_entropy = 0.0
        for xval in np.unique(Xd[:, i]):
            y_subset = y[Xd[:, i] == xval]
            new_entropy += entropy(y_subset) * (len(y_subset) / len(y))
        gains[i] = base - new_entropy

    return gains


# ============================================================
# Linear regression with missing value exclusion
# ============================================================
def lin_regression(X, y, sentinel=SENTINEL):
    """
    Compute R^2 for each feature, excluding missing values.
    """
    r2 = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        xj = X[:, j]
        mask = ~np.isnan(xj)
        
        if np.sum(mask) < 2:  # Need at least 2 points
            r2[j] = 0.0
            continue
        
        xj_obs = xj[mask]
        y_obs = y[mask]
        
        if np.all(xj_obs == xj_obs[0]):
            r2[j] = 0.0
            continue
        
        _, _, r_value, _, _ = stats.linregress(xj_obs, y_obs)
        r2[j] = r_value ** 2
    return r2


# ============================================================
# Neural net (your architecture), seed-controlled
# ============================================================
class Net(nn.Module):
    def __init__(self, m):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(m, 30)
        self.bn1 = nn.BatchNorm1d(30)
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(30, 30)
        self.bn2 = nn.BatchNorm1d(30)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        return x

def train_net(X_train, y_train, X_val, y_val, epochs=10, lr=0.005, seed=0):
    # Reproducible init/training per call (best effort)
    set_all_seeds(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long, device=device)

    net = Net(X_train.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))

    best_acc = -1.0
    best_state = None

    for _ in range(int(epochs)):
        net.train()
        optimizer.zero_grad()
        y_pred = net(X_train_t)
        loss_train = criterion(y_pred, y_train_t)
        loss_train.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            preds_val = torch.argmax(net(X_val_t), dim=1)
            val_acc = (preds_val == y_val_t).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            # keep ONLY the best weights (fast)
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)

    return best_acc, net

def cross_validation_net_scaled(X, y, continuous_idx, do_scale=True,
                               num_splits=5, epochs=35, lr=0.005, seed=0):
    # Shuffle folds with a fixed random_state -> less dependence on sample order
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

    total_acc = 0.0
    for fold_id, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if do_scale and continuous_idx:
            scaler = fit_scaler_on_train(X_train, continuous_idx)
            X_train = apply_scaler(X_train, scaler, continuous_idx)
            X_val   = apply_scaler(X_val,   scaler, continuous_idx)

        fold_seed = seed + 10_000 * (fold_id + 1)
        acc, _ = train_net(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, seed=fold_seed)
        total_acc += acc

    return total_acc / num_splits

def get_accuracy_net_scaled(X_train, y_train, X_test, y_test, continuous_idx,
                            do_scale=True, num_splits=5, epochs=35,
                            lrs=(0.005,), seed=0):
    # Hyperparameter selection by CV on training only
    best_lr = float(lrs[0])
    best_cv_acc = -1.0

    for lr in lrs:
        cv_acc = cross_validation_net_scaled(
            X_train, y_train, continuous_idx=continuous_idx,
            do_scale=do_scale, num_splits=num_splits,
            epochs=epochs, lr=float(lr), seed=seed
        )
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc
            best_lr = float(lr)

    # Train/val split for best epoch selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
    )

    if do_scale and continuous_idx:
        scaler = fit_scaler_on_train(X_tr, continuous_idx)
        X_tr   = apply_scaler(X_tr,   scaler, continuous_idx)
        X_val  = apply_scaler(X_val,  scaler, continuous_idx)
        X_test_use = apply_scaler(X_test, scaler, continuous_idx)
    else:
        X_test_use = X_test

    _, net = train_net(X_tr, y_tr, X_val, y_val, epochs=epochs, lr=best_lr, seed=seed + 777)

    X_test_t = torch.tensor(X_test_use, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    with torch.no_grad():
        preds_test = torch.argmax(net(X_test_t), dim=1)

    test_acc = (preds_test == y_test_t).float().mean().item()

    mask_0 = (y_test_t == 0)
    mask_1 = (y_test_t == 1)
    acc_0 = (preds_test[mask_0] == y_test_t[mask_0]).float().mean().item() if mask_0.any() else float("nan")
    acc_1 = (preds_test[mask_1] == y_test_t[mask_1]).float().mean().item() if mask_1.any() else float("nan")

    return test_acc, acc_0, acc_1, best_lr

def get_accuracy_net_scaled_fixed_lr(
    X_train, y_train, X_test, y_test,
    continuous_idx,
    do_scale=True,
    epochs=10,
    lr=0.005,
    seed=0
):
    """
    Fast path for single-feature NN scoring:
    - No LR grid search (fixed lr)
    - Shorter training (10 epochs)
    - Keeps leakage-safe scaling (fit on train only)
    - Uses a train/val split for epoch selection via early stopping
    """
    # Train/val split for best epoch selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
    )

    if do_scale and continuous_idx:
        scaler = fit_scaler_on_train(X_tr, continuous_idx)
        X_tr      = apply_scaler(X_tr,      scaler, continuous_idx)
        X_val     = apply_scaler(X_val,     scaler, continuous_idx)
        X_test_use= apply_scaler(X_test,    scaler, continuous_idx)
    else:
        X_test_use = X_test

    _, net = train_net(X_tr, y_tr, X_val, y_val, epochs=epochs, lr=float(lr), seed=seed + 777)

    X_test_t = torch.tensor(X_test_use, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    with torch.no_grad():
        preds_test = torch.argmax(net(X_test_t), dim=1)

    test_acc = (preds_test == y_test_t).float().mean().item()

    mask_0 = (y_test_t == 0)
    mask_1 = (y_test_t == 1)
    acc_0 = (preds_test[mask_0] == y_test_t[mask_0]).float().mean().item() if mask_0.any() else float("nan")
    acc_1 = (preds_test[mask_1] == y_test_t[mask_1]).float().mean().item() if mask_1.any() else float("nan")

    return test_acc, acc_0, acc_1

def net_single_scaled(
    X_train, y_train, X_test, y_test, continuous_idx,
    do_scale=True, num_splits=5, epochs=10, lrs=(0.005,), seed=0
):
    """
    Single-feature NN scoring (used by --single --clf net):
    - Fixed lr (first element of lrs) — no CV grid search
    - 10 epochs (sufficient for 1D classification)
    - Early stopping on held-out validation set
    
    NOTE: This is intentionally simpler than multi-feature NN evaluation
    for computational efficiency. With 1D input, the NN learns a nonlinear
    threshold; extensive HP tuning provides negligible ranking improvement.
    """
    cont_set = set(continuous_idx)
    accs = []

    fixed_lr = float(lrs[0]) if len(lrs) else 0.005

    for j in trange(X_train.shape[1]):
        Xtr = X_train[:, [j]]
        Xte = X_test[:,  [j]]

        is_cont = (j in cont_set)
        do_scale_j = bool(do_scale and is_cont)
        cont_idx = [0] if do_scale_j else []

        acc, _, _ = get_accuracy_net_scaled_fixed_lr(
            Xtr, y_train, Xte, y_test,
            continuous_idx=cont_idx,
            do_scale=do_scale_j,
            epochs=epochs,
            lr=fixed_lr,
            seed=seed + 100_000 + j
        )
        accs.append(acc)

    return np.asarray(accs, dtype=float)


# ============================================================
# Classic ML CV with leakage-safe scaling
# ============================================================
def cross_validation_scaled(clf_func, X, y, param, continuous_idx,
                            do_scale=False, num_splits=5, seed=0):
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    total_acc = 0.0

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        if do_scale and continuous_idx:
            scaler = fit_scaler_on_train(X_train, continuous_idx)
            X_train = apply_scaler(X_train, scaler, continuous_idx)
            X_val   = apply_scaler(X_val,   scaler, continuous_idx)

        clf = clf_func(param)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        total_acc += metrics.accuracy_score(y_val, y_pred)

    return total_acc / num_splits

def run_single_scaled(X_train, y_train, X_test, y_test, clf_func, params,
                      continuous_idx, do_scale=False, num_splits=5, seed=0, 
                      fixed_hp=False, return_best_params=False):
    """
    Single-feature scoring for svm/knn.
    Scaling is done leakage-safely: fit scaler on Xtr only, apply to Xte.
    """
    cont_set = set(continuous_idx)
    accs = []
    best_params_list = []

    for j in trange(X_train.shape[1]):
        Xtr = X_train[:, [j]]
        Xte = X_test[:,  [j]]

        is_cont = (j in cont_set)
        do_scale_j = bool(do_scale and is_cont)
        cont_idx = [0] if do_scale_j else []

        # hyperparameter selection by CV on Xtr only
       
        if fixed_hp:
            best_param = params[0] 
        else:
            best_param = params[0]
            best_cv = -1.0
            for p in params:
                cv = cross_validation_scaled(
                    clf_func, Xtr, y_train, p,
                    continuous_idx=cont_idx,
                    do_scale=do_scale_j,
                    num_splits=num_splits,
                    seed=seed + 50_000 + j
                )
                if cv > best_cv:
                    best_cv = cv
                    best_param = p
        best_params_list.append(best_param)
        # fit final model on full Xtr with optional scaling
        if do_scale_j:
            scaler = fit_scaler_on_train(Xtr, [0])
            Xtr_fit = apply_scaler(Xtr, scaler, [0])
            Xte_fit = apply_scaler(Xte, scaler, [0])
        else:
            Xtr_fit, Xte_fit = Xtr, Xte

        clf = clf_func(best_param)
        clf.fit(Xtr_fit, y_train)
        y_pred = clf.predict(Xte_fit)
        accs.append(metrics.accuracy_score(y_test, y_pred))

    if return_best_params:
        return np.asarray(accs, dtype=float), np.asarray(best_params_list)
    else:
        return np.asarray(accs, dtype=float)


# ============================================================
# CLI args
# ============================================================
def handle_args(argv):
    seed = 0
    num_shuffles = 10
    clf = ""
    params = ""
    single = False
    top = False
    bottom = False
    fixed_hp = False

    try:
        opts, _ = getopt.getopt(
            argv, "hc:p:stb",
            ["clf=", "params=", "single", "top", "bottom", "seed=", "device=", "num_shuffles=", "fixed_hp"]
        )
        device_arg = None
    except getopt.GetoptError:
        print("python3 cross_validation.py --clf <clf> --params <p1,p2,...> "
                "[--single --top --bottom --seed N --device cpu|mps|cuda --num_shuffles K --fixed_hp]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("python3 cross_validation.py --clf <clf> --params <p1,p2,...> "
                  "[--single --top --bottom --seed N --device cpu|mps|cuda --num_shuffles K --fixed_hp]")
            sys.exit()
        elif opt in ("-c", "--clf"):
            clf = arg
        elif opt in ("-p", "--params"):
            params = arg
        elif opt in ("-s", "--single"):
            single = True
        elif opt in ("-t", "--top"):
            top = True
        elif opt in ("-b", "--bottom"):
            bottom = True
        elif opt == "--seed":
            seed = int(arg)
        elif opt == "--device":
            device_arg = arg
        elif opt == "--num_shuffles":
            num_shuffles = int(arg)
        elif opt == "--fixed_hp":
            fixed_hp = True


    if clf == "svm":
        clf_func = lambda x, s=seed: svm.SVC(kernel="linear", C=float(x), random_state=s)
    elif clf == "knn":
        clf_func = lambda x: KNeighborsClassifier(n_neighbors=int(x))
    elif clf in ("entropy", "linreg", "net"):
        clf_func = None
    else:
        raise ValueError("Classifier options: svm, knn, entropy, linreg, net")

    # robust parsing: handles spaces and trailing commas
    if params.strip():
        params_list = [float(e.strip()) for e in params.split(",") if e.strip() != ""]
    else:
        params_list = []

    return clf_func, params_list, clf, single, top, bottom, seed, device_arg, num_shuffles, fixed_hp


# ============================================================
# Main (your pipeline)
# ============================================================
def main(argv):
    clf_func, params, clf_str, single, top, bottom, seed, device_arg, num_shuffles, fixed_hp = handle_args(argv)

    # Single place where we seed everything for this run
    set_all_seeds(seed)
    global device
    device = pick_device(clf_str, force_device=device_arg)
    print(f"Using device: {device}")

    # Load full X, y (binary=True returns original labels {1,3})
    X_full, y_full = get_data(subsamp=False, binary=True)
    n_features_full = X_full.shape[1]

    # feature names & continuous indices in full matrix
    feature_names = load_feature_names()
    continuous_idx_full, missing = build_continuous_idx(feature_names, CONTINUOUS_FEATURES)
    if missing:
        print("Warning: CONTINUOUS_FEATURES not found in feature_names.csv:")
        print(missing)

    # Top/bottom selection:
    # - if your matrix is not the paper's 215 features, you MUST supply your own top list
    if top or bottom:
        if n_features_full == 215:
            paper_top = [81, 44, 47, 53, 80, 48, 46, 177, 78, 55, 73, 70, 74, 116, 52, 72, 82, 75, 22, 105, 95, 127, 51, 106, 83, 108, 90, 100, 98, 89]
            top_features = [x - 1 for x in paper_top]
        else:
            top_path = "../results/feature_selection/top_features.txt"
            if not os.path.exists(top_path):
                raise FileNotFoundError(
                    f"--top/--bottom requested but X has {n_features_full} features (not 215). "
                    f"Provide 0-based indices in: {top_path}"
                )
            top_features = np.loadtxt(top_path, dtype=int).tolist()

        top_features = [int(i) for i in top_features if 0 <= int(i) < n_features_full]
        top_features = list(dict.fromkeys(top_features))  # unique, preserve order
        bottom_features = [i for i in range(n_features_full) if i not in set(top_features)]

        if top:
            selected = top_features
        else:
            selected = bottom_features

        X_full = X_full[:, selected]
        continuous_idx = remap_indices_after_column_subset(continuous_idx_full, selected)
    else:
        continuous_idx = continuous_idx_full

    # ============================================================
    # Convert labels {1, 3} -> {0, 1} WITHOUT subsampling
    # Classes are near-balanced (71 vs 70), no need to drop patients
    # ============================================================
    y_binary = np.where(y_full == 1, 0, 1).astype(int)
    print(f"Dataset: {X_full.shape[0]} patients, {X_full.shape[1]} features")
    print(f"Classes: {(y_binary==0).sum()} non-responders (0), {(y_binary==1).sum()} high-responders (1)")

    # Build the shuffled datasets (all patients each time, no subsampling)
    Xs, ys = [], []
    for i in range(num_shuffles):
        perm = np.random.RandomState(seed=seed + i).permutation(X_full.shape[0])
        Xs.append(np.asarray(X_full[perm]))
        ys.append(np.asarray(y_binary[perm]))

    # -----------------------------
    # ENTROPY (information gain)
    # -----------------------------
    if clf_str == "entropy":
        entropies = np.zeros(Xs[0].shape[1], dtype=float)

        for i in range(num_shuffles):
            X = Xs[i]; y = ys[i]

            # split seed depends on seed + i
            X_train, _, y_train, _ = train_test_split(
                X, y, test_size=0.2, random_state=seed + i, stratify=y
            )

            # discretizers fit on TRAIN only
            bin_edges, value_maps = fit_entropy_discretizers(
                X_train, continuous_idx=continuous_idx, n_bins=5, sentinel=SENTINEL
            )
            Xd_train = apply_entropy_discretizers(
                X_train, bin_edges, value_maps, continuous_idx=continuous_idx, sentinel=SENTINEL
            )

            ent = calc_entropy_gain_discrete(Xd_train, y_train)
            entropies = entropies * (i) / float(i + 1) + ent / float(i + 1)

        np.savetxt("entropies.txt", entropies, fmt="%10.5f")
        return

    # -----------------------------
    # LINREG (R^2 scoring)
    # -----------------------------
    if clf_str == "linreg":
        if single:
            r2_values = np.zeros(Xs[0].shape[1], dtype=float)
            for i in range(num_shuffles):
                r2_values = r2_values * (i) / float(i + 1) + lin_regression(Xs[i], ys[i], sentinel=SENTINEL) / float(i + 1)
            np.savetxt("r2.txt", r2_values, fmt="%10.5f")
        else:
            scores = []
            for i in range(num_shuffles):
                # For overall model: exclude rows with ANY missing values (casewise deletion)
                X = Xs[i]
                y = ys[i]
                
                # Find rows without any missing values
                complete_rows = np.all(X != SENTINEL, axis=1)
                X_complete = X[complete_rows]
                y_complete = y[complete_rows]
                
                if len(X_complete) < 2:
                    print(f"Warning: Shuffle {i} has only {len(X_complete)} complete cases. Skipping.")
                    continue
                
                model = LinearRegression().fit(X_complete, y_complete)
                scores.append(model.score(X_complete, y_complete))
            
            if len(scores) > 0:
                print(f"Average R² (on {len(scores)} shuffles): {sum(scores) / len(scores)}")
            else:
                print("No complete cases found across shuffles.")
        return

    # -----------------------------
    # NET
    # -----------------------------
    if clf_str == "net":
        if not params:
            params = [0.005]  # default lr candidates for Adam

        if single:
            accs = np.zeros(Xs[0].shape[1], dtype=float)
            for i in trange(num_shuffles):
                X = Xs[i]; y = ys[i]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed + i, stratify=y
                )
                
                # Impute missing values
                X_train_imp, X_test_imp = impute_missing(X_train, X_test, continuous_idx, sentinel=SENTINEL)
                
                # Single-feature scoring: fixed lr=0.005, 10 epochs, no inner CV
                a = net_single_scaled(
                    X_train_imp, y_train, X_test_imp, y_test,
                    continuous_idx=continuous_idx,
                    do_scale=True,
                    epochs=10,
                    lrs=(0.005,),
                    seed=seed + 10_000 * (i + 1)
                )
                accs = accs * (i) / float(i + 1) + a / float(i + 1)
            np.savetxt("net_single.txt", accs, fmt="%10.5f")
        else:
            accs = accs0 = accs1 = 0.0
            for i in trange(num_shuffles):
                X = Xs[i]; y = ys[i]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed + i, stratify=y
                )
                
                # Impute missing values
                X_train_imp, X_test_imp = impute_missing(X_train, X_test, continuous_idx, sentinel=SENTINEL)
                
                acc, a0, a1, _ = get_accuracy_net_scaled(
                    X_train_imp, y_train, X_test_imp, y_test,
                    continuous_idx=continuous_idx,
                    do_scale=True,
                    lrs=tuple(params),
                    seed=seed + 10_000 * (i + 1)
                )
                accs += acc; accs0 += a0; accs1 += a1
            print("Accuracy overall: ", accs / num_shuffles)
            print("Accuracy non:     ", accs0 / num_shuffles)
            print("Accuracy high:    ", accs1 / num_shuffles)
        return

    # -----------------------------
    # SVM / KNN
    # -----------------------------
    if clf_str in {"svm", "knn"}:
        if not params:
            raise ValueError(f"--params is required for clf={clf_str}")

        do_scale = (clf_str in {"svm", "knn"})

        if single:
            accs = np.zeros(Xs[0].shape[1], dtype=float)
            all_best_params = []
            
            for i in range(num_shuffles):
                X = Xs[i]; y = ys[i]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed + i, stratify=y
                )
                
                # Impute missing values
                X_train_imp, X_test_imp = impute_missing(X_train, X_test, continuous_idx, sentinel=SENTINEL)
                
                a, bp = run_single_scaled(
                    X_train_imp, y_train, X_test_imp, y_test,
                    clf_func=clf_func, params=params,
                    continuous_idx=continuous_idx,
                    do_scale=do_scale,
                    num_splits=5,
                    seed=seed + 10_000 * (i + 1),
                    fixed_hp=fixed_hp,
                    return_best_params=True
                )
                accs = accs * (i) / float(i + 1) + a / float(i + 1)
                all_best_params.append(bp)
            
            # Save accuracies
            np.savetxt(f"{clf_str}_single.txt", accs, fmt="%10.5f")
            
            # Save best params
            if all_best_params:
                all_best_params = np.vstack(all_best_params)
                from scipy.stats import mode
                mode_params = mode(all_best_params, axis=0, keepdims=False)[0]
                np.savetxt(f"{clf_str}_best_params.txt", mode_params, fmt="%.6f")

        else:
            accs = accs0 = accs1 = 0.0
            best_params = []
            for i in trange(num_shuffles):
                X = Xs[i]; y = ys[i]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=seed + i, stratify=y
                )
                
                # Impute missing values
                X_train_imp, X_test_imp = impute_missing(X_train, X_test, continuous_idx, sentinel=SENTINEL)
                
                # hyperparameter selection
                best_acc = -1.0
                best_p = params[0]
                for p in params:
                    cv = cross_validation_scaled(
                        clf_func, X_train_imp, y_train, p,
                        continuous_idx=continuous_idx,
                        do_scale=do_scale,
                        num_splits=5,
                        seed=seed + 10_000 * (i + 1)
                    )
                    if cv > best_acc:
                        best_acc = cv
                        best_p = p

                # final fit (with optional scaling fit on TRAIN only)
                if do_scale and continuous_idx:
                    scaler = fit_scaler_on_train(X_train_imp, continuous_idx)
                    Xtr_fit = apply_scaler(X_train_imp, scaler, continuous_idx)
                    Xte_fit = apply_scaler(X_test_imp,  scaler, continuous_idx)
                else:
                    Xtr_fit, Xte_fit = X_train_imp, X_test_imp

                clf = clf_func(best_p)
                clf.fit(Xtr_fit, y_train)
                y_pred = clf.predict(Xte_fit)

                accs += metrics.accuracy_score(y_test, y_pred)
                accs0 += metrics.accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
                accs1 += metrics.accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
                best_params.append(best_p)

            print("Accuracy overall: ", accs / num_shuffles)
            print("Accuracy non:     ", accs0 / num_shuffles)
            print("Accuracy high:    ", accs1 / num_shuffles)
            print(best_params)
        return

    raise ValueError(f"Unhandled clf: {clf_str}")


if __name__ == "__main__":
    main(sys.argv[1:])