"""
plot_roc_curves.py
==================
Generate per-classifier ROC curves with 95% confidence bands
using the 22 stable features and 10×10 Repeated Stratified K-Fold CV.

Produces a 3-panel figure (Linear SVM, k-NN, Decision Tree) matching
the publication-ready style.

Usage:
    python plot_roc_curves.py
"""

import os
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — adjust paths to match your project layout
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG = {
    'stable_features_path': (SCRIPT_DIR / '../results_fe/stability_top30_exp14/stable_top_features.csv').resolve(),
    'data_dir':             (SCRIPT_DIR / '../data').resolve(),
    'feature_names_path':   (SCRIPT_DIR / '../data/feature_names.csv').resolve(),
    'continuous_cols_path':  (SCRIPT_DIR / '../dataset/continuous_cols.txt').resolve(),
    'output_dir':           (SCRIPT_DIR / '../results/multi_feature_evaluation/').resolve(),

    'n_folds': 10,
    'n_repeats': 10,
    'random_state': 42,
    'confidence_level': 0.95,

    'models': {
        'svm':  {'name': 'Linear SVM', 'C': 1.0,  'color': '#4682B4'},  # steelblue
        'knn':  {'name': 'k-NN',       'n_neighbors': 5, 'color': '#E8740C'},  # orange
        'tree': {'name': 'Decision Tree', 'max_depth': 5, 'color': '#2E8B57'},  # sea green
    },
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)


# ============================================================
# Seeding
# ============================================================
def set_all_seeds(seed):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Data helpers (identical to 4_multifeature_evaluation.py)
# ============================================================
def load_continuous_idx(feature_names, continuous_cols_path):
    with open(continuous_cols_path) as f:
        cont_names = [line.strip() for line in f if line.strip()]
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return sorted([name_to_idx[n] for n in cont_names if n in name_to_idx])


def impute_missing(X_train, X_test, continuous_idx):
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    cont_set = set(continuous_idx)

    for j in range(X_train.shape[1]):
        train_col = X_train[:, j]
        observed = train_col[~np.isnan(train_col)]
        if len(observed) == 0:
            fill_value = 0.0
        elif j in cont_set:
            fill_value = np.median(observed)
        else:
            fill_value = stats.mode(observed, keepdims=True)[0][0]

        X_train_imp[np.isnan(X_train_imp[:, j]), j] = fill_value
        X_test_imp[np.isnan(X_test_imp[:, j]), j] = fill_value

    return X_train_imp, X_test_imp


def fit_scaler_on_train(X_train, continuous_idx):
    if not continuous_idx:
        return None
    Xc = X_train[:, continuous_idx].astype(float)
    means = np.nanmean(Xc, axis=0)
    stds = np.nanstd(Xc, axis=0, ddof=0)
    means = np.where(np.isnan(means), 0.0, means)
    stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds)

    scaler = StandardScaler()
    scaler.mean_ = means
    scaler.scale_ = stds
    scaler.var_ = stds ** 2
    scaler.n_features_in_ = len(continuous_idx)
    return scaler


def apply_scaler(X, scaler, continuous_idx):
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
# Collect ROC curves across folds
# ============================================================
def collect_roc_curves(X, y, feature_indices, continuous_idx_full,
                       clf_func, clf_params, n_folds=10, n_repeats=10, seed=42):
    """
    Run repeated stratified K-fold and collect interpolated ROC curves.

    Returns
    -------
    mean_fpr : ndarray (101,)  — common FPR grid [0, 0.01, ..., 1.0]
    all_tprs : list of ndarray — one interpolated TPR per fold
    all_aucs : list of float   — one AUC per fold
    """
    feature_indices = list(feature_indices)
    X_subset = X[:, feature_indices]

    cont_full_set = set(continuous_idx_full)
    cont_subset = [new for new, old in enumerate(feature_indices) if old in cont_full_set]

    rkf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)

    mean_fpr = np.linspace(0, 1, 101)
    all_tprs = []
    all_aucs = []

    for train_idx, test_idx in rkf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Impute
        X_train_imp, X_test_imp = impute_missing(X_train, X_test, cont_subset)

        # Scale
        if cont_subset:
            scaler = fit_scaler_on_train(X_train_imp, cont_subset)
            X_train_s = apply_scaler(X_train_imp, scaler, cont_subset)
            X_test_s = apply_scaler(X_test_imp, scaler, cont_subset)
        else:
            X_train_s, X_test_s = X_train_imp, X_test_imp

        # Train
        clf = clf_func(**clf_params)
        clf.fit(X_train_s, y_train)

        # Scores for ROC
        if hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(X_test_s)
        elif hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test_s)[:, 1]
        else:
            y_score = clf.predict(X_test_s).astype(float)

        # Compute ROC and interpolate onto common grid
        try:
            fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
            auc_val = metrics.auc(fpr, tpr)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0   # force start at origin

            all_tprs.append(interp_tpr)
            all_aucs.append(auc_val)
        except Exception:
            pass  # skip degenerate folds

    return mean_fpr, all_tprs, all_aucs


# ============================================================
# Bootstrap CI (same as 4_multifeature_evaluation.py)
# ============================================================
def bootstrap_ci(values, confidence=0.95, n_bootstrap=10000, seed=42):
    """Compute bootstrap confidence interval for the MEAN."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return {'mean': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

    bootstrap_means = [np.mean(rng.choice(values, size=len(values), replace=True))
                       for _ in range(n_bootstrap)]

    alpha = 1 - confidence
    return {
        'mean': np.mean(values),
        'ci_lower': np.percentile(bootstrap_means, 100 * alpha / 2),
        'ci_upper': np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    }


# ============================================================
# Plotting
# ============================================================
def plot_roc_panels(roc_data, n_features, output_dir):
    """
    Create a 3-panel ROC figure with mean curve + 95% CI band.

    Parameters
    ----------
    roc_data : dict  {model_key: (mean_fpr, all_tprs, all_aucs, color, name)}
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)

    for ax, (model_key, (mean_fpr, all_tprs, all_aucs, color, name)) in zip(axes, roc_data.items()):
        tprs = np.array(all_tprs)
        mean_tpr = tprs.mean(axis=0)
        mean_tpr[-1] = 1.0  # force end at (1,1)

        mean_auc = np.mean(all_aucs)

        # 95% CI for TPR band — bootstrap on the mean at each FPR point
        rng = np.random.RandomState(CONFIG['random_state'])
        n_boot = 10000
        n_folds_total = tprs.shape[0]
        alpha = 1 - CONFIG['confidence_level']

        boot_mean_tprs = np.zeros((n_boot, len(mean_fpr)))
        for b in range(n_boot):
            idx = rng.choice(n_folds_total, size=n_folds_total, replace=True)
            boot_mean_tprs[b] = tprs[idx].mean(axis=0)

        tpr_lower = np.percentile(boot_mean_tprs, 100 * alpha / 2, axis=0)
        tpr_upper = np.percentile(boot_mean_tprs, 100 * (1 - alpha / 2), axis=0)

        # AUC CI — same bootstrap method as 4_multifeature_evaluation.py
        auc_ci = bootstrap_ci(all_aucs, CONFIG['confidence_level'], 10000,
                              CONFIG['random_state'])

        # Plot
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2, label='Mean ROC')
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.18,
                        label='95% CI')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)

        ax.set_title(f"{name}\nAUC = {mean_auc:.3f} [{auc_ci['ci_lower']:.3f}, {auc_ci['ci_upper']:.3f}]",
                     fontsize=12, fontweight='bold')

        ax.grid(False)
        ax.set_aspect('equal')

    fig.suptitle(
        f"ROC Curves by Classifier ({n_features} Stable Features, "
        f"{CONFIG['n_repeats']}×{CONFIG['n_folds']} CV)",
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    out_png = output_dir / 'roc_curves_per_classifier.png'
    out_pdf = output_dir / 'roc_curves_per_classifier.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"   Saved: {out_png}")
    print(f"   Saved: {out_pdf}")


# ============================================================
# Main
# ============================================================
def main():
    set_all_seeds(CONFIG['random_state'])
    output_dir = Path(CONFIG['output_dir'])

    # ---- Load data ----
    print("Loading data...")
    data_dir = Path(CONFIG['data_dir'])

    with open(data_dir / "dataset.csv", "r", newline="") as f:
        reader = csv.reader(f)
        data = [[float(x) if x not in ('nan', 'NaN', '') else np.nan for x in row]
                for row in reader]

    with open(data_dir / "labels.csv", "r", newline="") as f:
        reader = csv.reader(f)
        labels = [int(row[0]) for row in reader]

    X = np.array(data, dtype=float)
    y = np.array(labels, dtype=int)

    if set(np.unique(y)) == {1, 3}:
        y = np.where(y == 1, 0, 1)

    n_samples, n_features_total = X.shape
    print(f"   Dataset: {n_samples} × {n_features_total}")

    # Feature names & continuous indices
    fn_path = Path(CONFIG['feature_names_path'])
    feature_names = pd.read_csv(fn_path)['feature_name'].tolist() if fn_path.exists() \
        else [f"feature_{i}" for i in range(n_features_total)]
    continuous_idx = load_continuous_idx(feature_names, CONFIG['continuous_cols_path'])

    # ---- Load stable features ----
    df_stable = pd.read_csv(CONFIG['stable_features_path'])
    idx_col = 'feature_idx' if 'feature_idx' in df_stable.columns else 'feature_index'
    selected_indices = df_stable[idx_col].values
    n_selected = len(selected_indices)
    print(f"   Stable features: {n_selected}")

    # ---- Collect ROC data per classifier ----
    roc_data = {}

    for model_key, mcfg in CONFIG['models'].items():
        print(f"\n   Collecting ROC for {mcfg['name']}...")

        if model_key == 'svm':
            clf_func = SVC
            clf_params = {'kernel': 'linear', 'C': mcfg['C'],
                          'random_state': CONFIG['random_state']}
        elif model_key == 'knn':
            clf_func = KNeighborsClassifier
            clf_params = {'n_neighbors': mcfg['n_neighbors']}
        elif model_key == 'tree':
            clf_func = DecisionTreeClassifier
            clf_params = {'criterion': 'entropy', 'max_depth': mcfg['max_depth'],
                          'random_state': CONFIG['random_state']}

        mean_fpr, all_tprs, all_aucs = collect_roc_curves(
            X, y, selected_indices, continuous_idx,
            clf_func, clf_params,
            CONFIG['n_folds'], CONFIG['n_repeats'], CONFIG['random_state']
        )

        mean_auc = np.mean(all_aucs)
        print(f"       Mean AUC = {mean_auc:.3f}  ({len(all_aucs)} folds)")

        roc_data[model_key] = (mean_fpr, all_tprs, all_aucs, mcfg['color'], mcfg['name'])

    # ---- Plot ----
    print("\nPlotting...")
    plot_roc_panels(roc_data, n_selected, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()