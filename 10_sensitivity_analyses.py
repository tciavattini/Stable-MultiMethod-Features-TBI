"""
sensitivity_analyses.py
========================
Three sensitivity analyses for the Lyme disease ML paper:

1. STABILITY THRESHOLD SENSITIVITY (§ requested by reviewer)
   - Vary the feature selection threshold from 50% to 90%
   - Report: n features, classification performance at each threshold
   - Demonstrates robustness of the 70% choice

2. MISSINGNESS SENSITIVITY (§ requested by reviewer)
   - Exclude high-missingness features (>30%, >50%) from stable set
   - Re-evaluate classification performance
   - Tests whether imputed features drive the results

3. TREATMENT DURATION CONFOUNDER CHECK
   - Test if treatment duration (Q52_antib_duration) differs between groups
   - Mann-Whitney U test between high responders and non-responders
   - Provides evidence for/against confounding

Outputs:
    ../results/sensitivity_analyses/
      ├── threshold_sensitivity_performance.csv
      ├── threshold_sensitivity_plot.pdf/png
      ├── missingness_sensitivity.csv
      ├── treatment_duration_confounder.txt
      └── sensitivity_summary.txt

Usage:
    python sensitivity_analyses.py

Requires same data files as 4_multifeature.py
"""

import os
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
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG = {
    # Input paths — same as 4_multifeature.py
    'data_dir': (SCRIPT_DIR / '../data').resolve(),
    'feature_names_path': (SCRIPT_DIR / '../data/feature_names.csv').resolve(),
    'continuous_cols_path': (SCRIPT_DIR / '../data/continuous_cols.txt').resolve(),
    'output_dir': (SCRIPT_DIR / '../results/sensitivity_analyses/').resolve(),

    # Stability results directory (contains per-seed selection counts)
    'stability_dir': (SCRIPT_DIR / '../results/stability_analysis').resolve(),

    # Patient classification data (contains treatment duration)
    'df_binary_path': (SCRIPT_DIR / '../results/patient_classification/df_binary_percentile.pkl').resolve(),

    # Engineered dataframe (contains full feature set with missingness info)
    'df_ready_path': (SCRIPT_DIR / '../data/df_ready.pkl').resolve(),

    # CV settings — MUST match 4_multifeature.py
    'n_folds': 10,
    'n_repeats': 10,
    'random_state': 42,

    # Fixed hyperparameters — same as 4_multifeature.py
    'models': {
        'svm':  {'name': 'Linear SVM',    'func': SVC,                  'params': {'kernel': 'linear', 'C': 1.0, 'random_state': 42}},
        'knn':  {'name': 'k-NN',          'func': KNeighborsClassifier,  'params': {'n_neighbors': 5}},
        'tree': {'name': 'Decision Tree', 'func': DecisionTreeClassifier, 'params': {'max_depth': 5, 'random_state': 42}},
    },

    # Threshold sensitivity range
    'thresholds': [0.50, 0.60, 0.70, 0.80, 0.90],

    # Missingness thresholds
    'missingness_thresholds': [0.30, 0.50],

    # Total resampling iterations (from stability_top30.py)
    'R': 100,
}

Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)


# ============================================================
# Data loading
# ============================================================
def load_data():
    import csv
    data_dir = CONFIG['data_dir']
    with open(data_dir / "dataset.csv", "r", newline="") as f:
        data = [[float(x) if x not in ("nan", "NaN", "") else np.nan for x in row] for row in csv.reader(f)]
    with open(data_dir / "labels.csv", "r", newline="") as f:
        labels = [int(row[0]) for row in csv.reader(f)]
    X = np.array(data, dtype=float)
    y = np.array(labels, dtype=int)
    if set(np.unique(y)) == {1, 3}:
        y = np.where(y == 1, 0, 1)
    return X, y


def load_feature_names():
    return pd.read_csv(CONFIG['feature_names_path'])["feature_name"].astype(str).tolist()


def load_continuous_idx(feature_names):
    with open(CONFIG['continuous_cols_path']) as f:
        cont_names = [line.strip() for line in f if line.strip()]
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return sorted([name_to_idx[n] for n in cont_names if n in name_to_idx])


# ============================================================
# Imputation / scaling — identical to 4_multifeature.py
# ============================================================
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


# ============================================================
# Evaluation — simplified version of 4_multifeature.py
# ============================================================
def evaluate_feature_set(X, y, feature_indices, continuous_idx_full, model_config, seed=42):
    """
    10x10 repeated stratified CV. Returns dict with accuracy and AUC means + CIs.
    """
    feature_indices = list(feature_indices)
    if len(feature_indices) == 0:
        return {'accuracy_mean': np.nan, 'auc_mean': np.nan, 'n_features': 0}

    X_subset = X[:, feature_indices]

    cont_full_set = set(continuous_idx_full)
    cont_subset = [i for i, orig in enumerate(feature_indices) if orig in cont_full_set]

    rkf = RepeatedStratifiedKFold(
        n_splits=CONFIG['n_folds'], n_repeats=CONFIG['n_repeats'], random_state=seed
    )

    accuracies = []
    aucs = []

    clf_func = model_config['func']
    clf_params = model_config['params']

    for train_idx, test_idx in rkf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Impute
        X_train_imp, X_test_imp = impute_missing(X_train, X_test, cont_subset)

        # Scale
        if cont_subset:
            scaler = StandardScaler()
            X_train_imp[:, cont_subset] = scaler.fit_transform(X_train_imp[:, cont_subset])
            X_test_imp[:, cont_subset] = scaler.transform(X_test_imp[:, cont_subset])

        # Train and predict
        clf = clf_func(**clf_params)
        clf.fit(X_train_imp, y_train)
        y_pred = clf.predict(X_test_imp)

        accuracies.append(accuracy_score(y_test, y_pred))

        try:
            if hasattr(clf, 'predict_proba'):
                y_prob = clf.predict_proba(X_test_imp)[:, 1]
            elif hasattr(clf, 'decision_function'):
                y_prob = clf.decision_function(X_test_imp)
            else:
                y_prob = y_pred.astype(float)
            aucs.append(roc_auc_score(y_test, y_prob))
        except Exception:
            aucs.append(np.nan)

    accuracies = np.array(accuracies)
    aucs = np.array(aucs)

    return {
        'n_features': len(feature_indices),
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies, ddof=1),
        'accuracy_ci_lower': np.percentile(accuracies, 2.5),
        'accuracy_ci_upper': np.percentile(accuracies, 97.5),
        'auc_mean': np.nanmean(aucs),
        'auc_std': np.nanstd(aucs, ddof=1),
        'auc_ci_lower': np.nanpercentile(aucs, 2.5),
        'auc_ci_upper': np.nanpercentile(aucs, 97.5),
    }


# ============================================================
# ANALYSIS 1: Stability Threshold Sensitivity
# ============================================================
def stability_threshold_sensitivity(X, y, feature_names, continuous_idx):
    """
    For each threshold in [50%, 60%, 70%, 80%, 90%]:
      1. Identify stable features (selected in ≥ threshold% of 100 iterations)
      2. Evaluate classification performance with those features
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1: STABILITY THRESHOLD SENSITIVITY")
    print("=" * 70)

    stability_dir = CONFIG['stability_dir']
    R = CONFIG['R']

    # Load selection counts from per-seed top-k files
    n_features = len(feature_names)
    selection_counts = np.zeros(n_features, dtype=int)

    for seed in range(R):
        topk_path = stability_dir / f"seed_{seed:03d}" / "top30_idx.txt"
        if topk_path.exists():
            topk = np.loadtxt(topk_path, dtype=int)
            selection_counts[topk] += 1
        else:
            print(f"   WARNING: Missing {topk_path}")

    print(f"   Loaded selection counts from {R} seeds")
    print(f"   Max selection count: {selection_counts.max()}")

    # Also load mean consensus ranks for feature ordering
    stable_features_path = CONFIG['stability_dir'] / "stable_top_features.csv"
    if stable_features_path.exists():
        df_stable = pd.read_csv(stable_features_path)
        rank_col = 'mean_consensus_rank' if 'mean_consensus_rank' in df_stable.columns else None
    else:
        df_stable = None
        rank_col = None

    results = []

    for threshold in CONFIG['thresholds']:
        min_count = int(threshold * R)
        stable_idx = np.where(selection_counts >= min_count)[0]
        n_stable = len(stable_idx)

        stable_names = [feature_names[i] for i in stable_idx if i < len(feature_names)]

        print(f"\n   Threshold: {threshold:.0%} (≥{min_count} iterations)")
        print(f"   Stable features: {n_stable}")

        if n_stable == 0:
            print(f"   SKIPPED: No features meet this threshold")
            results.append({
                'threshold': threshold,
                'min_count': min_count,
                'n_stable': 0,
                'features': '',
            })
            continue

        # Evaluate with Linear SVM (primary classifier)
        model_config = CONFIG['models']['svm']
        res = evaluate_feature_set(X, y, stable_idx, continuous_idx, model_config, CONFIG['random_state'])

        print(f"   SVM Accuracy: {res['accuracy_mean']:.3f} [{res['accuracy_ci_lower']:.3f}, {res['accuracy_ci_upper']:.3f}]")
        print(f"   SVM AUC:      {res['auc_mean']:.3f} [{res['auc_ci_lower']:.3f}, {res['auc_ci_upper']:.3f}]")

        row = {
            'threshold': threshold,
            'min_count': min_count,
            'n_stable': n_stable,
            'features': '; '.join(stable_names[:10]) + ('...' if n_stable > 10 else ''),
            **{f'svm_{k}': v for k, v in res.items()},
        }

        # Also evaluate with k-NN
        model_config_knn = CONFIG['models']['knn']
        res_knn = evaluate_feature_set(X, y, stable_idx, continuous_idx, model_config_knn, CONFIG['random_state'])
        row.update({f'knn_{k}': v for k, v in res_knn.items()})
        print(f"   k-NN Accuracy: {res_knn['accuracy_mean']:.3f} [{res_knn['accuracy_ci_lower']:.3f}, {res_knn['accuracy_ci_upper']:.3f}]")

        results.append(row)

    df_results = pd.DataFrame(results)
    return df_results


def plot_threshold_sensitivity(df_results, output_dir):
    """Plot accuracy and AUC vs threshold."""
    df_plot = df_results[df_results['n_stable'] > 0].copy()
    if len(df_plot) == 0:
        print("   No data to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Number of features vs threshold
    ax = axes[0]
    ax.bar(df_plot['threshold'] * 100, df_plot['n_stable'], width=8, color='steelblue', alpha=0.8)
    ax.set_xlabel('Stability threshold (%)', fontsize=11)
    ax.set_ylabel('Number of stable features', fontsize=11)
    ax.set_title('A. Feature count', fontsize=12, fontweight='bold')
    ax.set_xticks([50, 60, 70, 80, 90])
    for i, row in df_plot.iterrows():
        ax.annotate(f"n={int(row['n_stable'])}", (row['threshold'] * 100, row['n_stable']),
                    textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)

    # Panel B: SVM Accuracy vs threshold
    ax = axes[1]
    x = df_plot['threshold'].values * 100
    y = df_plot['svm_accuracy_mean'].values
    lo = df_plot['svm_accuracy_ci_lower'].values
    hi = df_plot['svm_accuracy_ci_upper'].values

    ax.plot(x, y, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.fill_between(x, lo, hi, alpha=0.2, color='steelblue')
    ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Stability threshold (%)', fontsize=11)
    ax.set_ylabel('Accuracy (10×10 CV)', fontsize=11)
    ax.set_title('B. Linear SVM accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks([50, 60, 70, 80, 90])
    ax.set_ylim([0.45, 0.80])
    ax.grid(True, alpha=0.3)

    # Panel C: SVM AUC vs threshold
    ax = axes[2]
    y_auc = df_plot['svm_auc_mean'].values
    lo_auc = df_plot['svm_auc_ci_lower'].values
    hi_auc = df_plot['svm_auc_ci_upper'].values

    ax.plot(x, y_auc, 'o-', color='#d62728', linewidth=2, markersize=8)
    ax.fill_between(x, lo_auc, hi_auc, alpha=0.2, color='#d62728')
    ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Stability threshold (%)', fontsize=11)
    ax.set_ylabel('AUC (10×10 CV)', fontsize=11)
    ax.set_title('C. Linear SVM AUC', fontsize=12, fontweight='bold')
    ax.set_xticks([50, 60, 70, 80, 90])
    ax.set_ylim([0.45, 0.85])
    ax.grid(True, alpha=0.3)

    plt.suptitle('Sensitivity to Feature Selection Stability Threshold',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_sensitivity_plot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'threshold_sensitivity_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# ANALYSIS 2: Missingness Sensitivity
# ============================================================
def missingness_sensitivity(X, y, feature_names, continuous_idx):
    """
    Exclude features with >30% or >50% missingness from the stable set,
    then re-evaluate classification performance.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: MISSINGNESS SENSITIVITY")
    print("=" * 70)

    # Load stable features (70% threshold = primary analysis)
    stable_path = CONFIG['stability_dir'] / "stable_top_features.csv"
    df_stable = pd.read_csv(stable_path)
    stable_idx = df_stable['feature_idx'].to_numpy(dtype=int)
    stable_names = df_stable['feature_name'].tolist()

    # Compute missingness per feature in the classification cohort (N=141)
    n_samples = X.shape[0]
    missingness = np.isnan(X).mean(axis=0)

    print(f"   Stable features: {len(stable_idx)}")
    print(f"   Dataset: {n_samples} samples × {X.shape[1]} features")

    results = []

    # Baseline: all stable features
    model_config = CONFIG['models']['svm']
    res_baseline = evaluate_feature_set(X, y, stable_idx, continuous_idx, model_config, CONFIG['random_state'])
    results.append({
        'analysis': 'All stable features',
        'n_features': len(stable_idx),
        'excluded_features': '',
        **res_baseline,
    })
    print(f"\n   Baseline (all {len(stable_idx)} stable): Acc={res_baseline['accuracy_mean']:.3f}, AUC={res_baseline['auc_mean']:.3f}")

    # For each missingness threshold, exclude features above it
    for miss_thresh in CONFIG['missingness_thresholds']:
        excluded = []
        kept_idx = []

        for i, feat_idx in enumerate(stable_idx):
            feat_miss = missingness[feat_idx] * 100
            feat_name = stable_names[i] if i < len(stable_names) else f"idx_{feat_idx}"

            if feat_miss > miss_thresh * 100:
                excluded.append(f"{feat_name} ({feat_miss:.1f}%)")
            else:
                kept_idx.append(feat_idx)

        n_excluded = len(excluded)
        n_kept = len(kept_idx)

        print(f"\n   Threshold: >{miss_thresh:.0%} missingness")
        print(f"   Excluded: {n_excluded} features")
        if excluded:
            for e in excluded:
                print(f"      - {e}")
        print(f"   Remaining: {n_kept} features")

        if n_kept == 0:
            results.append({
                'analysis': f'Exclude >{miss_thresh:.0%} missing',
                'n_features': 0,
                'excluded_features': '; '.join(excluded),
                'accuracy_mean': np.nan, 'auc_mean': np.nan,
            })
            continue

        res = evaluate_feature_set(X, y, kept_idx, continuous_idx, model_config, CONFIG['random_state'])
        results.append({
            'analysis': f'Exclude >{miss_thresh:.0%} missing',
            'n_features': n_kept,
            'excluded_features': '; '.join(excluded),
            **res,
        })
        print(f"   SVM Accuracy: {res['accuracy_mean']:.3f} [{res['accuracy_ci_lower']:.3f}, {res['accuracy_ci_upper']:.3f}]")
        print(f"   SVM AUC:      {res['auc_mean']:.3f} [{res['auc_ci_lower']:.3f}, {res['auc_ci_upper']:.3f}]")

        # Compute delta from baseline
        delta_acc = res['accuracy_mean'] - res_baseline['accuracy_mean']
        delta_auc = res['auc_mean'] - res_baseline['auc_mean']
        print(f"   Δ Accuracy: {delta_acc:+.3f}")
        print(f"   Δ AUC:      {delta_auc:+.3f}")

    return pd.DataFrame(results)


# ============================================================
# ANALYSIS 3: Treatment Duration Confounder Check
# ============================================================
def treatment_duration_confounder():
    """
    Test if treatment duration (Q52_antib_duration) differs between
    high responders and non-responders.

    Also checks if treatment duration is among the 149 predictive features.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 3: TREATMENT DURATION CONFOUNDER CHECK")
    print("=" * 70)

    results_text = []
    results_text.append("TREATMENT DURATION CONFOUNDER ANALYSIS")
    results_text.append("=" * 50)

    # Try to load the binary outcome dataframe
    df_binary_path = CONFIG['df_binary_path']

    if not df_binary_path.exists():
        msg = f"Cannot find {df_binary_path}. Skipping treatment duration analysis."
        print(f"   {msg}")
        results_text.append(msg)
        return '\n'.join(results_text)

    df = pd.read_pickle(df_binary_path)
    print(f"   Loaded {len(df)} patients from {df_binary_path.name}")

    # Check if treatment duration column exists
    duration_col = None
    candidates = ['Q52_antib_duration', 'Q52antibduration', 'antib_duration', 'treatment_duration']
    for c in candidates:
        if c in df.columns:
            duration_col = c
            break

    if duration_col is None:
        # Search for partial match
        for c in df.columns:
            if 'duration' in c.lower() or 'q52' in c.lower():
                duration_col = c
                break

    if duration_col is None:
        msg = "Treatment duration column not found in dataset."
        print(f"   {msg}")
        print(f"   Available columns containing 'duration' or 'Q52':")
        for c in df.columns:
            if 'duration' in c.lower() or 'q52' in c.lower():
                print(f"      - {c}")
        results_text.append(msg)
        results_text.append("\nIMPLICATION: Treatment duration is not available in the dataset.")
        results_text.append("Add to limitations: 'Treatment duration (12-40 weeks; Xi et al. [18])")
        results_text.append("was not available as a variable and could not be controlled for.'")
        return '\n'.join(results_text)

    print(f"   Treatment duration column: {duration_col}")

    # Convert to numeric
    df[duration_col] = pd.to_numeric(df[duration_col], errors='coerce')

    # Split by outcome
    high = df[df['y'] == 1][duration_col].dropna()
    non = df[df['y'] == 0][duration_col].dropna()

    n_high_valid = len(high)
    n_non_valid = len(non)
    n_total_valid = n_high_valid + n_non_valid
    n_total = len(df)
    n_missing = n_total - n_total_valid

    print(f"\n   Treatment duration data:")
    print(f"   Total patients: {n_total}")
    print(f"   Valid values: {n_total_valid} ({100*n_total_valid/n_total:.1f}%)")
    print(f"   Missing: {n_missing} ({100*n_missing/n_total:.1f}%)")

    results_text.append(f"\nTreatment duration column: {duration_col}")
    results_text.append(f"Total patients: {n_total}")
    results_text.append(f"Valid values: {n_total_valid} ({100*n_total_valid/n_total:.1f}%)")
    results_text.append(f"Missing: {n_missing} ({100*n_missing/n_total:.1f}%)")

    if n_high_valid < 5 or n_non_valid < 5:
        msg = f"Insufficient data for comparison (high: {n_high_valid}, non: {n_non_valid})."
        print(f"   {msg}")
        results_text.append(f"\n{msg}")
        return '\n'.join(results_text)

    # Descriptive statistics
    print(f"\n   High responders (n={n_high_valid}):")
    print(f"      Mean ± SD: {high.mean():.1f} ± {high.std():.1f}")
    print(f"      Median [IQR]: {high.median():.1f} [{high.quantile(0.25):.1f}, {high.quantile(0.75):.1f}]")

    print(f"   Non-responders (n={n_non_valid}):")
    print(f"      Mean ± SD: {non.mean():.1f} ± {non.std():.1f}")
    print(f"      Median [IQR]: {non.median():.1f} [{non.quantile(0.25):.1f}, {non.quantile(0.75):.1f}]")

    results_text.append(f"\nDescriptive Statistics:")
    results_text.append(f"  High responders (n={n_high_valid}): {high.mean():.1f} ± {high.std():.1f} (mean ± SD)")
    results_text.append(f"  Non-responders (n={n_non_valid}): {non.mean():.1f} ± {non.std():.1f} (mean ± SD)")

    # Mann-Whitney U test
    stat, p_value = mannwhitneyu(high, non, alternative='two-sided')

    # Effect size: rank-biserial correlation
    n1, n2 = len(high), len(non)
    r_rb = 1 - (2 * stat) / (n1 * n2)

    print(f"\n   Mann-Whitney U test:")
    print(f"      U = {stat:.1f}")
    print(f"      p = {p_value:.4f}")
    print(f"      rank-biserial r = {r_rb:.3f}")

    results_text.append(f"\nMann-Whitney U test:")
    results_text.append(f"  U = {stat:.1f}, p = {p_value:.4f}")
    results_text.append(f"  rank-biserial r = {r_rb:.3f}")

    # Check if in the 149 features
    feature_names = load_feature_names()
    is_in_features = duration_col in feature_names

    results_text.append(f"\nIs treatment duration in the 149 predictive features? {'YES' if is_in_features else 'NO'}")

    if is_in_features:
        results_text.append("  → Partially controlled: included as a predictor")
    else:
        results_text.append("  → Not controlled: not included as a predictor")

    # Interpretation
    results_text.append(f"\n{'='*50}")
    results_text.append("INTERPRETATION FOR PAPER:")
    results_text.append(f"{'='*50}")

    if p_value >= 0.05:
        interpretation = (
            f"Treatment duration did not differ significantly between high responders "
            f"(mean {high.mean():.1f} ± {high.std():.1f}) and non-responders "
            f"(mean {non.mean():.1f} ± {non.std():.1f}; Mann-Whitney U = {stat:.1f}, "
            f"p = {p_value:.3f}), suggesting that treatment duration is unlikely to "
            f"confound the baseline discriminative signature."
        )
    else:
        interpretation = (
            f"Treatment duration differed significantly between high responders "
            f"(mean {high.mean():.1f} ± {high.std():.1f}) and non-responders "
            f"(mean {non.mean():.1f} ± {non.std():.1f}; Mann-Whitney U = {stat:.1f}, "
            f"p = {p_value:.3f}, rank-biserial r = {r_rb:.3f}). This represents a "
            f"potential confounder that should be acknowledged in the limitations."
        )

    print(f"\n   {interpretation}")
    results_text.append(interpretation)

    return '\n'.join(results_text)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("SENSITIVITY ANALYSES")
    print("=" * 70)

    output_dir = Path(CONFIG['output_dir'])

    # Load data
    print("\nLoading data...")
    X, y = load_data()
    feature_names = load_feature_names()
    continuous_idx = load_continuous_idx(feature_names)
    print(f"   Dataset: {X.shape}")

    # ── Analysis 1: Stability Threshold Sensitivity ──
    df_threshold = stability_threshold_sensitivity(X, y, feature_names, continuous_idx)
    df_threshold.to_csv(output_dir / 'threshold_sensitivity_performance.csv', index=False)
    plot_threshold_sensitivity(df_threshold, output_dir)

    # ── Analysis 2: Missingness Sensitivity ──
    df_missingness = missingness_sensitivity(X, y, feature_names, continuous_idx)
    df_missingness.to_csv(output_dir / 'missingness_sensitivity.csv', index=False)

    # ── Analysis 3: Treatment Duration Confounder ──
    duration_report = treatment_duration_confounder()
    with open(output_dir / 'treatment_duration_confounder.txt', 'w') as f:
        f.write(duration_report)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary_lines = [
        "SENSITIVITY ANALYSES SUMMARY",
        "=" * 50,
        "",
        "1. STABILITY THRESHOLD SENSITIVITY",
        "-" * 40,
    ]

    for _, row in df_threshold.iterrows():
        if row['n_stable'] > 0:
            acc = row.get('svm_accuracy_mean', np.nan)
            auc_val = row.get('svm_auc_mean', np.nan)
            summary_lines.append(
                f"   {row['threshold']:.0%}: {int(row['n_stable'])} features, "
                f"Acc={acc:.3f}, AUC={auc_val:.3f}"
            )
        else:
            summary_lines.append(f"   {row['threshold']:.0%}: 0 features (threshold too strict)")

    summary_lines.extend([
        "",
        "2. MISSINGNESS SENSITIVITY",
        "-" * 40,
    ])
    for _, row in df_missingness.iterrows():
        acc = row.get('accuracy_mean', np.nan)
        auc_val = row.get('auc_mean', np.nan)
        summary_lines.append(
            f"   {row['analysis']}: {int(row['n_features'])} features, "
            f"Acc={acc:.3f}, AUC={auc_val:.3f}"
        )

    summary_lines.extend([
        "",
        "3. TREATMENT DURATION CONFOUNDER",
        "-" * 40,
        "   See treatment_duration_confounder.txt for full results.",
    ])

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    with open(output_dir / 'sensitivity_summary.txt', 'w') as f:
        f.write(summary_text)

    print(f"\nAll outputs saved to: {output_dir}")
    print("   - threshold_sensitivity_performance.csv")
    print("   - threshold_sensitivity_plot.pdf/png")
    print("   - missingness_sensitivity.csv")
    print("   - treatment_duration_confounder.txt")
    print("   - sensitivity_summary.txt")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()