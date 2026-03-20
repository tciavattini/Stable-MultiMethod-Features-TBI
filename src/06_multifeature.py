"""
4_multifeature_evaluation.py
============================
Multi-feature classification evaluation for the stable feature set.

Compares: All features vs Selected features vs Random features
across SVM, KNN, and Decision Tree classifiers.

CHANGES FROM NOTEBOOK VERSION:
  - Path updated to stability_analysis
  - Continuous features loaded from continuous_cols.txt (matches cross_validation.py)
  - RepeatedStratifiedKFold (10×10 = 100 folds) for stable estimates
  - Permutation test seed uses CONFIG random_state
  - Saves all outputs as CSV + publication-ready table + plot

Usage:
    python 4_multifeature_evaluation.py
"""

import os
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script mode
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, permutation_test

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG = {
    # Paths
    'stable_features_path': (SCRIPT_DIR / '../results/stability_analysis/stable_top_features.csv').resolve(),
    'data_dir': (SCRIPT_DIR / '../data').resolve(),
    'feature_names_path': (SCRIPT_DIR / '../data/feature_names.csv').resolve(),
    'continuous_cols_path': (SCRIPT_DIR / '../data/continuous_cols.txt').resolve(),
    'output_dir': (SCRIPT_DIR / '../results/multifeature_classification/').resolve(),

    # Evaluation parameters
    'n_folds': 10,
    'n_repeats': 10,            # NEW: repeated CV → 10×10 = 100 folds total
    'random_state': 42,
    'n_random_sets': 100,
    'n_bootstrap': 10000,
    'confidence_level': 0.95,

    # Feature counts to test
    'feature_counts': [5, 10, 15, 20, 25, 30],

    # Models with pre-specified hyperparameters (no inner CV tuning)
    # NOTE: Feature SELECTION used CV-tuned HPs (SVM C∈{0.01,0.1,1,10}, KNN k∈{1..19}).
    # Feature EVALUATION uses fixed HPs to avoid double-dipping.
    'models': {
        'svm': {'name': 'Linear SVM', 'C': 1.0},
        'knn': {'name': 'K-NN', 'n_neighbors': 5},
        'tree': {'name': 'Decision Tree', 'max_depth': 5},
    }
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
# Data helpers (matching cross_validation.py exactly)
# ============================================================
def load_continuous_idx(feature_names, continuous_cols_path):
    """Load continuous feature indices from file — matches cross_validation.py."""
    with open(continuous_cols_path) as f:
        cont_names = [line.strip() for line in f if line.strip()]
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    idx = sorted([name_to_idx[n] for n in cont_names if n in name_to_idx])
    missing = [n for n in cont_names if n not in name_to_idx]
    if missing:
        print(f"   Warning: {len(missing)} continuous feature names not found in feature_names.csv")
    return idx


def impute_missing(X_train, X_test, continuous_idx):
    """Impute missing values: median for continuous, mode for discrete.
    Fit on training data only to prevent leakage."""
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
    """Fit scaler on train set only."""
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
    """Apply scaler to continuous columns only."""
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
# Statistical functions
# ============================================================
def bootstrap_ci(values, confidence=0.95, n_bootstrap=10000, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) < 2:
        return {'mean': np.nan, 'std': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

    bootstrap_means = [np.mean(rng.choice(values, size=len(values), replace=True))
                       for _ in range(n_bootstrap)]

    alpha = 1 - confidence
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1),
        'ci_lower': np.percentile(bootstrap_means, 100 * alpha / 2),
        'ci_upper': np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    }


def format_ci(mean, ci_lower, ci_upper, decimals=3):
    """Format metric with CI."""
    return f"{mean:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"


def compare_paired(fold_values_a, fold_values_b, alpha=0.05):
    """Paired statistical comparison (works with any number of paired observations)."""
    a, b = np.array(fold_values_a), np.array(fold_values_b)
    diff = a - b

    t_stat, p_ttest = ttest_rel(a, b)

    try:
        _, p_wilcoxon = wilcoxon(a, b)
    except ValueError:
        p_wilcoxon = np.nan

    try:
        def stat_func(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        perm_result = permutation_test(
            (a, b), stat_func, n_resamples=9999,
            alternative='two-sided', random_state=CONFIG['random_state']
        )
        p_perm = perm_result.pvalue
    except Exception:
        p_perm = np.nan

    std_diff = np.std(diff, ddof=1)
    cohens_d = np.mean(diff) / std_diff if std_diff > 0 else np.nan

    return {
        'mean_diff': np.mean(diff),
        'cohens_d': cohens_d,
        't_stat': t_stat,
        'p_ttest': p_ttest,
        'p_wilcoxon': p_wilcoxon,
        'p_permutation': p_perm,
        'significant': p_ttest < alpha
    }


# ============================================================
# Evaluation functions
# ============================================================
def evaluate_feature_set(X, y, feature_indices, continuous_idx_full,
                         clf_func, clf_params, n_folds=10, n_repeats=10, seed=42):
    """Evaluate a feature set with repeated stratified k-fold cross-validation.

    Returns fold-level metrics (n_folds × n_repeats observations) and summary stats.
    """
    feature_indices = list(feature_indices)
    X_subset = X[:, feature_indices]

    # Remap continuous indices to subset positions
    continuous_idx_full_set = set(continuous_idx_full)
    continuous_idx_subset = [
        new_idx for new_idx, old_idx in enumerate(feature_indices)
        if old_idx in continuous_idx_full_set
    ]

    rkf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)

    fold_metrics = {k: [] for k in ['accuracy', 'auc', 'sensitivity', 'specificity',
                                     'acc_class0', 'acc_class1']}

    for train_idx, test_idx in rkf.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Impute and scale (train-only)
        X_train_imp, X_test_imp = impute_missing(X_train, X_test, continuous_idx_subset)

        if continuous_idx_subset:
            scaler = fit_scaler_on_train(X_train_imp, continuous_idx_subset)
            X_train_scaled = apply_scaler(X_train_imp, scaler, continuous_idx_subset)
            X_test_scaled = apply_scaler(X_test_imp, scaler, continuous_idx_subset)
        else:
            X_train_scaled, X_test_scaled = X_train_imp, X_test_imp

        # Train and predict
        clf = clf_func(**clf_params)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        # AUC: use decision_function for SVM, predict_proba for others
        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(clf, 'decision_function'):
            y_prob = clf.decision_function(X_test_scaled)
        else:
            y_prob = y_pred

        # Metrics
        fold_metrics['accuracy'].append(metrics.accuracy_score(y_test, y_pred))

        try:
            fold_metrics['auc'].append(metrics.roc_auc_score(y_test, y_prob))
        except Exception:
            fold_metrics['auc'].append(np.nan)

        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
        fold_metrics['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        fold_metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)

        mask0, mask1 = (y_test == 0), (y_test == 1)
        fold_metrics['acc_class0'].append(
            metrics.accuracy_score(y_test[mask0], y_pred[mask0]) if mask0.sum() > 0 else np.nan)
        fold_metrics['acc_class1'].append(
            metrics.accuracy_score(y_test[mask1], y_pred[mask1]) if mask1.sum() > 0 else np.nan)

    # Compute CIs from all fold-level observations
    results = {'fold_metrics': fold_metrics}
    for metric_name, values in fold_metrics.items():
        ci = bootstrap_ci(values, CONFIG['confidence_level'], CONFIG['n_bootstrap'], seed)
        results[f'{metric_name}_mean'] = ci['mean']
        results[f'{metric_name}_std'] = ci['std']
        results[f'{metric_name}_ci_lower'] = ci['ci_lower']
        results[f'{metric_name}_ci_upper'] = ci['ci_upper']

    return results


def evaluate_random_sets(X, y, n_features_to_select, continuous_idx_full,
                          clf_func, clf_params, n_random_sets=100,
                          n_folds=10, n_repeats=10, seed=42):
    """Evaluate multiple random feature sets (same CV scheme as selected)."""
    rng = np.random.RandomState(seed)
    n_total = X.shape[1]

    all_results = []
    for _ in range(n_random_sets):
        random_indices = rng.choice(n_total, size=n_features_to_select, replace=False)
        res = evaluate_feature_set(X, y, random_indices, continuous_idx_full,
                                    clf_func, clf_params, n_folds, n_repeats, seed)
        all_results.append(res)

    # Average per fold position (across random sets)
    n_total_folds = len(all_results[0]['fold_metrics']['accuracy'])
    averaged = {k: [] for k in ['accuracy', 'auc', 'sensitivity', 'specificity']}

    for fold_idx in range(n_total_folds):
        for metric in averaged.keys():
            values = [r['fold_metrics'][metric][fold_idx] for r in all_results]
            averaged[metric].append(np.nanmean(values))

    distribution = np.array([r['accuracy_mean'] for r in all_results])

    results = {
        'averaged_fold_metrics': averaged,
        'distribution': distribution,
        'distribution_mean': np.mean(distribution),
        'distribution_std': np.std(distribution, ddof=1),
        'distribution_min': np.min(distribution),
        'distribution_max': np.max(distribution),
    }

    for metric_name, values in averaged.items():
        ci = bootstrap_ci(values, CONFIG['confidence_level'], CONFIG['n_bootstrap'], seed)
        results[f'{metric_name}_mean'] = ci['mean']
        results[f'{metric_name}_std'] = ci['std']
        results[f'{metric_name}_ci_lower'] = ci['ci_lower']
        results[f'{metric_name}_ci_upper'] = ci['ci_upper']

    return results


# ============================================================
# Main
# ============================================================
def main():
    set_all_seeds(CONFIG['random_state'])
    output_dir = Path(CONFIG['output_dir'])

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    data_dir = Path(CONFIG['data_dir'])
    dataset_path = data_dir / "dataset.csv"
    labels_path = data_dir / "labels.csv"

    print(f"\n   Loading dataset from: {dataset_path}")

    with open(dataset_path, "r", newline="") as f:
        reader = csv.reader(f)
        data = [[float(x) if x not in ('nan', 'NaN', '') else np.nan for x in row]
                for row in reader]

    with open(labels_path, "r", newline="") as f:
        reader = csv.reader(f)
        labels = [int(row[0]) for row in reader]

    X_full = np.array(data, dtype=float)
    y_full = np.array(labels, dtype=int)

    # Convert labels {1, 3} → {0, 1} if needed
    if set(np.unique(y_full)) == {1, 3}:
        y_full = np.where(y_full == 1, 0, 1)

    n_samples, n_features = X_full.shape

    print(f"   Dataset: {n_samples} samples × {n_features} features")
    print(f"   Classes: {(y_full == 0).sum()} non-resp (0), {(y_full == 1).sum()} high-resp (1)")

    # Load feature names
    feature_names_path = Path(CONFIG['feature_names_path'])
    if feature_names_path.exists():
        feature_names = pd.read_csv(feature_names_path)['feature_name'].tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Load continuous feature indices from file (matches cross_validation.py)
    continuous_idx = load_continuous_idx(feature_names, CONFIG['continuous_cols_path'])
    print(f"   Continuous features (from continuous_cols.txt): {len(continuous_idx)}")

    # --------------------------------------------------
    # Load stable features
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("LOADING STABLE FEATURES")
    print("=" * 70)

    stable_path = Path(CONFIG['stable_features_path'])
    print(f"\n   Loading from: {stable_path}")

    if not stable_path.exists():
        raise FileNotFoundError(
            f"\n{'=' * 60}\n"
            f"ERROR: Stable features file not found!\n"
            f"Path: {stable_path}\n\n"
            f"Run stability_top30.py first to generate this file.\n"
            f"{'=' * 60}"
        )

    df_stable = pd.read_csv(stable_path)

    # Flexible column detection
    idx_col = 'feature_idx' if 'feature_idx' in df_stable.columns else 'feature_index'
    name_col = 'feature_name' if 'feature_name' in df_stable.columns else 'feature'

    selected_indices = df_stable[idx_col].values
    selected_names = df_stable[name_col].values
    n_selected = len(selected_indices)

    print(f"\n   Loaded {n_selected} stable features")
    print(f"\n   Top 5 features:")
    for i in range(min(5, n_selected)):
        print(f"      {i + 1}. {selected_names[i]} (idx={selected_indices[i]})")

    # --------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    X_selected = X_full[:, selected_indices]
    n_sel = len(selected_indices)
    corr_matrix = np.full((n_sel, n_sel), np.nan)

    for i in range(n_sel):
        for j in range(n_sel):
            col_i = X_selected[:, i]
            col_j = X_selected[:, j]
            mask = ~(np.isnan(col_i) | np.isnan(col_j))
            if mask.sum() > 2:
                corr_matrix[i, j] = np.corrcoef(col_i[mask], col_j[mask])[0, 1]

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            r = corr_matrix[i, j]
            if not np.isnan(r) and abs(r) > 0.7:
                high_corr_pairs.append({
                    'feature_1': selected_names[i],
                    'feature_2': selected_names[j],
                    'correlation': r
                })

    if high_corr_pairs:
        print(f"\n   Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.7):")
        for pair in high_corr_pairs[:5]:
            print(f"      {pair['feature_1']} <-> {pair['feature_2']}: r={pair['correlation']:.3f}")
        pd.DataFrame(high_corr_pairs).to_csv(output_dir / 'high_correlation_pairs.csv', index=False)
    else:
        print("\n   No highly correlated pairs found (all |r| ≤ 0.7)")

    df_corr_matrix = pd.DataFrame(corr_matrix, index=selected_names, columns=selected_names)
    df_corr_matrix.to_csv(output_dir / 'selected_features_correlation_matrix.csv')

    # --------------------------------------------------
    # Feature set comparison
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("FEATURE SET COMPARISON")
    print(f"   (10×10 Repeated Stratified K-Fold = 100 fold-level observations)")
    print("=" * 70)

    all_results = []
    comparison_results = []

    for model_name, model_config in CONFIG['models'].items():
        print(f"\n{'─' * 50}")
        print(f"Model: {model_config['name']}")
        print(f"{'─' * 50}")

        if model_name == 'svm':
            clf_func = SVC
            clf_params = {'kernel': 'linear', 'C': model_config['C'],
                          'random_state': CONFIG['random_state']}
        elif model_name == 'knn':
            clf_func = KNeighborsClassifier
            clf_params = {'n_neighbors': model_config['n_neighbors']}
        elif model_name == 'tree':
            clf_func = DecisionTreeClassifier
            clf_params = {'criterion': 'entropy', 'max_depth': model_config['max_depth'],
                          'random_state': CONFIG['random_state']}
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # A) All features
        print(f"\n   [A] All features ({n_features})...")
        res_all = evaluate_feature_set(
            X_full, y_full, range(n_features), continuous_idx,
            clf_func, clf_params, CONFIG['n_folds'], CONFIG['n_repeats'],
            CONFIG['random_state']
        )
        print(f"       Accuracy: {format_ci(res_all['accuracy_mean'], res_all['accuracy_ci_lower'], res_all['accuracy_ci_upper'])}")
        print(f"       AUC:      {format_ci(res_all['auc_mean'], res_all['auc_ci_lower'], res_all['auc_ci_upper'])}")

        all_results.append({
            'model': model_config['name'], 'feature_set': f'All ({n_features})',
            'n_features': n_features,
            **{k: res_all[k] for k in res_all if k != 'fold_metrics'}
        })

        # B) Selected features
        print(f"\n   [B] Selected features ({n_selected})...")
        res_selected = evaluate_feature_set(
            X_full, y_full, selected_indices, continuous_idx,
            clf_func, clf_params, CONFIG['n_folds'], CONFIG['n_repeats'],
            CONFIG['random_state']
        )
        print(f"       Accuracy: {format_ci(res_selected['accuracy_mean'], res_selected['accuracy_ci_lower'], res_selected['accuracy_ci_upper'])}")
        print(f"       AUC:      {format_ci(res_selected['auc_mean'], res_selected['auc_ci_lower'], res_selected['auc_ci_upper'])}")
        print(f"       Sens/Spec: {res_selected['sensitivity_mean']:.3f} / {res_selected['specificity_mean']:.3f}")

        all_results.append({
            'model': model_config['name'], 'feature_set': f'Selected ({n_selected})',
            'n_features': n_selected,
            **{k: res_selected[k] for k in res_selected if k != 'fold_metrics'}
        })

        # C) Random features
        print(f"\n   [C] Random features ({n_selected}) — averaging {CONFIG['n_random_sets']} sets...")
        res_random = evaluate_random_sets(
            X_full, y_full, n_selected, continuous_idx,
            clf_func, clf_params, CONFIG['n_random_sets'],
            CONFIG['n_folds'], CONFIG['n_repeats'], CONFIG['random_state']
        )
        print(f"       Accuracy: {format_ci(res_random['accuracy_mean'], res_random['accuracy_ci_lower'], res_random['accuracy_ci_upper'])}")
        print(f"       Distribution: {res_random['distribution_mean']:.3f} ± {res_random['distribution_std']:.3f}")

        all_results.append({
            'model': model_config['name'], 'feature_set': f'Random ({n_selected})*',
            'n_features': n_selected,
            **{k: res_random[k] for k in res_random
               if k not in ['fold_metrics', 'averaged_fold_metrics', 'distribution']}
        })

        # Statistical comparisons (100 paired observations from repeated CV)
        print(f"\n   STATISTICAL COMPARISONS (100 paired fold-level observations):")

        comp_all = compare_paired(
            res_selected['fold_metrics']['accuracy'],
            res_all['fold_metrics']['accuracy']
        )
        print(f"\n   ▸ Selected vs All:")
        print(f"       Δ accuracy: {comp_all['mean_diff']:+.3f}")
        print(f"       Cohen's d:  {comp_all['cohens_d']:.3f}")
        print(f"       Paired t-test:     p={comp_all['p_ttest']:.4f}")
        print(f"       Wilcoxon signed:   p={comp_all['p_wilcoxon']:.4f}")
        print(f"       Permutation test:  p={comp_all['p_permutation']:.4f}")

        comp_random = compare_paired(
            res_selected['fold_metrics']['accuracy'],
            res_random['averaged_fold_metrics']['accuracy']
        )
        print(f"\n   ▸ Selected vs Random:")
        print(f"       Δ accuracy: {comp_random['mean_diff']:+.3f}")
        print(f"       Cohen's d:  {comp_random['cohens_d']:.3f}")
        print(f"       Paired t-test:     p={comp_random['p_ttest']:.4f}")
        print(f"       Wilcoxon signed:   p={comp_random['p_wilcoxon']:.4f}")
        print(f"       Permutation test:  p={comp_random['p_permutation']:.4f}")

        percentile = (res_random['distribution'] < res_selected['accuracy_mean']).mean() * 100
        print(f"\n   ▸ Selected outperforms {percentile:.1f}% of random sets")

        comparison_results.extend([
            {'model': model_config['name'], 'comparison': 'Selected vs All', **comp_all},
            {'model': model_config['name'], 'comparison': 'Selected vs Random',
             **comp_random, 'percentile_rank': percentile}
        ])

    # --------------------------------------------------
    # Feature count analysis (SVM only — primary classifier)
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("FEATURE COUNT ANALYSIS (Linear SVM, C=1.0)")
    print("=" * 70)

    feature_counts = []
    clf_func = SVC
    clf_params = {'kernel': 'linear', 'C': 1.0, 'random_state': CONFIG['random_state']}

    for k in CONFIG['feature_counts']:
        if k > n_selected:
            continue

        print(f"   Top {k:2d} features...", end=' ')
        res = evaluate_feature_set(
            X_full, y_full, selected_indices[:k], continuous_idx,
            clf_func, clf_params, CONFIG['n_folds'], CONFIG['n_repeats'],
            CONFIG['random_state']
        )
        print(f"{format_ci(res['accuracy_mean'], res['accuracy_ci_lower'], res['accuracy_ci_upper'])}")

        feature_counts.append({
            'n_features': k,
            'accuracy_mean': res['accuracy_mean'],
            'accuracy_std': res['accuracy_std'],
            'accuracy_ci_lower': res['accuracy_ci_lower'],
            'accuracy_ci_upper': res['accuracy_ci_upper'],
            'auc_mean': res['auc_mean']
        })

    df_counts = pd.DataFrame(feature_counts)

    if len(df_counts) > 0:
        best = df_counts.loc[df_counts['accuracy_mean'].idxmax()]
        print(f"\n   Best: {best['n_features']:.0f} features at "
              f"{format_ci(best['accuracy_mean'], best['accuracy_ci_lower'], best['accuracy_ci_upper'])}")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_dir / 'feature_set_comparison.csv', index=False)

    df_comparisons = pd.DataFrame(comparison_results)
    df_comparisons.to_csv(output_dir / 'statistical_comparisons.csv', index=False)

    df_counts.to_csv(output_dir / 'feature_count_analysis.csv', index=False)

    # Publication table
    pub_rows = []
    for _, row in df_results.iterrows():
        pub_rows.append({
            'Model': row['model'],
            'Feature Set': row['feature_set'],
            'Accuracy (95% CI)': format_ci(row['accuracy_mean'], row['accuracy_ci_lower'], row['accuracy_ci_upper']),
            'AUC (95% CI)': format_ci(row['auc_mean'], row['auc_ci_lower'], row['auc_ci_upper']),
            'Sensitivity': f"{row['sensitivity_mean']:.3f}" if not np.isnan(row.get('sensitivity_mean', np.nan)) else '—',
            'Specificity': f"{row['specificity_mean']:.3f}" if not np.isnan(row.get('specificity_mean', np.nan)) else '—',
        })

    df_pub = pd.DataFrame(pub_rows)
    df_pub.to_csv(output_dir / 'publication_table.csv', index=False)

    print(f"\n   Saved to: {output_dir}")
    print(f"   - feature_set_comparison.csv")
    print(f"   - statistical_comparisons.csv")
    print(f"   - feature_count_analysis.csv")
    print(f"   - publication_table.csv")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    svm_sel = df_results[(df_results['model'] == 'Linear SVM') &
                          (df_results['feature_set'].str.startswith('Selected'))].iloc[0]
    svm_all = df_results[(df_results['model'] == 'Linear SVM') &
                          (df_results['feature_set'].str.startswith('All'))].iloc[0]
    svm_rnd = df_results[(df_results['model'] == 'Linear SVM') &
                          (df_results['feature_set'].str.startswith('Random'))].iloc[0]

    print(f"""
LINEAR SVM RESULTS (10×10 Repeated Stratified K-Fold):

All features ({n_features}):      {format_ci(svm_all['accuracy_mean'], svm_all['accuracy_ci_lower'], svm_all['accuracy_ci_upper'])}
Selected features ({n_selected}): {format_ci(svm_sel['accuracy_mean'], svm_sel['accuracy_ci_lower'], svm_sel['accuracy_ci_upper'])}
Random features ({n_selected}):   {format_ci(svm_rnd['accuracy_mean'], svm_rnd['accuracy_ci_lower'], svm_rnd['accuracy_ci_upper'])}

* Random baseline averaged over {CONFIG['n_random_sets']} random feature sets
* Paired tests based on 100 fold-level observations (10 repeats × 10 folds)
""")

    # --------------------------------------------------
    # Plot: Accuracy vs Number of Features
    # --------------------------------------------------
    if len(df_counts) > 0:
        df_plot = df_counts.sort_values("n_features")

        x = df_plot["n_features"].to_numpy()
        y = df_plot["accuracy_mean"].to_numpy()
        lo = df_plot["accuracy_ci_lower"].to_numpy()
        hi = df_plot["accuracy_ci_upper"].to_numpy()

        plt.figure(figsize=(6.5, 4.2))
        plt.plot(x, y, marker="o", linewidth=1.5, color='steelblue')
        plt.fill_between(x, lo, hi, alpha=0.2, color='steelblue')

        plt.xlabel("Number of top-ranked features", fontsize=11)
        plt.ylabel("Accuracy (10×10 repeated CV)", fontsize=11)
        plt.title("Accuracy vs Number of Selected Features", fontsize=12, fontweight='bold')

        plt.ylim(0.4, 0.9)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "feature_count_accuracy.png", dpi=300)
        plt.savefig(output_dir / "feature_count_accuracy.pdf")
        plt.close()

        print(f"Saved plot: {output_dir / 'feature_count_accuracy.png'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()