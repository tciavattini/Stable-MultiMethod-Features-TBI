"""
Evaluates classification performance of feature sets selected by
alternative methods (LASSO, Elastic Net, Random Forest, Combined),
mirroring the cross-validation framework of the main Vendrow pipeline.

Must be run AFTER alternative_feature_selection.py has produced
stable feature CSVs in ../results/alternative_feature_selection/

Output:
  - ../results/alternative_feature_selection/classification_performance.csv
  - ../results/alternative_feature_selection/classification_stats.csv
  - ../results/alternative_feature_selection/classification_summary.txt

"""

import os
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Configuration — mirror your main pipeline exactly
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / "../data").resolve()
DATASET_DIR = (SCRIPT_DIR / "../data").resolve()
ALT_FS_DIR = (SCRIPT_DIR / "../results/alternative_feature_selection").resolve()
VENDROW_DIR = (SCRIPT_DIR / "../results/stability_analysis").resolve()

# CV settings — MUST match cross_validation.py
N_SPLITS = 10
N_REPEATS = 10
N_FOLDS_TOTAL = N_SPLITS * N_REPEATS  # 100

# Bootstrap CI settings
N_BOOTSTRAP = 10_000
CI_LEVEL = 0.95

# Random baseline
N_RANDOM_SUBSETS = 100

# Fixed hyperparameters — same as cross_validation.py
SVM_C = 1.0
KNN_K = 5
DT_MAX_DEPTH = 5

SEED_BASE = 42

# ============================================================
# Seeding
# ============================================================
def set_all_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Data loading — identical to alternative_feature_selection.py
# ============================================================
def load_data():
    import csv
    dataset_path = DATA_DIR / "dataset.csv"
    labels_path = DATA_DIR / "labels.csv"

    with open(dataset_path, "r", newline="") as f:
        reader = csv.reader(f)
        data = [
            [float(x) if x not in ("nan", "NaN", "") else np.nan for x in row]
            for row in reader
        ]

    with open(labels_path, "r", newline="") as f:
        reader = csv.reader(f)
        labels = [int(row[0]) for row in reader]

    X = np.array(data, dtype=float)
    y = np.array(labels, dtype=int)

    if set(np.unique(y)) == {1, 3}:
        y = np.where(y == 1, 0, 1)

    return X, y


def load_feature_names():
    path = DATA_DIR / "feature_names.csv"
    return pd.read_csv(path)["feature_name"].astype(str).tolist()


def load_continuous_idx(feature_names):
    cont_path = DATASET_DIR / "continuous_cols.txt"
    with open(cont_path) as f:
        cont_names = [line.strip() for line in f if line.strip()]
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    return sorted([name_to_idx[n] for n in cont_names if n in name_to_idx])


# ============================================================
# Imputation — identical to alternative_feature_selection.py
# ============================================================
def impute_train_test(X_train, X_test, continuous_idx):
    from scipy.stats import mode as scipy_mode

    X_tr = X_train.copy()
    X_te = X_test.copy()
    cont_set = set(continuous_idx)

    for j in range(X_tr.shape[1]):
        col = X_tr[:, j]
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            fill = 0.0
        elif j in cont_set:
            fill = np.median(observed)
        else:
            fill = scipy_mode(observed, keepdims=True)[0][0]
        X_tr[np.isnan(X_tr[:, j]), j] = fill
        X_te[np.isnan(X_te[:, j]), j] = fill

    return X_tr, X_te


# ============================================================
# Load stable feature indices from CSV
# ============================================================
def load_stable_features(csv_path: Path) -> np.ndarray:
    """
    Returns array of feature indices sorted by rank (ascending = best).
    """
    df = pd.read_csv(csv_path)

    # ── Resolve index column ──────────────────────────────────────────
    idx_candidates = ["feature_idx", "feature_index", "idx", "index", "Feature_idx"]
    idx_col = None
    for c in idx_candidates:
        if c in df.columns:
            idx_col = c
            break
    # Fallback: first integer-like column
    if idx_col is None:
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]):
                idx_col = c
                break
    if idx_col is None:
        raise KeyError(
            f"Cannot find feature index column in {csv_path.name}.\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── Resolve rank column ───────────────────────────────────────────
    rank_candidates = ["mean_rank", "consensus_rank", "rank",
                       "mean_consensus_rank", "Mean Consensus Rank", "mean_score"]
    rank_col = None
    for c in rank_candidates:
        if c in df.columns:
            rank_col = c
            break
    if rank_col is not None:
        df = df.sort_values(rank_col, ascending=True)
    # If no rank column: preserve original order (assumed already sorted)

    return df[idx_col].to_numpy(dtype=int)


# ============================================================
# Build classifiers (fixed hyperparameters)
# ============================================================
def get_classifiers():
    return {
        "LinearSVM": SVC(C=SVM_C, max_iter=5000, random_state=SEED_BASE, kernel='linear'),
        "kNN": KNeighborsClassifier(n_neighbors=KNN_K),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=DT_MAX_DEPTH, random_state=SEED_BASE, criterion='entropy'
        ),
    }


# ============================================================
# Single CV evaluation
# Returns 100 fold-level dicts: accuracy, auc, sens, spec
# ============================================================
def evaluate_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: np.ndarray,
    continuous_idx: list,
    clf_name: str,
    seed: int = SEED_BASE,
) -> list:
    """
    Run 10×10 repeated stratified CV on a given feature subset.

    Returns
    -------
    fold_results : list of dicts, length = N_FOLDS_TOTAL
        Each dict: {accuracy, auc, sensitivity, specificity}
    """
    # Map continuous_idx to the local column positions within feature_idx
    feat_set = set(feature_idx)
    local_cont_idx = [
        i for i, orig in enumerate(feature_idx) if orig in set(continuous_idx)
    ]

    # Subset features
    X_sub = X[:, feature_idx]

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed
    )

    fold_results = []

    for train_idx, test_idx in rskf.split(X_sub, y):
        X_tr_raw, X_te_raw = X_sub[train_idx], X_sub[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Impute (train statistics only)
        X_tr, X_te = impute_train_test(X_tr_raw, X_te_raw, local_cont_idx)

        # Scale continuous features (train statistics only)
        if local_cont_idx:
            scaler = StandardScaler()
            X_tr[:, local_cont_idx] = scaler.fit_transform(X_tr[:, local_cont_idx])
            X_te[:, local_cont_idx] = scaler.transform(X_te[:, local_cont_idx])

        # Build fresh classifier each fold (fixed hyperparams)
        clf_dict = get_classifiers()
        clf = clf_dict[clf_name]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        # AUC: use decision_function or predict_proba if available
        try:
            scores = clf.decision_function(X_te)
        except AttributeError:
            try:
                scores = clf.predict_proba(X_te)[:, 1]
            except AttributeError:
                scores = y_pred.astype(float)

        acc = accuracy_score(y_te, y_pred)
        try:
            auc = roc_auc_score(y_te, scores)
        except Exception:
            auc = np.nan

        # Sensitivity = recall for positive class (label=1)
        tp = np.sum((y_pred == 1) & (y_te == 1))
        fn = np.sum((y_pred == 0) & (y_te == 1))
        tn = np.sum((y_pred == 0) & (y_te == 0))
        fp = np.sum((y_pred == 1) & (y_te == 0))

        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        fold_results.append({
            "accuracy": acc,
            "auc": auc,
            "sensitivity": sens,
            "specificity": spec,
        })

    return fold_results


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0):
    """Percentile bootstrap CI for the mean."""
    rng = np.random.RandomState(seed)
    values = values[~np.isnan(values)]
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - CI_LEVEL) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return float(np.mean(values)), float(lo), float(hi)


# ============================================================
# Random baseline — same folds, random features
# ============================================================
def evaluate_random_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_features_select: int,
    n_total_features: int,
    continuous_idx: list,
    clf_name: str,
    n_random: int = N_RANDOM_SUBSETS,
    seed: int = SEED_BASE,
) -> list:
    """
    Evaluate n_random random subsets of size n_features_select.
    Returns averaged fold-level accuracy list (length = N_FOLDS_TOTAL).
    Mirrors the same folds by using the same CV seed.
    """
    # Collect fold-level accuracy for each random subset
    all_acc = []  # shape: (n_random, N_FOLDS_TOTAL)

    rng = np.random.RandomState(seed)
    for i in range(n_random):
        rand_idx = rng.choice(n_total_features, size=n_features_select, replace=False)
        fold_res = evaluate_feature_set(
            X, y, rand_idx, continuous_idx, clf_name, seed=seed
        )
        all_acc.append([r["accuracy"] for r in fold_res])

    # Mean across random subsets for each fold position
    mean_acc_per_fold = np.array(all_acc).mean(axis=0)
    return mean_acc_per_fold.tolist()


# ============================================================
# Paired statistical tests
# ============================================================
def paired_tests(a: np.ndarray, b: np.ndarray, n_perm: int = 9999, seed: int = 0):
    """
    Paired t-test, Wilcoxon signed-rank, and permutation test.
    Returns dict: delta, cohen_d, p_ttest, p_wilcoxon, p_perm
    """
    diffs = a - b
    delta = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    cohen_d = delta / sd if sd > 0 else np.nan

    _, p_t = stats.ttest_rel(a, b)

    try:
        _, p_w = stats.wilcoxon(diffs)
    except Exception:
        p_w = np.nan

    # Permutation test: randomly flip signs of differences
    rng = np.random.RandomState(seed)
    obs_stat = np.abs(delta)
    perm_stats = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_stats.append(np.abs(np.mean(diffs * signs)))
    p_perm = float(np.mean(np.array(perm_stats) >= obs_stat))

    return {
        "delta": delta,
        "cohen_d": cohen_d,
        "p_ttest": float(p_t),
        "p_wilcoxon": float(p_w),
        "p_perm": p_perm,
    }


# ============================================================
# Main
# ============================================================
def main():
    set_all_seeds(SEED_BASE)
    ALT_FS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLASSIFICATION PERFORMANCE — ALTERNATIVE FEATURE SETS")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    X, y = load_data()
    feature_names = load_feature_names()
    continuous_idx = load_continuous_idx(feature_names)
    n_total = X.shape[1]

    print(f"  Dataset: {X.shape[0]} × {n_total}")
    print(f"  CV: {N_SPLITS}-fold × {N_REPEATS} repeats = {N_FOLDS_TOTAL} observations")

    # ── Load stable feature sets ───────────────────────────────────────────
    #
    # Expected CSVs from alternative_feature_selection.py:
    #   stable_features_LASSO.csv
    #   stable_features_ElasticNet.csv
    #   stable_features_RandomForest.csv
    #   stable_features_Combined_3method.csv
    #
    # Plus the Vendrow consensus from the main pipeline:
    #   ../results/stability_analysis/stable_top_features.csv

    feature_sets = {}

    # Alternative methods
    for method in ["LASSO", "ElasticNet", "RandomForest", "Combined_3method"]:
        csv_path = ALT_FS_DIR / f"stable_features_{method}.csv"
        if csv_path.exists():
            feature_sets[method] = load_stable_features(csv_path)
            print(f"  Loaded {method}: {len(feature_sets[method])} stable features")
        else:
            print(f"  WARNING: {csv_path} not found — skipping {method}")

    # Vendrow consensus
    vendrow_path = VENDROW_DIR / "stable_top_features.csv"
    if vendrow_path.exists():
        feature_sets["Vendrow"] = load_stable_features(vendrow_path)
        print(f"  Loaded Vendrow: {len(feature_sets['Vendrow'])} stable features")
    else:
        print(f"  WARNING: Vendrow stable features not found at {vendrow_path}")

    # Full feature set
    feature_sets["Full_149"] = np.arange(n_total)

    if not feature_sets:
        raise RuntimeError("No feature sets loaded. Check file paths.")

    # ── Evaluate each method × classifier ─────────────────────────────────
    print("\n" + "=" * 70)
    print("RUNNING CV EVALUATIONS")
    print("=" * 70)

    classifiers = list(get_classifiers().keys())

    # results[method][clf] = list of fold dicts
    results = {m: {} for m in feature_sets}

    for method, feat_idx in feature_sets.items():
        print(f"\n  ── {method} ({len(feat_idx)} features) ──")
        for clf_name in classifiers:
            print(f"    {clf_name} ...", end=" ", flush=True)
            fold_res = evaluate_feature_set(
                X, y, feat_idx, continuous_idx, clf_name, seed=SEED_BASE
            )
            results[method][clf_name] = fold_res
            mean_acc = np.mean([r["accuracy"] for r in fold_res])
            mean_auc = np.nanmean([r["auc"] for r in fold_res])
            print(f"acc={mean_acc:.3f}  auc={mean_auc:.3f}")

    # ── Random baseline (using same n as Vendrow for fair comparison) ──────
    n_select = len(feature_sets.get("Vendrow", feature_sets[list(feature_sets.keys())[0]]))

    print(f"\n  ── Random baseline ({n_select} features, {N_RANDOM_SUBSETS} subsets) ──")
    random_acc = {}
    for clf_name in classifiers:
        print(f"    {clf_name} ...", end=" ", flush=True)
        avg_acc = evaluate_random_baseline(
            X, y, n_select, n_total, continuous_idx, clf_name, seed=SEED_BASE
        )
        random_acc[clf_name] = avg_acc
        print(f"mean acc={np.mean(avg_acc):.3f}")

    # ── Compute summary statistics ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    perf_rows = []

    # Header
    print(f"  {'Method':<20} {'Classifier':<14} {'Acc':>6} {'[95% CI]':>16} "
          f"{'AUC':>6} {'Sens':>6} {'Spec':>6}")
    print("  " + "-" * 75)

    for method, clf_dict in results.items():
        for clf_name, fold_res in clf_dict.items():
            acc_vals = np.array([r["accuracy"] for r in fold_res])
            auc_vals = np.array([r["auc"] for r in fold_res])
            sens_vals = np.array([r["sensitivity"] for r in fold_res])
            spec_vals = np.array([r["specificity"] for r in fold_res])

            acc_m, acc_lo, acc_hi = bootstrap_ci(acc_vals)
            auc_m, auc_lo, auc_hi = bootstrap_ci(auc_vals)
            sens_m = float(np.nanmean(sens_vals))
            spec_m = float(np.nanmean(spec_vals))

            print(f"  {method:<20} {clf_name:<14} "
                  f"{acc_m:.3f} [{acc_lo:.3f},{acc_hi:.3f}] "
                  f"{auc_m:.3f} {sens_m:.3f} {spec_m:.3f}")

            perf_rows.append({
                "method": method,
                "classifier": clf_name,
                "n_features": len(feature_sets[method]),
                "accuracy": round(acc_m, 3),
                "acc_ci_lo": round(acc_lo, 3),
                "acc_ci_hi": round(acc_hi, 3),
                "auc": round(auc_m, 3),
                "auc_ci_lo": round(auc_lo, 3),
                "auc_ci_hi": round(auc_hi, 3),
                "sensitivity": round(sens_m, 3),
                "specificity": round(spec_m, 3),
            })

    # Add random baseline rows
    for clf_name in classifiers:
        acc_vals = np.array(random_acc[clf_name])
        acc_m, acc_lo, acc_hi = bootstrap_ci(acc_vals)
        print(f"  {'Random_baseline':<20} {clf_name:<14} "
              f"{acc_m:.3f} [{acc_lo:.3f},{acc_hi:.3f}]  ---   ---   ---")
        perf_rows.append({
            "method": "Random_baseline",
            "classifier": clf_name,
            "n_features": n_select,
            "accuracy": round(acc_m, 3),
            "acc_ci_lo": round(acc_lo, 3),
            "acc_ci_hi": round(acc_hi, 3),
            "auc": np.nan,
            "auc_ci_lo": np.nan,
            "auc_ci_hi": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
        })

    df_perf = pd.DataFrame(perf_rows)
    perf_path = ALT_FS_DIR / "classification_performance.csv"
    df_perf.to_csv(perf_path, index=False)
    print(f"\n  Saved: {perf_path}")

    # ── Paired statistical tests ───────────────────────────────────────────
    # Compare each alternative method against:
    #   (a) Full feature set
    #   (b) Random baseline
    #   (c) Vendrow consensus (if available)

    print("\n" + "=" * 70)
    print("PAIRED STATISTICAL COMPARISONS (fold-level accuracy)")
    print("=" * 70)

    stat_rows = []
    comparisons = []

    # Build comparison pairs
    alt_methods = [m for m in feature_sets if m not in ("Full_149",)]
    for method in alt_methods:
        comparisons.append((method, "Full_149"))
        comparisons.append((method, "Random_baseline"))
        if "Vendrow" in feature_sets and method != "Vendrow":
            comparisons.append((method, "Vendrow"))

    for (method_a, method_b) in comparisons:
        for clf_name in classifiers:
            # Get fold-level accuracy arrays
            if method_a == "Random_baseline":
                a_acc = np.array(random_acc[clf_name])
            else:
                a_acc = np.array([r["accuracy"] for r in results[method_a][clf_name]])

            if method_b == "Random_baseline":
                b_acc = np.array(random_acc[clf_name])
            else:
                b_acc = np.array([r["accuracy"] for r in results[method_b][clf_name]])

            # Ensure same length (should always be 100)
            min_len = min(len(a_acc), len(b_acc))
            test_res = paired_tests(a_acc[:min_len], b_acc[:min_len])

            row = {
                "method_A": method_a,
                "method_B": method_b,
                "classifier": clf_name,
                **test_res,
            }
            stat_rows.append(row)

            print(f"  {method_a} vs {method_b} | {clf_name}: "
                  f"Δ={test_res['delta']:+.3f}  d={test_res['cohen_d']:.2f}  "
                  f"p_t={test_res['p_ttest']:.2e}  p_perm={test_res['p_perm']:.4f}")

    df_stats = pd.DataFrame(stat_rows)
    stats_path = ALT_FS_DIR / "classification_stats.csv"
    df_stats.to_csv(stats_path, index=False)
    print(f"\n  Saved: {stats_path}")



if __name__ == "__main__":
    main()