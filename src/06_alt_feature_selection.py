"""
LASSO, Elastic Net, and Random Forest feature selection with stability analysis.

Designed to run alongside the existing Vendrow consensus pipeline and produce
directly comparable results (same data, same resampling, same metrics).

Input:
    - ../data/dataset.csv        (same as cross_validation.py)
    - ../data/labels.csv
    - ../data/feature_names.csv
    - ../dataset/continuous_cols.txt

Output:
    - ../results/alternative_feature_selection/  (all results)
"""

import os
import random
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm, trange

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / "../data").resolve()
DATASET_DIR = (SCRIPT_DIR / "../data").resolve()
OUT_ROOT = (SCRIPT_DIR / "../results/alternative_feature_selection").resolve()

R = 100             # resampling iterations (same as stability_top30.py)
NUM_SHUFFLES = 10   # shuffles per iteration
TOPK = 30           # top features per iteration
MIN_FREQ = 0.70     # stability threshold
SEED_BASE = 0
N_JOBS = 6          # parallel jobs for Random Forest

# ============================================================
# Seeding
# ============================================================
def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Data loading
# ============================================================
def load_data():
    """Load dataset and labels, convert {1,3} → {0,1}."""
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

    # Convert {1, 3} → {0, 1}
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
# Imputation
# ============================================================
def impute_train_test(X_train, X_test, continuous_idx):
    """Impute missing values: median for continuous, mode for categorical.
    Fit on training only."""
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
# Feature selection methods
# ============================================================

def lasso_feature_importance(X_train, y_train, X_test, continuous_idx, seed=0):
    """
    LASSO logistic regression with CV-tuned regularization.
    Returns absolute coefficients as importance scores.
    """
    X_tr, X_te = impute_train_test(X_train, X_test, continuous_idx)

    # Scale continuous features (fit on train only)
    scaler = StandardScaler()
    X_tr_scaled = X_tr.copy()
    X_te_scaled = X_te.copy()
    if continuous_idx:
        X_tr_scaled[:, continuous_idx] = scaler.fit_transform(X_tr[:, continuous_idx])
        X_te_scaled[:, continuous_idx] = scaler.transform(X_te[:, continuous_idx])

    # Find best C via inner 5-fold CV
    best_c = 1.0
    best_score = -1.0
    for c in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = []
        for tr_idx, val_idx in kf.split(X_tr_scaled, y_train):
            model = LogisticRegression(
                penalty="l1", C=c, solver="saga", max_iter=5000,
                random_state=seed, tol=1e-4
            )
            model.fit(X_tr_scaled[tr_idx], y_train[tr_idx])
            scores.append(model.score(X_tr_scaled[val_idx], y_train[val_idx]))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_c = c

    # Fit final model with best C
    model = LogisticRegression(
        penalty="l1", C=best_c, solver="saga", max_iter=5000,
        random_state=seed, tol=1e-4
    )
    model.fit(X_tr_scaled, y_train)
    importance = np.abs(model.coef_[0])

    return importance, best_c


def elastic_net_feature_importance(X_train, y_train, X_test, continuous_idx, seed=0):
    """
    Elastic Net logistic regression (L1 + L2) with CV-tuned parameters.
    Returns absolute coefficients as importance scores.
    """
    X_tr, X_te = impute_train_test(X_train, X_test, continuous_idx)

    scaler = StandardScaler()
    X_tr_scaled = X_tr.copy()
    if continuous_idx:
        X_tr_scaled[:, continuous_idx] = scaler.fit_transform(X_tr[:, continuous_idx])

    # Grid: C controls overall regularization, l1_ratio controls L1 vs L2 mix
    best_c = 1.0
    best_l1 = 0.5
    best_score = -1.0

    for c in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            scores = []
            for tr_idx, val_idx in kf.split(X_tr_scaled, y_train):
                # SGDClassifier with log_loss = logistic regression
                # l1_ratio and alpha map to elastic net
                # alpha ≈ 1/C for SGD
                model = SGDClassifier(
                    loss="log_loss", penalty="elasticnet",
                    alpha=1.0 / c, l1_ratio=l1_ratio,
                    max_iter=5000, random_state=seed, tol=1e-4
                )
                model.fit(X_tr_scaled[tr_idx], y_train[tr_idx])
                scores.append(model.score(X_tr_scaled[val_idx], y_train[val_idx]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_c = c
                best_l1 = l1_ratio

    model = SGDClassifier(
        loss="log_loss", penalty="elasticnet",
        alpha=1.0 / best_c, l1_ratio=best_l1,
        max_iter=5000, random_state=seed, tol=1e-4
    )
    model.fit(X_tr_scaled, y_train)
    importance = np.abs(model.coef_[0])

    return importance, (best_c, best_l1)


def random_forest_feature_importance(X_train, y_train, X_test, continuous_idx, seed=0):
    """
    Random Forest with permutation-based feature importance.
    Uses out-of-bag score for model selection.
    """
    X_tr, X_te = impute_train_test(X_train, X_test, continuous_idx)

    # No scaling needed for RF (tree-based)
    # Tune n_estimators and max_depth via inner CV
    best_params = (200, None)
    best_score = -1.0

    for n_est in [100, 200, 500]:
        for max_d in [None, 5, 10]:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            scores = []
            for tr_idx, val_idx in kf.split(X_tr, y_train):
                model = RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_d,
                    random_state=seed, n_jobs=1, max_features="sqrt"
                )
                model.fit(X_tr[tr_idx], y_train[tr_idx])
                scores.append(model.score(X_tr[val_idx], y_train[val_idx]))
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = (n_est, max_d)

    # Fit final model
    model = RandomForestClassifier(
        n_estimators=best_params[0], max_depth=best_params[1],
        random_state=seed, n_jobs=1, max_features="sqrt"
    )
    model.fit(X_tr, y_train)
    importance = model.feature_importances_

    return importance, best_params


# ============================================================
# Stability metrics 
# ============================================================
def jaccard(a, b):
    a, b = set(a), set(b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def nogueira_stability(topk_lists, p, k=None):
    """Nogueira stability index."""
    m = len(topk_lists)
    if m < 2:
        return float("nan")
    if k is None:
        k = len(topk_lists[0])
    if k == 0 or p <= 1:
        return float("nan")

    Z = np.zeros((m, p), dtype=np.int8)
    for r, sel in enumerate(topk_lists):
        Z[r, np.asarray(sel, dtype=int)] = 1

    s2 = Z.var(axis=0, ddof=1)
    s2_bar = float(s2.mean())
    denom = (k / p) * (1.0 - (k / p))
    if denom <= 0:
        return float("nan")
    return 1.0 - (s2_bar / denom)


# ============================================================
# Run one resampling iteration
# ============================================================
def run_one_iteration(X, y, continuous_idx, seed, num_shuffles, topk):
    """
    Run LASSO, Elastic Net, and RF for one seed.
    Mirrors the Vendrow pipeline: multiple shuffles, average scores, rank, top-k.
    """
    n_features = X.shape[1]

    # Accumulate importance scores across shuffles
    lasso_scores = np.zeros(n_features)
    enet_scores = np.zeros(n_features)
    rf_scores = np.zeros(n_features)

    for i in range(num_shuffles):
        shuffle_seed = seed * 1000 + i
        perm = np.random.RandomState(seed=shuffle_seed).permutation(X.shape[0])
        X_shuf, y_shuf = X[perm], y[perm]

        X_train, X_test, y_train, y_test = train_test_split(
            X_shuf, y_shuf, test_size=0.2,
            random_state=shuffle_seed, stratify=y_shuf
        )

        # LASSO
        imp_l, _ = lasso_feature_importance(
            X_train, y_train, X_test, continuous_idx, seed=shuffle_seed
        )
        lasso_scores += imp_l / num_shuffles

        # Elastic Net
        imp_e, _ = elastic_net_feature_importance(
            X_train, y_train, X_test, continuous_idx, seed=shuffle_seed
        )
        enet_scores += imp_e / num_shuffles

        # Random Forest
        imp_r, _ = random_forest_feature_importance(
            X_train, y_train, X_test, continuous_idx, seed=shuffle_seed
        )
        rf_scores += imp_r / num_shuffles

    # Convert to ranks (higher importance = rank 1)
    def to_ranks(scores):
        return pd.Series(scores).rank(ascending=False, method="average").to_numpy()

    R_lasso = to_ranks(lasso_scores)
    R_enet = to_ranks(enet_scores)
    R_rf = to_ranks(rf_scores)

    # Individual top-k for each method
    topk_lasso = np.argsort(R_lasso)[:topk]
    topk_enet = np.argsort(R_enet)[:topk]
    topk_rf = np.argsort(R_rf)[:topk]

    # Combined consensus (3-method)
    S_combined = (R_lasso + R_enet + R_rf) / 3.0
    topk_combined = np.argsort(S_combined)[:topk]

    return {
        "lasso_scores": lasso_scores,
        "enet_scores": enet_scores,
        "rf_scores": rf_scores,
        "R_lasso": R_lasso,
        "R_enet": R_enet,
        "R_rf": R_rf,
        "topk_lasso": topk_lasso,
        "topk_enet": topk_enet,
        "topk_rf": topk_rf,
        "S_combined": S_combined,
        "topk_combined": topk_combined,
    }


# ============================================================
# Main
# ============================================================
def main():
    set_all_seeds(SEED_BASE)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ALTERNATIVE FEATURE SELECTION: LASSO / Elastic Net / Random Forest")
    print("=" * 70)

    # Load data
    X, y = load_data()
    feature_names = load_feature_names()
    continuous_idx = load_continuous_idx(feature_names)
    n_features = X.shape[1]

    print(f"  Dataset: {X.shape[0]} samples × {n_features} features")
    print(f"  Classes: {np.bincount(y)} (0=non-responder, 1=high-responder)")
    print(f"  Continuous features: {len(continuous_idx)}")
    print(f"  Resampling iterations: {R}")
    print(f"  Shuffles per iteration: {NUM_SHUFFLES}")
    print(f"  Top-k per iteration: {TOPK}")
    print()

    # -------------------------
    # Run all iterations (parallel)
    # -------------------------
    from concurrent.futures import ProcessPoolExecutor, as_completed

    max_workers = N_JOBS  # adjust based on your system

    # Containers: indexed by seed for correct ordering
    results_by_seed = [None] * R

    print(f"  Running {R} iterations with {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_one_iteration,
                X, y, continuous_idx,
                seed=SEED_BASE + seed,
                num_shuffles=NUM_SHUFFLES,
                topk=TOPK
            ): seed
            for seed in range(R)
        }

        for fut in tqdm(as_completed(futures), total=R, desc="Resampling iterations"):
            seed = futures[fut]
            results_by_seed[seed] = fut.result()

    # Unpack into separate lists
    topk_lasso_all = [r["topk_lasso"] for r in results_by_seed]
    topk_enet_all = [r["topk_enet"] for r in results_by_seed]
    topk_rf_all = [r["topk_rf"] for r in results_by_seed]
    topk_combined_all = [r["topk_combined"] for r in results_by_seed]

    S_lasso_all = [r["R_lasso"] for r in results_by_seed]
    S_enet_all = [r["R_enet"] for r in results_by_seed]
    S_rf_all = [r["R_rf"] for r in results_by_seed]
    S_combined_all = [r["S_combined"] for r in results_by_seed]

    # -------------------------
    # Compute stability metrics for each method
    # -------------------------
    methods = {
        "LASSO": (topk_lasso_all, np.vstack(S_lasso_all)),
        "ElasticNet": (topk_enet_all, np.vstack(S_enet_all)),
        "RandomForest": (topk_rf_all, np.vstack(S_rf_all)),
        "Combined_3method": (topk_combined_all, np.vstack(S_combined_all)),
    }

    print("\n" + "=" * 70)
    print("STABILITY METRICS")
    print("=" * 70)

    all_stable_sets = {}

    for method_name, (topk_lists, S_matrix) in methods.items():
        # Nogueira
        nog = nogueira_stability(topk_lists, p=n_features, k=TOPK)

        # Pairwise Jaccard
        jac_vals = []
        for i, j in combinations(range(R), 2):
            jac_vals.append(jaccard(topk_lists[i], topk_lists[j]))
        jac_vals = np.array(jac_vals)

        # Pairwise Spearman on rank vectors
        rho_vals = []
        for i, j in combinations(range(R), 2):
            r, _ = spearmanr(S_matrix[i], S_matrix[j])
            rho_vals.append(r)
        rho_vals = np.array(rho_vals)

        # Mean ranks and stable features
        mean_ranks = S_matrix.mean(axis=0)

        # Selection frequency
        sel_counts = np.zeros(n_features, dtype=int)
        for sel in topk_lists:
            sel_counts[np.asarray(sel, dtype=int)] += 1
        sel_freq = sel_counts / R

        min_count = int(MIN_FREQ * R)
        stable_idx = np.where(sel_counts >= min_count)[0]
        stable_sorted = stable_idx[np.argsort(mean_ranks[stable_idx])]

        all_stable_sets[method_name] = set(stable_sorted)

        print(f"\n  {method_name}:")
        print(f"    Nogueira stability:      {nog:.4f}")
        print(f"    Mean Jaccard:            {jac_vals.mean():.4f}")
        print(f"    Mean Spearman ρ:         {np.nanmean(rho_vals):.4f}")
        print(f"    Stable features (≥{MIN_FREQ:.0%}): {len(stable_sorted)}")

        # Save stable features
        rows = []
        for idx in stable_sorted:
            rows.append({
                "feature_idx": int(idx),
                "feature_name": feature_names[idx] if idx < len(feature_names) else f"idx_{idx}",
                "selection_freq": float(sel_freq[idx]),
                "mean_rank": float(mean_ranks[idx]),
            })
        df = pd.DataFrame(rows)
        out_path = OUT_ROOT / f"stable_features_{method_name}.csv"
        df.to_csv(out_path, index=False)
        print(f"    Saved: {out_path}")

    # -------------------------
    # Load Vendrow consensus results for comparison
    # -------------------------
    print("\n" + "=" * 70)
    print("COMPARISON WITH VENDROW CONSENSUS (5-method)")
    print("=" * 70)

    vendrow_path = (SCRIPT_DIR / "../results/stability_analysis/stable_top_features.csv").resolve()

    if vendrow_path.exists():
        df_vendrow = pd.read_csv(vendrow_path)
        vendrow_set = set(df_vendrow["feature_idx"].tolist())
        vendrow_names = dict(zip(df_vendrow["feature_idx"], df_vendrow["feature_name"]))

        for method_name, stable_set in all_stable_sets.items():
            overlap = vendrow_set & stable_set
            jac_vs_vendrow = jaccard(vendrow_set, stable_set)
            print(f"\n  {method_name} vs Vendrow consensus:")
            print(f"    Vendrow stable: {len(vendrow_set)}, {method_name} stable: {len(stable_set)}")
            print(f"    Overlap: {len(overlap)} features")
            print(f"    Jaccard: {jac_vs_vendrow:.4f}")

            if overlap:
                overlap_names = [
                    feature_names[i] if i < len(feature_names) else f"idx_{i}"
                    for i in sorted(overlap)
                ]
                print(f"    Shared features: {', '.join(overlap_names)}")

            # Features unique to each
            only_vendrow = vendrow_set - stable_set
            only_alt = stable_set - vendrow_set

            if only_vendrow:
                names_v = [feature_names[i] for i in sorted(only_vendrow) if i < len(feature_names)]
                print(f"    Only in Vendrow ({len(only_vendrow)}): {', '.join(names_v)}")
            if only_alt:
                names_a = [feature_names[i] for i in sorted(only_alt) if i < len(feature_names)]
                print(f"    Only in {method_name} ({len(only_alt)}): {', '.join(names_a)}")

        # Save comparison table
        comparison_rows = []
        all_features = set()
        for s in all_stable_sets.values():
            all_features |= s
        all_features |= vendrow_set

        for idx in sorted(all_features):
            row = {
                "feature_idx": int(idx),
                "feature_name": feature_names[idx] if idx < len(feature_names) else f"idx_{idx}",
                "in_vendrow": idx in vendrow_set,
            }
            for method_name, stable_set in all_stable_sets.items():
                row[f"in_{method_name}"] = idx in stable_set
            # Count how many methods selected it
            row["n_methods"] = sum([
                idx in vendrow_set,
                idx in all_stable_sets.get("LASSO", set()),
                idx in all_stable_sets.get("ElasticNet", set()),
                idx in all_stable_sets.get("RandomForest", set()),
            ])
            comparison_rows.append(row)

        df_comp = pd.DataFrame(comparison_rows).sort_values("n_methods", ascending=False)
        comp_path = OUT_ROOT / "method_comparison.csv"
        df_comp.to_csv(comp_path, index=False)
        print(f"\n  Comparison table saved: {comp_path}")

        # Highlight features selected by ALL methods
        unanimous = df_comp[df_comp["n_methods"] == 4]
        if len(unanimous) > 0:
            print(f"\n  Features selected by ALL 4 methods ({len(unanimous)}):")
            for _, row in unanimous.iterrows():
                print(f"    {row['feature_name']}")
        else:
            print("\n  No features selected by all 4 methods.")

        # Features by ≥3 methods
        robust = df_comp[df_comp["n_methods"] >= 3]
        if len(robust) > 0:
            print(f"\n  Features selected by ≥3 methods ({len(robust)}):")
            for _, row in robust.iterrows():
                marker = "★" if row["n_methods"] == 4 else " "
                print(f"    {marker} {row['feature_name']} ({row['n_methods']}/4)")

    else:
        print(f"\n  WARNING: Vendrow results not found at {vendrow_path}")
        print("  Run stability_top30.py first, then re-run this script for comparison.")

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Results saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()