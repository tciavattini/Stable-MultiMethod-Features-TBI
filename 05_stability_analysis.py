# stability_top30.py
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import trange
from itertools import combinations
from scipy.stats import spearmanr

# -------------------------------------------------
# Paths
# -------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
CROSSVAL_PATH = SCRIPT_DIR / "04_cross_validation.py"

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def scores_to_ranks(scores, higher_is_better=True):
    s = pd.Series(scores)
    ascending = not higher_is_better
    return s.rank(ascending=ascending, method="average").to_numpy()

def jaccard(a, b):
    a = set(a); b = set(b)
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0

import numpy as np

def nogueira_stability_fixed_k(topk_lists, p, k=None):
    """
    Nogueira et al. (2018) stability index for feature selection.
    Assumes fixed selection size k across runs.

    Parameters
    ----------
    topk_lists : list of array-like
        Each element is a list/array of selected feature indices (length k).
    p : int
        Total number of features.
    k : int or None
        Number of selected features per run. If None, inferred from first run.

    Returns
    -------
    float
        Stability in (-inf, 1], typically [0, 1] for reasonable procedures.
    """
    m = len(topk_lists)
    if m < 2:
        return float("nan")

    if k is None:
        k = len(topk_lists[0])
    if k == 0 or p <= 1:
        return float("nan")
    if not all(len(sel) == k for sel in topk_lists):
        raise ValueError("All runs must select the same number of features (fixed k).")

    # Build selection matrix Z (m x p)
    Z = np.zeros((m, p), dtype=np.int8)
    for r, sel in enumerate(topk_lists):
        sel = np.asarray(sel, dtype=int)
        if np.any(sel < 0) or np.any(sel >= p):
            raise ValueError("Some selected indices are out of bounds [0, p).")
        Z[r, sel] = 1

    # Sample variance per feature (unbiased, ddof=1)
    # For binary indicators, this matches the definition above.
    s2 = Z.var(axis=0, ddof=1)  # shape (p,)
    s2_bar = float(s2.mean())

    denom = (k / p) * (1.0 - (k / p))
    if denom <= 0:
        return float("nan")

    S = 1.0 - (s2_bar / denom)
    return float(S)

# -------------------------------------------------
# One-seed runner
# -------------------------------------------------
def run_one_seed(seed, num_shuffles, params, device, workdir, crossval_path):
    """
    Runs all per-feature scorers (--single) for one seed.
    Output files are written inside workdir (because cwd=workdir).
    
    Methods run (5 total):
      1. SVM (linear, 5-fold inner CV for C)
      2. KNN (5-fold inner CV for k)
      3. Neural Network (fixed lr=0.005, 10 epochs, no inner CV)
      4. Linear Regression R² (in-sample, filter metric)
      5. Information Gain / Entropy (train-only discretization)
    """
    workdir.mkdir(parents=True, exist_ok=True)

    def run(cmd, label):
        cmd = [str(c) for c in cmd]
        res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"[seed {seed}] Command failed ({label}):\n{' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
            )

    run(
        ["python", crossval_path, "--clf", "svm",
         "--params", params["svm"],
         "--single", "--seed", seed, "--num_shuffles", num_shuffles],
        label="SVM single"
    )

    run(
        ["python", crossval_path, "--clf", "knn",
         "--params", params["knn"],
         "--single", "--seed", seed, "--num_shuffles", num_shuffles],
        label="KNN single"
    )

    run(
        ["python", crossval_path, "--clf", "net",
         "--params", params["net"],
         "--single", "--seed", seed, "--num_shuffles", num_shuffles,
         "--device", device],
        label="NET single"
    )

    run(
        ["python", crossval_path, "--clf", "linreg",
         "--single", "--seed", seed, "--num_shuffles", num_shuffles],
        label="LINREG"
    )

    run(
        ["python", crossval_path, "--clf", "entropy",
         "--seed", seed, "--num_shuffles", num_shuffles],
        label="ENTROPY"
    )


def run_one_seed_fixed(seed, num_shuffles, params, device, workdir, crossval_path):
    """
    Same as run_one_seed but with --fixed_hp flag (no hyperparameter tuning).
    Uses first value in params list for each classifier.
    """
    workdir.mkdir(parents=True, exist_ok=True)

    def run(cmd, label):
        cmd = [str(c) for c in cmd]
        res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"[seed {seed}] Command failed ({label}):\n{' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
            )

    run(
        ["python", crossval_path, "--clf", "svm",
         "--params", params["svm"],
         "--single", "--seed", seed, "--num_shuffles", num_shuffles,
         "--fixed_hp"],
        label="SVM single (fixed)"
    )

    run(
        ["python", crossval_path, "--clf", "knn",
         "--params", params["knn"],
         "--single", "--seed", seed, "--num_shuffles", num_shuffles,
         "--fixed_hp"],
        label="KNN single (fixed)"
    )

    run(
        ["python", crossval_path, "--clf", "net",
         "--params", params["net"],
         "--single", "--seed", seed, "--num_shuffles", num_shuffles,
         "--device", device,
         "--fixed_hp"],
        label="NET single (fixed)"
    )

    run(
        ["python", crossval_path, "--clf", "linreg",
         "--single", "--seed", seed, "--num_shuffles", num_shuffles,
         "--fixed_hp"],
        label="LINREG (fixed)"
    )

    run(
        ["python", crossval_path, "--clf", "entropy",
         "--seed", seed, "--num_shuffles", num_shuffles],
        label="ENTROPY"  # Entropy doesn't have hyperparameters
    )

from concurrent.futures import ProcessPoolExecutor, as_completed

def run_seed_job(seed, num_shuffles, params, device, out_root, crossval_path, topk):
    """
    Wrapper parallelizzabile: una seed completa.
    """
    run_dir = out_root / f"seed_{seed:03d}"

    run_one_seed(
        seed=seed,
        num_shuffles=num_shuffles,
        params=params,
        device=device,
        workdir=run_dir,
        crossval_path=crossval_path,
    )

    topk_idx, S = consensus_topk_from_folder(run_dir, k=topk)
    np.savetxt(run_dir / f"top{topk}_idx.txt", topk_idx, fmt="%d")

    return seed, topk_idx, S

def run_seed_job_fixed(seed, num_shuffles, params, device, out_root, crossval_path, topk):
    """
    Wrapper for fixed hyperparameter runs.
    """
    run_dir = out_root / f"seed_{seed:03d}"

    run_one_seed_fixed(
        seed=seed,
        num_shuffles=num_shuffles,
        params=params,
        device=device,
        workdir=run_dir,
        crossval_path=crossval_path,
    )

    topk_idx, S = consensus_topk_from_folder(run_dir, k=topk)
    np.savetxt(run_dir / f"top{topk}_idx.txt", topk_idx, fmt="%d")

    return seed, topk_idx, S

# -------------------------------------------------
# Consensus ranking
# -------------------------------------------------
def consensus_topk_from_folder(folder, k=30):
    """
    Load single-feature scores from one seed's folder and compute consensus top-k.
    
    Consensus is computed from 5 methods:
      1. SVM accuracy (higher = better)
      2. KNN accuracy (higher = better)
      3. Neural Network accuracy (higher = better)
      4. Linear Regression R² (higher = better)
      5. Information Gain / Entropy (higher = better)
    
    Each method's scores are converted to ranks, then averaged.
    """
    svm  = np.loadtxt(folder / "svm_single.txt")
    knn  = np.loadtxt(folder / "knn_single.txt")
    net  = np.loadtxt(folder / "net_single.txt")
    r2   = np.loadtxt(folder / "r2.txt")
    ent  = np.loadtxt(folder / "entropies.txt")

    m = len(svm)
    for name, arr in [("knn", knn), ("net", net), ("r2", r2), ("ent", ent)]:
        if len(arr) != m:
            raise ValueError(f"Length mismatch in {folder}: svm={m}, {name}={len(arr)}")

    R_svm  = scores_to_ranks(svm)
    R_knn  = scores_to_ranks(knn)
    R_net  = scores_to_ranks(net)
    R_r2   = scores_to_ranks(r2)
    R_ent  = scores_to_ranks(ent)

    # 5-method consensus ranking (average of ranks)
    S = (R_svm + R_knn + R_net + R_r2 + R_ent) / 5.0

    topk = np.argsort(S)[:k]
    return topk, S

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    # -------------------------
    # CONFIG
    # -------------------------
    R = 100                 # number of seeds to run
    num_shuffles = 10       # data shuffles per seed
    topk = 30               # how many top features to select per seed
    min_count = int(0.7 * R)   # frequency threshold (70% = selected in 70/100 seeds)
    device_for_net = "cpu"     # recommended for reproducibility

    # Hyperparameter grids (only for SVM and KNN; NN uses fixed lr=0.005)
    params = {
        "svm":  "0.01,0.1,1,10",
        "knn":  "1,3,5,7,9,11,13,15,17,19",
        "net":  "0.005",  # fixed lr, passed but only first value used
    }

    feature_names_path = (SCRIPT_DIR / "../data/feature_names.csv").resolve()
    names = pd.read_csv(feature_names_path)["feature_name"].astype(str).tolist()

    out_root = (SCRIPT_DIR / "../results/stability_analysis").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Containers
    topk_lists = [None] * R
    S_all = [None] * R

    # -------------------------
    # Run seeds in parallel
    # -------------------------
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    max_workers = 6   # adjust based on your system (4-6 for M4 Max)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_seed_job,
                seed,
                num_shuffles,
                params,
                device_for_net,
                out_root,
                CROSSVAL_PATH,
                topk
            )
            for seed in range(R)
        ]

        for fut in tqdm(as_completed(futures), total=R, desc="Seeds (parallel)"):
            seed, topk_idx, S = fut.result()
            topk_lists[seed] = topk_idx
            S_all[seed] = S

    S_all = np.vstack(S_all)  # (R, p)
    mean_S = S_all.mean(axis=0)
    p = S_all.shape[1]

    # -------------------------
    # Stability metrics
    # -------------------------
    # 1) Pairwise Jaccard on top-k sets
    jac = []
    for i, j in combinations(range(R), 2):
        jac.append(jaccard(topk_lists[i], topk_lists[j]))
    jac = np.asarray(jac, dtype=float)

    # 2) Nogueira stability
    nog = nogueira_stability_fixed_k(topk_lists, p=p, k=topk)

    # 3) Rank correlation across seeds (Spearman on consensus ranks)
    rho = []
    for i, j in combinations(range(R), 2):
        r, _ = spearmanr(S_all[i], S_all[j])
        rho.append(r)
    rho = np.asarray(rho, dtype=float)

    # Save stability metrics summary
    metrics_out = out_root / "stability_metrics.txt"
    with open(metrics_out, "w") as f:
        f.write(f"R={R}\n")
        f.write(f"num_shuffles={num_shuffles}\n")
        f.write(f"topk={topk}\n\n")
        f.write("Consensus methods: SVM, KNN, Neural Network, Linear R², Information Gain\n")
        f.write("(5 methods, no Decision Tree)\n\n")
        f.write("Pairwise Jaccard (top-k sets):\n")
        f.write(f"  mean={jac.mean():.4f}, median={np.median(jac):.4f}, std={jac.std(ddof=0):.4f}\n\n")
        f.write("Nogueira stability index:\n")
        f.write(f"  S={nog:.4f}\n\n")
        f.write("Pairwise Spearman correlation (consensus ranks):\n")
        f.write(f"  mean={np.nanmean(rho):.4f}, median={np.nanmedian(rho):.4f}, std={np.nanstd(rho):.4f}\n")

    print(f"\nSaved stability metrics: {metrics_out}")

    # -------------------------
    # Selection counts
    # -------------------------
    selection_counts = np.zeros(p, dtype=int)
    for sel in topk_lists:
        if sel is None:
            raise RuntimeError("At least one seed failed (topk_lists contains None). Check logs.")
        selection_counts[np.asarray(sel, dtype=int)] += 1

    # Threshold sensitivity analysis
    thresholds = [0.5, 0.6, 0.7, 0.8]
    threshold_results = []

    for thresh in thresholds:
        thresh_min_count = int(thresh * R)
        n_stable = (selection_counts >= thresh_min_count).sum()
        stable_idx = np.where(selection_counts >= thresh_min_count)[0]
        threshold_results.append({
            'threshold': thresh,
            'min_count': thresh_min_count,
            'n_stable_features': n_stable,
            'stable_indices': stable_idx.tolist()
        })

    pd.DataFrame(threshold_results).to_csv(out_root / "threshold_sensitivity.csv", index=False)

    keep_idx = np.where(selection_counts >= min_count)[0]

    # =========================================================================
    # SENSITIVITY ANALYSIS: Fixed Hyperparameters
    # =========================================================================
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS: Fixed Hyperparameters")
    print("="*70)
    print("Running stability analysis with fixed HP (no tuning)...")
    print(f"   SVM: C={params['svm'].split(',')[0]}")
    print(f"   KNN: k={params['knn'].split(',')[0]}")
    print(f"   Net: lr={params['net'].split(',')[0]}")
    
    out_root_fixed = (SCRIPT_DIR / "../results/stability_analysis_fixed_hp").resolve()
    out_root_fixed.mkdir(parents=True, exist_ok=True)
    
    # Containers for fixed HP run
    topk_lists_fixed = [None] * R
    S_all_fixed = [None] * R
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures_fixed = [
            executor.submit(
                run_seed_job_fixed,
                seed,
                num_shuffles,
                params,
                device_for_net,
                out_root_fixed,
                CROSSVAL_PATH,
                topk
            )
            for seed in range(R)
        ]

        for fut in tqdm(as_completed(futures_fixed), total=R, desc="Seeds fixed HP"):
            seed, topk_idx, S = fut.result()
            topk_lists_fixed[seed] = topk_idx
            S_all_fixed[seed] = S
    
    S_all_fixed = np.vstack(S_all_fixed)
    mean_S_fixed = S_all_fixed.mean(axis=0)
    
    # Compute stability for fixed HP run
    nog_fixed = nogueira_stability_fixed_k(topk_lists_fixed, p=p, k=topk)
    
    # Compare tuned vs fixed: Jaccard between stable feature sets
    selection_counts_fixed = np.zeros(p, dtype=int)
    for sel in topk_lists_fixed:
        selection_counts_fixed[np.asarray(sel, dtype=int)] += 1
    
    keep_idx_fixed = np.where(selection_counts_fixed >= min_count)[0]
    
    # Jaccard similarity between tuned and fixed stable sets
    jaccard_tuned_vs_fixed = jaccard(keep_idx, keep_idx_fixed)
    
    # Spearman correlation between mean consensus ranks
    rho_tuned_vs_fixed, _ = spearmanr(mean_S, mean_S_fixed)
    
    # Save comparison results
    sensitivity_results = {
        'tuned_n_stable': len(keep_idx),
        'fixed_n_stable': len(keep_idx_fixed),
        'jaccard_tuned_vs_fixed': jaccard_tuned_vs_fixed,
        'spearman_tuned_vs_fixed': rho_tuned_vs_fixed,
        'nogueira_tuned': nog,
        'nogueira_fixed': nog_fixed,
        'overlap_features': len(set(keep_idx) & set(keep_idx_fixed)),
    }
    
    sensitivity_path = out_root / "hp_sensitivity_comparison.txt"
    with open(sensitivity_path, "w") as f:
        f.write("HYPERPARAMETER SENSITIVITY ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Tuned HP - stable features: {sensitivity_results['tuned_n_stable']}\n")
        f.write(f"Fixed HP - stable features: {sensitivity_results['fixed_n_stable']}\n")
        f.write(f"Overlap: {sensitivity_results['overlap_features']} features\n\n")
        f.write(f"Jaccard similarity (stable sets): {jaccard_tuned_vs_fixed:.4f}\n")
        f.write(f"Spearman correlation (rankings): {rho_tuned_vs_fixed:.4f}\n\n")
        f.write(f"Nogueira stability (tuned): {nog:.4f}\n")
        f.write(f"Nogueira stability (fixed): {nog_fixed:.4f}\n")
    
    print(f"\nSensitivity analysis results:")
    print(f"   Jaccard (tuned vs fixed stable sets): {jaccard_tuned_vs_fixed:.4f}")
    print(f"   Spearman (ranking correlation): {rho_tuned_vs_fixed:.4f}")
    print(f"   Overlap: {sensitivity_results['overlap_features']}/{len(keep_idx)} features")
    print(f"\nSaved: {sensitivity_path}")
    
    # Save fixed HP stable features too
    keep_sorted_fixed = keep_idx_fixed[np.argsort(mean_S_fixed[keep_idx_fixed])]
    rows_fixed = []
    for idx in keep_sorted_fixed:
        rows_fixed.append({
            "feature_idx": int(idx),
            "feature_name": names[idx] if idx < len(names) else f"idx_{idx}",
            "selected_count": int(selection_counts_fixed[idx]),
            "selected_freq": float(selection_counts_fixed[idx]) / R,
            "mean_consensus_rank": float(mean_S_fixed[idx]),
        })
    
    df_fixed = pd.DataFrame(rows_fixed)
    df_fixed.to_csv(out_root_fixed / "stable_top_features.csv", index=False)
    
    # =========================================================================
    # END SENSITIVITY ANALYSIS
    # =========================================================================

    # -------------------------
    # Stable feature set by frequency threshold
    # -------------------------
    keep_sorted = keep_idx[np.argsort(mean_S[keep_idx])]

    rows = []
    for idx in keep_sorted:
        rows.append({
            "feature_idx": int(idx),
            "feature_name": names[idx] if idx < len(names) else f"idx_{idx}",
            "selected_count": int(selection_counts[idx]),
            "selected_freq": float(selection_counts[idx]) / R,
            "mean_consensus_rank": float(mean_S[idx]),
        })

    df = pd.DataFrame(rows)
    out_csv = out_root / "stable_top_features.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_csv}")
    if len(df) > 0:
        print(df.head(30).to_string(index=False))
    else:
        print("No features met the frequency threshold. Lower min_count or increase R.")

    # -------------------------
    # Hyperparameter distribution analysis
    # -------------------------
    print("\nAnalyzing hyperparameter distributions...")

    for clf_name in ["svm", "knn"]:  # Only SVM and KNN have tuned HPs
        all_best_params = []
        
        for seed in range(R):
            param_file = out_root / f"seed_{seed:03d}" / f"{clf_name}_best_params.txt"
            if param_file.exists():
                params_seed = np.loadtxt(param_file)
                all_best_params.append(params_seed)
        
        if all_best_params:
            all_best_params = np.vstack(all_best_params)  # (R, n_features)
            
            # For each feature, find most common hyperparameter
            from scipy.stats import mode
            mode_params = mode(all_best_params, axis=0, keepdims=False)[0]
            
            # Save summary
            hp_summary_path = out_root / f"{clf_name}_hp_summary.csv"
            df_hp = pd.DataFrame({
                'feature_idx': range(len(mode_params)),
                'feature_name': names[:len(mode_params)],
                'mode_best_param': mode_params,
                'param_std': all_best_params.std(axis=0)
            })
            df_hp.to_csv(hp_summary_path, index=False)
            print(f"   Saved: {hp_summary_path}")

if __name__ == "__main__":
    main()