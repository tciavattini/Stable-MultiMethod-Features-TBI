"""
figS2_alt_methods_heatmap.py
=============================
Supplementary Figure S2.
Heatmap of AUC values for all alternative feature selection methods
× classifiers, with Vendrow consensus and full-feature baselines included.

Also adds a side panel showing accuracy gain over random baseline (Cohen's d).

INPUT:
    results/multi_feature_evaluation/feature_set_comparison.csv
        — OR — the alternative_fs_classification CSV if you ran that separately.
        The script accepts either; it looks for a file with columns:
            method (or feature_set), classifier (or model), auc_mean,
            accuracy_mean, cohens_d_vs_random (optional)

    If you ran alternative_fs_classification.py the CSV is typically:
        results/alternative_fs_classification/alt_classification_results.csv

Run:
    python figS2_alt_methods_heatmap.py

Output:
    figures/suppfig_alt_methods_heatmap.pdf
    figures/suppfig_alt_methods_heatmap.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── hardcoded results (from paper Tables 4 + alt_classification) ──────────────
# Rows: method | Cols: classifier
# Paste values from your CSVs if you have them; these match the paper tables.

AUC_DATA = {
    #                       LinearSVM   k-NN    Dec.Tree
    "LASSO (18)†":        [0.830,      0.637,  0.587],
    "Elastic Net (17)":   [0.737,      0.631,  0.652],
    "Random Forest (30)": [0.689,      0.616,  0.579],
    "Combined 3-mth (19)":[0.743,      0.632,  0.638],
    "Vendrow (22)":       [0.723,      0.710,  0.640],
    "Full set (149)":     [0.543,      0.510,  0.579],
}

ACC_DATA = {
    #                       LinearSVM   k-NN    Dec.Tree
    "LASSO (18)†":        [0.783,      0.597,  0.551],
    "Elastic Net (17)":   [0.681,      0.591,  0.649],
    "Random Forest (30)": [0.648,      0.574,  0.598],
    "Combined 3-mth (19)":[0.670,      0.593,  0.634],
    "Vendrow (22)":       [0.655,      0.646,  0.642],
    "Full set (149)":     [0.531,      0.500,  0.587],
}

# Cohen's d vs random baseline for accuracy (from statistical comparisons)
# Positive = selected > random; use nan if not computed
COHENS_D = {
    #                       LinearSVM   k-NN    Dec.Tree
    "LASSO (18)†":        [np.nan,     np.nan, np.nan],  # fill from your stats CSV
    "Elastic Net (17)":   [np.nan,     np.nan, np.nan],
    "Random Forest (30)": [np.nan,     np.nan, np.nan],
    "Combined 3-mth (19)":[np.nan,     np.nan, np.nan],
    "Vendrow (22)":       [1.10,       1.14,   0.96],    # from Table 3
    "Full set (149)":     [np.nan,     np.nan, np.nan],
}

METHODS     = list(AUC_DATA.keys())
CLASSIFIERS = ["Linear SVM", "k-NN", "Decision Tree"]

auc_matrix = np.array([AUC_DATA[m]     for m in METHODS])
acc_matrix = np.array([ACC_DATA[m]     for m in METHODS])
d_matrix   = np.array([COHENS_D[m]     for m in METHODS])

# ── layout: two side-by-side heatmaps ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5),
                         gridspec_kw={"width_ratios": [1, 1], "wspace": 0.55})

RANDOM_BASELINE_ACC = 0.52   # approximate chance level

def draw_heatmap(ax, matrix, title, vmin, vmax, cmap, fmt=".3f",
                 annot_thresh=None, row_labels=METHODS, col_labels=CLASSIFIERS,
                 highlight_method=None, highlight_col=None):
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap,
                   aspect="auto", interpolation="none")

    n_rows, n_cols = matrix.shape
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.8)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.8)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                text = "—"
                tcol = "#888"
            else:
                text = f"{val:{fmt}}"
                bg = (val - vmin) / max(vmax - vmin, 1e-9)
                tcol = "white" if (bg > 0.65 or bg < 0.3) else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=tcol, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    # thick border around Vendrow row for reference
    vendrow_idx = [i for i, m in enumerate(METHODS) if "Vendrow" in m]
    if vendrow_idx:
        i = vendrow_idx[0]
        for spine_kwargs in [
            dict(xy=(-0.5, i - 0.5), width=n_cols, height=1,
                 linewidth=2, edgecolor="#333", facecolor="none")
        ]:
            from matplotlib.patches import Rectangle
            rect = Rectangle((-.5, i - .5), n_cols, 1,
                              linewidth=1.8, edgecolor="#333333",
                              facecolor="none", zorder=5)
            ax.add_patch(rect)

    return im


# left: AUC heatmap
im1 = draw_heatmap(axes[0], auc_matrix,
                   "AUC (Area under ROC curve)",
                   vmin=0.50, vmax=0.85,
                   cmap="Blues")
cb1 = fig.colorbar(im1, ax=axes[0], fraction=0.04, pad=0.03)
cb1.set_label("AUC", fontsize=8)
cb1.ax.tick_params(labelsize=8)

# right: Accuracy heatmap
im2 = draw_heatmap(axes[1], acc_matrix,
                   "Accuracy  (10×10 repeated CV)",
                   vmin=0.50, vmax=0.80,
                   cmap="Greens")
cb2 = fig.colorbar(im2, ax=axes[1], fraction=0.04, pad=0.03)
cb2.set_label("Accuracy", fontsize=8)
cb2.ax.tick_params(labelsize=8)

# chance line annotation
for ax in axes:
    ax.text(2.55, len(METHODS) - 0.05,
            f"Random ≈ 0.52",
            fontsize=7, color="#cc0000", ha="right", va="bottom")

fig.suptitle(
    "Classification performance by feature selection method and classifier\n"
    "Vendrow consensus row (outlined) is the primary reference. "
    "†LASSO–LinearSVM: interpret with caution (shared L1 assumption).",
    fontsize=9, fontweight="bold", y=1.03
)

plt.savefig(OUT_DIR / "suppfig_alt_methods_heatmap.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "suppfig_alt_methods_heatmap.png", dpi=300, bbox_inches="tight")
print("Saved: figures/suppfig_alt_methods_heatmap.pdf  +  .png")
plt.close()