"""
figS1_correlation_heatmap.py
=============================
Supplementary Figure S1.
Pearson correlation heatmap of the 22 stable features.

INPUT:
    results/multi_feature_evaluation/selected_features_correlation_matrix.csv
        — produced by 4_multifeature.py (always saved, even if not plotted there)
    OR (fallback) raw data + stable features CSV to recompute

Run:
    python figS1_correlation_heatmap.py

Output:
    figures/suppfig_correlation_heatmap.pdf
    figures/suppfig_correlation_heatmap.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import csv

# ── paths ─────────────────────────────────────────────────────────────────────
CORR_CSV         = Path("../results/multi_feature_evaluation/selected_features_correlation_matrix.csv")
DATASET_CSV      = Path("../data/dataset.csv")
LABELS_CSV       = Path("../data/labels.csv")
FEATURE_NAMES_CSV = Path("../data/feature_names.csv")
STABLE_CSV       = Path("../results_fe/stability_top30_exp14/stable_top_features.csv")
OUT_DIR          = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication-ready short labels (same dict as fig2 — keep in sync)
LABEL_MAP = {
    "T0_severe_fatigue_rate":       "Severe fatigue severity",
    "num_neurological_symp":        "Neurological symptoms (count)",
    "Q20. Swglands":                "Swollen glands",
    "T0_muscle_pain_rate":          "Muscle pain severity",
    "T0_mood_rate":                 "Mood severity",
    "Q27. Lighthead.":              "Lightheadedness",
    "Q47Anxiety":                   "Anxiety",
    "T0_symp_today_rate":           "Overall symptom severity",
    "Q35Facial":                    "Facial palsy",
    "Q53_antib_symp_improv":        "Sympt. improvement (antibiotics)",
    "Q42Tinnit":                    "Tinnitus",
    "num_total_symptoms":           "Total symptom count",
    "Q40Vision":                    "Visual disturbances",
    "CD8%":                         "CD8⁺ T cells (%)",
    "Q14_Impact_symp_employment":   "Other employment status",
    "age":                          "Age",
    "CD3%":                         "CD3⁺ T cells (%)",
    "Q12_trt_care_rate":            "Treatment received rate",
    "cardiac_symp":                 "Cardiac symptoms",
    "Q30. Intensity":               "Symptom intensity",
    "Q55_alternative_trt_success":  "Alt. treatment success",
    "BaB M IgG":                    "Babesia microti IgG",
}


# ── load or compute correlation matrix ───────────────────────────────────────
if CORR_CSV.exists():
    corr_df = pd.read_csv(CORR_CSV, index_col=0)
    feature_order = corr_df.index.tolist()
    corr_matrix   = corr_df.values
    print(f"Loaded correlation matrix from {CORR_CSV}  ({corr_matrix.shape[0]} features)")
else:
    print("Correlation CSV not found — recomputing from raw data...")

    # load raw data
    with open(DATASET_CSV) as f:
        data = [[float(x) if x not in ("nan","NaN","") else np.nan for x in row]
                for row in csv.reader(f)]
    X = np.array(data, dtype=float)

    feature_names = pd.read_csv(FEATURE_NAMES_CSV)["feature_name"].tolist()
    name_to_idx   = {n: i for i, n in enumerate(feature_names)}

    stable_df = pd.read_csv(STABLE_CSV)
    name_col  = next(c for c in ["feature_name","feature","name"] if c in stable_df.columns)
    feature_order = stable_df[name_col].tolist()

    idxs = [name_to_idx[n] for n in feature_order if n in name_to_idx]
    X_sel = X[:, idxs]
    n = len(idxs)
    corr_matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            mask = ~(np.isnan(X_sel[:,i]) | np.isnan(X_sel[:,j]))
            if mask.sum() > 2:
                corr_matrix[i,j] = np.corrcoef(X_sel[mask,i], X_sel[mask,j])[0,1]
    print(f"Recomputed {n}×{n} correlation matrix.")

# Mask the upper triangle (k=1 excludes the diagonal too, keeping diagonal visible)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
corr_lower = np.ma.masked_where(mask, corr_matrix)


# ── apply publication labels ───────────────────────────────────────────────
tick_labels = [LABEL_MAP.get(f, f) for f in feature_order]

# ── plot ──────────────────────────────────────────────────────────────────────
n = len(feature_order)
fig, ax = plt.subplots(figsize=(10, 8))

cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad(color="white")   # masked (upper triangle) cells render as white

# diverging colourmap centred at 0 — only lower triangle is drawn
im = ax.imshow(corr_lower, vmin=-1, vmax=1,
               cmap=cmap, aspect="auto", interpolation="none")

# annotate cells with r value (only for |r| > 0.4 to avoid clutter)
for i in range(n):
    for j in range(i + 1):          # j <= i  →  lower triangle + diagonal
        val = corr_matrix[i, j]
        if not np.isnan(val) and abs(val) > 0.4 and i != j:
            text_color = "white" if abs(val) > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

# axes — hide x-ticks above the diagonal (upper triangle is blank)
ax.set_xticks(range(n))
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(n))
ax.set_yticklabels(tick_labels, fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# colour bar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Pearson r", fontsize=11)
cbar.ax.tick_params(labelsize=9)

# highlight the one pair exceeding |r| > 0.70
high_thresh = 0.70
high_pairs  = []
for i in range(n):
    for j in range(i):
        val = corr_matrix[i, j]
        if not np.isnan(val) and abs(val) > high_thresh:
            high_pairs.append((i, j, val))

if high_pairs:
    note = "  ".join(
        f"{tick_labels[i]} ↔ {tick_labels[j]}: r = {r:.2f}"
        for i, j, r in high_pairs
    )
    ax.set_xlabel(f"Pairs with |r| > {high_thresh}: {note}", fontsize=7.5,
                  color="#8b0000", labelpad=10)

ax.set_title(
    "Pairwise Pearson correlation among the 22 stable features\n"
    "(annotated where |r| > 0.40; one pair exceeds |r| = 0.70)",
    fontsize=12, fontweight="bold", pad=12
)

plt.tight_layout()
plt.savefig(OUT_DIR / "suppfig_correlation_heatmap_half.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "suppfig_correlation_heatmap_half.png", dpi=300, bbox_inches="tight")
print("Saved: figures/suppfig_correlation_heatmap_half.pdf  +  .png")
plt.close()