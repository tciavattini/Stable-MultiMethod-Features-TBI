"""
fig4_violin_phenotype.py
=========================
Figure 4 — Main text.
Violin + strip plot for the 7 FDR-significant stable features,
comparing high responders vs non-responders.
"""

import numpy as np
import pandas as pd
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
import csv

# ── GLOBAL FONT SIZES — applied to everything automatically ──────────────────
plt.rcParams.update({
    "font.size":          17,    # base size for everything
    "axes.titlesize":     18,    # panel titles
    "axes.labelsize":     16,    # x/y axis labels
    "xtick.labelsize":    15,    # x tick labels
    "ytick.labelsize":    15,    # y tick labels
    "legend.fontsize":    16,    # legend text
    "figure.titlesize":   18,    # suptitle
})

# ── paths ─────────────────────────────────────────────────────────────────────
PHENOTYPE_CSV     = Path("../results/phenotype_stable/phenotype_stable_features.csv")
DATASET_CSV       = Path("../data/dataset.csv")
LABELS_CSV        = Path("../data/labels.csv")
FEATURE_NAMES_CSV = Path("../data/feature_names.csv")
OUT_DIR           = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load phenotype results ────────────────────────────────────────────────────
phen = pd.read_csv(PHENOTYPE_CSV)

ALPHA = 0.05
sig = phen[
    (phen["q_value"] < ALPHA) &
    (phen["cohens_d"].abs() >= 0.2)
].copy()
sig = sig.sort_values("q_value").reset_index(drop=True)

if len(sig) == 0:
    raise ValueError("No significant features found in phenotype CSV.")

print(f"Significant features to plot: {len(sig)}")

# ── load raw data ─────────────────────────────────────────────────────────────
with open(DATASET_CSV, "r") as f:
    data = [[float(x) if x not in ("nan", "NaN", "") else np.nan for x in row]
            for row in csv.reader(f)]
X = np.array(data, dtype=float)

with open(LABELS_CSV, "r") as f:
    labels = [int(row[0]) for row in csv.reader(f)]
y = np.array(labels, dtype=int)

if set(np.unique(y)) == {1, 3}:
    y = np.where(y == 1, 0, 1)

feature_names = pd.read_csv(FEATURE_NAMES_CSV)["feature_name"].tolist()
name_to_idx   = {n: i for i, n in enumerate(feature_names)}

# ── colours ───────────────────────────────────────────────────────────────────
GROUP_COLORS = {
    "High responders": "#2166ac",
    "Non-responders":  "#d6604d",
}
GROUP_LABELS = {0: "Non-responders", 1: "High responders"}
GROUP_ORDER  = ["Non-responders", "High responders"]

# ── layout: smaller figure = fonts look proportionally bigger ─────────────────
# 2 rows × 4 cols, each panel compact so text dominates
N_COLS  = 4
N_ROWS  = 2
PANEL_W = 3.8
PANEL_H = 5.2   # less tall

fig_w = PANEL_W * N_COLS
fig_h = PANEL_H * N_ROWS + 3.5  # extra height for title and legend breathing room

fig = plt.figure(figsize=(fig_w, fig_h))

gs = gridspec.GridSpec(
    N_ROWS, N_COLS, figure=fig,
    hspace=0.5,
    wspace=0.45,
    top=0.82, bottom=0.17,
    left=0.07, right=0.98,
)

# ── draw panels ───────────────────────────────────────────────────────────────
for panel_idx, (_, row) in enumerate(sig.iterrows()):
    r_idx = panel_idx // N_COLS
    c_idx = panel_idx  % N_COLS

    feat_raw = row["feature"]
    feat_pub = row.get("feature_publication", feat_raw)
    cohens_d = row["cohens_d"]
    q_val    = row["q_value"]

    ax = fig.add_subplot(gs[r_idx, c_idx])

    if feat_raw not in name_to_idx:
        ax.set_title(feat_pub)
        ax.axis("off")
        continue

    col_idx = name_to_idx[feat_raw]
    x_col   = X[:, col_idx]

    data_groups = {}
    for grp_val, grp_label in GROUP_LABELS.items():
        mask = (y == grp_val) & ~np.isnan(x_col)
        data_groups[grp_label] = x_col[mask]

    # ── violin ────────────────────────────────────────────────────────────────
    positions = [0, 1]
    parts = ax.violinplot(
        [data_groups[g] for g in GROUP_ORDER],
        positions=positions,
        widths=0.65,
        showmedians=False,
        showextrema=False,
    )
    for pc, grp in zip(parts["bodies"], GROUP_ORDER):
        pc.set_facecolor(GROUP_COLORS[grp])
        pc.set_alpha(0.45)
        pc.set_edgecolor("none")

    # ── box plot overlay ──────────────────────────────────────────────────────
    bp = ax.boxplot(
        [data_groups[g] for g in GROUP_ORDER],
        positions=positions,
        widths=0.18,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.6, color="#444"),
        capprops=dict(linewidth=1.6, color="#444"),
        flierprops=dict(marker="", markersize=0),
    )
    for patch, grp in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[grp])
        patch.set_alpha(0.75)

    # ── strip plot ────────────────────────────────────────────────────────────
    rng = np.random.RandomState(42)
    for pos, grp in zip(positions, GROUP_ORDER):
        vals   = data_groups[grp]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=18, alpha=0.35,
            color=GROUP_COLORS[grp],
            zorder=5, linewidths=0,
        )

    # ── axes cosmetics — rcParams handles font sizes automatically ────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(["Non-resp.", "High resp."])
    ax.set_xlim(-0.6, 1.6)
    ax.spines[["top", "right"]].set_visible(False)

    # ── TITLE ─────────────────────────────────────────────────────────────────
    wrapped = "\n".join(textwrap.wrap(feat_pub, width=18))
    ax.set_title(wrapped, fontweight="bold", pad=10, loc="center", linespacing=1.3)

    # ── STATS below x-axis ────────────────────────────────────────────────────
    direction = "↑ High resp." if cohens_d > 0 else "↑ Non-resp."
    q_fmt     = f"q = {q_val:.3f}" if q_val >= 0.001 else "q < 0.001"
    d_fmt     = f"d = {cohens_d:+.2f}"

    ax.set_xlabel(f"{d_fmt}  |  {q_fmt}\n{direction}", labelpad=6, color="#444444")

# ── hide unused panels ────────────────────────────────────────────────────────
n_total = N_ROWS * N_COLS
for extra in range(len(sig), n_total):
    ax_empty = fig.add_subplot(gs[extra // N_COLS, extra % N_COLS])
    ax_empty.axis("off")

# ── shared group legend ───────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=GROUP_COLORS[g], label=g, alpha=0.85)
    for g in GROUP_ORDER
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=2,
    framealpha=0.92,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.01),
)

# ── suptitle ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "FDR-significant stable features: distribution by response group\n"
    "(violin = density  |  box = IQR + median  |  dots = observations)",
    fontweight="bold", y=0.97,
)

plt.savefig(OUT_DIR / "violin_phenotype.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "violin_phenotype.png", dpi=150, bbox_inches="tight")
print("Saved: figures/violin_phenotype.pdf  +  .png")
plt.close()