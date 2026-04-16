"""
fig3_bar_accuracy_comparison.py
================================
Figure 3 — Main text.
Grouped bar chart comparing accuracy across feature configurations
and classifiers with 95% bootstrap CI error bars.

INPUT (CSV produced by 4_multifeature.py):
    results/multi_feature_evaluation/feature_set_comparison.csv

Required columns:
    model, feature_set, accuracy_mean, accuracy_ci_lower, accuracy_ci_upper

Run:
    python fig3_bar_accuracy_comparison.py

Output:
    figures/bar_accuracy_comparison.pdf
    figures/bar_accuracy_comparison.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_CSV = Path("../results/multi_feature_evaluation/feature_set_comparison.csv")
OUT_DIR     = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(RESULTS_CSV)

# Normalise feature_set labels into short keys
def label_to_key(s):
    s = str(s).lower()
    if "all" in s:      return "All (149)"
    if "random" in s:   return "Random (22)"
    if "selected" in s: return "Selected (22)"
    return s

df["config"] = df["feature_set"].apply(label_to_key)

# Normalise model names
MODEL_ORDER = ["Linear SVM", "K-NN", "Decision Tree"]
CONFIG_ORDER = ["All (149)", "Selected (22)", "Random (22)"]

# Colours: one per configuration
COLORS = {
    "All (149)":     "#bbbbbb",   # light grey
    "Selected (22)": "#2166ac",   # blue
    "Random (22)":   "#e0e0e0",   # very light grey
}
HATCHES = {
    "All (149)":     "",
    "Selected (22)": "",
    "Random (22)":   "///",
}

# ── layout ────────────────────────────────────────────────────────────────────
n_models  = len(MODEL_ORDER)
n_configs = len(CONFIG_ORDER)
bar_width = 0.22
group_gap = 0.08      # extra space between model groups

x_centers = np.arange(n_models) * (n_configs * bar_width + group_gap)

fig, ax = plt.subplots(figsize=(8, 4.8))

for ci, config in enumerate(CONFIG_ORDER):
    sub = df[df["config"] == config].set_index("model")

    x_pos = x_centers + (ci - n_configs / 2 + 0.5) * bar_width

    means  = []
    lo_err = []
    hi_err = []

    for model in MODEL_ORDER:
        if model in sub.index:
            row = sub.loc[model]
            means.append(row["accuracy_mean"])
            lo_err.append(row["accuracy_mean"] - row["accuracy_ci_lower"])
            hi_err.append(row["accuracy_ci_upper"] - row["accuracy_mean"])
        else:
            means.append(np.nan)
            lo_err.append(0)
            hi_err.append(0)

    bars = ax.bar(
        x_pos,
        means,
        width=bar_width,
        color=COLORS[config],
        hatch=HATCHES[config],
        edgecolor="black",
        linewidth=0.6,
        label=config,
        yerr=[lo_err, hi_err],
        capsize=3,
        error_kw={"elinewidth": 0.9, "ecolor": "#444444"},
        zorder=3,
    )

    # annotate accuracy value inside/above bar
    for rect, mean_val in zip(bars, means):
        if not np.isnan(mean_val):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                mean_val + max(hi_err) + 0.012,
                f"{mean_val:.2f}",
                ha="center", va="bottom",
                fontsize=6.5, color="#222222", rotation=90,
            )

# chance line
ax.axhline(0.50, color="#cc0000", linestyle="--", linewidth=1.2, zorder=2)
ax.text(x_centers[-1] + bar_width * 2, 0.503, "Chance (0.50)",
        color="#cc0000", fontsize=8, va="bottom")

# axes
ax.set_xticks(x_centers)
ax.set_xticklabels(MODEL_ORDER, fontsize=10)
ax.set_ylabel("Accuracy  (10×10 repeated CV)", fontsize=10)
ax.set_ylim(0.38, 0.85)
ax.set_xlim(x_centers[0] - bar_width * 2, x_centers[-1] + bar_width * 2.5)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)

# legend
legend_patches = [
    mpatches.Patch(facecolor=COLORS[c], edgecolor="black",
                   hatch=HATCHES[c], label=c)
    for c in CONFIG_ORDER
]
ax.legend(handles=legend_patches, title="Feature configuration",
          fontsize=8.5, title_fontsize=8.5,
          loc="upper left", framealpha=0.9)

ax.set_title(
    "Classification accuracy by feature configuration and classifier\n"
    "Error bars = 95% bootstrap confidence intervals",
    fontsize=10, fontweight="bold"
)

plt.tight_layout()
plt.savefig(OUT_DIR / "bar_accuracy_comparison.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "bar_accuracy_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: figures/bar_accuracy_comparison.pdf  +  .png")
plt.close()