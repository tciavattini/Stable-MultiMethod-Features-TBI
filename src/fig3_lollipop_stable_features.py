"""
Figure 3 — Main text.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
STABLE_CSV = Path("../results_fe/stability_top30_exp14/stable_top_features.csv")
OUT_DIR    = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── domain map ────────────────────────────────────────────────────────────────
DOMAIN_MAP = {
    "T0_severe_fatigue_rate":       "Symptom",
    "num_neurological_symp":        "Symptom",
    "Q20. Swglands":                "Symptom",
    "T0_muscle_pain_rate":          "Symptom",
    "T0_mood_rate":                 "Symptom",
    "Q27. Lighthead.":              "Symptom",
    "Q47Anxiety":                   "Symptom",
    "T0_symp_today_rate":           "Symptom",
    "Q35Facial":                    "Symptom",
    "Q53_antib_symp_improv":        "Treatment",
    "Q42Tinnit":                    "Symptom",
    "num_total_symptoms":           "Symptom",
    "Q40Vision":                    "Symptom",
    "CD8%":                         "Immunological",
    "Other_empl":                   "Demographic",
    "age":                          "Demographic",
    "CD3%":                         "Immunological",
    "Q12_trt_care_rate":            "Patient experience",
    "cardiac_symp":                 "Symptom",
    "Q30. Intensity":               "Symptom",
    "Q55_alternative_trt_success":  "Treatment",
    "BaB M IgG":                    "Immunological",
}

DOMAIN_COLORS = {
    "Symptom":       "#2166ac",
    "Immunological": "#d6604d",
    "Demographic":   "#4dac26",
    "Treatment":     "#8073ac",
    "Patient experience": "#e78ac3",
}
DEFAULT_COLOR = "#969696"

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
    "CD8%":                         "CD8\u207a T cells (%)",
    "Other_empl":                   "Other employment status",
    "age":                          "Age",
    "CD3%":                         "CD3\u207a T cells (%)",
    "Q12_trt_care_rate":            "Treatment received rate",
    "cardiac_symp":                 "Cardiac symptoms",
    "Q30. Intensity":               "Symptom intensity",
    "Q55_alternative_trt_success":  "Alternative treatment success",
    "BaB M IgG":                    r"Seropositivity for $\it{Babesia}$ IgG",
}

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(STABLE_CSV)

rank_col = next((c for c in ["mean_consensus_rank","mean_rank","rank"]
                 if c in df.columns), None)
freq_col = next((c for c in ["selected_freq","selection_frequency","freq","frequency"]
                 if c in df.columns), None)
name_col = next((c for c in ["feature_name","feature","name"]
                 if c in df.columns), None)

if None in (rank_col, freq_col, name_col):
    raise ValueError(f"Cannot detect columns. Found: {list(df.columns)}")

df = df.rename(columns={rank_col: "rank", freq_col: "freq", name_col: "feature"})
df = df.sort_values("rank").reset_index(drop=True)   # rank 1 = best

df["label"]  = df["feature"].map(LABEL_MAP).fillna(df["feature"])
df["domain"] = df["feature"].map(DOMAIN_MAP).fillna("Other")
df["lcolor"] = df["domain"].map(DOMAIN_COLORS).fillna(DEFAULT_COLOR)

n = len(df)
# y positions: best rank at top
y_pos = np.arange(n - 1, -1, -1)   # n-1 down to 0

# ── colour map for dots (frequency) ───────────────────────────────────────────
freq_min, freq_max = 0.70, 1.00
cmap   = cm.get_cmap("YlOrRd")
norm   = mcolors.Normalize(vmin=freq_min, vmax=freq_max)
dot_colors = [cmap(norm(f)) for f in df["freq"]]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 8))

# --- horizontal lines (stems) ------------------------------------------------
for i, (yi, row) in enumerate(zip(y_pos, df.itertuples())):
    ax.plot(
        [0, row.rank],
        [yi, yi],
        color=row.lcolor,
        linewidth=1.6,
        solid_capstyle="round",
        alpha=0.75,
        zorder=2,
    )

# --- dots at rank position ---------------------------------------------------
sc = ax.scatter(
    df["rank"],
    y_pos,
    c=dot_colors,
    s=110,
    zorder=4,
    edgecolors="#333333",
    linewidths=0.6,
)

# --- frequency annotation next to each dot -----------------------------------
for i, (yi, row) in enumerate(zip(y_pos, df.itertuples())):
    ax.text(
        row.rank + 1.0,
        yi,
        f"{row.freq:.0%}",
        va="center",
        ha="left",
        fontsize=9.5,
        color="#444444",
    )

# --- shaded region: top-10 (freq = 100%) ------------------------------------
# find the rank value at which freq drops below 1.0
always_selected = df[df["freq"] >= 0.999]
if len(always_selected) > 0:
    shade_xmax = always_selected["rank"].max() + 1.5
    ax.axvspan(0, shade_xmax, color="#ddeeff", alpha=0.40, zorder=0)
    ax.text(
        shade_xmax / 2, n - 0.2,
        "Selected in 100%\nof iterations",
        ha="center", va="top",
        fontsize=8, color="#2166ac", fontstyle="italic",
    )

# ── axes ─────────────────────────────────────────────────────────────────────
ax.set_yticks(y_pos)
ax.set_yticklabels(df["label"], fontsize=12)
ax.set_xlabel("Mean consensus rank  (lower = higher importance)", fontsize=12)
ax.set_xlim(left=0, right=df["rank"].max() + 4)
ax.set_ylim(-0.8, n - 0.2)
ax.xaxis.set_major_locator(plt.MultipleLocator(5))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.grid(axis="x", linestyle="--", alpha=0.30, zorder=0)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="y", length=0)

# ── legend 1: domain (line colour) ───────────────────────────────────────────
domain_patches = [
    mpatches.Patch(facecolor=col, label=dom, alpha=0.85)
    for dom, col in DOMAIN_COLORS.items()
    if dom in df["domain"].values
]
leg1 = ax.legend(
    handles=domain_patches,
    title="Feature domain",
    title_fontsize=10,
    fontsize=9,
    loc="upper right",
    framealpha=0.92,
    edgecolor="#cccccc",
)
ax.add_artist(leg1)

# ── legend 2: frequency colourbar ────────────────────────────────────────────
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.01, aspect=25)
cbar.set_label("Selection\nfrequency", fontsize=10)
cbar.ax.tick_params(labelsize=9)
cbar.set_ticks([0.70, 0.80, 0.90, 1.00])
cbar.set_ticklabels(["70%", "80%", "90%", "100%"])

ax.set_title(
    "Stable feature set: consensus rank and selection frequency\n"
    "(22 features selected in ≥70% of 100 resampling iterations)",
    fontsize=13, fontweight="bold", pad=10,
)

plt.tight_layout()
plt.savefig(OUT_DIR / "lollipop_stable_features_bigger.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "lollipop_stable_features_bigger.png", dpi=300, bbox_inches="tight")
print("Saved: figures/lollipop_stable_features_bigger.pdf  +  .png")
plt.close()