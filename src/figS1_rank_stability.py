"""
Supplementary Figure S1.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from itertools import combinations
from pathlib import Path

# ── global font / style settings ──────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,             # base size
    "axes.titlesize": 14,        # panel titles
    "axes.labelsize": 13,        # axis labels
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.titlesize": 15,
})

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_ROOT    = Path("../results_fe/stability_top30_exp14")
STABLE_CSV  = OUT_ROOT / "stable_top_features.csv"
OUT_DIR     = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

R           = 100      # number of seeds
TOPK        = 30       # features selected per seed
FREQ_THRESH = 0.70     # stable feature threshold

# Domain colour map (same as fig2)
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
    "Q14_Impact_symp_employment":   "Demographic",
    "age":                          "Demographic",
    "CD3%":                         "Immunological",
    "Q12_trt_care_rate":            "Treatment",
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
    "Other":         "#969696",
}
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
    "Q53_antib_symp_improv":        "Sympt. improv. (antibiotics)",
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


# ── helper: scores → ranks (matches stability_top30.py exactly) ──────────────
def scores_to_ranks(scores, higher_is_better=True):
    s = pd.Series(scores)
    ascending = not higher_is_better
    return s.rank(ascending=ascending, method="average").to_numpy()


# ── load stable features CSV ──────────────────────────────────────────────────
df_stable = pd.read_csv(STABLE_CSV)

# flexible column detection
name_col  = next(c for c in ["feature_name", "feature", "name"] if c in df_stable.columns)
freq_col  = next(c for c in ["selected_freq", "freq", "frequency"] if c in df_stable.columns)
rank_col  = next(c for c in ["mean_consensus_rank", "mean_rank", "rank"] if c in df_stable.columns)
idx_col   = next(c for c in ["feature_idx", "feature_index", "idx"] if c in df_stable.columns)

stable_names  = df_stable[name_col].tolist()
stable_freqs  = df_stable[freq_col].values
stable_ranks  = df_stable[rank_col].values
stable_idxs   = df_stable[idx_col].values.astype(int)

n_stable = len(stable_names)
p        = None   # total features — detected below


# ── load per-seed consensus score vectors ─────────────────────────────────────
print("Loading per-seed consensus score vectors…")
S_all      = []
topk_lists = []
loaded     = 0

for seed in range(R):
    run_dir = OUT_ROOT / f"seed_{seed:03d}"

    # Try reconstructing S from raw scorer files (preferred — matches pipeline)
    svm_f = run_dir / "svm_single.txt"
    knn_f = run_dir / "knn_single.txt"
    net_f = run_dir / "net_single.txt"
    r2_f  = run_dir / "r2.txt"
    ent_f = run_dir / "entropies.txt"

    if all(f.exists() for f in [svm_f, knn_f, net_f, r2_f, ent_f]):
        svm = np.loadtxt(svm_f)
        knn = np.loadtxt(knn_f)
        net = np.loadtxt(net_f)
        r2  = np.loadtxt(r2_f)
        ent = np.loadtxt(ent_f)

        if p is None:
            p = len(svm)

        S = (
            scores_to_ranks(svm) +
            scores_to_ranks(knn) +
            scores_to_ranks(net) +
            scores_to_ranks(r2) +
            scores_to_ranks(ent)
        ) / 5.0

        S_all.append(S)
        topk_lists.append(np.argsort(S)[:TOPK])
        loaded += 1

    else:
        # Fallback: load saved top-k indices only (no full S vector)
        topk_f = run_dir / f"top{TOPK}_idx.txt"
        if topk_f.exists():
            topk_lists.append(np.loadtxt(topk_f, dtype=int))
            loaded += 1

print(f"  Loaded {loaded}/{R} seeds  ({len(S_all)} with full S vectors)")

if p is None:
    p = int(stable_idxs.max()) + 10   # rough fallback


# ── Panel A: pairwise Spearman ρ distribution ─────────────────────────────────
print("Computing pairwise Spearman ρ…")
rho_vals = []

if len(S_all) >= 2:
    # Full S vectors available — compute all pairs
    for i, j in combinations(range(len(S_all)), 2):
        r, _ = spearmanr(S_all[i], S_all[j])
        rho_vals.append(r)
else:
    # Fallback: binary selection vectors
    Z = np.zeros((len(topk_lists), p), dtype=np.int8)
    for r_idx, sel in enumerate(topk_lists):
        Z[r_idx, sel] = 1
    for i, j in combinations(range(len(topk_lists)), 2):
        r, _ = spearmanr(Z[i], Z[j])
        rho_vals.append(r)

rho_vals = np.array(rho_vals)
print(
    f"  Spearman ρ:  mean={rho_vals.mean():.4f}  "
    f"median={np.median(rho_vals):.4f}  "
    f"std={rho_vals.std():.4f}  min={rho_vals.min():.4f}"
)


# ── Panel B: per-feature selection frequency (all features, sorted desc) ─────
if len(topk_lists) > 0:
    counts = np.zeros(p, dtype=int)
    for sel in topk_lists:
        counts[np.asarray(sel, dtype=int)] += 1
    freqs_all = counts / len(topk_lists)
    freqs_sorted = np.sort(freqs_all)[::-1]
else:
    freqs_sorted = stable_freqs   # fallback


# ── Panel C: scatter — mean rank vs frequency (stable features) ───────────────
domains = [DOMAIN_MAP.get(n, "Other") for n in stable_names]
colors  = [DOMAIN_COLORS[d] for d in domains]
labels  = [LABEL_MAP.get(n, n) for n in stable_names]


# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 6.5))
gs = gridspec.GridSpec(
    1, 3,
    figure=fig,
    wspace=0.42,
    width_ratios=[1.1, 1.3, 1.35]
)

# ── Panel A ───────────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0])

n_bins = min(40, max(10, len(rho_vals) // 200))
ax_a.hist(
    rho_vals,
    bins=n_bins,
    color="#4393c3",
    edgecolor="white",
    linewidth=0.5,
    alpha=0.85
)

ax_a.axvline(
    rho_vals.mean(),
    color="#b2182b",
    linewidth=2.0,
    linestyle="-",
    label=f"Mean = {rho_vals.mean():.3f}"
)
ax_a.axvline(
    np.median(rho_vals),
    color="#fdae61",
    linewidth=1.8,
    linestyle="--",
    label=f"Median = {np.median(rho_vals):.3f}"
)

ax_a.set_xlabel("Pairwise Spearman ρ\n(consensus rank vectors)")
ax_a.set_ylabel("Frequency (seed pairs)")
ax_a.set_title(
    "A  Rank concordance across\n100 resampling iterations",
    fontweight="bold",
    loc="left",
    pad=10
)
ax_a.legend(framealpha=0.95, loc="upper left")
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.tick_params(labelsize=11)

# ── Panel B ───────────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[1])

x_all = np.arange(len(freqs_sorted))
stable_mask = freqs_sorted >= FREQ_THRESH

ax_b.bar(
    x_all[~stable_mask],
    freqs_sorted[~stable_mask],
    color="#cccccc",
    width=1.0,
    linewidth=0
)
ax_b.bar(
    x_all[stable_mask],
    freqs_sorted[stable_mask],
    color="#2166ac",
    width=1.0,
    linewidth=0,
    label=f"Stable (≥{int(FREQ_THRESH * 100)}%): n={stable_mask.sum()}"
)

ax_b.axhline(
    FREQ_THRESH,
    color="#b2182b",
    linewidth=1.8,
    linestyle="--",
    label=f"Threshold = {FREQ_THRESH:.0%}"
)

ax_b.set_xlabel(f"Features (sorted by selection frequency, n={len(freqs_sorted)})")
ax_b.set_ylabel("Selection frequency")
ax_b.set_title(
    "B  Selection frequency across\n100 resampling iterations",
    fontweight="bold",
    loc="left",
    pad=10
)
ax_b.set_xlim(-1, len(freqs_sorted))
ax_b.set_ylim(0, 1.05)
ax_b.legend(framealpha=0.95, loc="upper right")
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.tick_params(labelsize=11)

# ── Panel C ───────────────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[2])

ax_c.scatter(
    stable_freqs,
    stable_ranks,
    c=colors,
    s=85,
    alpha=0.9,
    edgecolors="white",
    linewidths=0.7,
    zorder=3
)

# annotate top stable features
top_mask = stable_freqs >= 0.95
for i in np.where(top_mask)[0]:
    ax_c.annotate(
        labels[i],
        xy=(stable_freqs[i], stable_ranks[i]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=9,
        color="#222222",
        va="center"
    )

ax_c.invert_yaxis()   # lower rank = better = at top
ax_c.set_xlabel("Selection frequency")
ax_c.set_ylabel("Mean consensus rank (lower = better)")
ax_c.set_title(
    "C  Stable features — frequency\nvs mean consensus rank",
    fontweight="bold",
    loc="left",
    pad=10
)
ax_c.set_xlim(FREQ_THRESH - 0.04, 1.03)
ax_c.grid(linestyle="--", alpha=0.3, zorder=0)
ax_c.spines[["top", "right"]].set_visible(False)
ax_c.tick_params(labelsize=11)

# domain legend (Panel C)
domain_patches = [
    mpatches.Patch(
        facecolor=col,
        label=dom,
        edgecolor="white",
        alpha=0.9
    )
    for dom, col in DOMAIN_COLORS.items()
    if dom in domains
]
ax_c.legend(
    handles=domain_patches,
    title="Domain",
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
    framealpha=0.95
)

# ── figure title ──────────────────────────────────────────────────────────────
fig.suptitle(
    "Feature selection stability across 100 independent resampling iterations\n"
    f"(10 shuffles × 80/20 stratified splits; 5-method consensus; top-{TOPK} per iteration)",
    fontweight="bold",
    y=1.08
)

plt.savefig(OUT_DIR / "suppfig_rank_stability_bigger.pdf", bbox_inches="tight")
plt.savefig(OUT_DIR / "suppfig_rank_stability_bigger.png", dpi=300, bbox_inches="tight")
print("Saved: figures/suppfig_rank_stability_bigger.pdf  +  .png")
plt.close()