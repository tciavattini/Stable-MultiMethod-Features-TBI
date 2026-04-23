import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

# ============================================================
# 1) LOAD FULL DATASET 
# ============================================================
df_full = pd.read_pickle("../dataset/df_ready.pkl").copy()

# ============================================================
# 2) EXCLUSION LOGIC (>= 3 deltas among 4 domains)
# ============================================================
t0_domains = [
    "T0_severe_fatigue_rate",
    "T0_muscle_pain_rate",
    "T0_symp_today_rate",
    "T0_mood_rate",
]
t2_domains = [
    "T2_severe_fatigue_rate",
    "T2_muscle_pain_rate",
    "T2_symp_today_rate",
    "T2_mood_rate",
]

missing_t0 = [c for c in t0_domains if c not in df_full.columns]
missing_t2 = [c for c in t2_domains if c not in df_full.columns]
if missing_t0 or missing_t2:
    raise ValueError(f"Missing columns. T0 missing={missing_t0} | T2 missing={missing_t2}")

# Ensure numeric
for c in t0_domains + t2_domains:
    df_full[c] = pd.to_numeric(df_full[c], errors="coerce")

# Compute deltas (T2 - T0) for each domain
delta_cols = []
for c0, c2 in zip(t0_domains, t2_domains):
    dcol = "d_" + c2.replace("T2_", "")  
    df_full[dcol] = df_full[c2] - df_full[c0]
    delta_cols.append(dcol)

# Count how many deltas are available (non-missing)
df_full["n_delta_available"] = df_full[delta_cols].notna().sum(axis=1)

# Excluded if fewer than 3 deltas
df_full["excluded"] = (df_full["n_delta_available"] < 3).astype(int)

n_excluded = int(df_full["excluded"].sum())
n_included = int((df_full["excluded"] == 0).sum())
print(f"Excluded: {n_excluded}, Included: {n_included}")

df_exc = df_full.loc[df_full["excluded"] == 1].copy()
df_inc = df_full.loc[df_full["excluded"] == 0].copy()

# ============================================================
# 3) Baseline comparisons: excluded vs included
# ============================================================
variables = {
    "age": ("continuous", "Age (years)"),
    "gender": ("binary", "Female sex"),
    "Q5_tick_bite": ("binary", "Tick bite history"),
    "Q4_Chronic_previous": ("binary", "Previous chronic conditions"),
    "Q50_antibiotic": ("binary", "Prior antibiotic treatment"),
    "T0_severe_fatigue_rate": ("continuous", "Severe fatigue severity (T0)"),
    "T0_muscle_pain_rate": ("continuous", "Muscle pain severity (T0)"),
    "T0_symp_today_rate": ("continuous", "Overall symptom severity (T0)"),
    "T0_mood_rate": ("continuous", "Mood severity (T0)"),
    "num_total_symptoms": ("continuous", "Total number of symptoms"),
    "num_positive_markers": ("continuous", "Number of positive serological markers"),
}

def _as_binary_01(x: pd.Series) -> pd.Series:
    if x.dtype == bool:
        return x.astype(float)
    x_num = pd.to_numeric(x, errors="coerce")
    uniq = set(x_num.dropna().unique().tolist())
    if uniq.issubset({0, 1}):
        return x_num
    x_str = x.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "y": 1, "true": 1, "t": 1, "no": 0, "n": 0, "false": 0, "f": 0}
    x_map = x_str.map(mapping)
    x_map = x_map.where(~x.isna(), np.nan)
    return x_map.astype(float) if x_map.notna().sum() > 0 else x_num

def _format_mean_sd(s: pd.Series) -> str:
    return f"{s.mean():.1f} ± {s.std(ddof=1):.1f}" if len(s) > 0 else "---"

def _format_count_pct(pos: float, n: int) -> str:
    return f"{int(pos)} ({(pos/n)*100:.1f}%)" if n > 0 else "---"

def _pformat(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

results = []
for var, (vtype, label) in variables.items():
    if var not in df_full.columns:
        print(f"Skipping '{var}' (not found).")
        continue

    n_missing_total = int(df_full[var].isna().sum())

    if vtype == "continuous":
        exc_data = pd.to_numeric(df_exc[var], errors="coerce").dropna()
        inc_data = pd.to_numeric(df_inc[var], errors="coerce").dropna()

        exc_str = _format_mean_sd(exc_data)
        inc_str = _format_mean_sd(inc_data)

        if len(exc_data) >= 2 and len(inc_data) >= 2:
            _, p = mannwhitneyu(exc_data, inc_data, alternative="two-sided")
        else:
            p = np.nan

    else:  # binary
        exc_bin = _as_binary_01(df_exc[var]).dropna()
        inc_bin = _as_binary_01(df_inc[var]).dropna()

        exc_pos = float(exc_bin.sum())
        inc_pos = float(inc_bin.sum())

        exc_str = _format_count_pct(exc_pos, len(exc_bin))
        inc_str = _format_count_pct(inc_pos, len(inc_bin))

        a = int(exc_pos)
        b = int(len(exc_bin) - exc_pos)
        c = int(inc_pos)
        d = int(len(inc_bin) - inc_pos)
        table = np.array([[a, b], [c, d]])

        try:
            if table.min() < 5:
                _, p = fisher_exact(table)
            else:
                _, p, _, _ = chi2_contingency(table)
        except Exception:
            p = np.nan

    results.append({
        "Variable": label,
        f"Excluded (n={n_excluded})": exc_str,
        f"Included (n={n_included})": inc_str,
        "Missing (total)": n_missing_total,
        "p-value": _pformat(p),
    })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# ============================================================
# 4) SAVE
# ============================================================
out_csv = "../results/baseline_characteristics/excluded_vs_included.csv"
df_results.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")