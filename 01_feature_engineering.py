"""
Converted from notebook: 1_feature_engineering.ipynb
"""

# # Feature Engineering for Persistent Lyme Disease Study
# 
# **Purpose:** Transform raw survey + lab data into clean numeric features  
# **Input:** `dataset_correct.xlsx` (301 patients × 155+ raw columns)  
# **Output:** `df_ready.pkl` (301 patients × 152 numeric features)  
# 
# **Author:** [Your Name]  
# **Date:** 2025-01-28

import numpy as np
import pandas as pd
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

print("Libraries loaded successfully")

# ## 1. Load and Initial Cleaning

# Load raw data
file_path = "../dataset/dataset_correct_final.xlsx"
df = pd.read_excel(file_path)
print("Original shape:", df.shape)

# Drop metadata rows (first 6 rows)
df = df.drop(df.index[0:6]).reset_index(drop=True)
print("After dropping first 6 rows:", df.shape)

# Normalize column names
df.columns = df.columns.astype(str).str.strip()

# Strip whitespace from all string cells
def _strip_cell(x):
    if isinstance(x, str):
        x = x.strip()
        return x if x != "" else np.nan
    return x

df = df.map(_strip_cell)

# Replace standard missing value tokens globally
df = df.replace({"US": np.nan, "NA": np.nan, "nan": np.nan, "": np.nan})

print("\n✓ Initial cleaning complete")

# 1. Percentage of missing values per column
missing_pct = df.isna().mean() * 100
missing_pct = missing_pct.rename("missing_pct")

# 2. Median per feature (numeric columns only)
medians = df.median(numeric_only=True)
medians = medians.rename("median")

# 3. Combine into a single table
summary = pd.concat([missing_pct, medians], axis=1)

# Optional: sort by missingness
summary = summary.sort_values("missing_pct", ascending=False)

summary

# Drop columns with >70% missing values
threshold = 1
missing_ratio = df.isna().mean()
cols_to_drop = missing_ratio[missing_ratio == threshold].index.tolist()

if cols_to_drop:
    print(f"Dropping {len(cols_to_drop)} columns with >{int(threshold*100)}% missing:")
    print(f"  {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
else:
    print(f"✓ No columns exceed {int(threshold*100)}% missingness threshold")

print(f"\nFinal shape: {df.shape}")

# 1. Percentage of missing values per column
missing_pct = df.isna().mean() * 100
missing_pct = missing_pct.rename("missing_pct")

# 2. Median per feature (numeric columns only)
medians = df.median(numeric_only=True)
medians = medians.rename("median")

# 3. Combine into a single table
summary = pd.concat([missing_pct, medians], axis=1)

# Optional: sort by missingness
summary = summary.sort_values("missing_pct", ascending=False)

summary

# ## 2. Recode Core Binary Variables

# Gender: F=1, M=0
if "gender" in df.columns:
    df["gender"] = df["gender"].replace({"F": 1, "M": 0, "f": 1, "m": 0})
    df["gender"] = pd.to_numeric(df["gender"], errors="coerce")
    df["gender"] = df["gender"].where(
        df["gender"].isin([0, 1]) | df["gender"].isna(), 
        np.nan
    )
    print("✓ Gender recoded (F=1, M=0)")

# NVRLIgG: P=1, N=0
if "NVRLIgG" in df.columns:
    df["NVRLIgG"] = df["NVRLIgG"].replace({"US": np.nan, "NA": np.nan, "": np.nan, "nan": np.nan})
    df["NVRLIgG"] = df["NVRLIgG"].replace({"P": 1, "N": 0, "p": 1, "n": 0})
    df["NVRLIgG"] = pd.to_numeric(df["NVRLIgG"], errors="coerce")
    df["NVRLIgG"] = df["NVRLIgG"].where(
        df["NVRLIgG"].isin([0, 1]) | df["NVRLIgG"].isna(), 
        np.nan)
    print("✓ NVRLIgG recoded (P=1, N=0)")

# ## 3. Helper Function for Y/N Binary Columns

def _clean_binary_yn(cols):
    """
    Convert Y/N columns to 0/1.
    Returns list of columns that actually existed in df.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return cols
    
    # Replace missing tokens
    df[cols] = df[cols].replace(["US", "NA", "", " "], np.nan)
    
    # Strip strings if object dtype
    for c in cols:
        df[c] = df[c].str.strip()
        df[c] = df[c].replace(["US", "NA", "", " ", "na"], np.nan)
    
    # Convert Y/N to 1/0
    df[cols] = df[cols].replace({"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0, "P": 1})
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    df[cols] = df[cols].where(df[cols].isin([0, 1]) | df[cols].isna(), np.nan)
    
    return cols

print("✓ Helper function defined")

# ## 4. Engineer Symptom Domain Features

# Skin symptoms
skin_cols = ["Q15 Bulls Eye", "Q16. Rash"]
skin_cols = _clean_binary_yn(skin_cols)
if skin_cols:
    all_missing = df[skin_cols].isna().all(axis=1)
    df["skin_symp"] = np.where(
        all_missing, 
        np.nan, 
        (df[skin_cols] == 1).any(axis=1).astype(int)
    )
    df["num_skin_symp"] = df[skin_cols].sum(axis=1, min_count=1)
    print("✓ Skin symptoms engineered")

# General wellbeing
gw_cols = ["Q17. Sweats", "Q18 Sore throat", "Q19. Headac", "Q20. Swglands", "Q21. Sev Fat"]
gw_cols = _clean_binary_yn(gw_cols)
if gw_cols:
    all_missing = df[gw_cols].isna().all(axis=1)
    df["general_wellbeing_symp"] = np.where(
        all_missing, 
        np.nan, 
        (df[gw_cols] == 1).any(axis=1).astype(int)
    )
    df["num_general_wellbeing_symp"] = df[gw_cols].sum(axis=1, min_count=1)
    print("✓ General wellbeing symptoms engineered")

# Cardiac
card_cols = ["Q24. Chest P.", "Q25. ShortB", "Q26. Palpit", "Q27. Lighthead."]
card_cols = _clean_binary_yn(card_cols)
if card_cols:
    all_missing = df[card_cols].isna().all(axis=1)
    df["cardiac_symp"] = np.where(
        all_missing, 
        np.nan, 
        (df[card_cols] == 1).any(axis=1).astype(int)
    )
    df["num_cardiac_symp"] = df[card_cols].sum(axis=1, min_count=1)
    print("✓ Cardiac symptoms engineered")

# Rheumatological
rheum_cols = ["Q28. JointP", "Q29.Moving", "Q30. Intensity", "Q31. JointSw", "Q32. MuscW.le", "Q33.MuscP"]
rheum_cols = _clean_binary_yn(rheum_cols)
if rheum_cols:
    all_missing = df[rheum_cols].isna().all(axis=1)
    df["rheumatological_symp"] = np.where(
        all_missing, 
        np.nan, 
        (df[rheum_cols] == 1).any(axis=1).astype(int)
    )
    df["num_rheumatological_symp"] = df[rheum_cols].sum(axis=1, min_count=1)
    print("✓ Rheumatological symptoms engineered")

# Neurological
neuro_cols = ["Q35Facial", "Q36ArmHan", "Q37Numbn", "Q38Concen", "Q39Sleep", "Q40Vision", "Q41. Neck", "Q42Tinnit"]
neuro_cols = _clean_binary_yn(neuro_cols)
if neuro_cols:
    all_missing = df[neuro_cols].isna().all(axis=1)
    df["neurological_symp"] = np.where(
        all_missing, 
        np.nan, 
        (df[neuro_cols] == 1).any(axis=1).astype(int)
    )
    df["num_neurological_symp"] = df[neuro_cols].sum(axis=1, min_count=1)
    print("✓ Neurological symptoms engineered")

# Psychological
psy_cols = ["Q43Person", "Q44Mood", "Q46Anger", "Q47Anxiety"]
psy_cols = _clean_binary_yn(psy_cols)
if psy_cols:
    all_missing = df[psy_cols].isna().all(axis=1)
    df["psychological_symp"] = np.where(
        all_missing, 
        np.nan, 
        (df[psy_cols] == 1).any(axis=1).astype(int)
    )
    df["num_psychological_symp"] = df[psy_cols].sum(axis=1, min_count=1)
    print("✓ Psychological symptoms engineered")

# Total symptom count across all domains
domain_counts = [
    "num_skin_symp",
    "num_general_wellbeing_symp",
    "num_cardiac_symp",
    "num_rheumatological_symp",
    "num_neurological_symp",
    "num_psychological_symp",
]
domain_counts = [c for c in domain_counts if c in df.columns]
if domain_counts:
    df["num_total_symptoms"] = df[domain_counts].sum(axis=1, min_count=1)
    print("✓ Total symptom count created")

# ## 6. Co-infections (ELISPOT)

coinf_cols = [
    "Babesia", "Bartonella H", "Myco Pne", "Ehrl Ana", "Rickettsia", "Epstein B",
    "Chlamydia P", "Chlamyd trac", "Cytomigalo", "VZV IgG", "Anaplasma Phago",
    "Herpes Simplex", "Aspergillas", "Yersinia.1", "Candida"
]
coinf_cols = _clean_binary_yn(coinf_cols)

if coinf_cols:
    all_missing = df[coinf_cols].isna().all(axis=1)
    df["coinfections_elispot_any"] = np.where(
        all_missing, 
        np.nan, 
        (df[coinf_cols] == 1).any(axis=1).astype(int)
    )
    df["coinfections_elispot_tested"] = df[coinf_cols].notna().sum(axis=1)
    df.loc[all_missing, "coinfections_elispot_tested"] = np.nan
    print("✓ Co-infection ELISPOT features created")

# ============================================================
# 7) Borrelia ELISPOT (binary Y/N) -> any + pos_count
# ============================================================
bb_cols = ["BB Full Anti", "BB Osp Mix", "BBLFA"]
bb_cols = _clean_binary_yn(bb_cols)

if bb_cols:
    all_missing = df[bb_cols].isna().all(axis=1)
    df["borrelia_elispot"] = np.where(all_missing, np.nan, (df[bb_cols] == 1).any(axis=1).astype(int))
    df["num_borrelia_elispot_pos"] = df[bb_cols].sum(axis=1, min_count=1)

# ## 7. Serology Panel (IgG/IgM + Clusters)

sero_all = [
    "b.burg+afz+gar.IgG", "b.burg+afz+gar+IgM", "B.Burg Round Body IgG", "B.Burg Round body IgM",
    "BaB M IgG", "BaB M IgM", "Bart H IgG", "Bart H IgM", "Ehrl C IgG", "Ehrl IgM",
    "Rick Ak IgG", "Rick Ak IgM", "Coxs IgG", "Coxs IgM", "Epst B IgG", "Epst B IgM",
    "Hum Par IgG", "Hum Par IgM", "Mycop Pneu IgG", "Mycop Pneu IgM", "Chlamydia pneumoni",
    "HSV / IgG", "VZV/ IgG", "Toxoplasma", "Yersinia", "HSV/1gG"
]

sero_cols = [c for c in sero_all if c in df.columns]
if sero_cols:
    # Clean
    df[sero_cols] = df[sero_cols].replace(["US", "NA", "", " "], np.nan)
    for c in sero_cols:
        df[c] = df[c].str.strip()
        df[c] = df[c].replace(["US", "NA", "", " "], np.nan)

    # Map P/N to 1/0
    df[sero_cols] = df[sero_cols].replace({"P": 1, "N": 0, "p": 1, "n": 0})
    df[sero_cols] = df[sero_cols].apply(pd.to_numeric, errors="coerce")
    df[sero_cols] = df[sero_cols].where(
        df[sero_cols].isin([0, 1]) | df[sero_cols].isna(), 
        np.nan
    )

    # IgG and IgM totals
    igG_cols = [c for c in sero_cols if "igg" in c.lower()]
    igM_cols = [c for c in sero_cols if "igm" in c.lower()]

    if igG_cols:
        df["IgG_total"] = df[igG_cols].sum(axis=1, min_count=1)
    if igM_cols:
        df["IgM_total"] = df[igM_cols].sum(axis=1, min_count=1)

    # Overall serology counts
    df["sero_tested_count"] = df[sero_cols].notna().sum(axis=1)
    df["num_positive_markers"] = df[sero_cols].sum(axis=1, min_count=1)

    # Pathogen clusters
    clusters = {
        "cluster_bburg":      ["burg"],
        "cluster_babesia":    ["bab"],
        "cluster_bartonella": ["bart"],
        "cluster_ehrlichia":  ["ehrl"],
        "cluster_rickettsia": ["rick"],
        "cluster_coxsackie":  ["coxs"],
        "cluster_ebv":        ["epst"],
        "cluster_parvo":      ["hum par"],
        "cluster_mycoplasma": ["mycop"],
        "cluster_chlamydia":  ["chlamydia"],
        "cluster_hsv":        ["hsv"],
        "cluster_vzv":        ["vzv"],
        "cluster_toxoplasma": ["toxoplasma"],
        "cluster_yersinia":   ["yersinia"],
    }

    for new_name, patterns in clusters.items():
        patterns = [p.lower() for p in patterns]
        cc = [c for c in sero_cols if any(p in c.lower() for p in patterns)]
        if cc:
            df[new_name] = df[cc].sum(axis=1, min_count=1)
    
    print(f"✓ Serology features created: IgG/IgM totals + {len(clusters)} pathogen clusters")

cols = ["Q7_tick_bite_where"]
cols = [c for c in cols if c in df.columns]

for col in cols:
    s = df[col]

    # strip solo se stringa (evita errori)
    if s.dtype == "object" or pd.api.types.is_string_dtype(s):
        s = s.str.strip()

    # placeholder → missing
    s = s.replace({"US": np.nan, "NA": np.nan, "": np.nan, "nan": np.nan})

    # presence indicator: 1 se presente, NA se missing
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")
    out[s.notna()] = 1

    df[col] = out

# ## 8. Mixed Survey Variables

cols = [
    "Q55_alternative_trt_success", "Q3_Outdoor_hobbies", "Q4_Chronic_previous",
    "Q5_tick_bite", "Q9_GP", "Q9a_GP_satisfaction", "Q10_Consultant",
    "Q10a_consultant_satisfaction", "Q14_Impact_symp_employment",
    "Q48_blood_analysis", "Q50_antibiotic", "Q53_antib_symp_improv",
    "Q54_alternative_trt", "persistent"
]

present = [c for c in cols if c in df.columns]

# minimal missing cleanup
df[present] = df[present].replace(
    ["US", "NA", "", " "], pd.NA
)

# minimal Y/N mapping (non destructive)
df[present] = df[present].replace(
    {"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0}
)

# Q49_blood_analysis_where -> one-hot encode
if "Q49_blood_analysis_where" in df.columns:
    c = "Q49_blood_analysis_where"
    s = df[c]
    s = s.str.strip()
    s = s.replace({"US": np.nan, "NA": np.nan, "": np.nan, "nan": np.nan, "N": np.nan})
    df[c] = s
    df = pd.get_dummies(df, columns=[c], prefix="blood", dummy_na=False, dtype=float)

print("✓ Mixed survey variables processed")

# ## 9. Rename T0 Symptom Rate Columns

# Rename T0 rate columns for clarity
rename_map = {
    "Q23_symp_today_rate": "T0_symp_today_rate",
    "Q34_muscle_pain_rate": "T0_muscle_pain_rate",
    "Q45_mood_rate": "T0_mood_rate",
    "Q22_severe_fatigue_rate": "T0_severe_fatigue_rate",
}

# Only rename columns that exist
rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
if rename_map:
    df = df.rename(columns=rename_map)
    print(f"✓ Renamed {len(rename_map)} T0 symptom rate columns")

# ## 10. Antibiotic Exposure Features

# Antibiotic dose columns
abx_cols = [
    "Cefuroxime_dose", "Rifampicin_dose", "Lymecyclin_dose",
    "Azithromycin_dose", "Clarithromycin_dose", "Doxycycline_dose",
    "Amoxicillin_dose", "Valoid_dose", "Malarone_dose", "Diflucan_dose"
]

# Keep only columns that exist
abx_cols = [c for c in abx_cols if c in df.columns]

if len(abx_cols) == 0:
    print("⚠ WARNING: No antibiotic dose columns found")
else:
    # Ensure numeric
    df[abx_cols] = df[abx_cols].apply(pd.to_numeric, errors="coerce")

    # Count number of antibiotics administered (any non-missing dose)
    df["num_antibiotics_administered"] = df[abx_cols].notna().sum(axis=1)

    print(f"✓ Antibiotic exposure features created from {len(abx_cols)} antibiotics")
    print(f"  Distribution: {df['num_antibiotics_administered'].value_counts().sort_index().to_dict()}")

if "ANA" in df.columns:
    df["ANA"] = df["ANA"].replace(["US", "NA", "", " "], pd.NA)
    df["ANA"] = df["ANA"].replace({"Y": 1, "N": 0, "y": 1, "n": 0, "Yes": 1, "No": 0, "P":1})
    df["ANA"] = pd.to_numeric(df["ANA"], errors="coerce")
    df["ANA"] = df["ANA"].where(df["ANA"].isin([0, 1]) | df["ANA"].isna(), np.nan)

# ## 11. Convert Boolean Columns to Numeric

# Convert any boolean columns (from get_dummies) to int
bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns
if len(bool_cols) > 0:
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"✓ Converted {len(bool_cols)} boolean columns to int")

# ## 12. Force Convert All Columns to Numeric

# Ensure IgA is numeric (manual fix from your code)
if "IgA" in df.columns:
    df["IgA"] = pd.to_numeric(df["IgA"], errors="coerce")

# Force all remaining non-numeric columns to numeric
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric_cols:
    print(f"⚠ Converting {len(non_numeric_cols)} non-numeric columns to numeric:")
    print(f"  {non_numeric_cols}")
    
    for col in non_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Verify all columns are now numeric
still_non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if still_non_numeric:
    print(f"❌ ERROR: These columns are still non-numeric: {still_non_numeric}")
    raise ValueError("Failed to convert all columns to numeric")
else:
    print("✓ All columns successfully converted to numeric")

# 1. Percentage of missing values per column
missing_pct = df.isna().mean() * 100
missing_pct = missing_pct.rename("missing_pct")

# 2. Median per feature (numeric columns only)
medians = df.median(numeric_only=True)
medians = medians.rename("median")

# 3. Combine into a single table
summary = pd.concat([missing_pct, medians], axis=1)

# Optional: sort by missingness
summary = summary.sort_values("missing_pct", ascending=False)

summary

# ## 13. Drop High-Missingness Columns

# Drop columns with >70% missing values
threshold = 0.70
missing_ratio = df.isna().mean()
cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

if cols_to_drop:
    print(f"Dropping {len(cols_to_drop)} columns with >{int(threshold*100)}% missing:")
    print(f"  {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
else:
    print(f"✓ No columns exceed {int(threshold*100)}% missingness threshold")

print(f"\nFinal shape: {df.shape}")

# ============================================================
# Remove ID and Constant Columns
# ============================================================

# Columns to always drop
cols_to_remove = ['ID']  # Identifier, not a feature

# Add any constant columns (zero variance)
numeric_df = df.select_dtypes(include=[np.number])
constant_cols = [col for col in numeric_df.columns if numeric_df[col].std(skipna=True) == 0]

if constant_cols:
    print(f"⚠ Found {len(constant_cols)} constant columns: {constant_cols}")
    cols_to_remove.extend(constant_cols)

# Remove duplicates and filter to existing columns
cols_to_remove = list(set(cols_to_remove))
cols_to_remove = [c for c in cols_to_remove if c in df.columns]

if cols_to_remove:
    print(f"Dropping {len(cols_to_remove)} columns: {cols_to_remove}")
    df = df.drop(columns=cols_to_remove)
    print(f"Shape after removal: {df.shape}")
else:
    print("✓ No problematic columns to remove")

# ## 14. Data Quality Audit

print("=" * 60)
print("DATAFRAME QUALITY AUDIT")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes.value_counts()}")

# Non-numeric columns (should be none)
non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
print(f"\nNon-numeric columns: {len(non_numeric)}")
if non_numeric:
    print(f"  ❌ WARNING: {non_numeric}")

# Missing values
missing_counts = df.isna().sum()
missing_counts = missing_counts[missing_counts > 0]
print(f"\nColumns with missing values: {len(missing_counts)} / {df.shape[1]}")
print(f"  Total missing cells: {df.isna().sum().sum()} / {df.shape[0] * df.shape[1]}")

# Inf values
numeric_df = df.select_dtypes(include=[np.number])
inf_counts = np.isinf(numeric_df).sum()
inf_counts = inf_counts[inf_counts > 0]
print(f"\nColumns with Inf values: {len(inf_counts)}")
if len(inf_counts) > 0:
    print(f"  ❌ WARNING: {inf_counts.index.tolist()}")

# ## 15. Column Type Classification

# Classify columns by cardinality
summary = []
numeric_df = df.select_dtypes(include=[np.number])

for col in numeric_df.columns:
    s = numeric_df[col]
    summary.append({
        "column": col,
        "n_unique": s.nunique(dropna=True),
        "n_nan": s.isna().sum(),
        "std": s.std(skipna=True)
    })

summary_df = pd.DataFrame(summary)

# Identify column types
constant_cols = summary_df[summary_df["std"] == 0]["column"].tolist()
binary_cols = summary_df[summary_df["n_unique"] == 2]["column"].tolist()
low_card_cols = summary_df[(summary_df["n_unique"] > 2) & (summary_df["n_unique"] <= 10)]["column"].tolist()
continuous_cols_10 = summary_df[summary_df["n_unique"] > 10]["column"].tolist()
continuous_cols = summary_df[summary_df["n_unique"] > 2]["column"].tolist()

print("\n" + "=" * 60)
print("COLUMN TYPE CLASSIFICATION")
print("=" * 60)
print(f"Constant columns (std=0): {len(constant_cols)}")
if constant_cols:
    print(f"  {constant_cols}")

print(f"\nBinary columns (n_unique=2): {len(binary_cols)}")
print(f"  Sample: {binary_cols[:5]}...")

print(f"\nLow-cardinality (3-10 unique): {len(low_card_cols)}")
print(f"  Sample: {low_card_cols[:5]}..." if low_card_cols else "  (none)")

print(f"\nContinuous (>10 unique): {len(continuous_cols_10)}")
print(f"  Sample: {continuous_cols[:5]}...")

# Save audit
summary_df.to_csv("../dataset/column_audit.csv", index=False)
print("\n✓ Saved: ../dataset/column_audit.csv")

feature_types = {}

for col in df.columns:
    nunique = df[col].nunique(dropna=True)
    dtype = df[col].dtype

    if df[col].dropna().isin([0, 1]).all():
        feature_types[col] = "binary"

    elif dtype != "object" and nunique > 15:
        feature_types[col] = "continuous"

    elif dtype != "object" and 2 <= nunique <= 10:
        feature_types[col] = "discrete_unknown"

    else:
        feature_types[col] = "categorical"


rows = []

MAX_UNIQUE_TO_SHOW = 20  # per evitare celle enormi

for col, assigned_type in feature_types.items():
    uniques = df[col].dropna().unique()

    # converti in lista Python standard
    uniques_list = sorted(uniques.tolist())

    # tronca se troppo lunga
    if len(uniques_list) > MAX_UNIQUE_TO_SHOW:
        uniques_repr = (
            uniques_list[:MAX_UNIQUE_TO_SHOW]
            + ["..."]
            + [f"(n_unique={len(uniques_list)})"]
        )
    else:
        uniques_repr = uniques_list

    rows.append({
        "column_name": col,
        "assigned_type": assigned_type,
        "unique_values": str(uniques_repr)
    })

df_feature_types = pd.DataFrame(rows)

# ordina per tipo per facilitare la revisione
df_feature_types = df_feature_types.sort_values("assigned_type")

# salva in Excel
output_path = "../dataset/feature_types_auto_with_uniques.xlsx"
df_feature_types.to_excel(output_path, index=False)

print(f"File salvato in: {output_path}")

# ## 16. Save Column Type Lists

# Save continuous and binary column lists for downstream use
pd.Series(continuous_cols).to_csv("../dataset/continuous_cols.txt", index=False, header=False)
pd.Series(binary_cols).to_csv("../dataset/binary_cols.txt", index=False, header=False)
pd.Series(low_card_cols).to_csv("../dataset/low_card_cols.txt", index=False, header=False)

print(f"✓ Saved: ../dataset/continuous_cols.txt ({len(continuous_cols)} columns)")
print(f"✓ Saved: ../dataset/binary_cols.txt ({len(binary_cols)} columns)")
print(f"✓ Saved: ../dataset/low_card_cols.txt ({len(low_card_cols)} columns)")

print("\n" + "=" * 60)
print("LOW-CARDINALITY COLUMNS: VALUE COUNTS")
print("=" * 60)

for col in low_card_cols:
    print(f"\n{col}")
    vc = numeric_df[col].value_counts(dropna=False).sort_index()
    print(vc)

# ## 17. Final Export

# Save engineered dataframe
out_xlsx = "../dataset/cleaned_file.xlsx"
out_pkl = "../dataset/df_ready.pkl"
out_csv = "../dataset/df_ready.csv"

df.to_excel(out_xlsx, index=False)
df.to_pickle(out_pkl)
df.to_csv(out_csv, index=False)

print("=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"Final shape: {df.shape}")
print(f"\nSaved:")
print(f"  - {out_xlsx}")
print(f"  - {out_pkl}")
print(f"  - {out_csv}")
print(f"\nReady for next step: Sample filtering and outcome definition")

# 1. Percentage of missing values per column
missing_pct = df.isna().mean() * 100
missing_pct = missing_pct.rename("missing_pct")

# 2. Median per feature (numeric columns only)
medians = df.median(numeric_only=True)
medians = medians.rename("median")

# 3. Combine into a single table
summary = pd.concat([missing_pct, medians], axis=1)

# Optional: sort by missingness
summary = summary.sort_values("missing_pct", ascending=False)

# Calculate the median of missing percentages across all features
median_missing_pct = summary['missing_pct'].median()

print(f"\nMedian missing percentage across all features: {median_missing_pct:.1f}%")
print(f"\n📝 TEXT FOR PAPER:")
print(f"...with a median of {median_missing_pct:.1f}% per feature")

summary

