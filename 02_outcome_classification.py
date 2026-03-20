"""
Converted from notebook: 2_patients_classification.ipynb
"""

# # Outcome Definition: Percentile-Based Method (Tertiles)
# 
# **Purpose:** Define high responders vs non-responders using fixed percentile ranks  
# **Approach:** Adapted from Vendrow et al. (2023) GROC tertile-based classification  
# **Input:** `df_ready.pkl` (301 patients × 150 features)  
# **Output:** Binary outcome dataset (high vs non responders)  
# 
# **Method:**
# 1. Compute T2 - T0 differences for 4 symptom domains
# 2. Standardize to z-scores (direction-adjusted so positive = improvement)
# 3. Create composite score (mean of 4 z-scores)
# 4. **Classify by FIXED PERCENTILE RULE:**
#    - High responders: Top tertile (≥67th percentile)
#    - Low responders: Middle tertile (34th-66th percentile)
#    - Non-responders: Bottom tertile (≤33rd percentile)
# 5. Keep only High + Non for binary classification
# 
# **Key:** This is a **fixed rule** ("top/bottom third"), not data-driven thresholds
# 
# **Author:** [Your Name]  
# **Date:** 2025-01-28

import pandas as pd
import numpy as np
import os
from scipy import stats

print("✓ Libraries loaded")

# ## Configuration

# File paths
INPUT_PATH = "../data/df_ready.pkl"
OUTPUT_DIR = "../results/patient_classification/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters to analyze (T0 baseline measures)
PARAMETERS_T0 = [
    'T0_severe_fatigue_rate',
    'T0_muscle_pain_rate',
    'T0_symp_today_rate',
    'T0_mood_rate'
]

# Define improvement direction for each parameter
IMPROVEMENT_DIRECTION = {
    'severe_fatigue_rate': False,  # 1=no fatigue, 10=extreme → lower is better
    'muscle_pain_rate': False,      # 1=no pain, 10=severe → lower is better
    'symp_today_rate': True,        # 1=poorly, 10=very well → higher is better
    'mood_rate': True               # 1=low mood, 10=very good → higher is better
}

# Classification percentiles (FIXED RULE - like Vendrow's tertiles)
PERCENTILE_HIGH = 2/3  # Top tertile (≥67th percentile)
PERCENTILE_LOW = 1/3   # Bottom tertile (≤33rd percentile)

# Minimum domains required for valid composite score
MIN_DOMAINS = 3  # Require at least 3 of 4 domains to have data

print("✓ Configuration set")
print(f"   Classification rule: Fixed percentile tertiles")
print(f"      High responders: ≥{100*PERCENTILE_HIGH:.0f}th percentile (top third)")
print(f"      Non-responders: ≤{100*PERCENTILE_LOW:.0f}th percentile (bottom third)")
print(f"   Minimum domains: {MIN_DOMAINS} of {len(PARAMETERS_T0)}")

# ## 1. Load Data

print("="*60)
print("OUTCOME DEFINITION: PERCENTILE-BASED (TERTILES)")
print("="*60)

df = pd.read_pickle(INPUT_PATH)
print(f"\n1. Loaded data: {df.shape}")
print(f"   Patients: {len(df)}")

# Verify required columns exist
parameters_base = [p.replace('T0_', '') for p in PARAMETERS_T0]
t0_cols = PARAMETERS_T0
t2_cols = [c.replace('T0_', 'T2_') for c in t0_cols]

missing_t0 = [c for c in t0_cols if c not in df.columns]
missing_t2 = [c for c in t2_cols if c not in df.columns]

if missing_t0:
    raise ValueError(f"Missing T0 columns: {missing_t0}")
if missing_t2:
    raise ValueError(f"Missing T2 columns: {missing_t2}")

print(f"   ✓ All required T0 and T2 columns present")

# ## 2. Compute Differences (T2 - T0)

print(f"\n2. Computing differences (T2 - T0)...")

for param_base in parameters_base:
    col_t0 = f'T0_{param_base}'
    col_t2 = f'T2_{param_base}'
    col_diff = f'diff_{param_base}'
    
    # Ensure numeric
    df[col_t0] = pd.to_numeric(df[col_t0], errors='coerce')
    df[col_t2] = pd.to_numeric(df[col_t2], errors='coerce')
    
    # Compute difference (only where both values present)
    df[col_diff] = np.where(
        df[col_t0].notna() & df[col_t2].notna(),
        df[col_t2] - df[col_t0],
        np.nan
    )
    
    n_valid = df[col_diff].notna().sum()
    mean_diff = df[col_diff].mean()
    print(f"   {param_base}:")
    print(f"      Valid: {n_valid} patients")
    print(f"      Mean raw diff: {mean_diff:.2f}")

print(f"\n   ✓ Differences computed for all parameters")

# ## 3. Standardize to Z-Scores (Direction-Adjusted)

print(f"\n3. Standardizing differences to z-scores...")
print(f"   (Adjusting direction so positive z = improvement)\n")

for param_base in parameters_base:
    col_diff = f'diff_{param_base}'
    col_z = f'z_{param_base}'
    
    # Get the raw difference
    raw_diff = df[col_diff].dropna()
    
    if len(raw_diff) < 2:
        print(f"   ⚠ {param_base}: <2 valid values, setting z=NaN")
        df[col_z] = np.nan
        continue
    
    # Compute z-score
    mean_diff = raw_diff.mean()
    std_diff = raw_diff.std()
    
    if std_diff == 0:
        print(f"   ⚠ {param_base}: zero variance, setting z=0")
        df[col_z] = 0
        continue
    
    df[col_z] = (df[col_diff] - mean_diff) / std_diff
    
    # Reverse direction if needed (so positive z = improvement)
    higher_is_better = IMPROVEMENT_DIRECTION[param_base]
    if not higher_is_better:
        df[col_z] = -df[col_z]
    
    print(f"   {param_base}:")
    print(f"      Raw: mean={mean_diff:.2f}, SD={std_diff:.2f}")
    print(f"      Direction: {'↑' if higher_is_better else '↓'} is better")
    print(f"      Z-scores: {df[col_z].notna().sum()} computed")
    print()

# ## 4. Create Composite Improvement Score

print(f"\n4. Creating composite improvement score...\n")

z_cols = [f'z_{p}' for p in parameters_base]

# Composite = mean of z-scores (equal weight to all 4 domains)
df['composite_z'] = df[z_cols].mean(axis=1, skipna=True)

# Count how many domains contributed
df['n_domains_available'] = df[z_cols].notna().sum(axis=1)

# Filter for patients with sufficient data
df_complete = df[df['n_domains_available'] >= MIN_DOMAINS].copy()

n_excluded_missing = len(df) - len(df_complete)
print(f"   Required: ≥{MIN_DOMAINS} of {len(parameters_base)} domains")
print(f"   Excluded: {n_excluded_missing} patients (insufficient data)")
print(f"   Remaining: {len(df_complete)} patients\n")

# Distribution statistics
print(f"   Composite z-score distribution:")
print(f"      Mean: {df_complete['composite_z'].mean():.3f}")
print(f"      Median: {df_complete['composite_z'].median():.3f}")
print(f"      SD: {df_complete['composite_z'].std():.3f}")
print(f"      Range: [{df_complete['composite_z'].min():.3f}, {df_complete['composite_z'].max():.3f}]")

# ## 5. Classify Patients by Percentile Rank (FIXED TERTILE RULE)

print(f"\n5. Classifying patients using FIXED PERCENTILE RULE...\n")

# Compute percentile rank for each patient (0 to 1)
df_complete['percentile_rank'] = df_complete['composite_z'].rank(pct=True)

# Apply FIXED tertile classification rule
df_complete['outcome_class'] = 'low-responder'

mask_high = df_complete['percentile_rank'] >= PERCENTILE_HIGH
mask_non = df_complete['percentile_rank'] <= PERCENTILE_LOW

df_complete.loc[mask_high, 'outcome_class'] = 'high-responder'
df_complete.loc[mask_non, 'outcome_class'] = 'non-responder'

print(f"   Classification rule (FIXED):")
print(f"      High responders: ≥{100*PERCENTILE_HIGH:.0f}th percentile (top tertile)")
print(f"      Low responders: {100*PERCENTILE_LOW:.0f}th-{100*PERCENTILE_HIGH:.0f}th percentile (middle tertile)")
print(f"      Non-responders: ≤{100*PERCENTILE_LOW:.0f}th percentile (bottom tertile)\n")

print(f"   Classification results:")
counts = df_complete['outcome_class'].value_counts().sort_index()
for cls in ['high-responder', 'low-responder', 'non-responder']:
    n = counts.get(cls, 0)
    pct = 100 * n / len(df_complete)
    print(f"      {cls}: {n} ({pct:.1f}%)")

# Report the actual z-score thresholds that correspond to these percentiles
threshold_high_value = df_complete['composite_z'].quantile(PERCENTILE_HIGH)
threshold_low_value = df_complete['composite_z'].quantile(PERCENTILE_LOW)
print(f"\n   Corresponding z-score cutoffs (for reference):")
print(f"      High: z ≥ {threshold_high_value:.3f}")
print(f"      Non: z ≤ {threshold_low_value:.3f}")

# ## 6. Create Binary Outcome (High vs Non)

print(f"\n6. Creating binary classification (High vs Non)...\n")

# Keep only extreme groups (exclude low responders)
df_binary = df_complete[
    df_complete['outcome_class'].isin(['high-responder', 'non-responder'])
].copy()

n_excluded_low = len(df_complete) - len(df_binary)

# Create binary outcome (1 = high, 0 = non)
df_binary['y'] = (df_binary['outcome_class'] == 'high-responder').astype(int)

print(f"   Excluded: {n_excluded_low} low responders (middle tertile)")
print(f"   Final sample: {len(df_binary)} patients\n")

n_high = (df_binary['y'] == 1).sum()
n_non = (df_binary['y'] == 0).sum()
imbalance_ratio = n_high / n_non if n_non > 0 else np.inf

print(f"   Distribution:")
print(f"      High responders (y=1): {n_high}")
print(f"      Non-responders (y=0): {n_non}")
print(f"      Class ratio (high/non): {imbalance_ratio:.2f}")

# Expected: approximately equal numbers (both ~33% of original)
if abs(imbalance_ratio - 1.0) > 0.3:
    print(f"\n   ⚠ Note: Imbalance ratio {imbalance_ratio:.2f} (expected ~1.0 for tertiles)")
    print(f"      This may occur due to ties at percentile boundaries")

# ## 7. Sample Exclusion Flow Diagram

print(f"\n7. Sample exclusion flow:\n")
print(f"   {len(df)} initial patients (from feature engineering)")
print(f"   - {n_excluded_missing} excluded (missing data: <{MIN_DOMAINS} domains)")
print(f"   = {len(df_complete)} with complete outcome data")
print(f"   - {n_excluded_low} excluded (low responders: middle tertile)")
print(f"   = {len(df_binary)} FINAL SAMPLE (top + bottom tertiles)")

# ## 8. Clinical Interpretation by Group

print(f"\n8. Clinical interpretation by group:\n")

for cls in ['high-responder', 'non-responder']:
    subset = df_binary[df_binary['outcome_class'] == cls]
    if len(subset) == 0:
        continue
    
    print(f"   {cls.upper()} (N={len(subset)}):")
    print(f"      Composite z-score: {subset['composite_z'].mean():.3f} ± {subset['composite_z'].std():.3f}")
    print(f"      Percentile range: {subset['percentile_rank'].min()*100:.0f}th-{subset['percentile_rank'].max()*100:.0f}th")
    print(f"      Raw improvements (mean):")
    
    for param_base in parameters_base:
        col_diff = f'diff_{param_base}'
        if col_diff in subset.columns:
            mean_diff = subset[col_diff].mean()
            # Adjust sign for interpretation
            direction = IMPROVEMENT_DIRECTION[param_base]
            if not direction:  # If lower is better, flip sign for reporting
                mean_diff = -mean_diff
            print(f"         {param_base}: {mean_diff:+.2f} points")
    print()

# ## 9. Save Results

print(f"\n9. Saving results...\n")

# Save all classifications (including low responders)
outcome_cols = ['composite_z', 'percentile_rank', 'n_domains_available', 'outcome_class']
df_complete[outcome_cols].to_excel(
    os.path.join(OUTPUT_DIR, 'outcome_percentile_all.xlsx'),
    index=True
)
print(f"   ✓ Saved: outcome_percentile_all.xlsx")

# Save binary outcome only (high + non)
binary_cols = ['composite_z', 'percentile_rank', 'outcome_class', 'y']
df_binary[binary_cols].to_excel(
    os.path.join(OUTPUT_DIR, 'outcome_percentile_binary.xlsx'),
    index=True
)
print(f"   ✓ Saved: outcome_percentile_binary.xlsx")

# Save full binary dataset (for feature selection)
df_binary.to_pickle(os.path.join(OUTPUT_DIR, 'df_binary_percentile.pkl'))
print(f"   ✓ Saved: df_binary_percentile.pkl")

print(f"\n   Note: Patient identifiers preserved as DataFrame index")

# ## 10. Validation Checks

print(f"\n10. Validation checks:\n")

# Check for duplicate indices
if df_binary.index.duplicated().any():
    print(f"   ❌ WARNING: Duplicate indices found!")
else:
    print(f"   ✓ No duplicate indices")

# Check for missing outcome
if df_binary['y'].isna().any():
    print(f"   ❌ WARNING: Missing outcome values!")
else:
    print(f"   ✓ No missing outcomes")

# Check subset relationship
original_indices = set(df.index)
binary_indices = set(df_binary.index)
is_subset = binary_indices.issubset(original_indices)
print(f"   ✓ Binary patients subset of original: {is_subset}")

# Check that high/non are from expected percentiles
high_subset = df_binary[df_binary['y'] == 1]
non_subset = df_binary[df_binary['y'] == 0]
print(f"\n   Percentile validation:")
print(f"      High responders: {high_subset['percentile_rank'].min()*100:.0f}th-{high_subset['percentile_rank'].max()*100:.0f}th percentile")
print(f"      Non-responders: {non_subset['percentile_rank'].min()*100:.0f}th-{non_subset['percentile_rank'].max()*100:.0f}th percentile")

# Check for reasonable sample sizes (should be ~33% each)
expected_per_group = len(df_complete) * (1/3)
if abs(n_high - expected_per_group) > 0.1 * expected_per_group:
    print(f"\n   ⚠ High responders: {n_high} (expected ~{expected_per_group:.0f})")
if abs(n_non - expected_per_group) > 0.1 * expected_per_group:
    print(f"   ⚠ Non-responders: {n_non} (expected ~{expected_per_group:.0f})")

# ## Summary

print("\n" + "="*60)
print("OUTCOME DEFINITION COMPLETE (PERCENTILE-BASED)")
print("="*60)

print(f"\nFinal dataset ready for feature selection:")
print(f"  File: df_binary_percentile.pkl")
print(f"  N = {len(df_binary)} patients")
print(f"  High responders (y=1): {n_high}")
print(f"  Non-responders (y=0): {n_non}")
print(f"  Outcome column: 'y'")

# Count features
outcome_cols_list = ['composite_z', 'percentile_rank', 'outcome_class', 'y', 'n_domains_available']
computed_cols = [c for c in df_binary.columns if c.startswith('diff_') or c.startswith('z_')]
metadata_cols_all = outcome_cols_list + computed_cols
feature_cols = [c for c in df_binary.columns if c not in metadata_cols_all]
print(f"  Features: {len(feature_cols)} columns")

print(f"\nInterpretation:")
print(f"  High responders: Top tertile (best-improving third of patients)")
print(f"  Non-responders: Bottom tertile (worst-improving third of patients)")
print(f"  Classification rule: FIXED percentile ranks (≥67th vs ≤33rd)")

print(f"\nAlignment with Vendrow et al.:")
print(f"  Vendrow: GROC scale with top/middle/bottom categories")
print(f"  This study: Composite z-score with top/middle/bottom tertiles")
print(f"  Both use: Fixed classification rules (not data-driven thresholds)")

print(f"\nNext step: Feature selection with proper nested CV")

age_mean = df_binary["age"].mean(skipna=True)
age_sd   = df_binary["age"].std(skipna=True)

print(f"Age: {age_mean:.1f} ± {age_sd:.1f}")

sex_table = (
    df_binary["gender"]
    .value_counts(dropna=True)
    .rename(index={1: "Female", 0: "Male"})
    .to_frame(name="n")
)

sex_table["%"] = sex_table["n"] / sex_table["n"].sum() * 100
sex_table["%"] = sex_table["%"].round(1)

sex_table

# Define the 4 baseline domains
baseline_domains = [
    'T0_severe_fatigue_rate',
    'T0_muscle_pain_rate', 
    'T0_symp_today_rate',
    'T0_mood_rate'
]

# Create a summary table
baseline_summary = []

for domain in baseline_domains:
    # Calculate statistics
    median = df_binary[domain].median()
    q1 = df_binary[domain].quantile(0.25)
    q3 = df_binary[domain].quantile(0.75)
    iqr = q3 - q1
    n_valid = df_binary[domain].notna().sum()
    
    # Format domain name (remove T0_ prefix for display)
    domain_name = domain.replace('T0_', '')
    
    baseline_summary.append({
        'Domain': domain_name,
        'N': n_valid,
        'Median': median,
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'Median (IQR)': f"{median:.1f} ({q1:.1f}-{q3:.1f})"
    })

# Create DataFrame
baseline_stats = pd.DataFrame(baseline_summary)

print("\n" + "="*70)
print("BASELINE SYMPTOM SEVERITY (Median and IQR)")
print("="*70)
print(baseline_stats.to_string(index=False))

# Calculate composite score statistics for high vs non-responders
print("\n" + "="*70)
print("COMPOSITE IMPROVEMENT SCORE STATISTICS")
print("="*70)

# Overall statistics (should be mean=0, SD=1 by construction)
overall_mean = df_binary['composite_z'].mean()
overall_sd = df_binary['composite_z'].std()

print(f"\nOverall composite score:")
print(f"  Mean: {overall_mean:.2f}")
print(f"  SD: {overall_sd:.2f}")

# High responders (y=1, ≥67th percentile)
high_responders = df_binary[df_binary['y'] == 1]
high_mean = high_responders['composite_z'].mean()
high_sd = high_responders['composite_z'].std()
high_n = len(high_responders)

print(f"\nHigh responders (≥67th percentile, n={high_n}):")
print(f"  Mean improvement: +{high_mean:.2f} SD")
print(f"  SD: {high_sd:.2f}")

# Non-responders (y=0, ≤33rd percentile)
non_responders = df_binary[df_binary['y'] == 0]
non_mean = non_responders['composite_z'].mean()
non_sd = non_responders['composite_z'].std()
non_n = len(non_responders)

print(f"\nNon-responders (≤33rd percentile, n={non_n}):")
print(f"  Mean improvement: {non_mean:.2f} SD")
print(f"  SD: {non_sd:.2f}")

# Difference between groups
difference = high_mean - non_mean

print(f"\n" + "-"*70)
print(f"Difference between groups: {difference:.2f} standard deviations")
print("-"*70)

# Format for paper
print(f"\n📝 TEXT FOR PAPER:")
print(f"The composite improvement score showed a continuous distribution")
print(f"(mean={overall_mean:.2f}, SD={overall_sd:.2f} by construction).")
print(f"High responders (≥67th percentile) had a mean improvement of")
print(f"+{high_mean:.2f} SD, while non-responders (≤33rd percentile) had")
print(f"a mean of {non_mean:.2f} SD, representing a {difference:.1f} standard")
print(f"deviation difference between groups.")

# Define feature categories
feature_categories = {
    'Demographics': ['age', 'gender', 'persistent', 'Q1_Residence_Ireland', 
                     'Q2_residence_outside_Ireland', 'Q3_Outdoor_hobbies',
                     'Working', 'Sick Leave', 'Retired', 'Caring res', 'Unemplo', 'Other_empl',
                     'Q14_Impact_symp_employment'],
    
    'Symptoms_baseline': ['T0_severe_fatigue_rate', 'T0_muscle_pain_rate', 
                          'T0_symp_today_rate', 'T0_mood_rate',
                          'Q15 Bulls Eye', 'Q16. Rash', 'Q17. Sweats', 'Q18 Sore throat',
                          'Q19. Headac', 'Q20. Swglands', 'Q21. Sev Fat', 
                          'Q24. Chest P.', 'Q25. ShortB', 'Q26. Palpit', 'Q27. Lighthead.',
                          'Q28. JointP', 'Q29.Moving', 'Q30. Intensity', 'Q31. JointSw',
                          'Q32. MuscW.le', 'Q33.MuscP', 'Q35Facial', 'Q36ArmHan',
                          'Q37Numbn', 'Q38Concen', 'Q39Sleep', 'Q40Vision', 'Q41. Neck',
                          'Q42Tinnit', 'Q43Person', 'Q44Mood', 'Q46Anger', 'Q47Anxiety'],
    
    'Laboratory': ['CD3%', 'CD3Total', 'CD8%', 'CD8-Suppr', 'CD4%', 'CD4-Helper',
                   'CD19Bcell', 'CD19%', 'H/SRATIO', 'CD57+NKCELLS',
                   'IgG', 'IgA', 'IgM', 'HgB', 'Platelets', 'neutrophils',
                   'Lymphocytes', 'WCC', 'RF', 'ANA', 'CRP', 'Iron', 'Transf',
                   '%transsat', 'Ferritin', 'Folate', 'CK', 'FT4', 'TSH'],
    
    'Treatments': ['Cefuroxime_dose', 'Rifampicin_dose', 'Lymecyclin_dose',
                   'Azithromycin_dose', 'Clarithromycin_dose', 'Doxycycline_dose',
                   'Amoxicillin_dose', 'LDN_dose', 'Melatonin_dose', 'Valoid_dose',
                   'Malarone_dose', 'Diflucan_dose', 'num_antibiotics_administered',
                   'Q50_antibiotic', 'Q52_antib_duration', 'Q53_antib_symp_improv',
                   'Q54_alternative_trt', 'Q55_alternative_trt_success'],
    
    'Co-infections': ['b.burg+afz+gar.IgG', 'b.burg+afz+gar+IgM', 
                      'B.Burg Round Body IgG', 'B.Burg Round body IgM',
                      'BaB M IgG', 'BaB M IgM', 'Bart H IgG', 'Bart H IgM',
                      'Ehrl C IgG', 'Ehrl IgM', 'Rick Ak IgG', 'Rick Ak IgM',
                      'Coxs IgG', 'Coxs IgM', 'Epst B IgG', 'Epst B IgM',
                      'Hum Par IgG', 'Hum Par IgM', 'Mycop Pneu IgG', 'Mycop Pneu IgM',
                      'cluster_bburg', 'cluster_babesia', 'cluster_bartonella',
                      'cluster_ehrlichia', 'cluster_mycoplasma', 'cluster_coxsackie',
                      'cluster_ebv', 'cluster_parvo', 'IgG_total', 'IgM_total',
                      'num_borrelia_elispot_pos', 'BB Full Anti', 'BB Osp Mix', 'BBLFA',
                      'Babesia', 'Bartonella H', 'Myco Pne', 'Ehrl Ana', 'Rickettsia',
                      'Epstein B', 'Chlamydia P', 'Chlamyd trac', 'Cytomigalo',
                      'VZV IgG', 'Anaplasma Phago', 'Herpes Simplex', 'Aspergillas',
                      'Yersinia.1', 'Candida', 'NVRLIgG']
}

# Create summary table
category_summary = []

for category, features in feature_categories.items():
    # Filter features that exist in the dataframe
    existing_features = [f for f in features if f in df_binary.columns]
    
    # Calculate statistics
    n_features = len(existing_features)
    pct_features = (n_features / len(df_binary.columns)) * 100
    
    # Missing data statistics for this category
    if existing_features:
        missing_data = df_binary[existing_features].isna().mean().mean() * 100
        median_missing = df_binary[existing_features].isna().mean().median() * 100
        min_missing = df_binary[existing_features].isna().mean().min() * 100
        max_missing = df_binary[existing_features].isna().mean().max() * 100
    else:
        missing_data = 0
        median_missing = 0
        min_missing = 0
        max_missing = 0
    
    category_summary.append({
        'Category': category,
        'N_features': n_features,
        'Percentage': f"{pct_features:.1f}%",
        'Mean_missing': f"{missing_data:.1f}%",
        'Median_missing': f"{median_missing:.1f}%",
        'Range_missing': f"{min_missing:.1f}-{max_missing:.1f}%"
    })

# Create DataFrame
category_table = pd.DataFrame(category_summary)

# Add total row
total_features = sum([len([f for f in features if f in df_binary.columns]) 
                      for features in feature_categories.values()])
overall_missing = df_binary.isna().mean().mean() * 100

total_row = pd.DataFrame([{
    'Category': 'TOTAL',
    'N_features': total_features,
    'Percentage': '100.0%',
    'Mean_missing': f"{overall_missing:.1f}%",
    'Median_missing': f"{df_binary.isna().mean().median() * 100:.1f}%",
    'Range_missing': f"{df_binary.isna().mean().min() * 100:.1f}-{df_binary.isna().mean().max() * 100:.1f}%"
}])

category_table = pd.concat([category_table, total_row], ignore_index=True)

print("\n" + "="*80)
print("FEATURE CATEGORIES AND MISSING DATA SUMMARY")
print("="*80)
print(category_table.to_string(index=False))
print("="*80)

category_table

