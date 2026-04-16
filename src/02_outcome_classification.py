# # Outcome Definition: Percentile-Based Method (Tertiles)
# 
# **Purpose:** Define high responders vs non-responders using fixed percentile ranks  
# **Approach:** Adapted from Vendrow et al. (2023) GROC tertile-based classification  
# **Input:** `../dataset/df_ready.pkl` (301 patients × 153 features)  
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


import pandas as pd
import numpy as np
import os
from scipy import stats

print("Libraries loaded")

# ## Configuration

# File paths
INPUT_PATH = "../dataset/df_ready.pkl"
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

print("Configuration set")
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

print(f"All required T0 and T2 columns present")

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

print(f"\n   Differences computed for all parameters")

# ## 3. Standardize to Z-Scores (Direction-Adjusted)

print(f"\n3. Standardizing differences to z-scores...")
print(f"   (Adjusting direction so positive z = improvement)\n")

for param_base in parameters_base:
    col_diff = f'diff_{param_base}'
    col_z = f'z_{param_base}'
    
    # Get the raw difference
    raw_diff = df[col_diff].dropna()
    
    if len(raw_diff) < 2:
        print(f"   {param_base}: <2 valid values, setting z=NaN")
        df[col_z] = np.nan
        continue
    
    # Compute z-score
    mean_diff = raw_diff.mean()
    std_diff = raw_diff.std()
    
    if std_diff == 0:
        print(f"   {param_base}: zero variance, setting z=0")
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


if abs(imbalance_ratio - 1.0) > 0.3:
    print(f"\n   Note: Imbalance ratio {imbalance_ratio:.2f} (expected ~1.0 for tertiles)")
    print(f"      This may occur due to ties at percentile boundaries")

# ## 7. Sample Exclusion Flow Diagram

print(f"\n7. Sample exclusion flow:\n")
print(f"   {len(df)} initial patients (from feature engineering)")
print(f"   - {n_excluded_missing} excluded (missing data: <{MIN_DOMAINS} domains)")
print(f"   = {len(df_complete)} with complete outcome data")
print(f"   - {n_excluded_low} excluded (low responders: middle tertile)")
print(f"   = {len(df_binary)} FINAL SAMPLE (top + bottom tertiles)")


# ## 8. Save Results

print(f"\n8. Saving results...\n")

# Save all classifications (including low responders)
outcome_cols = ['composite_z', 'percentile_rank', 'n_domains_available', 'outcome_class']
df_complete[outcome_cols].to_excel(
    os.path.join(OUTPUT_DIR, 'outcome_percentile_all.xlsx'),
    index=True
)
print(f"   Saved: outcome_percentile_all.xlsx")

# Save binary outcome only (high + non)
binary_cols = ['composite_z', 'percentile_rank', 'outcome_class', 'y']
df_binary[binary_cols].to_excel(
    os.path.join(OUTPUT_DIR, 'outcome_percentile_binary.xlsx'),
    index=True
)
print(f"   Saved: outcome_percentile_binary.xlsx")

# Save full binary dataset (for feature selection)
df_binary.to_pickle(os.path.join(OUTPUT_DIR, 'df_binary_percentile.pkl'))
print(f"   Saved: df_binary_percentile.pkl")

# ## 10. Validation Checks

print(f"\n10. Validation checks:\n")

# Check for duplicate indices
if df_binary.index.duplicated().any():
    print(f"   WARNING: Duplicate indices found!")
else:
    print(f"   No duplicate indices")

# Check for missing outcome
if df_binary['y'].isna().any():
    print(f"   WARNING: Missing outcome values!")
else:
    print(f"   No missing outcomes")

# Check subset relationship
original_indices = set(df.index)
binary_indices = set(df_binary.index)
is_subset = binary_indices.issubset(original_indices)
print(f"   Binary patients subset of original: {is_subset}")

# Check that high/non are from expected percentiles
high_subset = df_binary[df_binary['y'] == 1]
non_subset = df_binary[df_binary['y'] == 0]
print(f"\n   Percentile validation:")
print(f"      High responders: {high_subset['percentile_rank'].min()*100:.0f}th-{high_subset['percentile_rank'].max()*100:.0f}th percentile")
print(f"      Non-responders: {non_subset['percentile_rank'].min()*100:.0f}th-{non_subset['percentile_rank'].max()*100:.0f}th percentile")

# Check for reasonable sample sizes 
expected_per_group = len(df_complete) * (1/3)
if abs(n_high - expected_per_group) > 0.1 * expected_per_group:
    print(f"\n   High responders: {n_high} (expected ~{expected_per_group:.0f})")
if abs(n_non - expected_per_group) > 0.1 * expected_per_group:
    print(f"   Non-responders: {n_non} (expected ~{expected_per_group:.0f})")

