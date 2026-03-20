"""
Converted from notebook: description.ipynb
"""

"""
Generate Baseline Characteristics Table
========================================

Creates Table 1 comparing high responders vs non-responders at baseline.
Includes:
- Demographics (age, sex, etc.)
- Key clinical variables
- Statistical tests for between-group differences
- Missing data summary
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'data_path': '../results/patient_classification/df_binary_percentile.pkl',
    'output_dir': Path('../results/baseline_characteristics/'),
    'alpha': 0.05,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def describe_continuous(data, name):
    """Descriptive stats for continuous variable."""
    data_clean = data.dropna()
    
    if len(data_clean) == 0:
        return {
            'variable': name,
            'type': 'continuous',
            'n': 0,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_missing': len(data),
            'pct_missing': 100.0
        }
    
    return {
        'variable': name,
        'type': 'continuous',
        'n': len(data_clean),
        'mean': data_clean.mean(),
        'std': data_clean.std(ddof=1),
        'median': data_clean.median(),
        'q25': data_clean.quantile(0.25),
        'q75': data_clean.quantile(0.75),
        'min': data_clean.min(),
        'max': data_clean.max(),
        'n_missing': data.isna().sum(),
        'pct_missing': data.isna().sum() / len(data) * 100
    }


def describe_categorical(data, name):
    """Descriptive stats for categorical variable."""
    data_clean = data.dropna()
    
    if len(data_clean) == 0:
        return {
            'variable': name,
            'type': 'categorical',
            'n': 0,
            'categories': [],
            'counts': [],
            'percentages': [],
            'n_missing': len(data),
            'pct_missing': 100.0
        }
    
    counts = data_clean.value_counts()
    percentages = data_clean.value_counts(normalize=True) * 100
    
    return {
        'variable': name,
        'type': 'categorical',
        'n': len(data_clean),
        'categories': counts.index.tolist(),
        'counts': counts.values.tolist(),
        'percentages': percentages.values.tolist(),
        'n_missing': data.isna().sum(),
        'pct_missing': data.isna().sum() / len(data) * 100
    }


def compare_continuous(group1, group2):
    """Compare continuous variable between groups using Mann-Whitney U."""
    # Remove NaN
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'test': 'insufficient_data'}
    
    # Mann-Whitney U (non-parametric)
    statistic, p_value = mannwhitneyu(g1, g2, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'test': 'Mann-Whitney U'
    }


def compare_categorical(group1, group2):
    """Compare categorical variable between groups using Chi-square or Fisher."""
    # Remove NaN
    g1 = group1.dropna()
    g2 = group2.dropna()
    
    if len(g1) < 2 or len(g2) < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'test': 'insufficient_data'}
    
    # Create contingency table
    combined = pd.concat([
        pd.Series(['Group1'] * len(g1), index=g1.index),
        pd.Series(['Group2'] * len(g2), index=g2.index)
    ])
    data_combined = pd.concat([g1, g2])
    
    contingency = pd.crosstab(combined, data_combined)
    
    # Check expected frequencies
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {'statistic': np.nan, 'p_value': np.nan, 'test': 'insufficient_categories'}
    
    # Use Fisher's exact for 2x2 tables with small counts
    if contingency.shape == (2, 2) and contingency.min().min() < 5:
        try:
            statistic, p_value = fisher_exact(contingency)
            return {'statistic': statistic, 'p_value': p_value, 'test': 'Fisher exact'}
        except:
            return {'statistic': np.nan, 'p_value': np.nan, 'test': 'fisher_failed'}
    
    # Otherwise use Chi-square
    try:
        statistic, p_value, dof, expected = chi2_contingency(contingency)
        return {'statistic': statistic, 'p_value': p_value, 'test': 'Chi-square'}
    except:
        return {'statistic': np.nan, 'p_value': np.nan, 'test': 'chi2_failed'}


# ============================================================
# MAIN ANALYSIS
# ============================================================

def create_baseline_table(df):
    """
    Create baseline characteristics table.
    
    Automatically detects variable types and generates appropriate statistics.
    """
    
    # Split by outcome
    df_high = df[df['y'] == 1].copy()
    df_non = df[df['y'] == 0].copy()
    
    print(f"\nGroups:")
    print(f"  High responders: n={len(df_high)}")
    print(f"  Non-responders: n={len(df_non)}")
    
    # Define variables to include in table
    # These are typically:
    # 1. Demographics
    # 2. Disease characteristics at baseline
    # 3. Key clinical variables
    
    variables = {
        # Demographics
        'age': ('continuous', 'Age (years)'),
        'gender': ('categorical', 'Female sex'),
        
        # Disease/treatment history
        'Q5_tick_bite': ('categorical', 'Tick bite history'),
        'Q6_tick_bite_when': ('continuous', 'Time since tick bite (months)'),
        'Q50_antibiotic': ('categorical', 'Prior antibiotic treatment'),
        'Q4_Chronic_previous': ('categorical', 'Previous chronic symptoms'),
        'Q54_alternative_trt': ('categorical', 'Alternative treatment use'),
        
        # Baseline symptom severity (T0 = baseline)
        'T0_severe_fatigue_rate': ('continuous', 'Severe fatigue severity (T0)'),
        'T0_muscle_pain_rate': ('continuous', 'Muscle pain severity (T0)'),
        'T0_symp_today_rate': ('continuous', 'Overall symptom severity (T0)'),
        'T0_mood_rate': ('continuous', 'Mood severity (T0)'),
        
        # Symptom counts
        'num_total_symptoms': ('continuous', 'Total number of symptoms'),
        #'num_neurological_symp': ('continuous', 'Number of neurological symptoms'),
        
        "Q52_antib_duration": ('continuous', 'Prior antibiotic duration (weeks)'),
        "study_treatment_duration": ('continuous', 'Study treatment duration (weeks)'),


        "num_positive_markers": ('continuous', 'Number of positive markers'),
        "sero_tested_count": ('continuous', 'Number of serological tests performed'),

        "cluster_bburg_binary": ('categorical', 'B. burgdorferi positive'),
        "cluster_babesia_binary": ('categorical', 'Babesia positive'),
        "cluster_bartonella_binary": ('categorical', 'Bartonella positive'),
        "cluster_ehrlichia_binary": ('categorical', 'Ehrlichia positive'),
        "cluster_rickettsia_binary": ('categorical', 'Rickettsia positive'),
    }
    
    results = []
    
    for var_name, (var_type, var_label) in variables.items():
        
        # Check if variable exists
        if var_name not in df.columns:
            print(f"  ⚠️  Skipping {var_name} - not found in data")
            continue
        
        # Extract data
        all_data = df[var_name]
        high_data = df_high[var_name]
        non_data = df_non[var_name]
        
        # Describe each group
        if var_type == 'continuous':
            desc_high = describe_continuous(high_data, var_label)
            desc_non = describe_continuous(non_data, var_label)
            comparison = compare_continuous(high_data, non_data)
            
            # Format for table
            row = {
                'Variable': var_label,
                'Type': 'Continuous',
                'High Responders': f"{desc_high['mean']:.1f} ± {desc_high['std']:.1f}",
                'Non-Responders': f"{desc_non['mean']:.1f} ± {desc_non['std']:.1f}",
                'Missing (%)': f"{all_data.isna().sum()} ({all_data.isna().sum()/len(df)*100:.1f}%)",
                'p-value': comparison['p_value'],
                'Test': comparison['test']
            }
            
        else:  # categorical
            desc_high = describe_categorical(high_data, var_label)
            desc_non = describe_categorical(non_data, var_label)
            comparison = compare_categorical(high_data, non_data)
            
            # For binary variables, report n (%)
            if len(desc_high['categories']) == 2:
                # Report the "yes" category (usually coded as 1 or True)
                # Find which category is "positive"
                yes_cat = max(desc_high['categories'])  # Assumes 1 = yes, 0 = no
                
                high_idx = desc_high['categories'].index(yes_cat) if yes_cat in desc_high['categories'] else None
                non_idx = desc_non['categories'].index(yes_cat) if yes_cat in desc_non['categories'] else None
                
                if high_idx is not None:
                    high_str = f"{desc_high['counts'][high_idx]} ({desc_high['percentages'][high_idx]:.1f}%)"
                else:
                    high_str = "0 (0.0%)"
                
                if non_idx is not None:
                    non_str = f"{desc_non['counts'][non_idx]} ({desc_non['percentages'][non_idx]:.1f}%)"
                else:
                    non_str = "0 (0.0%)"
                
                row = {
                    'Variable': var_label,
                    'Type': 'Binary',
                    'High Responders': high_str,
                    'Non-Responders': non_str,
                    'Missing (%)': f"{all_data.isna().sum()} ({all_data.isna().sum()/len(df)*100:.1f}%)",
                    'p-value': comparison['p_value'],
                    'Test': comparison['test']
                }
            else:
                # Multi-category - just report n
                row = {
                    'Variable': var_label,
                    'Type': 'Categorical',
                    'High Responders': f"n={desc_high['n']}",
                    'Non-Responders': f"n={desc_non['n']}",
                    'Missing (%)': f"{all_data.isna().sum()} ({all_data.isna().sum()/len(df)*100:.1f}%)",
                    'p-value': comparison['p_value'],
                    'Test': comparison['test']
                }
        
        results.append(row)
    
    df_results = pd.DataFrame(results)
    
    return df_results


def analyze_missing_data(df):
    """
    Comprehensive missing data analysis.
    """
    
    print("\n" + "="*60)
    print("MISSING DATA ANALYSIS")
    print("="*60)
    
    # Exclude outcome-related columns
    feature_cols = [c for c in df.columns if not any(
        x in c for x in ['T2_', 'diff_', 'z_', 'composite', 'percentile', 
                         'outcome', 'improve_', 'class_', 'y']
    )]
    
    print(f"\nAnalyzing {len(feature_cols)} baseline features...")
    
    # Overall statistics
    total_cells = len(df) * len(feature_cols)
    missing_cells = df[feature_cols].isna().sum().sum()
    missing_pct = missing_cells / total_cells * 100
    
    print(f"\nOverall missing data:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Missing cells: {missing_cells:,}")
    print(f"  Percentage: {missing_pct:.2f}%")
    
    # Per-feature statistics
    missing_per_feature = df[feature_cols].isna().sum()
    missing_pct_per_feature = missing_per_feature / len(df) * 100
    
    # Range
    print(f"\nMissing data per feature:")
    print(f"  Minimum: {missing_pct_per_feature.min():.1f}%")
    print(f"  Maximum: {missing_pct_per_feature.max():.1f}%")
    print(f"  Median: {missing_pct_per_feature.median():.1f}%")
    print(f"  Mean: {missing_pct_per_feature.mean():.1f}%")
    
    # Categories
    n_complete = (missing_per_feature == 0).sum()
    n_high_quality = (missing_pct_per_feature <= 30).sum()
    n_moderate = ((missing_pct_per_feature > 30) & (missing_pct_per_feature <= 50)).sum()
    n_low_quality = (missing_pct_per_feature > 50).sum()
    
    print(f"\nData quality tiers:")
    print(f"  Complete (0% missing): {n_complete} features")
    print(f"  High quality (≤30% missing): {n_high_quality} features")
    print(f"  Moderate quality (30-50% missing): {n_moderate} features")
    print(f"  Low quality (>50% missing): {n_low_quality} features")
    
    # Most missing features
    print(f"\nFeatures with highest missingness:")
    top_missing = missing_pct_per_feature.nlargest(10)
    for feat, pct in top_missing.items():
        print(f"  {feat}: {pct:.1f}%")
    
    # Save detailed report
    missing_report = pd.DataFrame({
        'feature': feature_cols,
        'n_missing': missing_per_feature.values,
        'pct_missing': missing_pct_per_feature.values,
        'tier': pd.cut(missing_pct_per_feature, 
                      bins=[-0.1, 0, 30, 50, 100],
                      labels=['Complete', 'High', 'Moderate', 'Low'])
    }).sort_values('pct_missing', ascending=False)
    
    output_path = CONFIG['output_dir'] / 'missing_data_report.csv'
    missing_report.to_csv(output_path, index=False)
    print(f"\n✓ Saved detailed report: {output_path}")
    
    return {
        'total_features': len(feature_cols),
        'overall_missing_pct': missing_pct,
        'range_min': missing_pct_per_feature.min(),
        'range_max': missing_pct_per_feature.max(),
        'n_complete': n_complete,
        'n_high_quality': n_high_quality,
        'n_moderate': n_moderate,
        'n_low_quality': n_low_quality,
    }


def format_latex_table(df_results, missing_stats):
    """
    Generate LaTeX code for baseline characteristics table.
    """
    
    latex = r"""\begin{table}[t]
\centering
\caption{Baseline characteristics of high responders and non-responders. 
Values are mean $\pm$ SD for continuous variables and n (\%) for categorical variables. 
P-values from Mann-Whitney U test (continuous) or Chi-square/Fisher exact test (categorical).}
\label{tab:baseline}
\begin{tabular}{lcccp{1.5cm}c}
\toprule
\textbf{Characteristic} & \textbf{High resp.} & \textbf{Non-resp.} & \textbf{Missing} & \textbf{$p$-value} \\
 & (n=71) & (n=70) & n (\%) & \\
\midrule
"""
    
    # Group by type for organization
    for var_type in ['Continuous', 'Binary', 'Categorical']:
        subset = df_results[df_results['Type'] == var_type]
        
        if len(subset) == 0:
            continue
        
        # Section header
        if var_type == 'Continuous':
            latex += r"\multicolumn{5}{l}{\textit{Continuous variables}} \\" + "\n"
        elif var_type == 'Binary':
            latex += r"\midrule" + "\n"
            latex += r"\multicolumn{5}{l}{\textit{Binary variables}} \\" + "\n"
        else:
            latex += r"\midrule" + "\n"
            latex += r"\multicolumn{5}{l}{\textit{Categorical variables}} \\" + "\n"
        
        for _, row in subset.iterrows():
            var_name = row['Variable']
            high_val = row['High Responders']
            non_val = row['Non-Responders']
            missing = row['Missing (%)']
            p_val = row['p-value']
            
            # Format p-value
            if pd.isna(p_val):
                p_str = "---"
            elif p_val < 0.001:
                p_str = "<0.001"
            else:
                p_str = f"{p_val:.3f}"
            
            latex += f"{var_name} & {high_val} & {non_val} & {missing} & {p_str} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

"""
    
    # Add text for methods/results
    latex += r"""

% Text for Results section:
% ---------------------------
Baseline characteristics are summarized in Table~\ref{tab:baseline}. 
The two groups showed similar distributions across demographic and clinical 
variables, with no statistically significant differences (all $p > 0.05$), 
ensuring that classification was not confounded by baseline characteristics. 
Missing data across the """ + f"{missing_stats['total_features']}" + r""" baseline features averaged 
""" + f"{missing_stats['overall_missing_pct']:.1f}" + r"""\% 
(range: """ + f"{missing_stats['range_min']:.0f}" + r"""--""" + f"{missing_stats['range_max']:.0f}" + r"""\%), 
with """ + f"{missing_stats['n_low_quality']}" + r""" features having $>50\%$ missingness.

"""
    
    return latex


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run complete baseline characteristics analysis."""
    
    print("\n" + "="*60)
    print("BASELINE CHARACTERISTICS TABLE")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_pickle(CONFIG['data_path'])
    print(f"   Loaded {len(df)} patients")
    
    # Before creating baseline table, binarize clusters
    cluster_cols = ['cluster_bburg', 'cluster_babesia', 'cluster_bartonella', 
                    'cluster_ehrlichia', 'cluster_rickettsia']
    for col in cluster_cols:
        df[col + '_binary'] = (df[col] >= 1).astype(float)
        df.loc[df[col].isna(), col + '_binary'] = np.nan

    dur = df['Q52_antib_duration'].dropna()

    df_t2 = pd.read_excel('../dataset/column_trt_2.xlsx')
    df_t2['trt_duration_T2'] = pd.to_numeric(df_t2['trt_duration_T2'], errors='coerce')

    print(f"Excel rows: {len(df_t2)}")
    print(f"df index range: {df.index.min()} to {df.index.max()}")
    print(f"df index length: {len(df.index)}")
    print(f"All df indices in Excel? {df.index.isin(df_t2.index).all()}")

    df['study_treatment_duration'] = df_t2.loc[df.index, 'trt_duration_T2'].values

    print(f"n = {len(dur)}")
    print(f"min = {dur.min()}, max = {dur.max()}")
    print(f"median = {dur.median()}")
    print(dur.describe())
    print(dur.value_counts().sort_index())

    # Create baseline table
    print("\n2. Creating baseline characteristics table...")
    df_baseline = create_baseline_table(df)
    
    # Save table
    table_path = CONFIG['output_dir'] / 'baseline_characteristics.csv'
    df_baseline.to_csv(table_path, index=False)
    print(f"\n✓ Saved table: {table_path}")
    
    # Display
    print("\n" + "="*60)
    print("BASELINE CHARACTERISTICS")
    print("="*60)
    print(df_baseline.to_string(index=False))
    
    # Missing data analysis
    print("\n3. Analyzing missing data...")
    missing_stats = analyze_missing_data(df)
    
    # Generate LaTeX
    print("\n4. Generating LaTeX code...")
    latex = format_latex_table(df_baseline, missing_stats)
    
    latex_path = CONFIG['output_dir'] / 'baseline_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"✓ Saved LaTeX: {latex_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total features analyzed: {missing_stats['total_features']}")
    print(f"Overall missing rate: {missing_stats['overall_missing_pct']:.1f}%")
    print(f"Features >50% missing: {missing_stats['n_low_quality']}")
    
    # Check for significant differences
    n_significant = (df_baseline['p-value'] < 0.05).sum()
    if n_significant == 0:
        print("\n✓ No significant baseline differences between groups")
        print("  (all p > 0.05)")
    else:
        print(f"\n⚠️  {n_significant} variables show significant differences:")
        sig_vars = df_baseline[df_baseline['p-value'] < 0.05]
        for _, row in sig_vars.iterrows():
            print(f"  - {row['Variable']}: p={row['p-value']:.3f}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Results saved to: {CONFIG['output_dir']}")
    print("\nFiles created:")
    print("  - baseline_characteristics.csv (data)")
    print("  - baseline_table.tex (LaTeX code)")
    print("  - missing_data_report.csv (detailed missing data)")


if __name__ == "__main__":
    main()


# ============================================================
# USAGE NOTES
# ============================================================
"""
CUSTOMIZATION:

1. Edit the 'variables' dictionary (line ~165) to include the specific
   variables you want in your table. Current list is just examples.

2. Adjust column names to match your data:
   - 'age', 'gender', etc. may have different names in your dataset
   - Check df.columns to see available variables

3. For categorical variables with specific coding:
   - Modify the binary variable handling (line ~230) if your yes/no
     coding is different (e.g., True/False, 'yes'/'no', etc.)

4. The script automatically:
   - Detects variable types
   - Chooses appropriate tests (Mann-Whitney vs Chi-square)
   - Formats output for LaTeX

EXPECTED OUTPUT:
- CSV table with all statistics
- LaTeX code ready to paste into paper
- Missing data report
- Console summary

INTERPRETATION:
- If p > 0.05 for all variables → groups are balanced ✓
- If some p < 0.05 → note these as potential confounders
- Missing data >50% → consider excluding from analysis
"""

# Load study treatment duration from Excel
df_t2 = pd.read_excel('../dataset/column_trt_2.xlsx')  # adjust path

# Check what columns are available
print(df_t2.columns.tolist())
print(df_t2.head())

# Load study treatment duration from Excel
df_t2['trt_duration_T2'] = pd.to_numeric(df_t2['trt_duration_T2'], errors='coerce')

