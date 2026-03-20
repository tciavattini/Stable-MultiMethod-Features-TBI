"""
Phenotype Analysis: Statistical Comparison of Top Features
==========================================================

Statistical comparison between high responders and non-responders
for the top selected features from feature selection.

Methods:
- Mann-Whitney U test (non-parametric)
- Benjamini-Hochberg FDR correction  
- Cohen's d effect size (weighted pooled SD)
- Common language effect size

CHANGES FROM PREVIOUS VERSION:
  - Path updated to stability_analysis
  - Label conversion consistent with cross_validation.py: np.where(y==1, 0, 1)
  - Cohen's d uses weighted pooled SD: sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))

Run: python 6_phenotype_analysis.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Import data loading (adjust path if needed)
import sys
sys.path.insert(0, '.')
from data_loading import get_data

# ============================================================
# PUBLICATION-READY FEATURE NAMES
# ============================================================

try:
    feature_mapping = pd.read_csv(str(SCRIPT_DIR / "../data/feature_name_mapping.csv"))
    feature_names_publication = dict(zip(
        feature_mapping['original_name'],
        feature_mapping['publication_name']
    ))
    print("✓ Loaded feature name mapping from file")
except Exception:
    feature_names_publication = {
        # Demographics
        'age': 'Age',
        'gender': 'Gender',
        'Q1_Residence_Ireland': 'Residence in Ireland',
        'Q2_residence_outside_Ireland': 'Residence Outside Ireland',
        'Q3_Outdoor_hobbies': 'Outdoor Hobbies',
        'Q4_Chronic_previous': 'Previous Chronic Conditions',
        'Q5_tick_bite': 'Tick Bite History',
        'Q6_tick_bite_when': 'Time Since Tick Bite',
        'Q9_GP': 'GP Consultation',
        'Q9a_GP_satisfaction': 'GP Satisfaction',
        'Q10_Consultant': 'Specialist Consultation',
        'Q10a_consultant_satisfaction': 'Specialist Satisfaction',
        'Q11_Number_doc': 'Number of Doctors Consulted',
        'Q12_trt_care_rate': 'Treatment Care Rating',
        'Q14_Impact_symp_employment': 'Employment Impact',

        # Symptoms - Baseline
        'Q15 Bulls Eye': 'Bulls Eye Rash',
        'Q16. Rash': 'Rash',
        'Q17. Sweats': 'Night Sweats',
        'Q18 Sore throat': 'Sore Throat',
        'Q19. Headac': 'Headache',
        'Q20. Swglands': 'Swollen Glands',
        'Q21. Sev Fat': 'Severe Fatigue (binary)',
        'T0_severe_fatigue_rate': 'Severe Fatigue Severity',
        'T0_symp_today_rate': 'Overall Symptom Severity',
        'Q24. Chest P.': 'Chest Pain',
        'Q25. ShortB': 'Shortness of Breath',
        'Q26. Palpit': 'Palpitations',
        'Q27. Lighthead.': 'Lightheadedness',
        'Q28. JointP': 'Joint Pain',
        'Q29.Moving': 'Moving Joint Pain',
        'Q30. Intensity': 'Pain Intensity',
        'Q31. JointSw': 'Joint Swelling',
        'Q32. MuscW.le': 'Muscle Weakness',
        'Q33.MuscP': 'Muscle Pain (binary)',
        'T0_muscle_pain_rate': 'Muscle Pain Severity',
        'Q35Facial': 'Facial Palsy',
        'Q36ArmHan': 'Arm/Hand Numbness',
        'Q37Numbn': 'Numbness/Tingling',
        'Q38Concen': 'Concentration Difficulty',
        'Q39Sleep': 'Sleep Disturbance',
        'Q40Vision': 'Vision Problems',
        'Q41. Neck': 'Neck Stiffness',
        'Q42Tinnit': 'Tinnitus',
        'Q43Person': 'Personality Changes',
        'Q44Mood': 'Mood Changes (binary)',
        'T0_mood_rate': 'Mood Severity',
        'Q46Anger': 'Anger/Irritability',
        'Q47Anxiety': 'Anxiety',

        # Symptom aggregates
        'skin_symp': 'Skin Symptoms (any)',
        'num_skin_symp': 'Number of Skin Symptoms',
        'general_wellbeing_symp': 'General Symptoms (any)',
        'num_general_wellbeing_symp': 'Number of General Symptoms',
        'cardiac_symp': 'Cardiac Symptoms (any)',
        'num_cardiac_symp': 'Number of Cardiac Symptoms',
        'rheumatological_symp': 'Rheumatological Symptoms (any)',
        'num_rheumatological_symp': 'Number of Rheumatological Symptoms',
        'neurological_symp': 'Neurological Symptoms (any)',
        'num_neurological_symp': 'Number of Neurological Symptoms',
        'psychological_symp': 'Psychological Symptoms (any)',
        'num_psychological_symp': 'Number of Psychological Symptoms',
        'num_total_symptoms': 'Total Number of Symptoms',

        # Treatment history
        'Q48_blood_analysis': 'Blood Analysis Performed',
        'Q50_antibiotic': 'Previous Antibiotic Treatment',
        'Q52_antib_duration': 'Antibiotic Duration',
        'Q53_antib_symp_improv': 'Symptom Improvement with Antibiotics',
        'Q54_alternative_trt': 'Alternative Treatment',
        'Q55_alternative_trt_success': 'Alternative Treatment Success',

        # Serology - Borrelia
        'b.burg+afz+gar.IgG': 'B. burgdorferi IgG (combined)',
        'b.burg+afz+gar+IgM': 'B. burgdorferi IgM (combined)',
        'B.Burg Round Body IgG': 'B. burgdorferi Round Body IgG',
        'B.Burg Round body IgM': 'B. burgdorferi Round Body IgM',
        'cluster_bburg': 'B. burgdorferi Cluster',

        # Serology - Co-infections
        'BaB M IgG': 'Babesia microti IgG',
        'BaB M IgM': 'Babesia microti IgM',
        'cluster_babesia': 'Babesia Cluster',

        'Bart H IgG': 'Bartonella henselae IgG',
        'Bart H IgM': 'Bartonella henselae IgM',
        'cluster_bartonella': 'Bartonella Cluster',

        'Ehrl C IgG': 'Ehrlichia chaffeensis IgG',
        'Ehrl IgM': 'Ehrlichia IgM',
        'cluster_ehrlichia': 'Ehrlichia Cluster',

        'Rick Ak IgG': 'Rickettsia akari IgG',
        'Rick Ak IgM': 'Rickettsia akari IgM',
        'cluster_rickettsia': 'Rickettsia Cluster',

        'Coxs IgG': 'Coxsackie IgG',
        'Coxs IgM': 'Coxsackie IgM',
        'cluster_coxsackie': 'Coxsackie Cluster',

        'Epst B IgG': 'Epstein-Barr IgG',
        'Epst B IgM': 'Epstein-Barr IgM',
        'cluster_ebv': 'Epstein-Barr Cluster',

        'Hum Par IgG': 'Human Parvovirus IgG',
        'Hum Par IgM': 'Human Parvovirus IgM',
        'cluster_parvo': 'Parvovirus Cluster',

        'Mycop Pneu IgG': 'Mycoplasma pneumoniae IgG',
        'Mycop Pneu IgM': 'Mycoplasma pneumoniae IgM',
        'cluster_mycoplasma': 'Mycoplasma Cluster',

        # ELISpot
        'BB Full Anti': 'BB Full Antigen ELISpot',
        'BB Osp Mix': 'BB Osp Mix ELISpot',
        'BBLFA': 'BB LFA ELISpot',
        'borrelia_elispot': 'Borrelia ELISpot (any positive)',
        'num_borrelia_elispot_pos': 'Number of Positive Borrelia ELISpot',
        'NVRLIgG': 'NVRL IgG',

        # Serology aggregates
        'IgG_total': 'Total IgG Positive Markers',
        'IgM_total': 'Total IgM Positive Markers',
        'sero_tested_count': 'Number of Serological Tests',
        'num_positive_markers': 'Total Positive Serological Markers',

        # Immunology
        'CD3%': 'CD3+ T cells (%)',
        'CD3Total': 'CD3+ T cells (absolute)',
        'CD8%': 'CD8+ T cells (%)',
        'CD8-Suppr': 'CD8+ Suppressor T cells',
        'CD4%': 'CD4+ T cells (%)',
        'CD4-Helper': 'CD4+ Helper T cells',
        'CD19Bcell': 'CD19+ B cells (absolute)',
        'CD19%': 'CD19+ B cells (%)',
        'H/SRATIO': 'Helper/Suppressor Ratio',
        'CD57+NKCELLS': 'CD57+ NK cells',
        'IgG': 'Immunoglobulin G',
        'IgA': 'Immunoglobulin A',
        'IgM': 'Immunoglobulin M',

        # Hematology
        'HgB': 'Hemoglobin',
        'Platelets': 'Platelets',
        'neutrophils': 'Neutrophils',
        'Lymphocytes': 'Lymphocytes',
        'WCC': 'White Cell Count',

        # Biochemistry
        'RF': 'Rheumatoid Factor',
        'CRP': 'C-Reactive Protein',
        'Iron': 'Serum Iron',
        'Transf': 'Transferrin',
        '%transsat': 'Transferrin Saturation',
        'Ferritin': 'Ferritin',
        'Folate': 'Folate',
        'CK': 'Creatine Kinase',
        'FT4': 'Free T4',
        'TSH': 'Thyroid Stimulating Hormone',

        # Treatment administered
        'Cefuroxime_dose': 'Cefuroxime Dose',
        'Rifampicin_dose': 'Rifampicin Dose',
        'Lymecyclin_dose': 'Lymecycline Dose',
        'LDN_dose': 'Low Dose Naltrexone',
        'Melatonin_dose': 'Melatonin Dose',
        'num_antibiotics_administered': 'Number of Antibiotics Administered',

        # Blood test location
        'blood_G': 'Blood Test (Germany)',
        'blood_UK': 'Blood Test (UK)',
        'blood_USA': 'Blood Test (USA)',
    }
    print("✓ Using inline feature name mapping")


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG = {
    'feature_names_path': (SCRIPT_DIR / '../data/feature_names.csv').resolve(),
    'stable_features_path': (SCRIPT_DIR / '../results/stability_analysis/stable_top_features.csv').resolve(),
    'output_dir': (SCRIPT_DIR / '../results/phenotype_analysis').resolve(),
    'alpha': 0.05,
    'sentinel': np.nan,
    'min_samples': 20,
    'effect_size_threshold': 0.2,
    'missing_threshold_warning': 0.30,
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    qvals = np.empty_like(q)
    qvals[order] = q
    return np.clip(qvals, 0, 1)


def cohens_d(x1, x2):
    """Cohen's d with weighted pooled standard deviation.

    Uses the standard Hedges-style pooled SD:
        s_p = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))

    This is more appropriate than equal-weight pooling when group
    sizes differ, and is the standard formula in most textbooks.
    With n1=71, n2=70 the difference is negligible, but this is
    the more defensible choice for a reviewer.
    """
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    denom = n1 + n2 - 2
    if denom <= 0:
        return 0.0

    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / denom)
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0.0


def common_language_effect_size(x_high, x_non):
    """
    CLES from Mann-Whitney U: P(X_high > X_non).
    """
    U_greater, _ = mannwhitneyu(x_high, x_non, alternative="greater")
    _, p_two_sided = mannwhitneyu(x_high, x_non, alternative="two-sided")
    n_high = len(x_high)
    n_non = len(x_non)
    cles = U_greater / (n_high * n_non)
    return U_greater, p_two_sided, cles


def summarize(x):
    """Descriptive statistics."""
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
    }


# ============================================================
# PHENOTYPE ANALYSIS
# ============================================================

def phenotype_analysis_stable(X, y, feature_names, feature_indices, config):
    """
    Compare stable features between groups.

    Returns DataFrame with statistical results and confidence tier.
    """

    X_subset = X[:, feature_indices]
    names = np.asarray(feature_names)[feature_indices]

    X_high = X_subset[y == 1]
    X_non = X_subset[y == 0]

    n_total = len(y)
    rows = []
    excluded_features = []

    for j, fname in enumerate(names):
        xh_raw, xn_raw = X_high[:, j], X_non[:, j]

        # Calculate missing percentage BEFORE removing
        n_missing = np.isnan(xh_raw).sum() + np.isnan(xn_raw).sum()
        missing_pct = n_missing / n_total * 100

        # Remove missing values
        xh = xh_raw[~np.isnan(xh_raw)]
        xn = xn_raw[~np.isnan(xn_raw)]

        # Check minimum sample size
        if len(xh) < config['min_samples'] or len(xn) < config['min_samples']:
            excluded_features.append({
                'feature': fname,
                'reason': f'Insufficient samples (n_high={len(xh)}, n_non={len(xn)})',
                'missing_pct': missing_pct
            })
            continue

        # Determine confidence tier based on missing data
        if missing_pct <= 30:
            tier = 'High'
            tier_label = 'Tier 1'
        elif missing_pct <= 50:
            tier = 'Moderate'
            tier_label = 'Tier 2'
        else:
            tier = 'Low'
            tier_label = 'Tier 3'

        # Statistics
        high_stats = summarize(xh)
        non_stats = summarize(xn)
        d = cohens_d(xh, xn)
        U, p, cles = common_language_effect_size(xh, xn)

        rows.append({
            "feature": fname,
            "feature_publication": feature_names_publication.get(fname, fname),
            "tier": tier_label,
            "confidence": tier,
            "missing_pct": missing_pct,
            "n_high": len(xh),
            "n_non": len(xn),
            "n_total": len(xh) + len(xn),
            "high_mean": high_stats["mean"],
            "high_median": high_stats["median"],
            "high_std": high_stats["std"],
            "high_iqr": high_stats["iqr"],
            "non_mean": non_stats["mean"],
            "non_median": non_stats["median"],
            "non_std": non_stats["std"],
            "non_iqr": non_stats["iqr"],
            "cohens_d": d,
            "cles": cles,
            "mwu_U": U,
            "mwu_p": p,
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("   ⚠️ No features met minimum sample size criteria!")
        return df, pd.DataFrame(excluded_features)

    # FDR correction
    df["q_value"] = bh_fdr(df["mwu_p"].values, config['alpha'])

    # Categorize by effect size and significance
    def categorize(row):
        threshold = config['effect_size_threshold']
        if abs(row["cohens_d"]) < threshold or row["q_value"] >= config['alpha']:
            return "Non-discriminating"
        return "High-dominant" if row["cohens_d"] > 0 else "Non-dominant"

    df["category"] = df.apply(categorize, axis=1)
    df["abs_d"] = df["cohens_d"].abs()

    # Sort by significance
    df = df.sort_values("q_value")

    return df, pd.DataFrame(excluded_features)


# ============================================================
# EFFECT SIZE THRESHOLD ANALYSIS
# ============================================================

def analyze_effect_size_threshold(results, config):
    """Analyze the distribution of effect sizes to help choose appropriate threshold."""
    print("\n" + "=" * 70)
    print("EFFECT SIZE THRESHOLD ANALYSIS")
    print("=" * 70)

    significant = results[results['q_value'] < config['alpha']]

    if len(significant) == 0:
        print("\nNo statistically significant features found.")
        return None

    print(f"\nEffect size distribution (n={len(significant)} significant features):")
    print(f"  Min |d|: {significant['abs_d'].min():.3f}")
    print(f"  25th percentile: {significant['abs_d'].quantile(0.25):.3f}")
    print(f"  Median |d|: {significant['abs_d'].median():.3f}")
    print(f"  75th percentile: {significant['abs_d'].quantile(0.75):.3f}")
    print(f"  Max |d|: {significant['abs_d'].max():.3f}")
    print(f"  Mean |d|: {significant['abs_d'].mean():.3f}")

    print(f"\nImpact of different thresholds (current: {config['effect_size_threshold']}):")
    print(f"  {'Threshold':<12} {'Dominant':<12} {'Shared':<12} {'% Dominant':<12}")
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")

    for threshold in [0.1, 0.2, 0.3, 0.5, 0.8]:
        n_dominant = (significant['abs_d'] >= threshold).sum()
        n_shared = (significant['abs_d'] < threshold).sum()
        pct_dominant = 100 * n_dominant / len(significant)
        marker = " *" if threshold == config['effect_size_threshold'] else ""
        print(f"  {threshold:<12.1f} {n_dominant:<12d} {n_shared:<12d} {pct_dominant:<12.1f}{marker}")

    return significant['abs_d'].describe()


# ============================================================
# VISUALIZATION
# ============================================================

def create_visualizations(results, config):
    """Create volcano plot and effect size distribution plot with publication names."""

    OUT_DIR = config['output_dir']
    colors = {'High-dominant': 'red', 'Non-dominant': 'blue', 'Non-discriminating': 'gray'}

    # 1. Volcano Plot
    print(f"\n   Creating volcano plot...")
    fig, ax = plt.subplots(figsize=(12, 8))

    results['neg_log_q'] = -np.log10(results['q_value'] + 1e-300)

    for cat in results['category'].unique():
        subset = results[results['category'] == cat]
        ax.scatter(subset['cohens_d'], subset['neg_log_q'],
                   c=colors.get(cat, 'gray'), label=cat, alpha=0.6, s=50)

    ax.axhline(-np.log10(config['alpha']), color='black', linestyle='--',
               linewidth=1, label=f'q = {config["alpha"]}')
    ax.axvline(config['effect_size_threshold'], color='red', linestyle=':',
               linewidth=1, alpha=0.5)
    ax.axvline(-config['effect_size_threshold'], color='blue', linestyle=':',
               linewidth=1, alpha=0.5)

    top_features = results.nlargest(5, 'abs_d')
    for _, row in top_features.iterrows():
        if row['q_value'] < config['alpha']:
            ax.annotate(
                row['feature_publication'],
                xy=(row['cohens_d'], row['neg_log_q']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8
            )

    ax.set_xlabel("Cohen's d (effect size)", fontsize=12)
    ax.set_ylabel("-log10(q-value)", fontsize=12)
    ax.set_title("Volcano Plot: Feature Differences Between Groups", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "volcano_plot.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: volcano_plot.png")
    plt.close()

    # 2. Effect Size Distribution
    print(f"   Creating effect size distribution plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    significant = results[results['q_value'] < config['alpha']]
    if len(significant) > 0:
        ax1.hist(significant['abs_d'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(config['effect_size_threshold'], color='red', linestyle='--',
                    linewidth=2, label=f'Threshold = {config["effect_size_threshold"]}')
        ax1.set_xlabel('|Cohen\'s d|', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Effect Size Distribution\n(Significant Features Only)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')

    category_counts = results['category'].value_counts()
    colors_pie = [colors.get(cat, 'gray') for cat in category_counts.index]
    ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax2.set_title('Feature Category Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT_DIR / "effect_size_distribution.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: effect_size_distribution.png")
    plt.close()

    # 3. Top Features Bar Plot
    print(f"   Creating top features comparison plot...")
    top_n = min(10, len(results))
    top_feat = results.nsmallest(top_n, 'q_value')

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(top_feat))
    colors_bar = [colors.get(cat, 'gray') for cat in top_feat['category']]

    ax.barh(y_pos, top_feat['cohens_d'], color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feat['feature_publication'], fontsize=9)
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_title(f"Top {top_n} Features by Statistical Significance", fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "top_features_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   Saved: top_features_comparison.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PHENOTYPE ANALYSIS: STABLE FEATURES ONLY")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Alpha (FDR): {CONFIG['alpha']}")
    print(f"  Effect size threshold: {CONFIG['effect_size_threshold']}")
    print(f"  Minimum samples per group: {CONFIG['min_samples']}")
    print(f"  Cohen's d: weighted pooled SD")
    print(f"  Stable features: {CONFIG['stable_features_path']}")

    # Load data
    print("\n1. Loading data...")
    X_full, y_full = get_data(subsamp=False, binary=True)

    # Convert labels {1, 3} → {0, 1} — consistent with cross_validation.py
    if set(np.unique(y_full)) == {1, 3}:
        y_binary = np.where(y_full == 1, 0, 1).astype(int)
    else:
        y_binary = y_full.astype(int)

    print(f"   Dataset: {X_full.shape}")
    print(f"   High responders (1): {(y_binary == 1).sum()}")
    print(f"   Non-responders (0):  {(y_binary == 0).sum()}")

    # Load feature names
    feature_names = pd.read_csv(CONFIG['feature_names_path'])["feature_name"].tolist()

    # Load stable features
    print("\n2. Loading stable features...")
    df_stable = pd.read_csv(CONFIG['stable_features_path'])

    # Flexible column detection
    if 'feature_name' in df_stable.columns:
        selected_names = df_stable['feature_name'].tolist()
    elif 'feature' in df_stable.columns:
        selected_names = df_stable['feature'].tolist()
    else:
        raise ValueError(f"Cannot find feature column. Available: {df_stable.columns.tolist()}")

    # Get indices
    selected_indices = []
    missing_features = []
    for f in selected_names:
        if f in feature_names:
            selected_indices.append(feature_names.index(f))
        else:
            missing_features.append(f)

    if missing_features:
        print(f"   ⚠️ Features not found in dataset: {missing_features}")

    print(f"   Stable features loaded: {len(selected_indices)}")

    # Run analysis
    print(f"\n3. Running phenotype analysis...")
    results, excluded = phenotype_analysis_stable(
        X_full, y_binary, feature_names, selected_indices, CONFIG
    )

    # Report tier distribution
    print(f"\n4. Data Quality Summary:")
    print(f"   Features analyzed: {len(results)}")
    if len(excluded) > 0:
        print(f"   Features excluded: {len(excluded)}")

    print(f"\n   Confidence tier distribution:")
    for tier in ['Tier 1', 'Tier 2', 'Tier 3']:
        n = (results['tier'] == tier).sum()
        if n > 0:
            pct = n / len(results) * 100
            print(f"      {tier}: {n} features ({pct:.1f}%)")

    # Statistical results
    print(f"\n5. Statistical Results:")
    n_sig = (results['q_value'] < CONFIG['alpha']).sum()
    print(f"   Significant (q < {CONFIG['alpha']}): {n_sig}")
    print(f"   Non-significant: {len(results) - n_sig}")

    # Category breakdown
    print(f"\n6. Feature Categories:")
    for cat in ['High-dominant', 'Non-dominant', 'Non-discriminating']:
        n = (results['category'] == cat).sum()
        if n > 0:
            pct = n / len(results) * 100
            print(f"   {cat}: {n} ({pct:.1f}%)")

    # Top significant features
    sig_features = results[results['q_value'] < CONFIG['alpha']]
    if len(sig_features) > 0:
        print(f"\n7. Significant Features (q < {CONFIG['alpha']}):")
        for _, row in sig_features.iterrows():
            direction = "↑" if row['cohens_d'] > 0 else "↓"
            print(f"   {direction} {row['feature_publication']}")
            print(f"      d = {row['cohens_d']:.3f}, q = {row['q_value']:.4f}, "
                  f"confidence: {row['confidence']}")

    # Effect size threshold analysis
    analyze_effect_size_threshold(results, CONFIG)

    # Save results
    print(f"\n8. Saving results...")
    OUT_DIR = CONFIG['output_dir']

    # Full results
    results.to_excel(OUT_DIR / "phenotype_stable_features.xlsx", index=False)
    results.to_csv(OUT_DIR / "phenotype_stable_features.csv", index=False)

    # Publication-ready table
    pub_cols = ['feature_publication', 'tier', 'n_high', 'n_non',
                'high_median', 'high_iqr', 'non_median', 'non_iqr',
                'cohens_d', 'cles', 'q_value', 'category']
    pub_table = results[pub_cols].copy()
    pub_table.columns = ['Feature', 'Confidence', 'n (High)', 'n (Non)',
                         'Median (High)', 'IQR (High)', 'Median (Non)', 'IQR (Non)',
                         "Cohen's d", 'CLES', 'q-value', 'Category']
    pub_table.to_excel(OUT_DIR / "phenotype_stable_publication.xlsx", index=False)

    # Excluded features (if any)
    if len(excluded) > 0:
        excluded.to_csv(OUT_DIR / "excluded_features.csv", index=False)

    # Summary report
    with open(OUT_DIR / "analysis_summary.txt", 'w') as f:
        f.write("PHENOTYPE ANALYSIS: STABLE FEATURES\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now()}\n")
        f.write(f"Stable features source: {CONFIG['stable_features_path']}\n")
        f.write(f"Features Analyzed: {len(results)} (stable features only)\n")
        f.write(f"FDR Alpha: {CONFIG['alpha']}\n")
        f.write(f"Effect Size Threshold: {CONFIG['effect_size_threshold']}\n")
        f.write(f"Cohen's d formula: weighted pooled SD\n")
        f.write(f"Min Samples per Group: {CONFIG['min_samples']}\n\n")

        f.write("TIER DISTRIBUTION:\n")
        for tier in ['Tier 1', 'Tier 2', 'Tier 3']:
            n = (results['tier'] == tier).sum()
            f.write(f"  {tier}: {n}\n")

        f.write(f"\nSIGNIFICANT FEATURES (q < {CONFIG['alpha']}):\n")
        if len(sig_features) > 0:
            for _, row in sig_features.iterrows():
                f.write(f"  - {row['feature_publication']}: d={row['cohens_d']:.3f}, "
                        f"q={row['q_value']:.4f}\n")
        else:
            f.write("  None\n")

    print(f"   Results saved to: {OUT_DIR}")

    # Create visualizations
    print(f"\n9. Creating visualizations...")
    create_visualizations(results, CONFIG)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results, excluded


if __name__ == "__main__":
    results, excluded = main()