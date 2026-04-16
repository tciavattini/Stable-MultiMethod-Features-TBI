"""
Data Preparation for Feature Selection

Parameters
----------
df_binary_path : str
    Path to the outcome-defined binary dataset (high vs non responders)
output_folder : str
    Where to save dataset.csv, labels.csv, feature_names.csv

Returns
-------
X : np.ndarray
    Feature matrix (with NaN for missing values)
y : np.ndarray
    Labels (0=non-responder, 1=high-responder)
feature_names : list
    Names of features in X
    
"""

import os
import pandas as pd
import numpy as np

def prepare_data_for_feature_selection(
    df_binary_path="../results/patient_classification/df_binary_percentile.pkl",
    output_folder="../data/",
    coerce_all_features_to_numeric=True,
    drop_all_nan_columns=True
):
  
    os.makedirs(output_folder, exist_ok=True)
    
    # Load outcome-defined dataset
    df = pd.read_pickle(df_binary_path)
    print(f"Loaded {len(df)} patients from {df_binary_path}")
    
    # Extract labels (0=non, 1=high)
    if 'y' not in df.columns:
        raise ValueError("Dataset must have 'y' column (binary outcome)")
    y = df['y'].astype(int).values
    print(f"  Labels: {(y==0).sum()} non-responders, {(y==1).sum()} high-responders")
    
    # ============================================================
    # CRITICAL: Define features to EXCLUDE (prevent leakage)
    # ============================================================
    
    # 1. Outcome-related columns (computed from T2-T0)
    outcome_related = [
        'y', 'outcome_class', 'final_class',
        'composite_z', 'composite_improvement', 
        'percentile_rank', 'n_domains_available',
        'high_count', 'non_count', 'low_count', 'n_classes_ge_k'
    ]
    
    # 2. T2 measurements (follow-up timepoint that defines outcome)
    t2_features = [c for c in df.columns if c.startswith('T2_')]
    
    # 3. Difference scores (T2 - T0, these ARE the outcome)
    diff_features = [c for c in df.columns if c.startswith('diff_')]
    
    # 4. Z-scores (standardized differences, also outcome-related)
    z_features = [c for c in df.columns if c.startswith('z_')]
    
    # 5. Improvement scores (direction-adjusted differences)
    improve_features = [c for c in df.columns if c.startswith('improve_')]
    
    # 6. Per-parameter classification columns
    class_features = [c for c in df.columns if c.startswith('class_')]
    
    # Combine all exclusions
    all_exclude = set(
        outcome_related + 
        t2_features + 
        diff_features + 
        z_features + 
        improve_features + 
        class_features
    )
    
    print(f"\nExcluding {len(all_exclude)} outcome-related features:")
    print(f"  T2 measurements: {len(t2_features)}")
    print(f"  Difference scores: {len(diff_features)}")
    print(f"  Z-scores: {len(z_features)}")
    print(f"  Improvement scores: {len(improve_features)}")
    print(f"  Classification columns: {len(class_features)}")
    print(f"  Other outcome metadata: {len(outcome_related)}")
    
    # ============================================================
    # Select candidate features 
    # ============================================================
    
    candidate_cols = [c for c in df.columns if c not in all_exclude]
    print(f"\nCandidate features: {len(candidate_cols)}")
    
    # Force numeric conversion if requested
    if coerce_all_features_to_numeric:
        print("Converting all features to numeric...")
        for c in candidate_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Keep only numeric features
    feature_cols = [
        c for c in candidate_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"Numeric features: {len(feature_cols)}")
    
    # Optional: drop columns that are entirely NaN
    if drop_all_nan_columns:
        all_nan = [c for c in feature_cols if df[c].isna().all()]
        if len(all_nan) > 0:
            print(f"Dropping {len(all_nan)} all-NaN columns")
            feature_cols = [c for c in feature_cols if c not in set(all_nan)]
    
    print(f"\nFinal feature count: {len(feature_cols)}")
    
    # ============================================================
    # Extract feature matrix 
    # ============================================================
    
    X = df[feature_cols].copy()
    
    # Count missing values
    missing_counts = X.isna().sum()
    n_features_with_missing = (missing_counts > 0).sum()
    print(f"\nMissing value summary:")
    print(f"  Features with missing: {n_features_with_missing}/{len(feature_cols)}")
    print(f"  Total missing cells: {X.isna().sum().sum()}")
    
    # ============================================================
    # Validation checks
    # ============================================================
    
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    # Check 1: No outcome features leaked
    leaky_patterns = ['T2_', 'diff_', 'z_', 'composite', 'percentile', 
                     'outcome', 'improve_', 'class_']
    leaky_found = []
    for col in feature_cols:
        if any(pattern in col for pattern in leaky_patterns):
            leaky_found.append(col)
    
    if leaky_found:
        print("LEAKAGE DETECTED - These features should be excluded:")
        for f in leaky_found:
            print(f"  - {f}")
        raise ValueError("Data leakage detected! Fix exclusion list.")
    else:
        print("No outcome-related features found in feature set")
    
    # Check 2: All features are numeric
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        print(f"WARNING: {len(non_numeric)} non-numeric features:")
        for f in non_numeric[:5]:
            print(f"  - {f}")
        print("These will cause errors in sklearn!")
    else:
        print("All features are numeric")
    
    # Check 3: Feature diversity
    n_continuous = sum(X[c].nunique() > 2 for c in feature_cols)
    n_binary = sum(X[c].nunique() == 2 for c in feature_cols)
    n_categorical = len(feature_cols) - n_continuous - n_binary
    print(f"\nFeature types:")
    print(f"  Continuous (>2 unique): {n_continuous}")
    print(f"  Binary (2 unique): {n_binary}")
    print(f"  Categorical (3-10 unique): {n_categorical}")
    
    # Check 4: T0 features present
    t0_features = [c for c in feature_cols if c.startswith('T0_')]
    print(f"\nBaseline features:")
    print(f"  T0 features found: {len(t0_features)}")
    if len(t0_features) > 0:
        print(f"  Examples: {t0_features[:3]}")
    
    # ============================================================
    # Save in Vendrow-compatible format
    # ============================================================
    
    # Save dataset 
    X.to_csv(
        os.path.join(output_folder, "dataset.csv"), 
        index=False, 
        header=False,
        na_rep='nan'  
    )
    
    # Save labels (Vendrow format: 1=non, 3=high)
    # His subsample_binary expects {1,3} and converts to {0,1}
    y_vendrow = np.where(y == 0, 1, 3)  # 0→1 (non), 1→3 (high)
    pd.DataFrame(y_vendrow).to_csv(
        os.path.join(output_folder, "labels.csv"), 
        index=False, 
        header=False
    )
    
    # Save feature names
    pd.DataFrame({"feature_name": feature_cols}).to_csv(
        os.path.join(output_folder, "feature_names.csv"), 
        index=False
    )
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"Saved to: {output_folder}")
    print(f"  dataset.csv: {X.shape} (with NaN preserved)")
    print(f"  labels.csv: {len(y)} labels (1=non, 3=high)")
    print(f"  feature_names.csv: {len(feature_cols)} feature names")

    X.to_pickle(os.path.join(output_folder, "processed_features.pkl"))

    
    return X.values, y, feature_cols

# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    # Run the corrected preparation
    X, y, feature_names = prepare_data_for_feature_selection(
        df_binary_path="../results/patient_classification/df_binary_percentile.pkl",
        output_folder="../data/"
    )
    