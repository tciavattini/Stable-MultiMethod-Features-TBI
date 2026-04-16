# Verify no leakage in feature_names.csv

import pandas as pd

names = pd.read_csv('../data/feature_names.csv')['feature_name'].tolist()

leaky_patterns = ['T2_', 'diff_', 'z_', 'composite', 'percentile', 'outcome', 'class_']
leaky_features = [n for n in names if any(p in n for p in leaky_patterns)]

if leaky_features:
    print("LEAKAGE DETECTED - DO NOT RUN!")
    print("These features contain outcome information:")
    for f in leaky_features:
        print(f"  - {f}")
else:
    print("No leakage found - safe to run")
    print(f"Total features: {len(names)}")
