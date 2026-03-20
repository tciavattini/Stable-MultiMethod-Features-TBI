"""
data_loading.py
===============
Load the analysis-ready dataset and labels.

Input files (in data/ directory):
    - dataset.csv:       Feature matrix (141 patients x 149 features)
    - labels.csv:        Binary outcome labels
    - feature_names.csv: Feature names
    - continuous_cols.txt: Names of continuous features

All paths are resolved via config.py.
"""

import csv
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR, DATASET_PATH, LABELS_PATH


def load_csvs():
    """Load raw dataset and labels from CSV files."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

    with open(DATASET_PATH, "r", newline="") as f:
        data = list(csv.reader(f))

    with open(LABELS_PATH, "r", newline="") as f:
        labels = list(csv.reader(f))

    return data, labels


def get_data(sentinel=np.nan):
    """
    Load feature matrix X and label vector y as numpy arrays.

    Returns
    -------
    X : ndarray of shape (n_patients, n_features)
        Baseline feature matrix. Missing values encoded as np.nan.
    y : ndarray of shape (n_patients,)
        Binary outcome labels (integers).
    """
    data, labels = load_csvs()

    y = np.array([int(v) for v in np.array(labels).T[0]])

    X = []
    for row in data:
        X.append([
            float(x) if x not in ("nan", "NaN", "") else np.nan
            for x in row
        ])

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def load_feature_names(path=None):
    """Load feature names from CSV."""
    if path is None:
        path = DATA_DIR / "feature_names.csv"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"feature_names.csv not found: {path}")

    df = pd.read_csv(path)
    return df["feature_name"].astype(str).tolist()


def load_continuous_features(path=None):
    """Load list of continuous feature names from text file."""
    if path is None:
        path = DATA_DIR / "continuous_cols.txt"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"continuous_cols.txt not found: {path}")

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_continuous_idx(feature_names, continuous_features):
    """
    Map continuous feature names to column indices.

    Returns
    -------
    idx : list of int
        Sorted column indices of continuous features.
    missing : list of str
        Feature names in continuous_features not found in feature_names.
    """
    name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
    idx, missing = [], []
    for c in continuous_features:
        if c in name_to_idx:
            idx.append(name_to_idx[c])
        else:
            missing.append(c)
    return sorted(set(idx)), missing
