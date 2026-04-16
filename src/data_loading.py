from pathlib import Path
import numpy as np
import csv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"

def load_csvs():
    dataset_path = DATA_DIR / "dataset.csv"
    labels_path  = DATA_DIR / "labels.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    # Load data
    with open(dataset_path, "r", newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    # Load labels
    with open(labels_path, "r", newline="") as f:
        reader = csv.reader(f)
        labels = list(reader)

    return data, labels

def get_data(binary=False, n_per_class=None, sentinel=np.nan):

    data, labels = load_csvs()

    y = np.array(labels).T[0].tolist()
    y = [int(v) for v in y]

    X = []
    for row in data:
            X.append([float(x) if x not in ('nan', 'NaN', '') else np.nan for x in row])

    if not subsamp:
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    if n_per_class is None:
        raise ValueError("If subsamp=True you must pass n_per_class (int).")

    