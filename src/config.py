from pathlib import Path

# ============================================================
# Directory structure
# ============================================================
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# ============================================================
# Input data files
# ============================================================
DATASET_PATH = DATA_DIR / "dataset.csv"
LABELS_PATH = DATA_DIR / "labels.csv"
FEATURE_NAMES_PATH = DATA_DIR / "feature_names.csv"
CONTINUOUS_COLS_PATH = DATA_DIR / "continuous_cols.txt"

# ============================================================
# Output directories (created on demand by each script)
# ============================================================
STABILITY_DIR = RESULTS_DIR / "stability_analysis"
MULTIFEATURE_DIR = RESULTS_DIR / "multifeature_classification"
ALT_FS_DIR = RESULTS_DIR / "alternative_feature_selection"
PHENOTYPE_DIR = RESULTS_DIR / "phenotype_analysis"
SENSITIVITY_DIR = RESULTS_DIR / "sensitivity_analyses"
DESCRIPTIVE_DIR = RESULTS_DIR / "descriptive_statistics"
PATIENT_CLASS_DIR = RESULTS_DIR / "patient_classification"

# ============================================================
# Analysis parameters
# ============================================================

# Feature selection
N_ITERATIONS = 100          # Resampling iterations
N_SHUFFLES = 10             # Train/test splits per iteration
TOP_K = 30                  # Features selected per iteration
STABILITY_THRESHOLD = 0.70  # Minimum selection frequency

# Classification
CV_REPEATS = 10             # Repeated stratified CV
CV_FOLDS = 10               # Folds per repeat
RANDOM_STATE = 42           # Base random seed

# Outcome definition
MIN_DOMAINS = 3             # Minimum symptom domains for composite score
PERCENTILE_HIGH = 2 / 3     # High-responder threshold (67th percentile)
PERCENTILE_LOW = 1 / 3      # Non-responder threshold (33rd percentile)

# Missing value sentinel
SENTINEL = float("nan")

# Single-feature evaluation: hyperparameter grids
SVM_PARAMS = "0.01,0.1,1,10"
KNN_PARAMS = "1,3,5,7,9,11,13,15,17,19"
NET_PARAMS = "0.005"

# Multi-feature classification: fixed hyperparameters
SVM_C = 1.0
KNN_K = 5
DT_MAX_DEPTH = 5


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
