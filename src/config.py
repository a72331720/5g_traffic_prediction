"""Project-wide paths and settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# Dataset files
RAW_DATASET_FILE = RAW_DATA_DIR / "5g_network_data.csv"
PROCESSED_DATASET_FILE = PROCESSED_DATA_DIR / "5g_network_clean.csv"

# Column that holds the throughput to be predicted
TARGET_COLUMN = "DL_bitrate"
TIMESTAMP_COLUMN = "Timestamp"

# Train / test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Categorical columns (will be label-encoded) ────────────────────
CATEGORICAL_COLUMNS = [
    "NetworkMode",
    "Operatorname",
    "Application",
    "Mobility",
    "State",
]

# ── Numeric columns from raw data ──────────────────────────────────
NUMERIC_COLUMNS = [
    "Speed",
    "RSRP",
    "RSRQ",
    "SNR",
    "CQI",
    "RSSI",
    "UL_bitrate",
    "Longitude",
    "Latitude",
]

# ── Time-based features (created in preprocessing) ─────────────────
TIME_FEATURES = ["hour", "weekday", "is_weekend"]

# ── Lag & rolling features (created in preprocessing) ──────────────
LAG_FEATURES = ["lag_1", "lag_2", "lag_3"]
ROLLING_FEATURES = ["rolling_mean_3", "rolling_mean_6", "rolling_std_3", "rolling_std_6"]

# ── Final feature set used by models ───────────────────────────────
FEATURE_COLUMNS = (
    NUMERIC_COLUMNS
    + CATEGORICAL_COLUMNS

    + TIME_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
)

# ── LSTM settings ──────────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH = 60   # 60 time steps (~1 min of 1-second granularity data)
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 64
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_LEARNING_RATE = 0.001
LSTM_SESSION_GAP_SECONDS = 10  # gaps > 10 s indicate a new drive-test session
