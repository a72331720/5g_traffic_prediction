"""Load raw network traffic data."""
from pathlib import Path
import pandas as pd

from . import config


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw traffic CSV into a DataFrame."""
    path = Path(path) if path else config.RAW_DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Download a Kaggle network traffic dataset and place it there."
        )
    df = pd.read_csv(path, parse_dates=[config.TIMESTAMP_COLUMN])
    return df


def load_processed(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned dataset."""
    path = Path(path) if path else config.PROCESSED_DATASET_FILE
    return pd.read_csv(path, parse_dates=[config.TIMESTAMP_COLUMN])
