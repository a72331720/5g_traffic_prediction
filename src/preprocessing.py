"""Clean data, encode categoricals, and build features for 5G throughput prediction."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from . import config


# ── Cleaning ───────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, parse timestamps, handle missing values."""
    df = df.drop_duplicates().copy()
    if config.TIMESTAMP_COLUMN in df.columns:
        df[config.TIMESTAMP_COLUMN] = pd.to_datetime(df[config.TIMESTAMP_COLUMN])
        df = df.sort_values(config.TIMESTAMP_COLUMN).reset_index(drop=True)
    df = df.ffill().bfill()
    return df


# ── Categorical encoding ──────────────────────────────────────────
_label_encoders: dict[str, LabelEncoder] = {}


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns and cast booleans to int."""
    df = df.copy()
    for col in config.CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        _label_encoders[col] = le
    for col in config.BOOLEAN_COLUMNS:
        df[col] = df[col].astype(int)
    return df


# ── Time features ─────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand timestamp into hour / weekday / is_weekend features."""
    df = df.copy()
    ts = df[config.TIMESTAMP_COLUMN]
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


# ── Lag features ──────────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    """Add lagged target values for time-series modelling."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[config.TARGET_COLUMN].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


# ── Rolling features ─────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame, windows=(3, 6)) -> pd.DataFrame:
    """Add rolling mean and std of target using only past values (no leakage)."""
    df = df.copy()
    target = df[config.TARGET_COLUMN]
    for w in windows:
        df[f"rolling_mean_{w}"] = target.shift(1).rolling(w).mean()
        df[f"rolling_std_{w}"] = target.shift(1).rolling(w).std()
    df = df.dropna().reset_index(drop=True)
    return df


# ── Master pipeline ──────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: clean → encode → time → lag → rolling."""
    df = clean(df)
    df = encode_categorical(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def save_processed(df: pd.DataFrame) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DATASET_FILE, index=False)
