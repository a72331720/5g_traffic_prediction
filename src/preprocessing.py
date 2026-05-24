"""Clean data, encode categoricals, and build features for 5G throughput prediction."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder


from . import config


# ── Cleaning ───────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, handle missing values. Timestamps parsed by loader."""
    df = df.drop_duplicates().copy()
    if config.TIMESTAMP_COLUMN in df.columns:
        df = df.sort_values(config.TIMESTAMP_COLUMN).reset_index(drop=True)
    df = df.ffill().bfill()
    return df


# ── Categorical encoding ──────────────────────────────────────────
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns."""
    df = df.copy()
    for col in config.CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
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
def add_lag_features(df: pd.DataFrame, lags=(1, 2, 3),
                     timestamps: "pd.Series | None" = None) -> pd.DataFrame:
    """Add lagged target values for time-series modelling.

    If timestamps is provided, lag resets at session boundaries
    (gap > LSTM_SESSION_GAP_SECONDS), preventing cross-session leakage.
    """
    df = df.copy()
    if timestamps is not None:
        gap = pd.Timedelta(seconds=config.LSTM_SESSION_GAP_SECONDS)
        session_id = (timestamps.diff() > gap).cumsum()
        session_id.index = df.index
        for lag in lags:
            df[f"lag_{lag}"] = df[config.TARGET_COLUMN].groupby(
                session_id, group_keys=False).shift(lag)
    else:
        for lag in lags:
            df[f"lag_{lag}"] = df[config.TARGET_COLUMN].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df


# ── Rolling features ─────────────────────────────────────────────
def add_rolling_features(df: pd.DataFrame, windows=(3, 6),
                         timestamps: "pd.Series | None" = None) -> pd.DataFrame:
    """Add rolling mean and std of target using only past values (no leakage).

    If timestamps is provided, rolling resets at session boundaries
    (gap > LSTM_SESSION_GAP_SECONDS), preventing cross-session leakage.
    """
    df = df.copy()
    if timestamps is not None:
        gap = pd.Timedelta(seconds=config.LSTM_SESSION_GAP_SECONDS)
        session_id = (timestamps.diff() > gap).cumsum()
        session_id.index = df.index
    else:
        session_id = None

    target = df[config.TARGET_COLUMN]
    for w in windows:
        if session_id is not None:
            df[f"rolling_mean_{w}"] = target.groupby(
                session_id, group_keys=False
            ).transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
            df[f"rolling_std_{w}"] = target.groupby(
                session_id, group_keys=False
            ).transform(lambda x: x.shift(1).rolling(w, min_periods=w).std())
        else:
            df[f"rolling_mean_{w}"] = target.shift(1).rolling(w).mean()
            df[f"rolling_std_{w}"] = target.shift(1).rolling(w).std()
    df = df.dropna().reset_index(drop=True)
    return df


# ── One-hot encoding (for LSTM) ──────────────────────────────────
def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns. Drops original columns.

    Label encoding + StandardScaler imposes false ordinal distances on
    categoricals. One-hot gives the LSTM clean binary indicators instead.
    """
    df = df.copy()
    for col in config.CATEGORICAL_COLUMNS:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
    return df


def build_lstm_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Feature pipeline for LSTM: clean → time features → one-hot encode.

    No label encoding, no lag/rolling features. Returns the full DataFrame
    so callers can extract feature columns dynamically.
    """
    df = clean(df_raw)
    df = add_time_features(df)
    df = one_hot_encode(df)
    return df


def build_lstm_lag_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Feature pipeline for LSTM with engineered features: clean → one-hot → time → lag → rolling.

    Otherwise identical to build_features() but uses one-hot instead of label encoding.
    """
    df = clean(df_raw)
    df = one_hot_encode(df)
    df = add_time_features(df)
    df = add_lag_features(df, timestamps=df[config.TIMESTAMP_COLUMN])
    df = add_rolling_features(df, timestamps=df[config.TIMESTAMP_COLUMN])
    return df


# ── Master pipeline ──────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: clean → encode → time → lag → rolling."""
    df = clean(df)
    df = encode_categorical(df)
    df = add_time_features(df)
    df = add_lag_features(df, timestamps=df[config.TIMESTAMP_COLUMN])
    df = add_rolling_features(df, timestamps=df[config.TIMESTAMP_COLUMN])
    return df


def save_processed(df: pd.DataFrame) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DATASET_FILE, index=False)
