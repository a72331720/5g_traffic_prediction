"""Traffic pattern analysis: peak hours, 5G-specific breakdowns, congestion detection."""
import pandas as pd
import numpy as np

from . import config


# ── Basic patterns (hour / weekday) ───────────────────────────────
def hourly_pattern(df: pd.DataFrame) -> pd.Series:
    """Average throughput per hour of day."""
    return df.groupby("hour")[config.TARGET_COLUMN].mean()


def weekday_pattern(df: pd.DataFrame) -> pd.Series:
    """Average throughput per weekday (0=Mon, 6=Sun)."""
    return df.groupby("weekday")[config.TARGET_COLUMN].mean()


def peak_hours(df: pd.DataFrame, top_n: int = 3) -> list[int]:
    """Return the top-N busiest hours of day."""
    return hourly_pattern(df).sort_values(ascending=False).head(top_n).index.tolist()


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of the target column."""
    return df[config.TARGET_COLUMN].describe().to_frame()


def daily_trend(df: pd.DataFrame) -> pd.Series:
    """Average throughput per day — shows overall trend over the dataset span."""
    ts = df[config.TIMESTAMP_COLUMN]
    return df.groupby(ts.dt.date)[config.TARGET_COLUMN].mean()


def weekday_vs_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """Compare average throughput between weekdays and weekends."""
    grouped = df.groupby("is_weekend")[config.TARGET_COLUMN].agg(
        ["mean", "std", "max", "min", "count"]
    )
    grouped.index = ["Weekday", "Weekend"]
    return grouped


# ── Congestion ────────────────────────────────────────────────────
def congestion_hours(df: pd.DataFrame, threshold_pct: float = 75) -> pd.Series:
    """Identify hours where throughput drops below the given percentile threshold.

    For throughput prediction, congestion means *low* speed, so we flag
    records below the 25th percentile (i.e., bottom quartile).
    """
    threshold = np.percentile(df[config.TARGET_COLUMN], 100 - threshold_pct)
    congested = df[df[config.TARGET_COLUMN] <= threshold].copy()
    hourly = congested.groupby("hour").size().rename("congestion_count")
    hourly = hourly.reindex(range(24), fill_value=0)
    hourly.index.name = "hour"
    return hourly


def weekly_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: hour (rows) × weekday (cols) of average throughput."""
    pivot = df.pivot_table(
        values=config.TARGET_COLUMN,
        index="hour",
        columns="weekday",
        aggfunc="mean",
    )
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return pivot


# ── 5G-specific analysis ─────────────────────────────────────────
def network_mode_comparison(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compare key performance metrics across network modes (5G / LTE / HSPA+ / ...).

    Uses raw_df (before encoding) for readable labels.
    """
    metrics = [
        "DL_bitrate",
        "UL_bitrate",
        "RSRP",
        "RSRQ",
        "SNR",
        "CQI",
    ]
    available = [m for m in metrics if m in raw_df.columns]
    return raw_df.groupby("NetworkMode")[available].mean()


def mobility_comparison(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compare throughput by mobility type (Driving vs Static)."""
    return raw_df.groupby("Mobility")[["DL_bitrate", "UL_bitrate", "Speed"]].mean()


def application_comparison(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compare throughput by application type (Netflix / Amazon Prime / Download)."""
    return raw_df.groupby("Application")[["DL_bitrate", "UL_bitrate", "RSRP"]].mean()


def signal_vs_throughput(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between signal quality metrics and throughput."""
    cols = [
        "RSRP",
        "RSRQ",
        "SNR",
        "CQI",
        "RSSI",
        "DL_bitrate",
        "UL_bitrate",
        "Speed",
    ]
    available = [c for c in cols if c in raw_df.columns]
    return raw_df[available].corr()
