"""Plotting helpers for 5G traffic analysis and model predictions."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from . import config

sns.set_style("whitegrid")


# ── helper ────────────────────────────────────────────────────────
def _save_or_show(fig, save_name):
    if save_name:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# =====================================================================
# Traffic pattern plots (adapted from original)
# =====================================================================
def plot_hourly_pattern(hourly_series, save_name=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    hourly_series.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Average Download Speed by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Avg Download Speed (kbps)")
    _save_or_show(fig, save_name)


def plot_weekday_pattern(weekday_series, save_name=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    plot_data = weekday_series.copy()
    plot_data.index = labels[: len(plot_data)]
    plot_data.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Average Download Speed by Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Avg Download Speed (kbps)")
    ax.tick_params(axis="x", rotation=0)
    _save_or_show(fig, save_name)


def plot_daily_trend(daily_series, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(daily_series.index), daily_series.values, linewidth=1, color="steelblue")
    ax.set_title("Average Daily Download Speed Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Download Speed (kbps)")
    ax.tick_params(axis="x", rotation=30)
    _save_or_show(fig, save_name)


def plot_weekday_vs_weekend(stats_df, save_name=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    stats_df[["mean", "max"]].plot(kind="bar", ax=ax, color=["steelblue", "coral"])
    ax.set_title("Weekday vs Weekend Throughput")
    ax.set_xlabel("")
    ax.set_ylabel("Download Speed (kbps)")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(["Mean", "Max"])
    _save_or_show(fig, save_name)


def plot_weekly_heatmap(heatmap_df, save_name=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_df, cmap="YlOrRd", annot=True, fmt=".0f", linewidths=0.5, ax=ax)
    ax.set_title("Avg Download Speed: Hour × Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Hour")
    _save_or_show(fig, save_name)


def plot_congestion(congestion_series, threshold_pct=75, save_name=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    congestion_series.plot(kind="bar", ax=ax, color="coral")
    ax.set_title(f"Low-Throughput Hours (bottom {100-threshold_pct}th pct)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Congestion Count")
    _save_or_show(fig, save_name)


# =====================================================================
# 5G-specific comparison plots
# =====================================================================
def plot_network_mode_comparison(comp_df, save_name=None):
    """Grouped bar chart comparing 5G / LTE / HSPA+ across key metrics."""
    n_metrics = len(comp_df.columns)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    for ax, metric in zip(axes, comp_df.columns):
        comp_df[metric].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Network Mode Performance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_name)


def plot_mobility_comparison(mob_df, save_name=None):
    """Bar chart comparing Driving vs Static throughput."""
    fig, axes = plt.subplots(1, len(mob_df.columns), figsize=(5 * len(mob_df.columns), 4))
    if len(mob_df.columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, mob_df.columns):
        mob_df[col].plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Mobility")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Performance by Mobility Type", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_name)


def plot_application_comparison(app_df, save_name=None):
    """Bar chart comparing throughput by application type."""
    fig, axes = plt.subplots(1, len(app_df.columns), figsize=(5 * len(app_df.columns), 4))
    if len(app_df.columns) == 1:
        axes = [axes]
    for ax, col in zip(axes, app_df.columns):
        app_df[col].plot(kind="bar", ax=ax, color=["#2ca02c", "#ff7f0e", "#1f77b4"])
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Application")
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Performance by Application Type", fontsize=14, y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_name)


def plot_signal_correlation(corr_df, save_name=None):
    """Heatmap of signal-strength vs performance correlations."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Signal Strength vs Performance Correlation")
    _save_or_show(fig, save_name)


# =====================================================================
# Model prediction / evaluation plots
# =====================================================================
def plot_prediction(y_true, y_pred, save_name=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.array(y_true)[:300], label="Actual", linewidth=1)
    ax.plot(np.array(y_pred)[:300], label="Predicted", linewidth=1, alpha=0.8)
    ax.set_title("Predicted vs Actual Download Speed")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Download Speed (kbps)")
    ax.legend()
    _save_or_show(fig, save_name)


def plot_residuals(y_true, y_pred, save_name=None):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(residuals, linewidth=0.8, color="steelblue")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residuals Over Time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Residual (Actual - Predicted)")
    _save_or_show(fig, save_name)


def plot_error_distribution(y_true, y_pred, save_name=None):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=50, edgecolor="white", color="steelblue", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Error Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    _save_or_show(fig, save_name)


def plot_actual_vs_predicted(y_true, y_pred, save_name=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1.5, label="y = x")
    ax.set_title("Actual vs Predicted Download Speed")
    ax.set_xlabel("Actual (kbps)")
    ax.set_ylabel("Predicted (kbps)")
    ax.legend()
    _save_or_show(fig, save_name)


def plot_model_comparison(model_specs, save_name=None):
    """Grouped bar chart comparing models across RMSE / MAE / R².

    model_specs: list of (name, metrics_dict) tuples, e.g.
        [("LR", lr_metrics), ("RF", rf_metrics), ("LSTM", lstm_metrics)]
    """
    n_models = len(model_specs)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    metric_names = ["RMSE", "MAE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, m in zip(axes, metric_names):
        names = [spec[0] for spec in model_specs]
        vals = [spec[1][m] for spec in model_specs]
        bars = ax.bar(names, vals, color=colors[:n_models])
        ax.set_title(m, fontsize=13)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01, f"{v:.2f}",
                    ha="center", fontsize=10)
    fig.suptitle("Model Comparison", fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_name)


def plot_lstm_loss(train_losses, val_losses=None, save_name=None):
    """LSTM training & validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, linewidth=1.2, color="steelblue", label="Train")
    if val_losses is not None:
        ax.plot(val_losses, linewidth=1.2, color="coral", label="Val")
    ax.set_title("LSTM Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    _save_or_show(fig, save_name)


def plot_congestion_timeline(df, threshold_pct=75, save_name=None):
    """Throughput over time with congestion threshold highlighted."""
    threshold = np.percentile(df[config.TARGET_COLUMN], 100 - threshold_pct)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[config.TIMESTAMP_COLUMN], df[config.TARGET_COLUMN],
            linewidth=0.6, color="steelblue", alpha=0.8, label="Throughput")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Congestion threshold ({threshold:.0f} kbps)")
    ax.fill_between(df[config.TIMESTAMP_COLUMN], 0, df[config.TARGET_COLUMN],
                    where=df[config.TARGET_COLUMN] <= threshold,
                    color="red", alpha=0.15, label="Congestion zone")
    ax.set_title("5G Throughput Timeline with Congestion Detection")
    ax.set_xlabel("Time")
    ax.set_ylabel("Download Speed (kbps)")
    ax.legend(loc="upper right")
    _save_or_show(fig, save_name)
