"""Evaluation metrics for regression models."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred) -> dict:
    """Return RMSE, MAE, and R² as a dict."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
