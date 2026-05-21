"""Machine learning models for 5G throughput prediction.

Models: Linear Regression, Random Forest, LSTM (PyTorch).
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config

FEATURE_COLUMNS = config.FEATURE_COLUMNS


# =====================================================================
# Data splitting
# =====================================================================
def split_data(df: pd.DataFrame):
    """Chronological train/test split (no shuffle for time-series)."""
    X = df[FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        shuffle=False,
        random_state=config.RANDOM_STATE,
    )


# =====================================================================
# Linear Regression
# =====================================================================
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# =====================================================================
# Random Forest
# =====================================================================
def train_random_forest(X_train, y_train, n_estimators=200, max_depth=None):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train):
    """GridSearchCV over key Random Forest hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    print(f"Best CV R2: {gs.best_score_:.4f}")
    return gs.best_estimator_


# =====================================================================
# LSTM (PyTorch)
# =====================================================================
class LSTMModel(nn.Module):
    """LSTM for regression on sequential 5G network data."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]          # last time-step
        return self.fc(out).squeeze(-1)


def prepare_lstm_sequences(X: np.ndarray, y: np.ndarray | None, seq_len: int,
                           timestamps: "pd.Series | None" = None):
    """Slide a window of `seq_len` over X to produce (samples, seq_len, features).

    If `timestamps` is provided, sequences that cross a session boundary
    (gap > LSTM_SESSION_GAP_SECONDS) are skipped — they represent
    time-steps from different drive tests and contaminate training.
    """
    n = len(X)
    if timestamps is not None and n > seq_len:
        gap = pd.Timedelta(seconds=config.LSTM_SESSION_GAP_SECONDS)
        # session_id changes at every gap — sequence valid only if uniform
        session_id = (timestamps.diff() > gap).cumsum().values
    else:
        session_id = None

    Xs = []
    ys = [] if y is not None else None
    for i in range(n - seq_len):
        if session_id is not None and session_id[i] != session_id[i + seq_len]:
            continue
        Xs.append(X[i : i + seq_len])
        if y is not None:
            ys.append(y[i + seq_len])
    if y is not None:
        return np.array(Xs), np.array(ys)
    return np.array(Xs)


def train_lstm(X_train, y_train, X_test, y_test,
               train_ts: "pd.Series | None" = None,
               test_ts: "pd.Series | None" = None):
    """Train an LSTM model and return (model, scaler, y_scaler, train_losses, val_losses, y_test_orig).

    Key fixes vs. baseline:
      - session-boundary detection: sequences spanning a time gap are skipped
      - val split BEFORE sequence creation to eliminate train/val time overlap
      - shuffle=False in DataLoader to preserve temporal order
      - ReduceLROnPlateau scheduler for better convergence
    """
    seq_len = config.LSTM_SEQUENCE_LENGTH
    epochs = config.LSTM_EPOCHS
    batch_size = config.LSTM_BATCH_SIZE
    lr = config.LSTM_LEARNING_RATE
    patience = 15

    # Feature scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Target scaling (direct kbps, no log transform)
    y_train_np = np.array(y_train, dtype=np.float32)
    y_test_np = np.array(y_test, dtype=np.float32)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_np.reshape(-1, 1)).ravel()

    # Chronological val split BEFORE creating sequences (avoids overlap leakage)
    n_train = len(X_train_sc)
    val_size = max(int(n_train * 0.1), seq_len + 1)
    train_end = n_train - val_size

    X_tr_raw = X_train_sc[:train_end]
    y_tr_raw = y_train_scaled[:train_end]
    X_val_raw = X_train_sc[train_end:]
    y_val_raw = y_train_scaled[train_end:]
    tr_ts = train_ts.iloc[:train_end] if train_ts is not None else None
    val_ts = train_ts.iloc[train_end:] if train_ts is not None else None

    # Create sequences with session-boundary filtering
    X_train_seq, y_train_seq = prepare_lstm_sequences(
        X_tr_raw, y_tr_raw, seq_len, timestamps=tr_ts)
    X_val_seq, y_val_seq = prepare_lstm_sequences(
        X_val_raw, y_val_raw, seq_len, timestamps=val_ts)
    X_test_seq, y_test_seq = prepare_lstm_sequences(
        X_test_sc, y_test_np, seq_len, timestamps=test_ts)

    print(f"  Train seqs: {len(X_train_seq)}  Val seqs: {len(X_val_seq)}  Test seqs: {len(X_test_seq)}")

    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    n_features = X_train_sc.shape[1]
    model = LSTMModel(
        input_size=n_features,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
    )
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(X_val_seq, dtype=torch.float32))
            val_loss = criterion(
                val_preds, torch.tensor(y_val_seq, dtype=torch.float32)
            ).item()
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}")

        if stale >= patience:
            print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    y_test_orig = y_test_seq.astype(np.float32)
    return model, scaler, y_scaler, train_losses, val_losses, y_test_orig


def predict_lstm(model, scaler, X, y_scaler=None, seq_len=None,
                  timestamps: "pd.Series | None" = None):
    """Generate predictions from a trained LSTM. Returns 1-D array in original kbps scale.

    Applies session-boundary filtering if timestamps provided, matching the
    training-time sequence creation so predictions align with y_test from train_lstm.
    """
    if seq_len is None:
        seq_len = config.LSTM_SEQUENCE_LENGTH
    X_sc = scaler.transform(X)
    X_seq = prepare_lstm_sequences(X_sc, None, seq_len, timestamps=timestamps)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_seq, dtype=torch.float32)).numpy()
    if y_scaler is not None:
        preds = y_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    else:
        preds = preds_scaled
    return preds


# =====================================================================
# Feature importance (RF)
# =====================================================================
def plot_feature_importance(model, save_name: str | None = None):
    """Horizontal bar chart of feature importance (RF only)."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    if save_name:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / save_name, dpi=150, bbox_inches="tight")
        plt.close(fig)


# =====================================================================
# Save / load
# =====================================================================
def save_model(model, name: str) -> Path:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{name}.pkl"
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), config.MODELS_DIR / f"{name}.pt")
        return config.MODELS_DIR / f"{name}.pt"
    joblib.dump(model, path)
    return path


