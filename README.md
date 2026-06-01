# 📡 5G Network Traffic Prediction

> Predicting downlink throughput on real-world 5G drive-test data — and why a
> tree-based model beats an LSTM on this kind of telemetry.

## 🎯 Problem & Motivation

Mobile networks need second-by-second estimates of the throughput each user
can sustain. This drives many production systems: video bitrate adaptation
(ABR), QoE-aware scheduling, edge cache pre-fetching, and capacity planning.
4G prediction was already hard; 5G makes it harder — beam-forming,
mmWave/Sub-6 switching, and heavier mobility produce a heavy-tailed
throughput distribution (DL_bitrate mean ≈ 10.8 Mbps, max ≈ 533 Mbps in
this dataset) that classical regressors struggle with.

The question we ask in this project:

> Given channel measurements (RSRP, RSRQ, SNR, CQI, RSSI), context
> (location, speed, network mode), and recent throughput history, how
> well can we predict the next DL throughput sample on a real 5G network?

And the more interesting question that emerged once we started training:

> When a tree-based model beats an LSTM on time-series telemetry — is it
> because of the model class, or because the tree has access to engineered
> lag features the LSTM doesn't? We answer this with a controlled
> experiment in [§ Controlled Experiment](#controlled-experiment-lstm--lag-features).

## ⚡ TL;DR

- **Random Forest wins**: R² = 0.909 vs LSTM 0.39 vs LR 0.80 (Figure 1).
- The dataset is **feature-driven, not sequence-driven** — lag-1
  autocorrelation = 0.89 dominates the signal.
- Even with identical lag/rolling features, **LSTM still loses**
  (R² = 0.719) — so the gap is the model class, not feature deprivation.
- LSTM is also **less reproducible** across PyTorch versions
  (R² varies 0.39–0.52); RF stays within 0.907–0.909.

## 📊 Dataset

- **Source**: [uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset) [1] — real-world 5G
  drive-test traces from a major Irish mobile operator, collected via G-NetTrack Pro.
- **Mobility patterns**: Static and Driving (car).
- **Application patterns**: video streaming (Netflix, Amazon Prime) and file download.
- **Metrics**: channel (RSRP, RSRQ, SNR, CQI, RSSI), context (location, speed,
  network mode), and throughput (DL/UL bitrate). This is the first publicly
  available 5G dataset combining throughput with channel and context information.
- **Size**: 188,711 rows, 1-second granularity, 27 dates (2019-11-20 to 2020-02-27).
- **Target**: DL_bitrate (mean 10,758 kbps, max 532,905 kbps) — **heavy-tailed**.

> Only the real-world drive-test traces are used. The accompanying ns-3 simulation
> framework is not included in this analysis.

## 🛠 Methodology

### Preprocessing

1. **Session detection** — drive-test traces are concatenated across multiple
   sessions in the raw CSV. We treat any timestamp gap > 10 s as a session
   boundary so rolling/lag features and LSTM sliding windows never span two
   unrelated sessions. Cross-session sequences would be garbage — combinations
   of rows that never occurred together in reality.
2. **Categorical encoding** — `NetworkMode`, `State`, `Operatorname`,
   `Application`, `Mobility` are one-hot encoded for LSTM (label encoding
   misled the network) and label-encoded for tree models (RF handles either).
3. **Train/test split** — chronological 80/20 split done on **raw rows
   before** sliding-window construction. Splitting after window construction
   creates temporal leakage between train and test sequences.

### Feature Engineering

| Group | Features |
|---|---|
| Raw channel | RSRP, RSRQ, SNR, CQI, RSSI, UL_bitrate |
| Context | Speed, Longitude, Latitude, NetworkMode, Operatorname, Application, Mobility, State |
| Time | hour, weekday, is_weekend |
| **Lag** | lag_1, lag_2, lag_3 (DL_bitrate at t-1, t-2, t-3, **within session**) |
| **Rolling** | rolling_mean_3, rolling_mean_6, rolling_std_3, rolling_std_6 |

Lag and rolling features are computed **per session** to avoid bleeding across
drive-test boundaries.

### Models

- **Linear Regression** — baseline.
- **Random Forest** — sklearn defaults (n_estimators=100); fed all engineered
  features.
- **LSTM** — 2-layer, hidden=64, sequence length=60, batch=64, lr=0.001,
  ReduceLROnPlateau scheduler, MSE loss. Trained on raw channel + context only
  (no lag features), so it must learn the temporal signal from the sequence
  alone. `shuffle=False` on the DataLoader to preserve temporal order.
- **LSTM + lag** — same architecture as LSTM, but fed the same lag/rolling
  features as RF. This is the controlled experiment.

`torch.manual_seed` is set for LSTM reproducibility within a PyTorch version.

## 📈 Results

| Model             | RMSE (kbps) | MAE (kbps) | R²        |
|-------------------|-------------|------------|-----------|
| Linear Regression | 28,998      | 13,086     | 0.799     |
| Random Forest     | 19,538      | 8,086      | **0.909** |
| LSTM              | 50,604      | 24,158     | 0.389     |
| LSTM + lag        | 34,320      | 16,306     | 0.719     |

![Model comparison](results/figures/model_comparison.png)
*Figure 1 — Model comparison on the held-out test set. RF dominates across all three metrics; LSTM trails even when given identical lag features.*

![RF actual vs predicted](results/figures/actual_vs_predicted_RF.png)
*Figure 2 — Random Forest predictions vs ground-truth DL_bitrate on the held-out test set. Points cluster on the diagonal; residuals widen at the high-throughput tail (a known characteristic of heavy-tailed targets).*

![RF feature importance](results/figures/feature_importance.png)
*Figure 3 — Top features by RF impurity reduction. The lag-1 feature dominates, consistent with the lag-1 autocorrelation of 0.89 measured on the raw target.*

As shown in **Figure 1**, Random Forest is the clear winner, and **Figure 3**
explains why: the dominant predictor is `lag_1` — the previous second's
throughput — and tree splits exploit it directly without any architectural
overhead. This aligns with the broader finding that tree ensembles often
outperform deep learning on structured tabular data [3].

### Controlled Experiment: LSTM + Lag Features

To rule out that Random Forest's win comes purely from feature engineering
rather than the model class itself, we trained an LSTM variant with the
**same lag and rolling features** that RF uses (LSTM+lag). With identical
features, LSTM+lag reaches R² = 0.719 — still below both Random Forest (0.909)
and Linear Regression (0.799). This isolates the cause: the gap is the
model class, not feature deprivation. Sequence models cannot exploit lag-1
autocorrelation as efficiently as a tree split on the lag feature directly.

## ⚖️ Trade-offs

| Axis                            | Random Forest          | LSTM                           | Notes |
|---------------------------------|------------------------|--------------------------------|-------|
| **Accuracy on this dataset**    | R² 0.909               | R² 0.39–0.52                   | RF wins decisively |
| **Reproducibility**             | ±0.002 across runs     | ±0.13 across PyTorch versions  | RF is platform-stable |
| **Training cost**               | < 1 min CPU            | 15–20 min GPU / 30–60 min CPU  | RF is two orders cheaper |
| **Model size on disk**          | < 1 MB (`.pkl`)        | ~220 MB (`lstm.pt`)            | RF deploys easily |
| **Interpretability**            | Feature importance     | Opaque                         | RF gives operators a story |
| **Scaling to longer histories** | Re-engineer lag/rolling| Native (extend `seq_len`)      | LSTM has the structural advantage — but it doesn't pay off here |
| **Cold start (no history)**     | Falls back to channel features | Same                   | Both degrade; LSTM more so |

The trade-off summary: **for tabular drive-test telemetry on a 1-second grid,
the structural advantage LSTM has (variable-length history, learned
representations) is exactly the advantage that doesn't matter when the
target is dominated by lag-1.** A tree split on `lag_1` is cheaper, more
accurate, more reproducible, and easier to ship. We would expect LSTM to
catch up or overtake RF if (a) the target depended on much longer
histories, or (b) the dataset were large enough that hand-crafted lag
features became the bottleneck — neither is the case here.

## 📝 Methodology Notes

- **Evaluation set size**: LR/RF are evaluated on ~37,533 samples; LSTM
  variants on ~36,783 / ~36,633 (after session-boundary filtering). This is
  a deliberate trade-off — discarding cross-session sequences is the correct
  behaviour, not a data leak.
- **Why no log-transform on the target**: we tried `log1p(DL_bitrate)` to
  tame the heavy tail, then `expm1` on prediction. R² in kbps space dropped
  from 0.51 to −0.11 — `expm1` amplifies small log-space errors at the
  high-throughput end. Direct `StandardScaler` on kbps works better here.

## ✅ Conclusion

Random Forest is the clear winner for this 5G throughput prediction task
(R² = 0.909). The dataset is fundamentally feature-driven, not sequence-driven:
lag-1 autocorrelation (0.89) is the dominant signal, and tree-based models
exploit it directly via engineered features while recurrent architectures
must infer it from noisy RF measurements over a limited window.

LSTM instability further reinforces this conclusion: across PyTorch versions
the same code and seed produce R² ranging from 0.39 to 0.52, while RF stays
within 0.907–0.909 regardless of platform. For production 5G traffic
prediction with tabular drive-test data, Random Forest is both the most
accurate and most reliable choice. Similar findings have been reported in
prior work on ML-based throughput prediction in LTE and 5G networks [4],
where RF-based approaches using RSRP, RSRQ, and RSSI features demonstrated
strong predictive performance [5].

## 🚀 Quick Start

Requires Python 3.12.x (or 3.10+).

```bash
pip install -r requirements.txt
python -m src.main
```

- GPU (CUDA): ~15–20 min
- CPU only: ~30–60 min
- Figures saved to `results/figures/`
- Trained models saved to `results/models/`

## 🎥 Video Demo

*Coming soon — link will be added before submission.*

## 📁 Project Structure

```
5g_traffic_prediction/
├── data/
│   ├── raw/                # Original datasets (from GitHub)
│   └── processed/          # Cleaned / feature-engineered data
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── results/
│   ├── figures/            # Plots and visualizations
│   └── models/             # Trained model artifacts (.pkl / .pt)
├── src/
│   ├── __init__.py         # Package init
│   ├── config.py           # Paths and global settings
│   ├── data_loader.py      # Load raw data
│   ├── preprocessing.py    # Clean data and build features
│   ├── analysis.py         # Traffic pattern / peak-hour analysis
│   ├── model.py            # Linear Regression, Random Forest, LSTM
│   ├── evaluation.py       # RMSE / MAE / R²
│   ├── visualization.py    # Plotting helpers
│   └── main.py             # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

## 👥 Team & Roles

| Member      | Role                                     |
|-------------|------------------------------------------|
| WANG JIARUI | Full pipeline: dataset selection & migration, preprocessing, feature engineering, model building (LR/RF/LSTM), evaluation, visualization, notebooks, literature survey & references, video production |
| LIU ZHENYI  | Dataset collection & initial research    |
| REN XUHUI   | Report writing                           |
| YANG XIAOLU | Presentation slides (PPT)                |

## 📚 References

[1] D. Raca, D. Leahy, C.J. Sreenan and J.J. Quinlan. "Beyond Throughput, The
Next Generation: A 5G Dataset with Channel and Context Metrics." *ACM Multimedia
Systems Conference (MMSys)*, Istanbul, Turkey, June 2020.
DOI: [10.1145/3339825.3394938](https://doi.org/10.1145/3339825.3394938)

[2] L. Breiman. "Random Forests." *Machine Learning*, vol. 45, no. 1, pp. 5–32,
2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

[3] R. Shwartz-Ziv and A. Armon. "Tabular Data: Deep Learning is Not All You
Need." *Information Fusion*, vol. 81, pp. 84–90, 2022.
DOI: [10.1016/j.inffus.2021.11.011](https://doi.org/10.1016/j.inffus.2021.11.011)

[4] D. Minovski, N. Ögren, C. Åhlund and K. Mitra. "Throughput Prediction Using
Machine Learning in LTE and 5G Networks." *IEEE Transactions on Mobile Computing*,
vol. 22, no. 3, pp. 1825–1840, 2023.
DOI: [10.1109/TMC.2021.3099397](https://doi.org/10.1109/TMC.2021.3099397)

[5] G.A. Fernandez. "Machine Learning for Wireless Network Throughput
Prediction." *Advances in Machine Learning & Artificial Intelligence*, vol. 5,
no. 1, pp. 1–6, 2024.
https://scholarworks.utrgv.edu/mss_fac/447/
