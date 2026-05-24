# 📡 5G Network Traffic Prediction

This project analyses real-world 5G drive-test data and builds machine learning models to predict network throughput. Using 188,711 measurements from the uccmisl/5Gdataset (ACM MMSys 2020), we compare Linear Regression, Random Forest, and LSTM approaches. Random Forest achieves R² = 0.908, confirming that the task is feature-driven rather than sequence-driven.

## 📁 Project Structure

```
5g_traffic_prediction/
├── data/
│   ├── raw/                # Original datasets (from GitHub)
│   └── processed/          # Cleaned / feature-engineered data
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── results/
│   ├── figures/            # Plots and visualizations
│   └── models/             # Trained model artifacts (.pkl)
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
| WANG JIARUI | Full pipeline: dataset selection & migration, preprocessing, feature engineering, model building (LR/RF/LSTM), evaluation, visualization, notebooks, video production |
| LIU ZHENYI  | Dataset collection & initial research    |
| REN XUHUI   | Report writing                           |
| YANG XIAOLU | Presentation slides (PPT)                |

## 📊 Dataset

- **Source**: [uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset) — real 5G drive-test traces
- **Paper**: "Beyond Throughput: The Next Generation a 5G Dataset with Channel and Context Metrics" (Raca et al., ACM MMSys 2020)
- **Size**: 188,711 rows, 1-second granularity, 27 dates (2019-11-20 to 2020-02-27)
- **Raw features**: RSRP, RSRQ, SNR, CQI, RSSI, UL_bitrate, Speed, Longitude, Latitude, NetworkMode, Operatorname, Application, Mobility, State
- **Engineered features**: hour, weekday, is_weekend, lag_1, lag_2, lag_3, rolling_mean_3, rolling_mean_6, rolling_std_3, rolling_std_6
- **Target**: DL_bitrate (mean 10,758 kbps, max 532,905 kbps)

## 📈 Model Performance

| Model            | RMSE      | MAE     | R²     |
|------------------|-----------|---------|--------|
| Linear Regression | 28,981   | 13,073  | 0.799  |
| Random Forest     | 19,690   | 8,114   | 0.907  |
| LSTM             | 50,604   | 24,158  | 0.389  |
| LSTM + lag       | 31,922   | 14,717  | 0.757  |

![Model comparison](results/figures/model_comparison.png)

✅ **Random Forest is the recommended model.** This dataset is feature-driven rather
than sequence-driven: the strongest predictive signal is lag-1 autocorrelation
(0.89), which tree-based models exploit directly via engineered lag/rolling
features. Recurrent architectures must learn the same patterns from raw noisy
radio-frequency (RF) signal measurements over a limited window, making them a less natural fit here.
Even when provided with equivalent lag-based temporal features, LSTM (R²=0.757) still underperforms
both Linear Regression (R²=0.799) and Random Forest (R²=0.907), confirming that this dataset is
inherently feature-driven rather than sequence-driven.

![RF actual vs predicted](results/figures/actual_vs_predicted_RF.png)

![RF feature importance](results/figures/feature_importance.png)

## Known Limitations

- **Evaluation set size mismatch**: LR/RF are evaluated on ~37,642 samples;
  LSTM variants on ~36,783 (after session-boundary filtering). The difference
  is negligible but noted for transparency.

## 🚀 Quick Start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m src.main
```

- Runs the full pipeline (~30–60 min)
- Figures saved to `results/figures/`
- Trained models saved to `results/models/`

## 📅 Timeline

- Week 1–2: Dataset collection & initial research
- Week 3–4: Data preprocessing & traffic pattern analysis
- Week 5–6: Model building & traffic prediction
- Week 7: Visualization & result analysis
- Week 8: Final report & presentation
