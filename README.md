# 📡 5G Network Traffic Prediction

Course project: data analysis and machine learning based prediction of 5G network traffic.

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
| WANG JIARUI | Full pipeline: preprocessing, feature engineering, model building (LR/RF/LSTM), evaluation, visualization, notebooks, video production |
| LIU ZHENYI  | Dataset collection & initial research    |
| REN XUHUI   | Report writing                           |
| YANG XIAOLU | Presentation slides (PPT)                |

## 📊 Dataset

- **Source**: [uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset) — real 5G drive-test traces
- **Paper**: "Beyond Throughput: The Next Generation a 5G Dataset with Channel and Context Metrics" (Raca et al., ACM MMSys 2020)
- **Size**: 188,711 rows, 1-second granularity, 27 dates (2019-11-20 to 2020-02-27)
- **Raw features**: RSRP, RSRQ, SNR, CQI, RSSI, UL_bitrate, Speed, Longitude, Latitude, NetworkMode, Operatorname, Application, Mobility, State
- **Engineered features**: hour, weekday, is_weekend, lag_1, lag_2, lag_3, rolling_mean_3, rolling_mean_6, rolling_std_3, rolling_std_6
- **Target**: DL_bitrate (mean 10.8 Mbps, max 533 Mbps; raw data in kbps)

## 📈 Model Performance

| Model            | RMSE      | MAE     | R²     |
|------------------|-----------|---------|--------|
| Linear Regression | 28,954   | 13,051  | 0.799  |
| ⭐ Random Forest  | 19,588   | 8,082   | 0.908  |
| LSTM             | 44,964   | 21,489  | 0.518  |
| LSTM + lag       | 31,843   | 14,751  | 0.758  |

✅ **Random Forest is the recommended model.** This dataset is feature-driven rather
than sequence-driven: the strongest predictive signal is lag-1 autocorrelation
(0.89), which tree-based models exploit directly via engineered lag/rolling
features. Recurrent architectures must learn the same patterns from raw noisy
RF measurements over a limited window, making them a less natural fit here.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python -m src.main
```

## 📅 Timeline

- Week 1–2: Dataset collection & initial research
- Week 3–4: Data preprocessing & traffic pattern analysis
- Week 5–6: Model building & traffic prediction
- Week 7: Visualization & result analysis
- Week 8: Final report & presentation
