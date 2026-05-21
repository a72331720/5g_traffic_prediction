# 5G Network Traffic Prediction

Course project: data analysis and machine learning based prediction of 5G network traffic.

## Project Structure

```
5g_traffic_prediction/
├── data/
│   ├── raw/                # Original datasets (e.g. from Kaggle)
│   └── processed/          # Cleaned / feature-engineered data
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── results/
│   ├── figures/            # Plots and visualizations
│   └── models/             # Trained model artifacts (.pkl)
├── src/
│   ├── config.py           # Paths and global settings
│   ├── data_loader.py      # Load raw data
│   ├── preprocessing.py    # Clean data and build features
│   ├── analysis.py         # Traffic pattern / peak-hour analysis
│   ├── model.py            # Linear Regression, Random Forest
│   ├── evaluation.py       # RMSE / MAE / R²
│   ├── visualization.py    # Plotting helpers
│   └── main.py             # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

## Team & Roles

| Member      | Role                                     |
|-------------|------------------------------------------|
| LIU ZHENYI  | Machine learning model, data preprocessing |
| REN XUHUI   | Data analysis and visualization          |
| YANG XIAOLU | Literature research, report writing, record manager |
| WANG JIARUI | Presentation slides and documentation    |

## Dataset

- **Source**: [uccmisl/5Gdataset](https://github.com/uccmisl/5Gdataset) — real 5G drive-test traces
- **Paper**: "Beyond Throughput: The Next Generation a 5G Dataset with Channel and Context Metrics" (Raca et al., ACM MMSys 2020)
- **Size**: 188,711 rows, 1-second granularity, 27 dates (2019-11-20 to 2020-02-27)
- **Features**: RSRP, RSRQ, SNR, CQI, RSSI, DL/UL bitrate, Speed, Location
- **Target**: DL_bitrate (mean 10.8 Mbps, max 533 Mbps)

## Model Performance

| Model            | RMSE      | MAE     | R²     |
|------------------|-----------|---------|--------|
| Linear Regression | 28,943   | 13,043  | 0.799  |
| Random Forest    | 19,580   | 8,077   | 0.908  |
| LSTM             | 45,268   | 21,097  | 0.509  |

## Quick Start

```bash
pip install -r requirements.txt
python -m src.main
```

## Timeline

- Week 1–2: Dataset collection & initial research
- Week 3–4: Data preprocessing & traffic pattern analysis
- Week 5–6: Model building & traffic prediction
- Week 7: Visualization & result analysis
- Week 8: Final report & presentation
