# Notebooks

- `01_data_exploration.ipynb` — Dataset overview: structure, missing values, time range, network mode / application distribution, throughput distribution, RSRP vs throughput
- `02_feature_engineering.ipynb` — Build features step by step: categorical encoding, time features, correlation heatmap. Imports reusable pipeline from `src/preprocessing.py`
- `03_model_comparison.ipynb` — Train and compare LR, RF, LSTM, LSTM+lag. Visualisations: prediction curves, actual vs predicted scatter, residuals, error distribution, loss curves, model comparison bar chart
