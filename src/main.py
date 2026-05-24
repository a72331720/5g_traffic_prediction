"""End-to-end pipeline: load → preprocess → 5G analysis → model → evaluate → results."""
from . import data_loader, preprocessing, analysis, model, evaluation, visualization, config


def run():
    # ── 1. Load ───────────────────────────────────────────────────
    print("=" * 60)
    print("[1/7] Loading raw 5G data...")
    df_raw = data_loader.load_raw()
    print(f"  Raw shape: {df_raw.shape}")

    # ── 2. Preprocess ─────────────────────────────────────────────
    print("[2/7] Preprocessing & feature engineering...")
    df = preprocessing.build_features(df_raw)
    preprocessing.save_processed(df)
    print(f"  Processed shape: {df.shape}")
    print(f"  Features: {len(config.FEATURE_COLUMNS)}")

    # ── 3. 5G-specific analysis ───────────────────────────────────
    print("[3/7] 5G traffic pattern analysis...")

    print(analysis.summary_stats(df))
    print("Peak hours:", analysis.peak_hours(df))

    print("  -> hourly pattern")
    visualization.plot_hourly_pattern(
        analysis.hourly_pattern(df), save_name="hourly_pattern.png"
    )
    print("  -> weekday pattern")
    visualization.plot_weekday_pattern(
        analysis.weekday_pattern(df), save_name="weekday_pattern.png"
    )
    print("  -> weekday vs weekend")
    wvsw = analysis.weekday_vs_weekend(df)
    print(wvsw)
    visualization.plot_weekday_vs_weekend(wvsw, save_name="weekday_vs_weekend.png")

    print("  -> daily trend")
    visualization.plot_daily_trend(
        analysis.daily_trend(df), save_name="daily_trend.png"
    )
    print("  -> weekly heatmap")
    visualization.plot_weekly_heatmap(
        analysis.weekly_heatmap_data(df), save_name="weekly_heatmap.png"
    )
    print("  -> congestion hours")
    visualization.plot_congestion(
        analysis.congestion_hours(df), save_name="congestion_hours.png"
    )

    # ── 5G-specific comparisons (use raw_df for labels) ───────────
    print("  -> network mode comparison (5G vs LTE vs HSPA+ ...)")
    net_comp = analysis.network_mode_comparison(df_raw)
    print(net_comp)
    visualization.plot_network_mode_comparison(net_comp, save_name="network_mode_comparison.png")

    print("  -> mobility comparison (Driving vs Static)")
    mob_comp = analysis.mobility_comparison(df_raw)
    print(mob_comp)
    visualization.plot_mobility_comparison(mob_comp, save_name="mobility_comparison.png")

    print("  -> application comparison (Netflix vs Amazon Prime vs Download)")
    app_comp = analysis.application_comparison(df_raw)
    print(app_comp)
    visualization.plot_application_comparison(app_comp, save_name="application_comparison.png")

    print("  -> signal-strength correlation")
    sig_corr = analysis.signal_vs_throughput(df_raw)
    visualization.plot_signal_correlation(sig_corr, save_name="signal_correlation.png")

    # ── 4. Train/test split ───────────────────────────────────────
    print("[4/7] Splitting data...")
    X_train, X_test, y_train, y_test = model.split_data(df)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── 5. Train models ──────────────────────────────────────────
    print("[5/7] Training models...")

    print("  -> Linear Regression (baseline)")
    lr = model.train_linear_regression(X_train, y_train)

    print("  -> Random Forest (tuned with GridSearchCV)")
    rf = model.tune_random_forest(X_train, y_train)
    model.plot_feature_importance(rf, save_name="feature_importance.png")

    print("  -> LSTM (PyTorch)")
    # LSTM uses its own feature pipeline: one-hot categoricals (not label encoding)
    df_lstm = preprocessing.build_lstm_features(df_raw)

    # One-hot columns are named like "NetworkMode_5G", "Application_Netflix", etc.
    one_hot_cols = [c for c in df_lstm.columns
                    if any(c.startswith(f"{cat}_") for cat in config.CATEGORICAL_COLUMNS)]
    lstm_feature_cols = config.NUMERIC_COLUMNS + config.TIME_FEATURES + one_hot_cols

    # Chronological split on the LSTM-specific dataframe (more rows — no lag/rolling dropna)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = model.split_data(
        df_lstm, feature_cols=lstm_feature_cols
    )

    train_ts = df_lstm.loc[X_train_lstm.index, config.TIMESTAMP_COLUMN]
    test_ts = df_lstm.loc[X_test_lstm.index, config.TIMESTAMP_COLUMN]

    lstm_model, lstm_scaler, lstm_y_scaler, lstm_losses, lstm_val_losses, y_test_lstm_seq = model.train_lstm(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm,
        train_ts=train_ts, test_ts=test_ts,
    )
    visualization.plot_lstm_loss(lstm_losses, lstm_val_losses, save_name="lstm_loss.png")

    print("  -> LSTM + lag/rolling (one-hot + engineered features)")
    df_lstm_lag = preprocessing.build_lstm_lag_features(df_raw)
    one_hot_cols_lag = [c for c in df_lstm_lag.columns
                        if any(c.startswith(f"{cat}_") for cat in config.CATEGORICAL_COLUMNS)]
    lstm_lag_feature_cols = (config.NUMERIC_COLUMNS + config.TIME_FEATURES
                             + one_hot_cols_lag + config.LAG_FEATURES + config.ROLLING_FEATURES)

    X_train_l2, X_test_l2, y_train_l2, y_test_l2 = model.split_data(
        df_lstm_lag, feature_cols=lstm_lag_feature_cols
    )

    train_ts_l2 = df_lstm_lag.loc[X_train_l2.index, config.TIMESTAMP_COLUMN]
    test_ts_l2 = df_lstm_lag.loc[X_test_l2.index, config.TIMESTAMP_COLUMN]

    lstm_lag_model, lstm_lag_scaler, lstm_lag_y_scaler, lstm_lag_losses, lstm_lag_val_losses, y_test_l2_seq = model.train_lstm(
        X_train_l2, y_train_l2, X_test_l2, y_test_l2,
        train_ts=train_ts_l2, test_ts=test_ts_l2,
    )
    visualization.plot_lstm_loss(lstm_lag_losses, lstm_lag_val_losses, save_name="lstm_lag_loss.png")

    model.save_model(lr, "linear_regression")
    model.save_model(rf, "random_forest")
    model.save_model(lstm_model, "lstm")
    model.save_model(lstm_lag_model, "lstm_lag")

    # ── 6. Evaluation ────────────────────────────────────────────
    print("[6/7] Evaluation...")

    lr_preds = lr.predict(X_test)
    rf_preds = rf.predict(X_test)
    lstm_preds = model.predict_lstm(lstm_model, lstm_scaler, X_test_lstm,
                                     y_scaler=lstm_y_scaler, timestamps=test_ts)
    lstm_lag_preds = model.predict_lstm(lstm_lag_model, lstm_lag_scaler, X_test_l2,
                                         y_scaler=lstm_lag_y_scaler, timestamps=test_ts_l2)

    # Evaluate on comparable test windows.
    # LR/RF:  y_test[seq_len:] = 37,681 samples after removing first seq_len rows.
    # LSTM:   y_test_lstm_seq = 36,783 samples (also skips cross-session sequences).
    # LSTM+lag: y_test_l2_seq = 36,781 samples (same).
    # The ~900-sample gap is from session-boundary filtering, not a data leak.
    seq_len = config.LSTM_SEQUENCE_LENGTH
    y_test_eval = y_test[seq_len:]
    lr_metrics = evaluation.evaluate(y_test_eval, lr_preds[seq_len:])
    rf_metrics = evaluation.evaluate(y_test_eval, rf_preds[seq_len:])
    lstm_metrics = evaluation.evaluate(y_test_lstm_seq, lstm_preds)
    lstm_lag_metrics = evaluation.evaluate(y_test_l2_seq, lstm_lag_preds)

    evaluation.print_metrics("LinearRegression", lr_metrics)
    evaluation.print_metrics("RandomForest", rf_metrics)
    evaluation.print_metrics("LSTM", lstm_metrics)
    evaluation.print_metrics("LSTM+lag", lstm_lag_metrics)

    # Prediction plots
    visualization.plot_prediction(y_test_eval, lr_preds[seq_len:], save_name="prediction_LR.png")
    visualization.plot_prediction(y_test_eval, rf_preds[seq_len:], save_name="prediction_RF.png")
    visualization.plot_prediction(y_test_lstm_seq, lstm_preds, save_name="prediction_LSTM.png")
    visualization.plot_prediction(y_test_l2_seq, lstm_lag_preds, save_name="prediction_LSTM_lag.png")

    # ── 7. Result analysis ───────────────────────────────────────
    print("[7/7] Result analysis...")

    print("  -> 4-model comparison")
    visualization.plot_model_comparison(
        [("LR", lr_metrics), ("RF", rf_metrics),
         ("LSTM", lstm_metrics), ("LSTM+lag", lstm_lag_metrics)],
        save_name="model_comparison.png",
    )

    print("  -> actual vs predicted (RF)")
    visualization.plot_actual_vs_predicted(
        y_test_eval, rf_preds[seq_len:], save_name="actual_vs_predicted_RF.png"
    )

    print("  -> actual vs predicted (LSTM)")
    visualization.plot_actual_vs_predicted(
        y_test_lstm_seq, lstm_preds, save_name="actual_vs_predicted_LSTM.png"
    )

    print("  -> actual vs predicted (LSTM+lag)")
    visualization.plot_actual_vs_predicted(
        y_test_l2_seq, lstm_lag_preds, save_name="actual_vs_predicted_LSTM_lag.png"
    )

    print("  -> residuals (RF)")
    visualization.plot_residuals(y_test_eval, rf_preds[seq_len:], save_name="residuals_RF.png")

    print("  -> residuals (LSTM)")
    visualization.plot_residuals(y_test_lstm_seq, lstm_preds, save_name="residuals_LSTM.png")

    print("  -> residuals (LSTM+lag)")
    visualization.plot_residuals(y_test_l2_seq, lstm_lag_preds, save_name="residuals_LSTM_lag.png")

    print("  -> error distribution (RF)")
    visualization.plot_error_distribution(
        y_test_eval, rf_preds[seq_len:], save_name="error_distribution_RF.png"
    )

    print("  -> error distribution (LSTM)")
    visualization.plot_error_distribution(
        y_test_lstm_seq, lstm_preds, save_name="error_distribution_LSTM.png"
    )

    print("  -> error distribution (LSTM+lag)")
    visualization.plot_error_distribution(
        y_test_l2_seq, lstm_lag_preds, save_name="error_distribution_LSTM_lag.png"
    )

    print("  -> congestion timeline")
    visualization.plot_congestion_timeline(df, save_name="congestion_timeline.png")

    print("=" * 60)
    print("Done. Figures in results/figures/, models in results/models/")
    print("=" * 60)


if __name__ == "__main__":
    run()
