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
    X_train_lstm = X_train[config.LSTM_FEATURE_COLUMNS]
    X_test_lstm = X_test[config.LSTM_FEATURE_COLUMNS]

    # Timestamps for session-boundary detection (cross-session = garbage sequences)
    train_ts = df.loc[X_train.index, config.TIMESTAMP_COLUMN]
    test_ts = df.loc[X_test.index, config.TIMESTAMP_COLUMN]

    lstm_model, lstm_scaler, lstm_y_scaler, lstm_losses, lstm_val_losses, y_test_lstm = model.train_lstm(
        X_train_lstm, y_train, X_test_lstm, y_test,
        train_ts=train_ts, test_ts=test_ts,
    )
    visualization.plot_lstm_loss(lstm_losses, lstm_val_losses, save_name="lstm_loss.png")

    model.save_model(lr, "linear_regression")
    model.save_model(rf, "random_forest")
    model.save_model(lstm_model, "lstm")

    # ── 6. Evaluation ────────────────────────────────────────────
    print("[6/7] Evaluation...")

    lr_preds = lr.predict(X_test)
    rf_preds = rf.predict(X_test)
    lstm_preds = model.predict_lstm(lstm_model, lstm_scaler, X_test_lstm,
                                     y_scaler=lstm_y_scaler, timestamps=test_ts)

    # Evaluate on comparable test windows (LSTM discards seq_len rows + session breaks)
    seq_len = config.LSTM_SEQUENCE_LENGTH
    y_test_eval = y_test[seq_len:]
    lr_metrics = evaluation.evaluate(y_test_eval, lr_preds[seq_len:])
    rf_metrics = evaluation.evaluate(y_test_eval, rf_preds[seq_len:])
    # y_test_lstm and lstm_preds are both session-boundary-filtered → aligned
    lstm_metrics = evaluation.evaluate(y_test_lstm, lstm_preds)

    evaluation.print_metrics("LinearRegression", lr_metrics)
    evaluation.print_metrics("RandomForest", rf_metrics)
    evaluation.print_metrics("LSTM", lstm_metrics)

    # Prediction plots
    visualization.plot_prediction(y_test_eval, lr_preds[seq_len:], save_name="prediction_LR.png")
    visualization.plot_prediction(y_test_eval, rf_preds[seq_len:], save_name="prediction_RF.png")
    visualization.plot_prediction(y_test_lstm, lstm_preds, save_name="prediction_LSTM.png")

    # ── 7. Result analysis ───────────────────────────────────────
    print("[7/7] Result analysis...")

    print("  -> 3-model comparison")
    visualization.plot_model_comparison_3(
        lr_metrics, rf_metrics, lstm_metrics, save_name="model_comparison.png"
    )

    print("  -> actual vs predicted (RF)")
    visualization.plot_actual_vs_predicted(
        y_test, rf_preds, save_name="actual_vs_predicted.png"
    )

    print("  -> residuals (RF)")
    visualization.plot_residuals(y_test, rf_preds, save_name="residuals.png")

    print("  -> error distribution (RF)")
    visualization.plot_error_distribution(
        y_test, rf_preds, save_name="error_distribution.png"
    )

    print("  -> congestion timeline")
    visualization.plot_congestion_timeline(df, save_name="congestion_timeline.png")

    print("=" * 60)
    print("Done. Figures in results/figures/, models in results/models/")
    print("=" * 60)


if __name__ == "__main__":
    run()
