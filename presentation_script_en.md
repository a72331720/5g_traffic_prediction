# 5G Network Traffic Prediction — 15-Minute Presentation Script

> **Total duration**: 15 minutes  
> **Format**: Each slide shows recommended time. Text in [brackets] is the script — read directly or adapt to your own style.

---

## Slide 1 — Title（0:00 – 0:30）｜30 sec

[Good morning / afternoon everyone. Today I'll be presenting our course project: machine-learning-based prediction of 5G network traffic.

We worked with 188,711（one hundred [and] eighty-eight thousand, seven hundred [and] eleven） real-world drive-test measurements from a 5G dataset published at ACM MMSys（ACM Multimedia Systems Conference） 2020, and compared three modelling approaches: Linear Regression, Random Forest, and LSTM.

Over the next 15 minutes I'll walk through our full pipeline and share a few findings that surprised us along the way.]

---

## Slide 2 — Agenda（0:30 – 0:50）｜20 sec

[Here's a quick overview of what we'll cover. We start with background and motivation, then the dataset and preprocessing, followed by our three-model methodology, experimental results, 5G-specific traffic analysis, and finally conclusions and future directions.]

---

## Slide 3 — Background & Motivation（0:50 – 1:50）｜60 sec

[Why does traffic prediction matter?

Today's 5G networks are reactive — congestion is handled after it happens, which is already too late. If we can predict traffic ahead of time, operators can pre-allocate bandwidth and optimise Quality of Service proactively.

Our core research question is: is 5G throughput prediction fundamentally feature-driven or sequence-driven? In other words, is engineered tabular features enough for tree-based models to win, or do we need recurrent architectures to model temporal dependencies?

Our strategy: build both, and let the data decide.]

---

## Slide 4 — Dataset Overview（1:50 – 2:40）｜50 sec

[The dataset comes from the uccmisl/5Gdataset（UC misl） repository — the first publicly available 5G dataset combining throughput with channel and context metrics, collected via drive tests across an Irish mobile network.

The data spans November 2019 to February 2020, with 188,711 records at one-second granularity.

Our prediction target is DL_bitrate — downlink throughput — with a mean of roughly 10,758 kbps and a maximum of 532,905 kbps. That 50× range tells you immediately that this is a highly non-stationary dataset: network conditions vary dramatically across scenarios.]

---

## Slide 5 — Raw Features（2:40 – 3:20）｜40 sec

[The raw data has 14 columns in two groups.

Nine numeric features including signal quality indicators — RSRP, RSRQ, SNR, CQI, RSSI — plus speed, longitude, and latitude.

Five categorical features: network mode (5G / LTE / HSPA+), operator name, application type, mobility, and state.

One thing worth flagging here: conventional wisdom says stronger signal means faster speed. We'll see shortly that the correlation between signal strength and throughput is almost exactly zero — a counterintuitive finding we'll explain in the analysis section.]

---

## Slide 6 — Preprocessing Pipeline（3:20 – 4:20）｜60 sec

[Preprocessing is where most of the engineering effort went. Let me highlight the key decisions.

First, cleaning: drop duplicates, sort by timestamp, forward-fill missing values.

Second, feature engineering. We extract time features — hour, weekday, is_weekend — from the timestamp. We add lag features: lag_1, lag_2, lag_3, the throughput values from the previous one, two, and three seconds. And rolling features: windowed mean and standard deviation over three and six steps.

Third, and this is a critical design decision: session boundary isolation. The dataset contains multiple separate drive-test sessions separated by time gaps. Lag and rolling values computed naively would bleed information across sessions — previous session's data contaminating the next session's features. We detect boundaries by checking for timestamp gaps greater than ten seconds, and reset lag and rolling calculations at each boundary. LSTM sequence creation applies the same filter, discarding any sequence that spans a boundary.

Finally, train-test split: chronological 80/20, shuffle=False to prevent future data leakage.]

---

## Slide 7 — Feature Engineering Deep-Dive（4:20 – 5:10）｜50 sec

[Let's look at feature importance from our trained Random Forest.

lag_1 importance: 0.77 — nearly 80% of the model's predictive power comes from a single feature: the throughput value one second ago.

UL_bitrate ranks second at roughly 0.15, which makes sense — uplink and downlink tend to move together.

rolling_std_6 is third. Everything else combined is below 0.05.

Signal quality features — RSRP, SNR, CQI — are essentially zero.

This single chart is the key to understanding everything that follows. The dominant signal in this dataset is short-range autocorrelation. Whatever happened last second is the best predictor of what will happen next second. This sets up the central question: can LSTM — which has to infer this from raw noisy sequences — compete with a tree model that receives the lag value directly as a feature?]

---

## Slide 8 — Models Overview（5:10 – 6:00）｜50 sec

[We trained four models.

Linear Regression as the baseline — all engineered features, tests whether linear relationships are strong enough.

Random Forest as the primary model — tuned with GridSearchCV, cv=3, searching over n_estimators, max_depth, and min_samples_split.

Basic LSTM — uses one-hot encoded categoricals plus numeric features, 60-step sequence window, no lag or rolling features. This tests pure sequence modelling capability.

LSTM + lag — adds the same lag and rolling features as RF. This is a controlled experiment: if we give LSTM equivalent feature information, can it catch up with Random Forest?

The answer to that question is the core finding of this project.]

---

## Slide 9 — LSTM Hard-Won Lessons（6:00 – 7:00）｜60 sec

[LSTM implementation required fixing five critical bugs. I'll go through each one briefly.

Bug 1: shuffle=True in the DataLoader. This destroyed temporal order during training. Fixed by setting shuffle=False.

Bug 2: validation split done after sequence creation. This caused sequences to span the train-val boundary, leaking future information into validation. Fixed by splitting the raw arrays first, then creating sequences separately for each split.

Bug 3: sliding window sequences crossing session boundaries. A 60-step window drawn across a drive-test gap combines data from two completely unrelated measurement sessions. Fixed by skipping any sequence where the session ID changes within the window.

Bug 4: label encoding plus StandardScaler on categorical features. This imposes false ordinal distances — the model thinks 5G and LTE are numerically adjacent. Fixed by switching LSTM inputs to one-hot encoding.

Bug 5: cross-version non-determinism. Same code, same seed 42 — R² of 0.518 on PyTorch 2.11 with CPU, and 0.389 on PyTorch 2.5.1 with GPU. Different versions initialise weights differently from the same seed. Fixed with cudnn.deterministic=True. Results are now reproducible within each version. Full details on Slide 17.]

---

## Slide 10 — Results: 4-Model Comparison（7:00 – 7:50）｜50 sec

[Here are the results.

Linear Regression: R²=0.799, RMSE=28,998. A baseline of 0.8 is already impressive — it confirms that our feature engineering is doing real work.

Random Forest: R²=0.909, RMSE=19,538. Best overall.

Basic LSTM: R²=0.389, RMSE=50,604. Clearly the worst.

LSTM + lag: R²=0.719, RMSE=34,320. Adding lag features improved R² by 0.33 — but it still falls short of Linear Regression.

Two counterintuitive results worth pausing on. First, the simplest model beats deep learning. Second, even when LSTM receives the same features as Random Forest, it still can't beat a linear model. This rules out feature access as the explanation — the difference is in how each model type processes the information.]

---

## Slide 11 — Prediction Visualization（7:50 – 8:30）｜40 sec

[These are the first 300 time steps of predicted versus actual throughput for all four models.

Random Forest tracks the actual signal most closely across the full range.

Basic LSTM follows the general trend in low-speed segments but struggles with high-speed bursts.

LSTM + lag shows something interesting: around step 150 there's an extreme spike in predicted values — well beyond anything in the actual data. We traced this back in the raw dataset: that timestep corresponds to an Amazon Prime buffering burst, where DL_bitrate jumped from near zero to 7,161 kbps in a single step, then immediately dropped again. LSTM+lag read lag_1=7,161 and amplified it into an extreme prediction.

Lag features are powerful in stable regimes, but they can create instability at abrupt transitions — exactly the kind of burst traffic common in streaming applications.]

---

## Slide 12 — Random Forest: Best Model Analysis（8:30 – 9:10）｜40 sec

[Looking at RF in more detail.

The scatter plot shows predictions closely aligned with the actual values along the y=x diagonal. The error distribution is sharply peaked around zero and roughly symmetric — no systematic over- or under-prediction across the full test set.

One nuance worth mentioning: in the first 130 time steps of the test window, the actual throughput frequently drops to zero. We checked the raw data — that segment is a static Amazon Prime test with RSRP of −97 dBm, deep in weak-coverage territory. The streaming app's buffering mechanism causes intermittent zero throughput. RF slightly over-predicts in that segment because other features like SNR and network mode don't fully reflect the signal weakness — it votes across multiple features and pulls the prediction upward. This is visible in the scatter plot as a cluster of points above the diagonal near actual=0.]

---

## Slide 13 — LSTM Performance Analysis（9:10 – 9:50）｜40 sec

[LSTM's scatter plot tells a very different story. The point cloud is widely dispersed — for actual values between 0 and 100,000 kbps, predicted values scatter across 0 to 250,000 kbps. The model has no reliable alignment with the diagonal.

The residual plot shows another important pattern: residuals are near zero in the first five thousand steps — the low-throughput static test segment — and then explode into high-variance oscillations of up to ±350,000 kbps in the high-speed driving segments.

This is heteroscedasticity: the model's error variance depends strongly on the regime. LSTM predicts acceptably in stable low-speed conditions but breaks down in high-speed bursty 5G scenarios.

Early stopping triggered between epochs 35 and 55. The model wasn't undertrained — it genuinely found its optimum. The data structure simply doesn't give LSTM enough to work with.]

---

## Slide 16 — Why RF > LSTM: The Core Argument（9:50 – 11:00）｜70 sec

[Now let's address the central question directly. We have three levels of evidence for why Random Forest wins.

First, statistical evidence. Lag-1 autocorrelation equals 0.89. This means the previous second's throughput explains 89% of the variance in the next second's throughput. Random Forest receives lag_1 as a direct feature and exploits this immediately. Basic LSTM must infer the same relationship from 60 steps of noisy RF signal measurements — and those measurements have near-zero correlation with throughput, as we saw in the feature importance chart.

Second, empirical evidence from the controlled experiment. When we gave LSTM the same lag and rolling features as RF, its R² jumped from 0.389 to 0.719 — a gain of 0.33. But it still couldn't beat Linear Regression at 0.799. Why? Because the relationship between lag_1 and the target is nearly linear. LSTM's non-linear capacity is not just unhelpful here — it's actively adding complexity where none is needed.

Third, domain reasoning. 5G drive-test data is inherently non-stationary. A 60-step window can contain measurements from a highway, an urban junction, and a weak-coverage zone, with different applications running in each. LSTM must learn across this mixed distribution simultaneously. Random Forest operates feature-by-feature with lag_1 doing the heavy lifting, which is exactly the right tool for this structure.]

---

## Slide 17 — LSTM Reproducibility（11:00 – 11:40）｜40 sec

[One additional finding worth highlighting separately: LSTM's cross-version instability.

Same codebase. Same random seed 42. Same training logic.

PyTorch 2.11 on CPU produces R²=0.518.
PyTorch 2.5.1 on GPU produces R²=0.389.

A difference of 0.13 from a version upgrade.

The cause: different PyTorch versions initialise weights differently from the same seed. Basic LSTM has weak feature signal, so it is highly sensitive to initialisation — small differences in starting weights lead to convergence at very different local optima.

After adding cudnn.deterministic=True and cudnn.benchmark=False, results stabilised within each version. The cross-version gap remains, but it is now attributable to a documented cause rather than random noise.

Random Forest, by contrast, produced R²=0.907 to 0.909 across both environments — essentially unchanged.

For production deployment, model reliability matters as much as peak accuracy. On both counts, Random Forest wins.]

---

## Slide 14 / 15 — Traffic Pattern & 5G-Specific Analysis（11:40 – 12:40）｜60 sec

[Let's look at what the traffic analysis revealed about 5G network behaviour.

The signal correlation heatmap shows that RSRP correlates with DL_bitrate at −0.04, SNR at +0.04, CQI at −0.01. All essentially zero, with some slightly negative.

This is counterintuitive. The explanation is that the dataset mixes 5G, LTE, and HSPA+ measurements. A 5G connection at the cell edge may have weak signal but fast throughput because fewer users are competing. An LTE connection with strong signal may deliver lower throughput. Network mode variation masks the expected signal-speed relationship — which also explains why RSRP and SNR appear near zero in the RF feature importance chart.

Application type has a far larger effect. File download averages roughly 41,000 kbps. Amazon Prime averages around 2,000 kbps. Netflix around 800 kbps. A 50× gap between application types — driven entirely by application-layer traffic patterns, not network capability.

This means measured throughput is as much a product of what the application is doing as of what the network can deliver.]

---

## Slide 18 — Known Limitations（12:40 – 13:10）｜30 sec

[We documented our known limitations explicitly and addressed each one.

Session boundary leakage in lag and rolling features: resolved by adding a timestamps parameter that resets calculations at session gaps greater than ten seconds. Post-fix, LSTM+lag R² dropped from 0.757 to 0.719 — confirming that the previous figure was slightly inflated by cross-session leakage.

Evaluation set size mismatch: LR and RF are evaluated on approximately 37,533 samples; LSTM variants on around 36,783, because session-boundary filtering discards sequences that span gaps. The difference is small but noted explicitly in our code comments and README for transparency.]

---

## Slide 19 — Future Work（13:10 – 13:40）｜30 sec

[Several directions are worth exploring from here.

Real-time deployment: converting the offline pipeline to streaming inference for live network monitoring.

Multi-step forecasting: extending from one-step-ahead to N-step predictions.

Transformer-based models: Informer or PatchTST, which handle long-range dependencies more efficiently than LSTM.

External features: weather, calendar events, and urban density — factors that influence traffic but are absent from the current dataset.

Uncertainty quantification: replacing point predictions with confidence intervals, which is more actionable for operator capacity planning.]

---

## Slide 20 — Conclusion（13:40 – 14:30）｜50 sec

[Three key takeaways.

First: Random Forest is the best model at R²=0.909. The dataset is fundamentally feature-driven. Lag-1 autocorrelation of 0.89 is the dominant signal, and tree-based models exploit it directly through engineered features.

Second: this dataset is feature-driven, not sequence-driven. The LSTM control experiment makes this concrete — even with equivalent lag features, LSTM+lag at R²=0.719 cannot beat Linear Regression at 0.799. Data characteristics determine the right model class. Deep learning is not always the answer.

Third: engineering discipline matters as much as model choice. We identified and fixed five LSTM implementation bugs, discovered and documented cross-version PyTorch instability, and proactively disclosed two data integrity limitations. These findings are part of the contribution, not just noise around the main results.]

---

## Slide 21 — References（14:30 – 14:50）｜20 sec

[Before I close, I want to quickly acknowledge the key references that informed our work.

The dataset itself comes from Raca et al., published at ACM MMSys 2020 — it's the first publicly available 5G dataset combining throughput with channel and context metrics. Our model evaluation methods and performance findings are validated against prior published work, including Minovski et al. on ML-based throughput prediction in LTE and 5G published in IEEE Transactions on Mobile Computing.

These five references, along with Breiman's foundational Random Forests paper, provide the academic grounding for our methodology and conclusions. All DOIs are listed on the slide and in our GitHub repository.]

---

## Slide 22 — Q&A（14:50 – 15:00）｜10 sec

[That covers our full presentation. The slide also lists prepared answers for the most likely questions.

Thank you — happy to take any questions.]

---

## Appendix: Prepared Q&A

**Q: Why not XGBoost instead of Random Forest?**
> RF already achieves R²=0.909. XGBoost typically offers a further 1–3% gain with more tuning effort. It's a valid next step but doesn't change the core feature-driven conclusion. We chose RF for interpretability via feature importance.

**Q: Why sequence length 60?**
> 60 steps corresponds to one minute of history at one-second granularity. We tested shorter windows and saw no significant improvement with longer ones, which is consistent with the finding that useful signal is concentrated in lag_1 — the most recent step.

**Q: Why not apply log transformation to the target?**
> DL_bitrate is right-skewed. A log transform before StandardScaler would likely help LSTM converge more smoothly. We used StandardScaler on the raw values. This is a documented improvement opportunity — but it would not change the conclusion that RF outperforms LSTM on this dataset.

**Q: Why is validation loss lower than training loss throughout?**
> The validation set is the final 10% of the training split by time. That segment is 99.4% 5G data, while the training set contains substantial LTE, HSPA+, and other network modes. The validation set's standard deviation is roughly 2.6× smaller than the training set's — it is simply an easier prediction target, not evidence of data leakage.

**Q: How do you know the session boundary fix is correct?**
> The fix caused LSTM+lag R² to drop from 0.757 to 0.719. A correctly applied fix that removes artificial signal should cause performance to decrease slightly — and it did. RF gained slightly (0.907 → 0.909) from cleaner features. Both movements are in the expected direction.
