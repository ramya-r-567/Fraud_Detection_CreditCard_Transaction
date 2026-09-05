# 🛡️ Fraud Detection in Credit Card Transactions

A machine learning system that detects fraudulent credit card transactions in near-real-time, built as part of a fintech-focused ML project (Project 4: Fraud Detection in Credit Card Transactions).

## Problem Statement

A financial institution faces challenges with increasing fraudulent credit card transactions. This project implements a machine learning–based fraud detection system to identify and prevent fraudulent activity and safeguard financial assets — including a near-real-time scoring pipeline and a continuous monitoring approach for production use.

## What This Project Does

- Analyzes historical credit card transaction data and engineers behavioral features (transaction velocity, deviation from average spend, device/location changes, night-time activity, etc.)
- Handles class imbalance and trains multiple classifiers (Logistic Regression, Random Forest, Gradient Boosting)
- Selects a final model based on Precision-Recall AUC and threshold tuning rather than accuracy alone
- Scores transactions in near-real-time (~11ms average latency) with automatic logging for monitoring
- Tracks data/model drift over time using Population Stability Index (PSI)
- Ships an interactive Streamlit dashboard for live monitoring and manual transaction checks

## Repository Structure

```
├── data/          # Raw and cleaned/feature-engineered transaction data
├── models/        # Saved trained model artifacts
├── monitoring/     # Drift detection notebook + transaction score logs
├── notebooks/      # Model training, evaluation, and monitoring notebooks
├── src/            # Core scoring logic (score_transaction.py)
├── app.py          # Streamlit dashboard (live monitor + manual check)
└── requirements.txt
```

## Key Findings

- **Data leakage caught early:** the raw dataset contained ~4,800 exact duplicate rows that leaked across train/test splits and artificially inflated precision to near-perfect. After deduplication, honest performance settled at a more realistic ROC-AUC of 0.65–0.69 and PR-AUC of 0.12–0.17 — a good example of why cross-validation and leakage checks matter more than raw accuracy.
- **`flag_sum`** (a composite of five behavioral risk flags) was the strongest predictor — fraud rate rises from 3.2% at zero flags to 36.6% at three flags.
- **Threshold tuning mattered as much as model choice.** The final model uses a low decision threshold (0.05) to prioritize recall, catching ~70% of fraud at the cost of a higher false-alarm rate — an explicit, documented trade-off rather than an optimized-away one.

## Model

**Final model:** Gradient Boosting Classifier
**Decision threshold:** 0.05 (tuned for recall)
**Approach:** Evaluated via Precision-Recall AUC and F-beta scoring (not just accuracy), given the highly imbalanced nature of fraud data.

## Near-Real-Time Scoring

`src/score_transaction.py` scores a single transaction end-to-end and logs it automatically to `monitoring/score_log.csv`, simulating a production-style scoring service. Average scoring latency is under 15ms per transaction.

## Continuous Monitoring

`notebooks/03_monitoring.ipynb` implements drift detection using PSI (Population Stability Index) on incoming transaction features, with a minimum 200-sample safeguard (PSI is unreliable on smaller samples).

## Dashboard

Run the dashboard locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

- **Live Monitor tab:** simulates a live transaction feed (sampled from historical data) and flags risky transactions in real time.
- **Check One Transaction tab:** lets you manually enter details of a single transaction in plain language and get an instant risk assessment.

> Note: the live feed replays historical, already-labeled transactions to simulate real-time traffic for demonstration purposes. In a true production deployment, transaction outcomes (confirmed fraud/chargeback) are not known instantly — which is why drift monitoring is handled on a delayed basis rather than in real time.

## Tech Stack

Python · pandas · NumPy · scikit-learn · joblib · matplotlib · Streamlit

## Limitations & Future Work

- PR-AUC (0.12–0.17) reflects the genuine difficulty of this dataset post-leakage-fix; further gains would likely come from more transaction history, SMOTE/class-weight tuning, or gradient-boosted libraries like XGBoost/LightGBM/CatBoost.
- Precision is intentionally traded off for recall given the cost asymmetry of missed fraud vs. false alarms; a cost-based threshold could be tuned further for a specific business context.
