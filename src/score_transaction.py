import os
import joblib
import pandas as pd
import numpy as np
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_fraud_model.joblib")
SCORE_LOG_PATH = os.path.join(PROJECT_ROOT, "monitoring", "score_log.csv")

_loaded = joblib.load(MODEL_PATH)

if isinstance(_loaded, dict):
    _package = _loaded
else:
    # Compatibility with a bare Pipeline saved by cell 8
    _pipeline = _loaded
    _preprocessor = _pipeline.named_steps["preprocessor"]

    _feature_columns = []
    for _, _, columns in _preprocessor.transformers_:
        if columns is not None:
            _feature_columns.extend(list(columns))

    _package = {
        "model": _pipeline,
        "threshold": 0.05,
        "feature_columns": _feature_columns,
    }

if not {"model", "threshold", "feature_columns"}.issubset(_package):
    raise ValueError("The saved model package is missing required keys.")
_model = _package["model"]
_threshold = _package["threshold"]
_feature_columns = _package["feature_columns"]


def _engineer_features(raw: dict) -> pd.DataFrame:
    """Reproduce the exact feature engineering used in training, on one row."""
    row = dict(raw)  # shallow copy so we don't mutate the caller's dict

    ts = pd.to_datetime(row["timestamp"])
    row["hour"] = ts.hour
    row["day_of_week"] = ts.dayofweek
    row["is_weekend"] = int(ts.dayofweek >= 5)
    row["is_night"] = int(0 <= ts.hour <= 5)

    row["log_amount"] = np.log1p(row["amount"])
    row["amount_to_avg_ratio"] = row["amount"] / (row["avg_user_amount"] + 1e-6)
    row["log_transaction_frequency"] = np.log1p(row["transaction_frequency"])
    row["log_transaction_gap"] = np.log1p(row["transaction_gap_seconds"])

    flag_cols = [
        "unusual_amount_flag", "velocity_flag", "new_device_flag",
        "location_change_flag", "night_transaction_flag",
    ]
    row["flag_sum"] = sum(row[c] for c in flag_cols)

    # drop raw timestamp -- not a model input, only used to derive time features
    row.pop("timestamp", None)

    df_row = pd.DataFrame([row])

    # guarantee exact column set/order the trained pipeline expects
    missing = set(_feature_columns) - set(df_row.columns)
    if missing:
        raise ValueError(f"Missing required fields for scoring: {missing}")
    return df_row[_feature_columns]


def score_transaction(raw_transaction: dict) -> dict:
    """
    Score one transaction in near-real-time.
    Returns fraud probability, the flag decision, the threshold used,
    and scoring latency in milliseconds (useful evidence for your write-up
    that this meets a 'near-real-time' bar).
    """
    start = time.perf_counter()

    X_row = _engineer_features(raw_transaction)
    proba = _model.predict_proba(X_row)[0, 1]
    is_flagged = bool(proba >= _threshold)

    elapsed_ms = (time.perf_counter() - start) * 1000

    result = {
        "fraud_probability": round(float(proba), 4),
        "is_flagged": is_flagged,
        "threshold_used": _threshold,
        "scoring_latency_ms": round(elapsed_ms, 2),
    }

    log_score(raw_transaction, result)
    return result

def log_score(raw_transaction: dict, result: dict) -> None:
    """Append this scoring event to a CSV log for later monitoring/drift analysis."""
    os.makedirs(os.path.dirname(SCORE_LOG_PATH), exist_ok=True)

    log_row = {
        "scored_at": pd.Timestamp.now().isoformat(),
        "amount": raw_transaction.get("amount"),
        "merchant_category": raw_transaction.get("merchant_category"),
        "location": raw_transaction.get("location"),
        "fraud_probability": result["fraud_probability"],
        "is_flagged": result["is_flagged"],
        "scoring_latency_ms": result["scoring_latency_ms"],
    }
    log_df = pd.DataFrame([log_row])
    header_needed = not os.path.exists(SCORE_LOG_PATH)
    log_df.to_csv(SCORE_LOG_PATH, mode="a", header=header_needed, index=False)

if __name__ == "__main__":
    # quick smoke test
    sample = {
        "amount": 2223.52,
        "transaction_type": "transfer",
        "merchant_category": "electronics",
        "timestamp": "2026-09-03 14:22:10",
        "transaction_frequency": 8,
        "avg_user_amount": 2071.73,
        "deviation_from_avg": 151.79,
        "transaction_gap_seconds": 2813,
        "account_age_days": 794,
        "failed_attempts": 1,
        "device_type": "mobile",
        "location": "India",
        "is_foreign_transaction": 0,
        "unusual_amount_flag": 0,
        "velocity_flag": 0,
        "new_device_flag": 0,
        "location_change_flag": 0,
        "night_transaction_flag": 0,
    }
    print(score_transaction(sample))
