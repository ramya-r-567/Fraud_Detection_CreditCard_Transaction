import os
import sys
import time
import random
from datetime import datetime

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from score_transaction import score_transaction  # noqa: E402

st.set_page_config(page_title="Fraud Monitoring", page_icon="\U0001F6E1", layout="wide")

# ---------------------------------------------------------------------------
# Load a pool of realistic historical transactions to simulate a live feed
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "fraud_detection_clean.csv")

@st.cache_data
def load_sample_pool():
    df = pd.read_csv(DATA_PATH)
    return df

if "feed" not in st.session_state:
    st.session_state.feed = []  # list of dicts: scored transactions so far

st.title("\U0001F6E1 Fraud Monitoring Dashboard")
st.caption("Watches incoming transactions and flags risky ones in near-real-time.")

tab_live, tab_manual = st.tabs(["\U0001F534 Live Monitor", "\U0001F50D Check One Transaction"])

# ===========================================================================
# TAB 1: LIVE MONITOR
# ===========================================================================
with tab_live:
    pool = load_sample_pool()

    control_col, stat_col = st.columns([1, 3])
    with control_col:
        batch_size = st.selectbox("Transactions to process", [5, 10, 25, 50], index=1)
        run_clicked = st.button("\u25B6 Process Next Batch", type="primary", use_container_width=True)
        clear_clicked = st.button("Clear Feed", use_container_width=True)

    if clear_clicked:
        st.session_state.feed = []

    if run_clicked:
        sample = pool.sample(batch_size).to_dict(orient="records")
        for row in sample:
            raw_transaction = {
                "amount": row["amount"],
                "transaction_type": row["transaction_type"],
                "merchant_category": row["merchant_category"],
                "timestamp": datetime.now().isoformat(),
                "transaction_frequency": row["transaction_frequency"],
                "avg_user_amount": row["avg_user_amount"],
                "deviation_from_avg": row["deviation_from_avg"],
                "transaction_gap_seconds": row["transaction_gap_seconds"],
                "account_age_days": row["account_age_days"],
                "failed_attempts": row["failed_attempts"],
                "device_type": row["device_type"],
                "location": row["location"],
                "is_foreign_transaction": row["is_foreign_transaction"],
                "unusual_amount_flag": row["unusual_amount_flag"],
                "velocity_flag": row["velocity_flag"],
                "new_device_flag": row["new_device_flag"],
                "location_change_flag": row["location_change_flag"],
                "night_transaction_flag": row["night_transaction_flag"],
            }
            result = score_transaction(raw_transaction)
            st.session_state.feed.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Amount": f"${row['amount']:,.2f}",
                "Category": row["merchant_category"].title(),
                "Location": row["location"],
                "Risk": "\U0001F6A8 Flagged" if result["is_flagged"] else "\u2705 Normal",
                "Latency (ms)": result["scoring_latency_ms"],
                "_amount_raw": row["amount"],
                "_flagged": result["is_flagged"],
                "_actual_fraud": bool(row["is_fraud"]),  # historical label, for demo insight only
            })

    feed = st.session_state.feed

    total = len(feed)
    flagged = sum(1 for r in feed if r["_flagged"])
    amount_under_review = sum(r["_amount_raw"] for r in feed if r["_flagged"])
    avg_latency = (sum(r["Latency (ms)"] for r in feed) / total) if total else 0

    with stat_col:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transactions Scanned", total)
        m2.metric("Flagged for Review", flagged)
        m3.metric("Amount Under Review", f"${amount_under_review:,.0f}")
        m4.metric("Avg. Scoring Speed", f"{avg_latency:.1f} ms")

    st.divider()

    if total == 0:
        st.info("Click **Process Next Batch** to start the live feed.")
    else:
        display_df = pd.DataFrame(feed).drop(columns=["_amount_raw", "_flagged", "_actual_fraud"])

        def highlight_flagged(row):
            color = "background-color: #ffe6e6" if "Flagged" in row["Risk"] else ""
            return [color] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_flagged, axis=1),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        with st.expander("How well did the model do on this batch? (demo insight, uses historical outcomes)"):
            actual_fraud = sum(1 for r in feed if r["_actual_fraud"])
            caught = sum(1 for r in feed if r["_actual_fraud"] and r["_flagged"])
            st.write(
                f"Of **{actual_fraud}** transactions in this batch that were historically "
                f"confirmed fraud, the model flagged **{caught}** of them for review."
            )
            st.caption(
                "This comparison is only possible here because we're replaying historical, "
                "already-labeled data for demonstration. In production, true outcomes aren't "
                "known until after a chargeback or confirmation \u2014 which is why the monitoring "
                "notebook tracks this on a delay, not instantly."
            )

# ===========================================================================
# TAB 2: MANUAL CHECK
# ===========================================================================
with tab_manual:
    st.write("Check a specific transaction that isn't in the automatic feed.")

    with st.form("manual_form"):
        amount = st.number_input("How much was the transaction?", min_value=0.0, value=1500.0, step=10.0)
        usual_amount = st.number_input(
            "What does this customer usually spend?", min_value=0.0, value=1200.0, step=10.0
        )
        purchase_type = st.selectbox("What kind of purchase?", ["Payment", "Transfer", "Withdrawal"])
        category = st.selectbox("Category", ["Electronics", "Fashion", "Gaming", "Grocery", "Travel"])
        country = st.selectbox("Country", ["India", "USA", "UK", "UAE", "Germany"])
        device = st.radio("Device used", ["Phone", "Computer"], horizontal=True)

        st.write("A few quick yes/no questions:")
        c1, c2 = st.columns(2)
        with c1:
            is_new_device = st.checkbox("First time using this device?")
            is_new_location = st.checkbox("Different location than usual?")
            is_international = st.checkbox("International transaction?")
        with c2:
            is_late_night = st.checkbox("Happening late at night?")
            is_rapid = st.checkbox("Several transactions in quick succession?")
            had_failed_attempts = st.checkbox("Any recent failed attempts on this account?")

        submitted = st.form_submit_button("Check This Transaction", type="primary", use_container_width=True)

    if submitted:
        is_unusual_amount = amount > usual_amount * 1.5

        raw_transaction = {
            "amount": amount,
            "transaction_type": purchase_type.lower(),
            "merchant_category": category.lower(),
            "timestamp": datetime.now().isoformat(),
            "transaction_frequency": 6 if is_rapid else 2,
            "avg_user_amount": usual_amount,
            "deviation_from_avg": round(amount - usual_amount, 2),
            "transaction_gap_seconds": 120 if is_rapid else 3600,
            "account_age_days": 500,
            "failed_attempts": 2 if had_failed_attempts else 0,
            "device_type": "mobile" if device == "Phone" else "web",
            "location": country,
            "is_foreign_transaction": int(is_international),
            "unusual_amount_flag": int(is_unusual_amount),
            "velocity_flag": int(is_rapid),
            "new_device_flag": int(is_new_device),
            "location_change_flag": int(is_new_location),
            "night_transaction_flag": int(is_late_night),
        }

        result = score_transaction(raw_transaction)

        st.divider()
        if result["is_flagged"]:
            st.error("### \u26A0\uFE0F This transaction looks risky")
            st.write("We'd recommend a closer look before approving this one.")
        else:
            st.success("### \u2705 This transaction looks normal")
            st.write("Nothing unusual detected.")
        st.caption(f"Checked in {result['scoring_latency_ms']} milliseconds")