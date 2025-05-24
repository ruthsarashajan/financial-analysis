import pandas as pd
import numpy as np
import plotly.express as px
import os
import boto3

# ---- CONFIG ----
DATA_PATH = "data/processed/cleaned_dataset_final.csv"
EDA_OUTPUT_DIR = "eda_fraud_anomaly_trends"
PROCESSED_SAVE_PATH = "data/processed/dataset_with_fraud_anomaly_flags.csv"
S3_BUCKET = "financial-analysis-project"
S3_PREFIX = "eda_fraud_anomaly_trends"

os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

def upload_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")

# Load Data
df = pd.read_csv(DATA_PATH)
print("Data loaded. Shape:", df.shape)

# --- FRAUD LABEL CHECK & REMAP ---
fraud_col = None
for col in df.columns:
    if col.lower() in ['is_fraud', 'fraud', 'fraud_flag']:
        fraud_col = col
        break

if fraud_col is None:
    print("No fraud column found! Fraud visualizations will be skipped.")
else:
    # Map if not 0/1
    unique_vals = df[fraud_col].dropna().unique()
    if not set(unique_vals).issubset({0, 1}):
        df[fraud_col] = df[fraud_col].map({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0, '1': 1, '0': 0}).fillna(df[fraud_col])
        df[fraud_col] = df[fraud_col].astype(int)
    print("Fraud value counts:\n", df[fraud_col].value_counts())

    # ---- FRAUD VISUALIZATIONS ----

    # a) Fraud Rate by Merchant State
    if 'merchant_state' in df.columns:
        rates = df.groupby('merchant_state')[fraud_col].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=rates.index, y=rates.values*100,
                    title="Fraud Rate (%) by Merchant State (Top 10)",
                    labels={'x':'Merchant State', 'y':'Fraud Rate (%)'})
        fig.show()
        out_path = f"{EDA_OUTPUT_DIR}/fraud_rate_by_merchant_state.html"
        fig.write_html(out_path)
        upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/fraud_rate_by_merchant_state.html")

    # b) Fraud Distribution by Transaction Amount
    if 'amount' in df.columns:
        fig = px.box(df, x=fraud_col, y='amount', points='all',
                    title="Transaction Amount Distribution by Fraud Status",
                    labels={fraud_col:'Is Fraud', 'amount':'Transaction Amount'})
        fig.show()
        out_path = f"{EDA_OUTPUT_DIR}/boxplot_amount_by_fraud.html"
        fig.write_html(out_path)
        upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/boxplot_amount_by_fraud.html")

    # c) Fraud Over Time
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_only'] = df['date'].dt.date
        frauds_over_time = df.groupby('date_only')[fraud_col].sum().reset_index()
        fig = px.line(frauds_over_time, x='date_only', y=fraud_col,
                    title="Number of Fraudulent Transactions Over Time",
                    labels={'date_only':'Date', fraud_col:'Fraud Transactions'})
        fig.show()
        out_path = f"{EDA_OUTPUT_DIR}/fraud_over_time.html"
        fig.write_html(out_path)
        upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/fraud_over_time.html")

# --- SPENDING TRENDS ---

if 'date' in df.columns and 'amount' in df.columns:
    if 'date_only' not in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date_only'] = df['date'].dt.date
    spend_trend = df.groupby('date_only')['amount'].sum().reset_index()
    fig = px.line(spend_trend, x='date_only', y='amount',
                title="Total Spending Trend Over Time",
                labels={'date_only': 'Date', 'amount': 'Total Amount'})
    fig.show()
    out_path = f"{EDA_OUTPUT_DIR}/spending_trend_over_time.html"
    fig.write_html(out_path)
    upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/spending_trend_over_time.html")

if 'merchant_state' in df.columns and 'amount' in df.columns:
    state_spend = df.groupby('merchant_state')['amount'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=state_spend.index, y=state_spend.values,
                title="Top 10 Merchant States by Total Spending",
                labels={'x': 'Merchant State', 'y': 'Total Amount'})
    fig.show()
    out_path = f"{EDA_OUTPUT_DIR}/top_merchant_states_spending.html"
    fig.write_html(out_path)
    upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/top_merchant_states_spending.html")

# --- ANOMALY DETECTION (IQR) ---
if 'amount' in df.columns:
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['amount_anomaly'] = ((df['amount'] < lower) | (df['amount'] > upper)).astype(int)
    print(f"Detected {df['amount_anomaly'].sum()} anomalies in 'amount' using IQR.")

    # Boxplot with anomalies highlighted
    fig = px.box(
        df, y='amount', points="all",
        color=df['amount_anomaly'].map({0: "Normal", 1: "Anomaly"}),
        title="Boxplot of Transaction Amount (Anomalies Highlighted)"
    )
    fig.show()
    out_path = f"{EDA_OUTPUT_DIR}/boxplot_amount_anomaly.html"
    fig.write_html(out_path)
    upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/boxplot_amount_anomaly.html")

    # Distribution with anomalies highlighted
    fig = px.histogram(
        df, x='amount', color=df['amount_anomaly'].map({0: "Normal", 1: "Anomaly"}),
        nbins=50, barmode='overlay',
        title="Transaction Amount Distribution (Anomalies Highlighted)"
    )
    fig.show()
    out_path = f"{EDA_OUTPUT_DIR}/hist_amount_anomaly.html"
    fig.write_html(out_path)
    upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/hist_amount_anomaly.html")

    # Anomaly trend over time
    if 'date_only' in df.columns:
        anomaly_time = df.groupby('date_only')['amount_anomaly'].sum().reset_index()
        fig = px.line(anomaly_time, x='date_only', y='amount_anomaly',
                    title="Number of Amount Anomalies Over Time",
                    labels={'date_only': 'Date', 'amount_anomaly': 'Number of Amount Anomalies'})
        fig.show()
        out_path = f"{EDA_OUTPUT_DIR}/amount_anomaly_over_time.html"
        fig.write_html(out_path)
        upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/amount_anomaly_over_time.html")

# --- SAVE DATASET TO data/processed ---
df.to_csv(PROCESSED_SAVE_PATH, index=False)
print(f"Dataset with fraud and anomaly flags saved to: {PROCESSED_SAVE_PATH}")
upload_to_s3(PROCESSED_SAVE_PATH, S3_BUCKET, f"{S3_PREFIX}/dataset_with_fraud_anomaly_flags.csv")

print(f"\nEDA complete! Plots shown, saved locally in '{EDA_OUTPUT_DIR}', and uploaded to s3://{S3_BUCKET}/{S3_PREFIX}/")