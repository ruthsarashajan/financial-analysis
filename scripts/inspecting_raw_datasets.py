import pandas as pd
import numpy as np
import plotly.express as px
import os
import boto3

# ---- CONFIG ----
DATA_PATH = "data/processed/cleaned_dataset_final.csv"
EDA_OUTPUT_DIR = "eda_fraud_anomaly_trends"
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

# --- 1. FRAUD VISUALIZATION ---
if 'is_fraud' in df.columns:
    fig = px.histogram(df, x='is_fraud', color='is_fraud',
                       title="Fraud vs Non-Fraud Transactions",
                       labels={'is_fraud': 'Is Fraud'}, text_auto=True)
    fig.show()
    out_path = f"{EDA_OUTPUT_DIR}/fraud_distribution.html"
    fig.write_html(out_path)
    upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/fraud_distribution.html")

# --- 2. SPENDING TRENDS ---

# By Date
if 'date' in df.columns and 'amount' in df.columns:
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

# By Top Merchant States (if available)
if 'merchant_state' in df.columns and 'amount' in df.columns:
    state_spend = df.groupby('merchant_state')['amount'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=state_spend.index, y=state_spend.values,
                 title="Top 10 Merchant States by Total Spending",
                 labels={'x': 'Merchant State', 'y': 'Total Amount'})
    fig.show()
    out_path = f"{EDA_OUTPUT_DIR}/top_merchant_states_spending.html"
    fig.write_html(out_path)
    upload_to_s3(out_path, S3_BUCKET, f"{S3_PREFIX}/top_merchant_states_spending.html")

# --- 3. ANOMALY DETECTION (Statistical, No ML) ---
# Using IQR for 'amount'
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

# --- Save flagged data for Power BI or further analysis ---
out_csv = f"{EDA_OUTPUT_DIR}/dataset_with_fraud_anomaly_flags.csv"
df.to_csv(out_csv, index=False)
upload_to_s3(out_csv, S3_BUCKET, f"{S3_PREFIX}/dataset_with_fraud_anomaly_flags.csv")

print(f"\nEDA complete! Plots shown, saved locally in '{EDA_OUTPUT_DIR}', and uploaded to s3://{S3_BUCKET}/{S3_PREFIX}/")