import os
import pandas as pd
import json
import boto3

BUCKET = "financial-analysis-project"
RAW_S3_PREFIX = "data/raw/"
SAMPLED_LOCAL_DIR = "data/sampled/"
SAMPLED_S3_PREFIX = "data/sampled/"
SAMPLE_SIZE = 10000  # Adjust as needed

os.makedirs(SAMPLED_LOCAL_DIR, exist_ok=True)

TRANSACTIONS_S3 = f"s3://financial-analysis-project/data/raw/transactions_data.csv"
FRAUD_LABELS_S3 = f"s3://financial-analysis-project/data/raw/train_fraud_labels.json"

# 1. Sample transactions_data.csv from S3 and save locally
transactions_sample_path = os.path.join(SAMPLED_LOCAL_DIR, "transactions_data.csv")
transactions = pd.read_csv(TRANSACTIONS_S3, nrows=SAMPLE_SIZE)
transactions.to_csv(transactions_sample_path, index=False)
print(f"Sampled {SAMPLE_SIZE} rows from {TRANSACTIONS_S3} to {transactions_sample_path}")

# 2. Sample train_fraud_labels.json from S3 and save locally
import s3fs
fs = s3fs.S3FileSystem()
fraud_labels_sample_path = os.path.join(SAMPLED_LOCAL_DIR, "train_fraud_labels.json")
with fs.open(FRAUD_LABELS_S3, "r") as f:
    labels = json.load(f)
if isinstance(labels, list):
    labels_sample = labels[:SAMPLE_SIZE]
elif isinstance(labels, dict):
    labels_sample = dict(list(labels.items())[:SAMPLE_SIZE])
else:
    raise ValueError("Unsupported JSON format for fraud labels")
with open(fraud_labels_sample_path, "w") as f:
    json.dump(labels_sample, f)
print(f"Sampled {SAMPLE_SIZE} records from {FRAUD_LABELS_S3} to {fraud_labels_sample_path}")

# 3. Upload sampled files back to S3 in data/sampled/
s3 = boto3.client("s3")
s3.upload_file(transactions_sample_path, BUCKET, f"{SAMPLED_S3_PREFIX}transactions_data.csv")
print(f"Uploaded {transactions_sample_path} to s3://{BUCKET}/{SAMPLED_S3_PREFIX}transactions_data.csv")

s3.upload_file(fraud_labels_sample_path, BUCKET, f"{SAMPLED_S3_PREFIX}train_fraud_labels.json")
print(f"Uploaded {fraud_labels_sample_path} to s3://{BUCKET}/{SAMPLED_S3_PREFIX}train_fraud_labels.json")