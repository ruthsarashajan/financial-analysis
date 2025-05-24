import os
import pandas as pd
import json
import boto3

# === Folder Paths ===
SAMPLED_DIR = "data/sampled/"
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# === Output Settings ===
BUCKET = "financial-analysis-project"
PROCESSED_S3_PREFIX = "data/processed/"
MERGED_FILENAME = "merged_dataset_final.csv"
MERGED_LOCAL_PATH = os.path.join(PROCESSED_DIR, MERGED_FILENAME)

# === Load Data ===

# 1. transactions (sampled)
transactions = pd.read_csv(os.path.join(SAMPLED_DIR, "transactions_data.csv"))

# 2. train_fraud_labels (sampled, needs dict -> DataFrame conversion and cleaning)
with open(os.path.join(SAMPLED_DIR, "train_fraud_labels.json"), "r") as f:
    fraud_labels_raw = json.load(f)
if isinstance(fraud_labels_raw, list):
    # If it's a list of dicts (less common for label set)
    train_fraud_labels = pd.DataFrame(fraud_labels_raw)
elif isinstance(fraud_labels_raw, dict):
    train_fraud_labels = pd.DataFrame(list(fraud_labels_raw.items()), columns=["transaction_id", "is_fraud"])
    # Only keep rows with numeric transaction_id
    train_fraud_labels = train_fraud_labels[train_fraud_labels["transaction_id"].str.isnumeric()]
    # Cast to same dtype as transactions["id"]
    train_fraud_labels["transaction_id"] = train_fraud_labels["transaction_id"].astype(transactions["id"].dtype)
else:
    raise ValueError("Unsupported JSON format for train_fraud_labels.json")

# 3. cards (raw)
cards = pd.read_csv(os.path.join(RAW_DIR, "cards_data.csv"))

# 4. users (raw)
users = pd.read_csv(os.path.join(RAW_DIR, "users_data.csv"))

# 5. mcc_codes (raw, needs renaming)
with open(os.path.join(RAW_DIR, "mcc_codes.json"), "r") as f:
    mcc_codes_raw = json.load(f)
if isinstance(mcc_codes_raw, list):
    mcc_codes = pd.DataFrame(mcc_codes_raw)
    mcc_codes.rename(columns={"key": "mcc", "value": "mcc_description"}, inplace=True)
elif isinstance(mcc_codes_raw, dict):
    mcc_codes = pd.DataFrame(
        [{"mcc": k, "mcc_description": v} for k, v in mcc_codes_raw.items()]
    )
else:
    raise ValueError("Unsupported JSON format for mcc_codes.json")

# === Merge Process ===

# Merge transactions + cards (card_id ↔ id)
df = transactions.merge(cards, left_on="card_id", right_on="id", how="left", suffixes=("", "_card"))

# Merge with users (client_id ↔ id)
df = df.merge(users, left_on="client_id", right_on="id", how="left", suffixes=("", "_user"))

# Merge with mcc_codes (mcc ↔ mcc)
# Ensure both are the same dtype
if df["mcc"].dtype != mcc_codes["mcc"].dtype:
    try:
        mcc_codes["mcc"] = mcc_codes["mcc"].astype(df["mcc"].dtype)
    except Exception:
        mcc_codes["mcc"] = mcc_codes["mcc"].astype(str)
        df["mcc"] = df["mcc"].astype(str)
df = df.merge(mcc_codes, on="mcc", how="left")

# Merge with train_fraud_labels (transactions.id ↔ transaction_id)
df = df.merge(train_fraud_labels, left_on="id", right_on="transaction_id", how="left")

# === Save Locally ===
df.to_csv(MERGED_LOCAL_PATH, index=False)
print(f"Merged dataset saved to {MERGED_LOCAL_PATH}")

# === Upload to S3 ===
s3 = boto3.client("s3")
s3.upload_file(MERGED_LOCAL_PATH, BUCKET, f"{PROCESSED_S3_PREFIX}{MERGED_FILENAME}")
print(f"Uploaded merged dataset to s3://{BUCKET}/{PROCESSED_S3_PREFIX}{MERGED_FILENAME}")