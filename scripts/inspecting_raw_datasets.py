import os
import pandas as pd
import json

SAMPLED_DIR = "data/sampled/"
RAW_DIR = "data/raw/"

# Filepaths for your datasets
datasets = {
    "transactions (sampled)": os.path.join(SAMPLED_DIR, "transactions_data.csv"),
    "train_fraud_labels (sampled)": os.path.join(SAMPLED_DIR, "train_fraud_labels.json"),
    "cards (raw)": os.path.join(RAW_DIR, "cards_data.csv"),
    "users (raw)": os.path.join(RAW_DIR, "users_data.csv"),
    "mcc_codes (raw)": os.path.join(RAW_DIR, "mcc_codes.json"),
}

def print_columns_from_csv(path, name):
    df = pd.read_csv(path, nrows=5)  # Only read a few rows for speed
    print(f"\n{name}:")
    print("Columns:", df.columns.tolist())
    print("Sample rows:\n", df.head())

def print_columns_from_json(path, name):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=["key", "value"])
    else:
        print(f"\n{name}: Unsupported JSON format.")
        return
    print(f"\n{name}:")
    print("Columns:", df.columns.tolist())
    print("Sample rows:\n", df.head())

# Inspect each dataset
for name, path in datasets.items():
    if not os.path.exists(path):
        print(f"\n{name}: File not found at {path}")
        continue
    if path.endswith(".csv"):
        print_columns_from_csv(path, name)
    elif path.endswith(".json"):
        print_columns_from_json(path, name)
    else:
        print(f"\n{name}: Unknown file type.")

print("\nInspect the columns above and decide which ones to use for merging (usually IDs like 'transaction_id', 'user_id', 'card_id', 'mcc_code', etc.).")