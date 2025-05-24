import pandas as pd
import numpy as np
import boto3
import os

# -------------------- CONFIG -------------------- #
BUCKET = "financial-analysis-project"
S3_INPUT_KEY = "data/processed/merged_dataset_final.csv"
S3_OUTPUT_KEY = "data/processed/cleaned_dataset_final.csv"
LOCAL_OUTPUT_PATH = "data/processed/cleaned_dataset_final.csv"
# ------------------------------------------------ #

def upload_to_s3(local_path, bucket, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, s3_key)
    print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")

def main():
    # 1. Load data directly from S3
    s3_uri = f"s3://{BUCKET}/{S3_INPUT_KEY}"
    print(f"Reading from {s3_uri}")
    df = pd.read_csv(s3_uri)

    # 2. Remove duplicates
    df = df.drop_duplicates()

    # 3. Convert columns to correct dtypes
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if 'amount' in df.columns:
        df['amount'] = df['amount'].replace('[\$,]', '', regex=True).astype(float)

    # Numeric columns (update this list if your columns differ)
    to_numeric = [
        'zip', 'mcc', 'cvv', 'credit_limit', 'year_pin_last_changed', 'num_cards_issued',
        'current_age', 'retirement_age', 'birth_year', 'birth_month', 'latitude', 'longitude',
        'per_capita_income', 'yearly_income', 'total_debt', 'credit_score', 'num_credit_cards'
    ]
    for col in to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Standardize categorical columns
    for col in ['has_chip', 'card_on_dark_web', 'is_fraud']:
        if col in df.columns:
            df[col] = df[col].map({'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

    # 5. Handle missing values
    threshold = 0.7  # drop columns with >70% missing
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    print("Dropped columns due to too many missing values:", drop_cols)

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 6. Example feature engineering
    if 'date' in df.columns:
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek

    # 7. Save cleaned dataset locally
    os.makedirs(os.path.dirname(LOCAL_OUTPUT_PATH), exist_ok=True)
    df.to_csv(LOCAL_OUTPUT_PATH, index=False)
    print(f"Cleaned dataset saved locally at {LOCAL_OUTPUT_PATH}")

    # 8. Upload cleaned dataset to S3
    upload_to_s3(LOCAL_OUTPUT_PATH, BUCKET, S3_OUTPUT_KEY)

if __name__ == "__main__":
    main()