import pandas as pd
import numpy as np
import os
import boto3

# For S3 access, uncomment if reading from S3
s3_bucket = 'financial-analysis-project'
s3_key = 'data/sampled/data_sampled.csv'
s3_path = f's3://{s3_bucket}/{s3_key}'
df = pd.read_csv(s3_path, storage_options={'anon': False})


# 1. Drop duplicates
df = df.drop_duplicates()

# 2. Handle missing values (none in your sample, but included for completeness)
# Fill numerical columns with median, categorical with mode
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# 3. Convert data types if necessary (example: ensure 'step' is int)
df['step'] = df['step'].astype(int)
df['isFraud'] = df['isFraud'].astype(int)
df['isFlaggedFraud'] = df['isFlaggedFraud'].astype(int)

# 4. Encode categorical columns (example: 'type')
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# 5. Optionally, remove or mask sensitive columns (like 'nameOrig', 'nameDest')
df = df.drop(['nameOrig', 'nameDest'], axis=1)


# 6. Save cleaned data locally
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/data_cleaned.csv', index=False)
print("Data cleaned and saved to data/processed/data_cleaned.csv")

# 7. Optionally, upload cleaned data to S3
s3 = boto3.client('s3')
s3.upload_file('data/processed/data_cleaned.csv', s3_bucket, 'data/processed/data_cleaned.csv')
print("Cleaned data uploaded to S3.")