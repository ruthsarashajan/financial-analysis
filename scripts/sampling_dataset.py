import pandas as pd
import boto3
import os

# ------------ CONFIGURATION -------------
s3_bucket = 'financial-analysis-project'  # <-- CHANGE THIS
input_s3_key = 'data/raw/data.csv'  # <-- CHANGE IF NEEDED
output_s3_key = 'data/sampled/data_sampled.csv'
local_sampled_path = 'data/sampled/data_sampled.csv'
sample_size = 10000  # Number of rows for the sample

# ------------ LOAD FROM S3 --------------
s3_path = f's3://{s3_bucket}/{input_s3_key}'
df = pd.read_csv(s3_path, storage_options={'anon': False})

# ------------ SAMPLE THE DATA -----------
if len(df) > sample_size:
    sample_df = df.sample(n=sample_size, random_state=42)
else:
    sample_df = df

# ------------ SAVE LOCALLY --------------
os.makedirs(os.path.dirname(local_sampled_path), exist_ok=True)
sample_df.to_csv(local_sampled_path, index=False)
print(f"Sampled data saved locally at {local_sampled_path}")

# ------------ UPLOAD TO S3 --------------
s3 = boto3.client('s3')
s3.upload_file(local_sampled_path, s3_bucket, output_s3_key)
print(f"Sampled data uploaded to s3://{s3_bucket}/{output_s3_key}")

print("Sampling completed successfully.")
# ------------ END OF SCRIPT -------------