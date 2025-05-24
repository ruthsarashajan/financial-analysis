import pandas as pd

# Replace with your actual bucket and key
s3_bucket = 'financial-analysis-project' 
s3_key = 'data/sampled/data_sampled.csv'

s3_path = f's3://{s3_bucket}/{s3_key}'
df = pd.read_csv(s3_path, storage_options={'anon': False})

# Show basic info
print("Basic Info:")
print(df.info())
print("\n")

# Show first 5 rows
print("First 5 rows:")
print(df.head())
print("\n")

# Show summary statistics (numerical columns)
print("Summary statistics (numerical columns):")
print(df.describe())
print("\n")

# Show summary statistics (all columns, including non-numerical)
print("Summary statistics (all columns):")
print(df.describe(include='all'))
print("\n")

# Show missing values per column
print("Missing values per column:")
print(df.isnull().sum())
print("\n")

# Show column names
print("Column names:")
print(df.columns.tolist())
