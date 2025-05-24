import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import boto3

# ----------- CONFIGURATION -----------
S3_BUCKET = 'financial-analysis-project'        # <-- CHANGE THIS!
S3_KEY = 'data/processed/data_cleaned.csv'
LOCAL_PLOT_DIR = 'eda_plots'
S3_PLOT_DIR = 'eda_plots/'

# ----------- LOAD DATA -----------
s3_path = f's3://{S3_BUCKET}/{S3_KEY}'
df = pd.read_csv(s3_path, storage_options={'anon': False})

os.makedirs(LOCAL_PLOT_DIR, exist_ok=True)
plot_paths = []

# ----------- 1. Spending Trends Over Time -----------
if 'step' in df.columns:
    # Transaction volume over time
    tx_count = df.groupby('step').size().reset_index(name='count')
    fig1 = px.line(tx_count, x='step', y='count', title='Transaction Volume Over Time')
    fig1.show()
    plot1_path = os.path.join(LOCAL_PLOT_DIR, 'tx_volume_over_time.html')
    pio.write_html(fig1, file=plot1_path, auto_open=False)
    plot_paths.append(plot1_path)

    # Total transaction amount per time step
    if 'amount' in df.columns:
        amount_sum = df.groupby('step')['amount'].sum().reset_index()
        fig2 = px.line(amount_sum, x='step', y='amount', title='Total Transaction Amount Over Time')
        fig2.show()
        plot2_path = os.path.join(LOCAL_PLOT_DIR, 'amount_over_time.html')
        pio.write_html(fig2, file=plot2_path, auto_open=False)
        plot_paths.append(plot2_path)

        # Average transaction amount per step
        amount_avg = df.groupby('step')['amount'].mean().reset_index()
        fig3 = px.line(amount_avg, x='step', y='amount', title='Average Transaction Amount Over Time')
        fig3.show()
        plot3_path = os.path.join(LOCAL_PLOT_DIR, 'avg_amount_over_time.html')
        pio.write_html(fig3, file=plot3_path, auto_open=False)
        plot_paths.append(plot3_path)

# ----------- 2. Spending Trends By Type -----------
if 'type' in df.columns and 'amount' in df.columns:
    fig4 = px.histogram(df, x='type', y='amount', histfunc='sum', title='Total Amount by Transaction Type')
    fig4.show()
    plot4_path = os.path.join(LOCAL_PLOT_DIR, 'amount_by_type.html')
    pio.write_html(fig4, file=plot4_path, auto_open=False)
    plot_paths.append(plot4_path)
else:
    # If one-hot encoded
    type_cols = [col for col in df.columns if col.startswith("type_")]
    if type_cols and 'amount' in df.columns:
        type_amounts = {col: df.loc[df[col]==1, 'amount'].sum() for col in type_cols}
        df_type_amount = pd.DataFrame(list(type_amounts.items()), columns=['type', 'amount'])
        fig4 = px.bar(df_type_amount, x='type', y='amount', title='Total Amount by Transaction Type')
        fig4.show()
        plot4_path = os.path.join(LOCAL_PLOT_DIR, 'amount_by_type.html')
        pio.write_html(fig4, file=plot4_path, auto_open=False)
        plot_paths.append(plot4_path)

# ----------- 3. Anomaly Detection -----------
if 'amount' in df.columns:
    # Boxplot of transaction amounts
    fig5 = px.box(df, y='amount', title='Transaction Amounts - Outlier Detection')
    fig5.show()
    plot5_path = os.path.join(LOCAL_PLOT_DIR, 'amount_boxplot.html')
    pio.write_html(fig5, file=plot5_path, auto_open=False)
    plot_paths.append(plot5_path)

# Top 20 accounts by sent amount
if 'nameOrig' in df.columns and 'amount' in df.columns:
    top_orig = df.groupby('nameOrig')['amount'].sum().sort_values(ascending=False).head(20).reset_index()
    fig6 = px.bar(top_orig, x='nameOrig', y='amount', title='Top 20 Accounts by Sent Amount')
    fig6.show()
    plot6_path = os.path.join(LOCAL_PLOT_DIR, 'top20_sender.html')
    pio.write_html(fig6, file=plot6_path, auto_open=False)
    plot_paths.append(plot6_path)

# Top 20 accounts by received amount
if 'nameDest' in df.columns and 'amount' in df.columns:
    top_dest = df.groupby('nameDest')['amount'].sum().sort_values(ascending=False).head(20).reset_index()
    fig7 = px.bar(top_dest, x='nameDest', y='amount', title='Top 20 Accounts by Received Amount')
    fig7.show()
    plot7_path = os.path.join(LOCAL_PLOT_DIR, 'top20_receiver.html')
    pio.write_html(fig7, file=plot7_path, auto_open=False)
    plot_paths.append(plot7_path)

# ----------- 4. Fraud Analysis -----------
if 'step' in df.columns and 'isFraud' in df.columns:
    # Fraud rate over time
    fraud_time = df.groupby('step')['isFraud'].mean().reset_index()
    fig8 = px.line(fraud_time, x='step', y='isFraud', title='Fraud Rate Over Time')
    fig8.show()
    plot8_path = os.path.join(LOCAL_PLOT_DIR, 'fraud_rate_over_time.html')
    pio.write_html(fig8, file=plot8_path, auto_open=False)
    plot_paths.append(plot8_path)

if 'type' in df.columns and 'isFraud' in df.columns:
    fraud_by_type = df.groupby('type')['isFraud'].mean().reset_index()
    fig9 = px.bar(fraud_by_type, x='type', y='isFraud', title='Fraud Rate by Transaction Type')
    fig9.show()
    plot9_path = os.path.join(LOCAL_PLOT_DIR, 'fraud_rate_by_type.html')
    pio.write_html(fig9, file=plot9_path, auto_open=False)
    plot_paths.append(plot9_path)
elif 'isFraud' in df.columns and type_cols:
    fraud_by_type = []
    for col in type_cols:
        fraud_rate = df.loc[df[col]==1, 'isFraud'].mean()
        fraud_by_type.append({'type': col, 'isFraud': fraud_rate})
    fraud_by_type_df = pd.DataFrame(fraud_by_type)
    fig9 = px.bar(fraud_by_type_df, x='type', y='isFraud', title='Fraud Rate by Transaction Type')
    fig9.show()
    plot9_path = os.path.join(LOCAL_PLOT_DIR, 'fraud_rate_by_type.html')
    pio.write_html(fig9, file=plot9_path, auto_open=False)
    plot_paths.append(plot9_path)

if 'amount' in df.columns and 'isFraud' in df.columns:
    fig10 = px.histogram(df, x='amount', color='isFraud', nbins=100, 
                        title='Transaction Amount Distribution: Fraud vs Non-Fraud',
                        barmode='overlay', log_y=True)
    fig10.show()
    plot10_path = os.path.join(LOCAL_PLOT_DIR, 'amount_dist_fraud_vs_nonfraud.html')
    pio.write_html(fig10, file=plot10_path, auto_open=False)
    plot_paths.append(plot10_path)

# ----------- 5. Upload Plots to S3 -----------
s3_client = boto3.client('s3')

def upload_to_s3(local_path, s3_folder):
    filename = os.path.basename(local_path)
    s3_dest = os.path.join(s3_folder, filename).replace("\\", "/")
    s3_client.upload_file(local_path, S3_BUCKET, s3_dest)
    print(f"Uploaded {filename} to s3://{S3_BUCKET}/{s3_dest}")

for path in plot_paths:
    if os.path.exists(path):
        upload_to_s3(path, S3_PLOT_DIR)

print("EDA complete: Plots saved locally and uploaded to S3.")