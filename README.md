# Financial Transaction Data Analysis

A comprehensive project for analyzing and visualizing financial transaction data, focusing on fraud detection and anomaly identification using Python, Power BI, and AWS S3.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [How to Run This Project](#how-to-run-this-project)
- [Dashboards & Output](#dashboards--output)

---

## Project Overview

This project analyzes a large financial transactions dataset to uncover trends, perform fraud detection, and identify anomalies. It involves data acquisition, sampling, cleaning, exploratory data analysis (EDA), and the development of an interactive Power BI dashboard. The workflow demonstrates a practical approach to handling real-world financial data, focusing on actionable insights for cyber forensics and data analytics.

---

## Features

- **Data Acquisition:** Downloads and stores the PaySim transactional dataset using AWS S3.
- **Data Sampling:** Efficiently samples a manageable subset for analysis.
- **Data Cleaning & Preprocessing:** Handles missing values, encodes categories, and prepares data for analysis.
- **Exploratory Data Analysis (EDA):** Visualizes trends, distributions, and fraud patterns using Python (Plotly).
- **Anomaly Detection:** (Optional/Planned) Identifies outlier transactions using statistical methods.
- **Interactive Dashboard:** Power BI dashboard with filters, KPIs, charts, and fraud/anomaly insights.
- **Documentation:** Complete project documentation and reporting.

---

## Technologies Used

- [Python 3](https://www.python.org/)
  - pandas, numpy, matplotlib, seaborn, plotly
- [Power BI](https://powerbi.microsoft.com/)
- [AWS S3](https://aws.amazon.com/s3/) (for data storage)
- VS Code

---

## Project Workflow

1. **Data Acquisition**
   - Source dataset: [Kaggle Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets?resource=download)
   - Upload to AWS S3 for centralized access.

2. **Data Sampling & Preprocessing**
   - Randomly sample 10,000 rows for efficient processing.
   - Remove duplicates, handle missing values, encode categories.
   - Save cleaned data locally and to AWS S3.

3. **Exploratory Data Analysis (EDA)**
   - Perform EDA with Python and Plotly.
   - Generate plots for transaction amounts, fraud rates, and trends.
   - Identify suspicious/fraudulent transactions and possible anomalies.

4. **Dashboard Development**
   - Import cleaned data into Power BI.
   - Build interactive dashboard:
     - KPIs: Total Transactions, Total Amount, Fraud Transactions, Anomalies Detected
     - Filters: Transaction Type, Step, Amount Range, Fraud Status
     - Visuals: Trends, distributions, fraud rates, summary tables

5. **Reporting & Documentation**
   - Compile findings, EDA visuals, and dashboard screenshots.
   - Prepare a detailed project report and weekly progress reports.

---

## How to Run This Project

1. **Clone the repository** and set up your Python environment.
2. **Download the dataset** from Kaggle and upload to AWS S3.
3. **Run the data preprocessing scripts** (`data_cleaning_and_preprocessing.py`) to sample and clean the data.
4. **Run EDA scripts** (`eda.py`) to generate plots.
5. **Load the cleaned dataset** into Power BI and build the dashboard using the provided visuals and measures.
6. **(Optional) Implement anomaly detection** by adding Z-score or other statistical methods to flag outliers.

---

## Dashboards & Output

- Example Power BI dashboard visualizing fraud statistics, trends, and anomalies (see `/powerbi` or Appendix).
- EDA plots available in `/eda_plots` or Appendix.

---

## Author

Ruth Sara Shajan  
Register Number: 22BCACDC60  
Internal Guide: Ms. Prathiskha

---
