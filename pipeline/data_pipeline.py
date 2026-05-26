import pandas as pd
import os

print("--- PHASE 1: BRONZE TO SILVER PIPELINE ---")

input_path = os.path.join('data', 'raw', 'Walmart.csv')
df = pd.read_csv(input_path)

print(f"Raw Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Week_of_Year'] = df['Date'].dt.isocalendar().week

df['CPI'] = df['CPI'].ffill()
df['Unemployment'] = df['Unemployment'].ffill()

print(f"\nUnique Stores: {df['Store'].nunique()}")
print(f"Max Weekly Sales: ${df['Weekly_Sales'].max():,.2f}")
print(f"Min Weekly Sales: ${df['Weekly_Sales'].min():,.2f}")

output_path = os.path.join('data', 'staged', 'sales_staged.parquet')
df.to_parquet(output_path, index=False)

print(f"\nSUCCESS! Silver data saved to {output_path}")