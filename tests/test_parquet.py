import pandas as pd
import os

file_path = os.path.join('data', 'staged', 'sales_staged.parquet')
df = pd.read_parquet(file_path)

print("--- FILE METRICS ---")
print(f"Shape: {df.shape}")
print("\n--- DATA TYPES ---")
print(df.dtypes)
print("\n--- FIRST 3 ROWS ---")
print(df.head(3))