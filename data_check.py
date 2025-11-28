import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('liver_cirrhosis.csv')

print("First 10 rows:")
print(df.head(10))
print("\nData types:")
print(df.dtypes)
print(f"\nShape: {df.shape}")
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSummary statistics:")
print(df.describe(include='all'))

# Check target
if 'Stage' in df.columns:
    print(f"\nStage unique values: {df['Stage'].unique()}")
    print(f"Stage value counts: {df['Stage'].value_counts()}")
else:
    print("Stage column not found!")
