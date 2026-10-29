
import pandas as pd

file_path = "data/raw/GitHubActions_Docs.csv"

df = pd.read_csv(file_path)

print("\n===== DATASET INFORMATION =====")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head().to_string())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)
