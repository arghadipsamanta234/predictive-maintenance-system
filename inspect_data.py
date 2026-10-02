
import pandas as pd
from pathlib import Path

# Locate the dataset
DATA_PATH = Path("data") / "ai4i2020.csv"

# Check whether the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH.resolve()}\n"
        "Place ai4i2020.csv inside the data folder."
    )

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("\n===== FIRST 5 RECORDS =====")
print(df.head())

print("\n===== DATASET SIZE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== MACHINE FAILURE COUNTS =====")
print(df["Machine failure"].value_counts())

print("\n===== FAILURE PERCENTAGE =====")
print(
    (df["Machine failure"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe())

print("\nDataset inspection completed successfully!")
