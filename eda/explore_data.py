import pandas as pd
import os

# Define dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/ai4i2020.csv")

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Display basic info
print("📊 Shape of dataset:", df.shape)
print("\n🧠 Column Info:")
print(df.info())

# Check for missing values
print("\n🩺 Missing values:\n", df.isnull().sum())

# Preview the data
print("\n🔍 First 5 rows:\n", df.head())

# Unique values in the failure column
print("\n⚠️ Failure Column Distribution:")
print(df['Machine failure'].value_counts())

# Optional: save summary to logs
summary_path = os.path.join(BASE_DIR, "../logs/eda_summary.txt")
with open(summary_path, "w") as f:
    f.write("Dataset Shape: {}\n".format(df.shape))
    f.write(str(df.describe()))
