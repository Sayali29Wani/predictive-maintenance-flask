import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Set up path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/ai4i2020.csv")

# Load data
df = pd.read_csv(DATA_PATH)

# Drop non-useful columns
df = df.drop(columns=["UDI", "Product ID"])

# Encode 'Type' column (L, M, H)
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])

# Define features and target
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save datasets (optional for tracking)
X_train.to_csv(os.path.join(BASE_DIR, "../data/X_train.csv"), index=False)
X_test.to_csv(os.path.join(BASE_DIR, "../data/X_test.csv"), index=False)
y_train.to_csv(os.path.join(BASE_DIR, "../data/y_train.csv"), index=False)
y_test.to_csv(os.path.join(BASE_DIR, "../data/y_test.csv"), index=False)

print("✅ Data prepared and split.")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
