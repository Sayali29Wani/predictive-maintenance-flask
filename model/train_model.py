import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
LOG_DIR = os.path.join(BASE_DIR, "../logs")
MODEL_PATH = os.path.join(BASE_DIR, "../logs/rf_model.pkl")

# Load data
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)

# Evaluate
print("✅ Model Trained Successfully!")
print("🎯 Accuracy: {:.2f}%".format(rf.score(X_test, y_test) * 100))

# Classification report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "confusion_matrix.png"))
plt.close()

# Feature Importance
importances = rf.feature_importances_
features = X_train.columns
plt.figure(figsize=(10, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, "feature_importance.png"))
plt.close()

# Save model
joblib.dump(rf, MODEL_PATH)
print(f"🧠 Model saved to: {MODEL_PATH}")

import joblib

# Save model to logs folder
MODEL_PATH = os.path.join(LOG_DIR, "rf_model.pkl")
joblib.dump(rf, MODEL_PATH)
print(f"🧠 Model saved to: {MODEL_PATH}")

