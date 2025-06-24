# 🔧 Predictive Maintenance Flask App

This project is an end-to-end **Predictive Maintenance System** built with Machine Learning and deployed using **Flask + Render**. It predicts **machine failure** based on sensor data and operational settings.



---

## 🧠 Features

- 🔍 Exploratory Data Analysis (EDA)
- 📊 Machine Learning with Random Forest
- ✅ High Accuracy (~99.9%)
- 📁 Model Serialization using Joblib
- 🌐 Interactive Web UI via Flask
- ☁️ Deployment on Render

---

## 📁 Project Structure

predictive_maintenance/
│
├── data/ # Dataset (ai4i2020.csv)
├── eda/ # Data exploration
│ └── explore_data.py
├── model/ # Model training & saving
│ ├── prepare_data.py
│ └── train_model.py
├── logs/ # Saved model and visualizations
│ ├── rf_model.pkl
│ ├── confusion_matrix.png
│ └── feature_importance.png
├── flask_app/ # Flask app for deployment
│ ├── app.py
│ └── templates/
│ └── index.html
├── requirements.txt # Project dependencies
└── README.md # Project documentation

This solution is ideal for smart factories looking to:

Predict equipment failure

Prevent unplanned downtime

Improve operational efficiency

Implement cost-effective maintenance strategies

