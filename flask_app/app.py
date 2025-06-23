from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
import os
import joblib

model_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'rf_model.pkl')
model = joblib.load(model_path)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        features = [
            float(request.form["type"]),
            float(request.form["air_temp"]),
            float(request.form["process_temp"]),
            float(request.form["rpm"]),
            float(request.form["torque"]),
            float(request.form["tool_wear"]),
            int(request.form["twf"]),
            int(request.form["hdf"]),
            int(request.form["pwf"]),
            int(request.form["osf"]),
            int(request.form["rnf"]),
        ]

        # Predict
        prediction = model.predict([features])[0]
        status = "⚠️ MACHINE FAILURE" if prediction == 1 else "✅ WORKING FINE"
        return render_template("index.html", prediction=status)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
