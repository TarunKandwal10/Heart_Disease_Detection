# app.py
# Heart Disease Prediction - Flask REST API
# This script creates a web API that accepts patient data and returns predictions.

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Home route - shows a simple form
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if request is JSON (API call) or form data (web form)
        if request.is_json:
            data = request.get_json()
        else:
            # Get data from form
            data = {
                "age": float(request.form["age"]),
                "sex": float(request.form["sex"]),
                "cp": float(request.form["cp"]),
                "trestbps": float(request.form["trestbps"]),
                "chol": float(request.form["chol"]),
                "fbs": float(request.form["fbs"]),
                "restecg": float(request.form["restecg"]),
                "thalach": float(request.form["thalach"]),
                "exang": float(request.form["exang"]),
                "oldpeak": float(request.form["oldpeak"]),
                "slope": float(request.form["slope"]),
                "ca": float(request.form["ca"]),
                "thal": float(request.form["thal"])
            }

        # Extract features in correct order
        features = [
            data["age"], data["sex"], data["cp"], data["trestbps"],
            data["chol"], data["fbs"], data["restecg"], data["thalach"],
            data["exang"], data["oldpeak"], data["slope"], data["ca"],
            data["thal"]
        ]

        # Make prediction
        import pandas as pd
        feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
                        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
        input_df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(input_df)[0]

        # Return result
        if prediction == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease"

        # If it was a form submission, render the result page
        if not request.is_json:
            return render_template("index.html", prediction_text=result)

        # If it was an API call, return JSON
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
