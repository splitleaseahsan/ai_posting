import pandas as pd
import os
import numpy as np
from flask import Flask, request, jsonify
from scipy.sparse import hstack
import joblib
from xgboost import XGBClassifier

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_success_model.pkl")
TFIDF_TITLE_PATH = os.path.join(SCRIPT_DIR, "tfidf_title.pkl")
TFIDF_DESC_PATH = os.path.join(SCRIPT_DIR, "tfidf_desc.pkl")

# Load trained model and TF-IDF vectorizers
model = joblib.load(MODEL_PATH)
tfidf_title = joblib.load(TFIDF_TITLE_PATH)
tfidf_desc = joblib.load(TFIDF_DESC_PATH)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # Convert text fields to string
        input_data["Post_Title"] = input_data["Post_Title"].astype(str).fillna("Unknown")
        input_data["Description"] = input_data["Description"].astype(str).fillna("Unknown")

        # Apply TF-IDF transformations
        title_tfidf = tfidf_title.transform(input_data["Post_Title"])
        desc_tfidf = tfidf_desc.transform(input_data["Description"])

        # Combine features
        X_combined = hstack((title_tfidf, desc_tfidf))

        # Make prediction
        prediction = model.predict(X_combined)[0]
        prediction_proba = model.predict_proba(X_combined)[0]

        return jsonify({
            "Success_Prediction": int(prediction),
            "Confidence": {
                "Failure": float(prediction_proba[0]),
                "Success": float(prediction_proba[1])
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
