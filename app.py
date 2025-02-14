import os
import json
import pandas as pd
import ast
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from scipy.sparse import hstack
import matplotlib.pyplot as plt

load_dotenv()

app = Flask(__name__)

# Set OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load Craigslist Data
csv_file_path = "craigslist_data.csv"
try:
    df = pd.read_csv(csv_file_path)
    df_successful = df[df["Success"] == 1]  # Filter for successful listings
except Exception as e:
    print(f"Error loading CSV: {e}")
    df_successful = None

# XGBoost Model Paths
MODEL_PATH = "xgboost_success_model.pkl"
TFIDF_TITLE_PATH = "tfidf_title.pkl"
TFIDF_DESC_PATH = "tfidf_desc.pkl"

# Load XGBoost model and TFIDF vectorizers
xgb_model = joblib.load(MODEL_PATH)
tfidf_title = joblib.load(TFIDF_TITLE_PATH)
tfidf_desc = joblib.load(TFIDF_DESC_PATH)

def get_style_inspiration():
    """Extracts tone and structure from successful listings."""
    if df_successful is None or df_successful.empty:
        return None

    sample_posts = df_successful.sample(n=min(5, len(df_successful)), random_state=42)

    titles = "\n".join(sample_posts["Post_Title"].astype(str))
    descriptions = "\n".join(sample_posts["Description"].astype(str))
    locations = sample_posts["Location"].dropna().unique().tolist()
    prices = sample_posts["Post_Price"].dropna().unique().tolist()
    bedrooms = sample_posts["Bedrooms"].dropna().unique().tolist()
    bathrooms = sample_posts["Bathrooms"].dropna().unique().tolist()

    # Extract and flatten amenities while ensuring all values are strings
    amenities_list = sample_posts[["Laundry", "Parking", "Private Bath"]].dropna().astype(str).values.tolist()
    flattened_amenities = [str(item) for sublist in amenities_list for item in sublist]  # Convert to strings

    return {
        "titles": titles,
        "descriptions": descriptions,
        "locations": locations,
        "prices": prices,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "amenities": flattened_amenities
    }

@app.route("/create_craigslist", methods=["POST"])
def predict_full_listing():
    """Enhances and rewrites a real estate listing inspired by successful Craigslist posts."""
    data = request.json

    # Get inspiration from successful posts
    inspiration = get_style_inspiration()
    if inspiration is None:
        return jsonify({"error": "No successful Craigslist data available"}), 500

    # If no input is provided, randomly generate a new listing inspired by successful posts
    if not data:
        import random
        post_title = random.choice(inspiration["titles"].split("\n"))
        description = random.choice(inspiration["descriptions"].split("\n"))
        location = random.choice(inspiration["locations"]) if inspiration["locations"] else "Not specified"
        price = random.choice(inspiration["prices"]) if inspiration["prices"] else "N/A"
        bedrooms = random.choice(inspiration["bedrooms"]) if inspiration["bedrooms"] else "N/A"
        bathrooms = random.choice(inspiration["bathrooms"]) if inspiration["bathrooms"] else "N/A"
        amenities = random.sample(inspiration["amenities"], min(3, len(inspiration["amenities"]))) if inspiration["amenities"] else ["N/A"]
    else:
        post_title = data.get("Post_Title", "")
        description = data.get("Description", "")
        location = data.get("Location", "Not specified")
        price = data.get("Post_Price", "N/A")
        bedrooms = data.get("Bedrooms", "N/A")
        bathrooms = data.get("Bathrooms", "N/A")

        amenities = [
            str(data.get("Laundry", "N/A")),
            str(data.get("Parking", "N/A")),
            "Private Bath" if data.get("Private Bath") else "Shared Bath"
        ]

    # **Ensure all amenities are strings to avoid TypeError**
    amenities = [str(amenity) for amenity in amenities]  # Convert all items to string

    prompt = f"""
    You are a professional real estate copywriter with expertise in writing high-performing Craigslist listings.
    Rewrite and optimize the given real estate listing using the proven style, structure, and tone of successful posts.

    ### **Style Inspiration**
    - **Successful Titles:** 
    {inspiration["titles"]}

    - **Successful Descriptions:** 
    {inspiration["descriptions"]}

    ### **Input Listing**
    - **Title:** {post_title}
    - **Description:** {description}
    - **Location:** {location}
    - **Price:** {price}
    - **Bedrooms:** {bedrooms}
    - **Bathrooms:** {bathrooms}
    - **Amenities:** {", ".join(amenities)}

    ### **Instructions**
    - **Title:** Rewrite using the style of successful Craigslist listings.
    - **Description:** Improve it with an engaging, persuasive tone.
    - **Additional Suggestions:** Enhance the listing with extra details.
    - Keep it clear, concise, and naturally flowing.

    ### **Desired Output (JSON Format)**
    {{
        "title": "Optimized Craigslist-style title",
        "description": "Engaging, persuasive description",
        "additional_suggestions": {{
            "price": "{price}",
            "bedrooms": "{bedrooms}",
            "bathrooms": "{bathrooms}",
            "location": "{location}",
            "amenities": {amenities}
        }}
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            api_key=OPENAI_API_KEY,
            messages=[{"role": "system", "content": "You are a skilled real estate marketer."},
                      {"role": "user", "content": prompt}],
            max_tokens=500
        )

        if "choices" not in response or not response["choices"]:
            return jsonify({"error": "Empty response from OpenAI"}), 500

        ai_content = response["choices"][0]["message"]["content"].strip()

        ai_content_fixed = ai_content.replace("'", '"')  # Converts bad single-quoted JSON to valid JSON

        try:
            parsed_response = json.loads(ai_content_fixed)
        except json.JSONDecodeError:
            try:
                parsed_response = ast.literal_eval(ai_content)
            except (SyntaxError, ValueError):
                return jsonify({"error": "Still invalid JSON from OpenAI", "raw_response": ai_content}), 500

        return jsonify(parsed_response)

    except openai.error.OpenAIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Predict the success of a Craigslist listing based on the XGBoost model."""
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
        prediction = xgb_model.predict(X_combined)[0]
        prediction_proba = xgb_model.predict_proba(X_combined)[0]

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
