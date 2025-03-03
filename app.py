import os
import json
import ast
import re
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# Additional imports for the new model:
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

load_dotenv()

app = Flask(__name__)
CORS(app)

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
    flattened_amenities = [str(item) for sublist in amenities_list for item in sublist]

    return {
        "titles": titles,
        "descriptions": descriptions,
        "locations": locations,
        "prices": prices,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "amenities": flattened_amenities
    }


# -------------------------------
# T&C Violation Model Integration
# -------------------------------

# Updated T&C violation clauses – expanded to capture more types of violations.
tc_violation_clauses = [
    "Posts that contain fraudulent, deceptive, or scam-related content are not allowed.",
    "Any post that uses misleading or false information will be removed.",
    "Inappropriate language, hate speech, and spam are strictly prohibited.",
    "Posts that promote illegal activities or violate community guidelines will be flagged.",
    "Posts containing obvious misspellings or deliberate obfuscation of scam-related words are not allowed.",
    "Posts that contain rude, threatening, or harassing language are not allowed.",
    "Posts that include violent or aggressive statements, such as threats to fight, attack, or harm others, will be flagged.",
    "Posts that contain unintelligible gibberish or random character strings are not allowed.",
    "Posts with insulting or demeaning language are not allowed.",
    "Posts with unrealistic or misleading financial claims are prohibited.",
    "Hateful, racist, or discriminatory language is strictly forbidden.",
    "Posts containing repetitive spam-like phrases are not allowed.",
    "Posts that use obfuscated or leetspeak language to disguise their content are not allowed.",
    "Posts that include the word 'fraudulent' or imply fraud are not allowed."
]

# Load the pre-trained SentenceTransformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')
# Pre-compute embeddings for the T&C violation clauses.
tc_embeddings = st_model.encode(tc_violation_clauses, convert_to_tensor=True)


def is_gibberish(token: str) -> bool:
    """
    Detect gibberish using a dynamic heuristic:
      - Remove non-alphabetic characters.
      - If a token has at least 3 alphabetic characters and its vowel ratio is below 0.2, consider it gibberish.
    """
    token_clean = re.sub(r'[^a-z]', '', token.lower())
    if len(token_clean) < 3:
        return False  # Too short to judge
    vowels = sum(1 for c in token_clean if c in 'aeiou')
    vowel_ratio = vowels / len(token_clean)
    return vowel_ratio < 0.2


def adjust_similarity_with_fuzzy(similarity: float, text: str,
                                 targets: list = ["scam", "threat", "fight",
                                                  "idiot", "guaranteed", "despise",
                                                  "buy now", "fraudulent"]) -> float:
    """
    Adjust the similarity score by:
      - Computing fuzzy match scores for each target word.
      - Checking for gibberish tokens in the text.
    Returns the maximum of the semantic similarity, fuzzy match scores, or a bonus if gibberish is detected.
    """
    max_fuzzy = 0.0
    for target in targets:
        fuzzy_score = fuzz.partial_ratio(target.lower(), text.lower()) / 100.0
        if fuzzy_score > max_fuzzy:
            max_fuzzy = fuzzy_score

    # Check for gibberish tokens in the text.
    gibberish_bonus = 0.0
    tokens = text.split()
    for token in tokens:
        if is_gibberish(token):
            gibberish_bonus = 0.8  # Boost similarity if gibberish is detected.
            break

    return max(similarity, max_fuzzy, gibberish_bonus)


def predict_post_tc_hybrid(post_title: str, description: str, similarity_threshold: float = 0.7) -> dict:
    """
    Predict whether a post should be flagged based on:
      - Semantic similarity of the combined post (title + description) to the T&C violation clauses.
      - Fuzzy matching scores for key terms.
      - A bonus if gibberish tokens are detected.

    Returns a dictionary with similarity scores and predicted outcome.
    """
    post_text = post_title + " " + description
    post_embedding = st_model.encode(post_text, convert_to_tensor=True)
    similarities = util.cos_sim(post_embedding, tc_embeddings)
    max_similarity = float(similarities.max())
    adjusted_similarity = adjust_similarity_with_fuzzy(max_similarity, post_text)
    prediction = "Flagged" if adjusted_similarity >= similarity_threshold else "Successful (Not Flagged)"

    return {
        "Post_Title": post_title,
        "Description": description,
        "Maximum_Semantic_Similarity": round(max_similarity, 2),
        "Adjusted_Similarity": round(adjusted_similarity, 2),
        "Predicted_Outcome": prediction
    }


# -------------------------------
# Flask Endpoints
# -------------------------------

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
        amenities = random.sample(inspiration["amenities"], min(3, len(inspiration["amenities"]))) if inspiration[
            "amenities"] else ["N/A"]
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

    # Ensure all amenities are strings to avoid TypeError
    amenities = [str(amenity) for amenity in amenities]

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


@app.route('/predict_tc_violation', methods=['POST'])
def predict_tc_violation():
    """
    Predict if a post violates T&C based on its title and description.
    Expected JSON payload:
    {
      "Post_Title": "example title",
      "Description": "example description"
    }
    """
    try:
        data = request.get_json()
        post_title = data.get("Post_Title", "")
        description = data.get("Description", "")
        if not post_title and not description:
            return jsonify({"error": "Post_Title or Description must be provided."}), 400

        result = predict_post_tc_hybrid(post_title, description, similarity_threshold=0.7)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
