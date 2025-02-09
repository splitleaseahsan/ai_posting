import pandas as pd
import random
from flask import Flask, request, jsonify
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import json

app = Flask(__name__)

# Set your OpenAI API key
OPENAI_API_KEY = ""


# Load successful posts from the CSV
def load_successful_posts(csv_file="/mnt/data/craigslist_data.csv"):
    """Loads successful listings with relevant features."""
    try:
        df = pd.read_csv(csv_file)
        successful_posts = df[df["Success"] == 1].dropna(subset=["Post_Title", "Description"])
        if successful_posts.empty:
            print("No successful listings found in the dataset.")
        return successful_posts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame()


# Load successful posts at startup
SUCCESSFUL_POSTS = load_successful_posts()


def find_best_match(new_post):
    """Finds the most similar successful post based on key features."""
    if SUCCESSFUL_POSTS.empty:
        return None

    # Convert categorical features into text
    successful_texts = SUCCESSFUL_POSTS.apply(lambda row:
                                              f"{row['Borough']} {row['Location']} {row['Bedrooms']} {row['Bathrooms']} {row['Private Room']} {row['Private Bath']} {row['Parking']}",
                                              axis=1)

    new_post_text = f"{new_post.get('Borough', '')} {new_post.get('Location', '')} {new_post.get('Bedrooms', '')} {new_post.get('Bathrooms', '')} {new_post.get('Private Room', '')} {new_post.get('Private Bath', '')} {new_post.get('Parking', '')}"

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(successful_texts.tolist() + [new_post_text])

    # Compute cosine similarity
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]

    # Get the best match
    best_match_idx = similarities.argmax()
    return SUCCESSFUL_POSTS.iloc[best_match_idx]


def enhance_text_gpt(title, description, example_title, example_description):
    """Uses OpenAI GPT to generate better titles and descriptions."""

    prompt = f"""
    Generate a JSON-formatted real estate listing based on this example:

    **Example Listing:**
    - Title: {example_title}
    - Description: {example_description}

    **New Listing:**
    - Title: {title or 'Unknown'}
    - Description: {description or 'Unknown'}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        api_key=OPENAI_API_KEY,
        messages=[
            {"role": "system",
             "content": "You are an expert in real estate listings. Format the response as a valid JSON object."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return json.loads(response["choices"][0]["message"]["content"].strip())


def generate_ai_listing():
    """Generates a fully AI-driven listing when no successful listings exist."""

    prompt = """Generate a JSON-formatted real estate listing including:
    - title
    - description
    - borough
    - location (address, latitude, longitude)
    - attributes (bedrooms, bathrooms, private room, private bath, parking options).
    Keep the language simple and engaging."""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        api_key=OPENAI_API_KEY,
        messages=[
            {"role": "system",
             "content": "You are an expert in real estate listings. Format the response as a valid JSON object."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400
    )

    return json.loads(response["choices"][0]["message"]["content"].strip())


@app.route("/predict_full_listing", methods=["POST"])
def predict_full_listing():
    """API to generate a full listing using successful posts as inspiration."""
    data = request.json

    # Handle case where no data is provided
    if not data:
        if SUCCESSFUL_POSTS.empty:
            ai_generated_listing = generate_ai_listing()
            return jsonify({"Generated_Listing_with_no_data": ai_generated_listing})

        # Pick a random successful listing
        best_match = SUCCESSFUL_POSTS.sample(n=1).iloc[0]
    else:
        # Extract input details
        borough = data.get("Borough", "")
        location = data.get("Location", "")
        bedrooms = data.get("Bedrooms", 1)
        bathrooms = data.get("Bathrooms", 1)
        private_room = data.get("Private Room", False)
        private_bath = data.get("Private Bath", False)
        parking = data.get("Parking", "no parking")

        # Create a new post dictionary
        new_post = {
            "Borough": borough,
            "Location": location,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Private Room": private_room,
            "Private Bath": private_bath,
            "Parking": parking
        }

        # Find the best matching successful post
        best_match = find_best_match(new_post)

        if best_match is None:
            if SUCCESSFUL_POSTS.empty:
                ai_generated_listing = generate_ai_listing()
                return jsonify({"Generated_Listing_data": ai_generated_listing})
            best_match = SUCCESSFUL_POSTS.sample(n=1).iloc[0]  # Fallback to a random successful post

    # Use AI to generate a slightly new version of the successful listing
    enhanced_listing = enhance_text_gpt(best_match["Post_Title"], best_match["Description"], best_match["Post_Title"],
                                        best_match["Description"])

    return jsonify({"Generated_Listing": enhanced_listing})


if __name__ == "__main__":
    app.run(debug=True)
