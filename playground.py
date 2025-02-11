import os
import json
from flask import Flask, request, jsonify
import openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


@app.route("/predict_full_listing", methods=["POST"])
def predict_full_listing():
    """API to enhance and rewrite a real estate listing to make it more elegant and catchy."""
    data = request.json
    post_title = data.get("Post_Title", "")
    description = data.get("Description", "")

    prompt = f"""
    You are an expert real estate copywriter with a deep understanding of successful Craigslist postings. Your task is to refine the given listing to make it more compelling, engaging, and effective while keeping the language simple and accessible.

    ### **Input Listing**
    - **Title:** {post_title}
    - **Description:** {description}

    ### **Instructions**
    - **Title:** Rewrite the title to match the style of popular Craigslist listings—short, direct, and attention-grabbing. Use words that create urgency, highlight key features, or add an enticing hook. (Examples: "Spacious 2BR w/ City Views – Move-in Ready!" or "Charming Studio in Heart of Downtown – $1,500/mo").
    - **Description:** Improve the description to make it more vivid, persuasive, and engaging while keeping it natural and easy to read. Focus on highlighting the best aspects of the property in a way that resonates with potential renters/buyers.
    - Avoid overly complex or formal language—keep it clear, casual, and compelling.
    - Maintain all key details but enhance the wording for better impact.

    ### **Desired Output (JSON Format)**
    {{
        "title": "Optimized title in Craigslist style",
        "description": "Improved description that is engaging and persuasive"
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        api_key=OPENAI_API_KEY,
        messages=[
            {"role": "system", "content": "You are a skilled real estate marketer. Provide a JSON response."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return jsonify(json.loads(response["choices"][0]["message"]["content"].strip()))


if __name__ == "__main__":
    app.run(debug=True)
