import re
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

# Pre-compute embeddings for the updated T&C violation clauses.
tc_embeddings = model.encode(tc_violation_clauses, convert_to_tensor=True)


def is_gibberish(token: str) -> bool:
    """
    Detect gibberish using a dynamic heuristic:
      - Remove non-alphabetic characters.
      - If a token has at least 3 alphabetic characters and its vowel ratio is below 0.2, consider it gibberish.
    This way, tokens of any length (≥ 3) with almost no vowels (i.e. random strings) are flagged.
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


def predict_post_tc_hybrid(post_title: str, description: str, similarity_threshold: float = 0.7) -> None:
    """
    Predict whether a post should be flagged based on:
      - Semantic similarity of the combined post (title + description) to the T&C violation clauses.
      - Fuzzy matching scores for key terms.
      - A bonus if gibberish tokens are detected.

    If the adjusted similarity meets or exceeds the threshold, the post is flagged.
    """
    post_text = post_title + " " + description
    post_embedding = model.encode(post_text, convert_to_tensor=True)
    similarities = util.cos_sim(post_embedding, tc_embeddings)
    max_similarity = float(similarities.max())
    adjusted_similarity = adjust_similarity_with_fuzzy(max_similarity, post_text)
    prediction = "Flagged" if adjusted_similarity >= similarity_threshold else "Successful (Not Flagged)"

    print("\nPrediction for the provided post:")
    print("Title:", post_title)
    print("Description:", description)
    print("Maximum Semantic Similarity: {:.2f}".format(max_similarity))
    print("Adjusted Similarity (with fuzzy & gibberish): {:.2f}".format(adjusted_similarity))
    print("Predicted Outcome:", prediction)


# Example usage:
if __name__ == '__main__':
    # Example 1: Post with unusual/random strings and suspicious language.
    sample_title1 = "Skyline Splendor! Stunningly Renovated 2BR/1BA Luxe Abode in the Heart of Manhattan"
    sample_description1 = "Welcome to your new home brimming with modern elegance! Prepare to be swept off your feet with this exquisite apartment."
    predict_post_tc_hybrid(sample_title1, sample_description1, similarity_threshold=0.7)

    # Example 2: Post with threatening language that should be flagged.
    sample_title2 = "Sacmm Alert Skyline Splendor! Stunningly Renovated 2BR/1BA Luxe Abode in the Heart of Manhattan"
    sample_description2 = (
        "Welcome to your new home brimming with modern elegance! Prepare to be swept off your feet with this exquisite apartment. "
        "I will fight if anyone talks loud and disrespectfully.")
    predict_post_tc_hybrid(sample_title2, sample_description2, similarity_threshold=0.7)

