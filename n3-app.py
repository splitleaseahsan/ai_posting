import re
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define T&C violation clauses – update these based on actual Craigslist T&C.
tc_violation_clauses = [
    "Posts that contain fraudulent, deceptive, or scam-related content are not allowed.",
    "Any post that uses misleading or false information will be removed.",
    "Inappropriate language, hate speech, and spam are strictly prohibited.",
    "Posts that promote illegal activities or violate community guidelines will be flagged.",
    "Posts containing obvious misspellings or deliberate obfuscation of scam-related words are not allowed.",
    "Posts that contain rude, threatening, or harassing language are not allowed.",
    "Posts that include violent or aggressive statements, such as threats to fight, attack, or harm others, will be flagged."
]

# Pre-compute embeddings for the T&C violation clauses.
tc_embeddings = model.encode(tc_violation_clauses, convert_to_tensor=True)


def is_gibberish(token: str) -> bool:
    """
    A simple heuristic to detect gibberish:
    If a token (after removing non-alphabetic characters) is longer than 8 characters
    and has a very low ratio of vowels (< 20%), consider it gibberish.
    """
    token = token.lower()
    token = re.sub(r'[^a-z]', '', token)
    if len(token) > 8:
        vowels = sum(1 for c in token if c in 'aeiou')
        ratio = vowels / len(token)
        if ratio < 0.2:
            return True
    return False


def adjust_similarity_with_fuzzy(similarity: float, text: str, targets: list = ["scam", "threat", "fight"]) -> float:
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

    # Check for gibberish tokens.
    gibberish_bonus = 0.0
    tokens = text.split()
    for token in tokens:
        if is_gibberish(token):
            gibberish_bonus = 0.8  # Boost similarity if gibberish is detected.
            break

    return max(similarity, max_fuzzy, gibberish_bonus)


def predict_post_tc_hybrid(post_title: str, description: str, similarity_threshold: float = 0.7) -> None:
    """
    Predicts whether a post should be flagged based on:
      - Semantic similarity of the post to T&C violation clauses.
      - Fuzzy matching scores for key terms ("scam", "threat", "fight").
      - Detection of gibberish tokens.

    If the adjusted similarity meets or exceeds the threshold, the post is flagged.

    Parameters:
      - post_title: Title of the post.
      - description: Description of the post.
      - similarity_threshold: The cutoff for flagging a post.
    """
    # Combine title and description.
    post_text = post_title + " " + description

    # Compute the embedding of the post.
    post_embedding = model.encode(post_text, convert_to_tensor=True)

    # Compute cosine similarities between the post and each T&C violation clause.
    similarities = util.cos_sim(post_embedding, tc_embeddings)
    max_similarity = float(similarities.max())

    # Adjust the similarity using fuzzy matching and gibberish detection.
    adjusted_similarity = adjust_similarity_with_fuzzy(max_similarity, post_text, targets=["scam", "threat", "fight"])

    # Determine the outcome.
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

