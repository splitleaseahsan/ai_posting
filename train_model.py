import pandas as pd
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "craigslist_data.csv")

MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_success_model.pkl")
TFIDF_TITLE_PATH = os.path.join(SCRIPT_DIR, "tfidf_title.pkl")
TFIDF_DESC_PATH = os.path.join(SCRIPT_DIR, "tfidf_desc.pkl")

# Load dataset
df = pd.read_csv(CSV_FILE_PATH)

# Ensure text fields are strings and fill missing values
df["Post_Title"] = df["Post_Title"].astype(str).fillna("Unknown")
df["Description"] = df["Description"].astype(str).fillna("Unknown")

# TF-IDF with n-grams
tfidf_title = TfidfVectorizer(ngram_range=(1,2), max_features=500)
tfidf_desc = TfidfVectorizer(ngram_range=(1,2), max_features=1000)

title_tfidf = tfidf_title.fit_transform(df["Post_Title"])
desc_tfidf = tfidf_desc.fit_transform(df["Description"])

# Combine features
X_combined = hstack((title_tfidf, desc_tfidf))
y = df["Success"]

# Check class distribution
count_negative = y.value_counts()[0]
count_positive = y.value_counts()[1]
scale_pos_weight = (count_negative / count_positive) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train XGBoost with class weights
xgb_model = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, max_depth=8, learning_rate=0.1, n_estimators=150)
xgb_model.fit(X_train, y_train)

# Plot feature importance
plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=20)
plt.title('XGBoost Feature Importance')
plt.show()

# Save model and vectorizers
joblib.dump(xgb_model, MODEL_PATH)
joblib.dump(tfidf_title, TFIDF_TITLE_PATH)
joblib.dump(tfidf_desc, TFIDF_DESC_PATH)

print("✅ Simplified XGBoost model trained and saved successfully!")
