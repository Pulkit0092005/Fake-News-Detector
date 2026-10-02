import pandas as pd
import numpy as np
import joblib
import re

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("datasets/cleaned_news.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

X = df["clean_content"]
y = df["label"]


# =========================
# 2. Feature Engineering
# =========================

def extract_features(text):

    words = text.split()

    word_count = len(words)

    char_count = len(text)

    sentence_count = max(len(re.findall(r"[.!?]", text)), 1)

    avg_word_length = (
        sum(len(word) for word in words) / word_count
        if word_count > 0 else 0
    )

    exclamation_count = text.count("!")
    question_count = text.count("?")

    uppercase_count = sum(1 for c in text if c.isupper())

    digit_count = sum(1 for c in text if c.isdigit())

    uppercase_ratio = (
        uppercase_count / char_count
        if char_count > 0 else 0
    )

    digit_ratio = (
        digit_count / char_count
        if char_count > 0 else 0
    )

    return [
        word_count,
        char_count,
        sentence_count,
        avg_word_length,
        exclamation_count,
        question_count,
        uppercase_ratio,
        digit_ratio
    ]


print("\nExtracting linguistic features...")

linguistic_features = np.array(
    [extract_features(text) for text in X]
)

print("Linguistic features shape:", linguistic_features.shape)


# =========================
# 3. Train/Test Split
# =========================

X_train, X_test, y_train, y_test, features_train, features_test = train_test_split(
    X,
    y,
    linguistic_features,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================
# 4. Word TF-IDF
# =========================

print("\nStarting Word TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Word TF-IDF completed!")
print("TF-IDF shape:", X_train_tfidf.shape)


# =========================
# 5. Combine Features
# =========================

print("\nCombining TF-IDF + Linguistic Features...")

X_train_combined = hstack([
    X_train_tfidf,
    csr_matrix(features_train)
])

X_test_combined = hstack([
    X_test_tfidf,
    csr_matrix(features_test)
])

print("Combined training shape:", X_train_combined.shape)
print("Combined testing shape:", X_test_combined.shape)


# =========================
# 6. Logistic Regression
# =========================

print("\nStarting Logistic Regression training...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_combined, y_train)

print("Model training completed!")


# =========================
# 7. Evaluation
# =========================

y_pred = model.predict(X_test_combined)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("VERSION 3 RESULTS")
print("================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 8. Save V3
# =========================

joblib.dump(model, "model/model_v3.pkl")
joblib.dump(vectorizer, "model/vectorizer_v3.pkl")

print("\nVersion 3 model saved successfully!")