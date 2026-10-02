import pandas as pd
import numpy as np
import joblib

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
# 1. Load V4 Dataset
# =========================

df = pd.read_csv("datasets/v4_news.csv")

print("V4 dataset loaded!")
print("Shape:", df.shape)

titles = df["clean_title"].fillna("").astype(str)
articles = df["clean_text"].fillna("").astype(str)
y = df["label"]


# =========================
# 2. Train/Test Split
# =========================

title_train, title_test, article_train, article_test, y_train, y_test = train_test_split(
    titles,
    articles,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", title_train.shape)
print("Testing data:", title_test.shape)


# =========================
# 3. Title TF-IDF
# =========================

print("\nStarting Title TF-IDF...")

title_vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words="english"
)

title_train_tfidf = title_vectorizer.fit_transform(title_train)
title_test_tfidf = title_vectorizer.transform(title_test)

print("Title TF-IDF completed!")
print("Shape:", title_train_tfidf.shape)


# =========================
# 4. Article TF-IDF
# =========================

print("\nStarting Article TF-IDF...")

article_vectorizer = TfidfVectorizer(
    max_features=35000,
    ngram_range=(1, 2),
    stop_words="english"
)

article_train_tfidf = article_vectorizer.fit_transform(article_train)
article_test_tfidf = article_vectorizer.transform(article_test)

print("Article TF-IDF completed!")
print("Shape:", article_train_tfidf.shape)


# =========================
# 5. Headline-Article
#    Consistency Feature
# =========================

print("\nCalculating headline-article consistency...")


def calculate_consistency(title, article):
    title_words = set(title.split())
    article_words = set(article.split())

    if len(title_words) == 0:
        return 0

    common_words = title_words.intersection(article_words)

    return len(common_words) / len(title_words)


train_consistency = np.array([
    calculate_consistency(title, article)
    for title, article in zip(title_train, article_train)
]).reshape(-1, 1)

test_consistency = np.array([
    calculate_consistency(title, article)
    for title, article in zip(title_test, article_test)
]).reshape(-1, 1)


print("Consistency feature created!")
print("Training consistency shape:", train_consistency.shape)
print("Testing consistency shape:", test_consistency.shape)


# =========================
# 6. Combine Features
# =========================

print("\nCombining Title + Article + Consistency...")

X_train_combined = hstack([
    title_train_tfidf,
    article_train_tfidf,
    csr_matrix(train_consistency)
])

X_test_combined = hstack([
    title_test_tfidf,
    article_test_tfidf,
    csr_matrix(test_consistency)
])

print("Combined training shape:", X_train_combined.shape)
print("Combined testing shape:", X_test_combined.shape)


# =========================
# 7. Logistic Regression
# =========================

print("\nStarting Logistic Regression training...")

model = LogisticRegression(
    max_iter=1500,
    solver="liblinear"
)

model.fit(X_train_combined, y_train)

print("Model training completed!")


# =========================
# 8. Evaluation
# =========================

y_pred = model.predict(X_test_combined)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("VERSION 4 RESULTS")
print("================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 9. Save V4 Model
# =========================

joblib.dump(model, "model/model_v4.pkl")

joblib.dump(
    title_vectorizer,
    "model/title_vectorizer_v4.pkl"
)

joblib.dump(
    article_vectorizer,
    "model/article_vectorizer_v4.pkl"
)

print("\nVersion 4 model saved successfully!")