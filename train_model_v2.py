import pandas as pd
import joblib

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
# 2. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# =========================
# 3. Character TF-IDF
# =========================

print("\nStarting Character TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=50000,
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Character TF-IDF completed!")
print("Training shape:", X_train_tfidf.shape)
print("Testing shape:", X_test_tfidf.shape)


# =========================
# 4. Logistic Regression
# =========================

print("\nStarting Logistic Regression training...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")


# =========================
# 5. Evaluation
# =========================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("VERSION 2 RESULTS")
print("================================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# 6. Save V2 Separately
# =========================

joblib.dump(model, "model/model_v2.pkl")
joblib.dump(vectorizer, "model/vectorizer_v2.pkl")

print("\nVersion 2 model saved successfully!")