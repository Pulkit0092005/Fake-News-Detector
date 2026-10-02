import pandas as pd
from sklearn.model_selection import train_test_split

# Cleaned dataset load karo
df = pd.read_csv("datasets/cleaned_news.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Features aur target
X = df["clean_content"]
y = df["label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    stop_words="english"
)

# Training data par TF-IDF fit karo
X_train_tfidf = vectorizer.fit_transform(X_train)

# Testing data ko transform karo
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF completed!")

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


from sklearn.linear_model import LogisticRegression

print("\nStarting Logistic Regression training...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_tfidf, y_train)

print("Model training completed!")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Test data par prediction
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

# Model aur TF-IDF vectorizer save karo
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")