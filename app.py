from flask import Flask, render_template, request
import joblib
import numpy as np

from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)


# =========================
# Load V4 Model
# =========================

model = joblib.load("model/model_v4.pkl")

title_vectorizer = joblib.load(
    "model/title_vectorizer_v4.pkl"
)

article_vectorizer = joblib.load(
    "model/article_vectorizer_v4.pkl"
)


# =========================
# Headline-Article Consistency
# =========================

def calculate_consistency(title, article):

    title_words = set(title.split())
    article_words = set(article.split())

    if len(title_words) == 0:
        return 0

    common_words = title_words.intersection(article_words)

    return len(common_words) / len(title_words)


# =========================
# Home Page
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# Prediction
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    title = request.form["title"]
    article = request.form["article"]

    # Convert title to TF-IDF
    title_tfidf = title_vectorizer.transform([title])

    # Convert article to TF-IDF
    article_tfidf = article_vectorizer.transform([article])

    # Calculate consistency
    consistency = calculate_consistency(
        title.lower(),
        article.lower()
    )

    consistency_feature = csr_matrix(
        np.array([[consistency]])
    )

    # Combine all features
    combined_features = hstack([
        title_tfidf,
        article_tfidf,
        consistency_feature
    ])

    # Prediction
    prediction = model.predict(combined_features)[0]

    # Probability
    probabilities = model.predict_proba(combined_features)[0]

    confidence = probabilities[prediction] * 100


    # Label mapping
    # 0 = Fake
    # 1 = Real

    if prediction == 0:
        result = "Likely Fake News"
    else:
        result = "Likely Real News"


    print("\n==============================")
    print("V4 PREDICTION")
    print("==============================")

    print("Prediction:", prediction)
    print("Result:", result)
    print("Confidence:", round(confidence, 2), "%")
    print("Consistency:", round(consistency * 100, 2), "%")


    return render_template(
        "index.html",
        prediction=prediction,
        result=result,
        confidence=round(confidence, 2),
        title=title,
        article=article
    )


# =========================
# Run Flask
# =========================

if __name__ == "__main__":
    app.run(debug=True)