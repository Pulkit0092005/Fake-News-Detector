# 📰 Fake News Detector

A Machine Learning and Natural Language Processing based web application that predicts whether a given news article is **Likely Fake News** or **Likely Real News**.

The system analyzes both the **news headline** and the **complete article** using TF-IDF feature extraction and a Logistic Regression classifier. It also calculates headline–article word consistency as an additional feature.

> ⚠️ This project provides an ML-based prediction and does not guarantee the factual truth of a news article.

---

## 🚀 Project Overview

Fake news can spread rapidly through online platforms and can mislead readers.

The objective of this project is to develop a simple web-based system that can analyze textual news content and classify it as likely fake or real using Machine Learning and NLP techniques.

The project uses the **WELFake Dataset** for training and evaluation.

---

## ✨ Features

- 📰 Headline and article-based news analysis
- 🤖 Machine Learning classification
- 🧠 NLP-based text processing
- 📊 TF-IDF feature extraction
- 🔗 Headline–article consistency feature
- 📈 Prediction confidence
- 🌐 Flask-based web interface
- 🎨 Responsive dark-themed UI
- ⚡ Local prediction without external APIs
- 📚 Multiple model experiments from V1 to V4

---

## 🧠 How the System Works

The system follows this pipeline:

**News Input → Text Processing → TF-IDF Features → Headline-Article Consistency → Logistic Regression → Prediction**

### 1. Input

The user enters:

- News Headline
- Complete News Article

### 2. Text Processing

The text is cleaned using NLP preprocessing techniques such as:

- Lowercasing
- URL removal
- HTML tag removal
- Special character removal
- Extra whitespace removal

### 3. TF-IDF Feature Extraction

TF-IDF converts the textual information into numerical features that can be processed by the Machine Learning model.

The final V4 model uses separate TF-IDF representations for:

- News headline
- News article

### 4. Headline–Article Consistency

The system calculates the proportion of unique headline words that also appear in the article.

This helps provide an additional signal about the relationship between the headline and article content.

> Note: This is a lexical word-overlap feature, not a deep semantic similarity measure.

### 5. Classification

The extracted features are combined and passed to a **Logistic Regression** classifier.

The model predicts:

- `0` → Likely Fake News
- `1` → Likely Real News

---

## 📊 Model Experiments

Several approaches were implemented and compared during development.

| Model | Approach | Accuracy |
|---|---|---:|
| V1 | Word TF-IDF + Logistic Regression | 95.94% |
| V2 | Character TF-IDF + Logistic Regression | 95.90% |
| V3 | Word TF-IDF + Linguistic Features | 95.55% |
| **V4** | **Headline + Article TF-IDF + Consistency + Logistic Regression** | **97.16%** |

### Final Model — V4

The V4 model separately processes the headline and article and adds the headline–article consistency feature.

**Test Accuracy: 97.16%**

Classification report:

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Fake (0) | 0.98 | 0.96 | 0.97 |
| Real (1) | 0.96 | 0.98 | 0.97 |

The reported accuracy is based on the held-out test split of the WELFake dataset and should not be interpreted as a guarantee of factual correctness on real-world news.

---

## 🖥️ Website Preview

### 1. Home Page

![Home Page](Screenshot%202026-10-02%20164628.png)

The home page provides the main interface of the Fake News Detector.

Users can enter a news headline and the complete news article before starting the analysis.

---

### 2. News Input and Analysis

![News Input](Screenshot%202026-10-02%20164651.png)

Users provide the required news information through the input fields.

After submission, the application processes the headline and article using the trained NLP and Machine Learning pipeline.

The text is transformed into TF-IDF features and the headline–article consistency is calculated.

---

### 3. Prediction Result

![Prediction Result](Screenshot%202026-10-02%20164708.png)

After analyzing the submitted news, the trained Logistic Regression model generates a prediction.

The application displays whether the news is:

- **Likely Fake News**
- **Likely Real News**

It also displays the model's prediction confidence.

---

### 4. How It Works

![How It Works](Screenshot%202026-10-02%20164722.png)

This section explains the complete working pipeline of the project.

The system processes the headline and article separately, extracts TF-IDF features, calculates headline–article consistency, and finally uses Logistic Regression for classification.

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Logistic Regression
- TF-IDF Vectorization

### Natural Language Processing
- NLTK
- Text preprocessing
- Word-based feature extraction

### Web Development
- Flask
- HTML
- CSS

### Data Processing
- Pandas
- NumPy
- SciPy

### Model Storage
- Joblib

---

## 📂 Project Structure

```text
FAKE NEWS DETECTOR/
│
├── Datasets/
│   └── WELFake_Dataset.csv
│
├── Model/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── model_v2.pkl
│   ├── vectorizer_v2.pkl
│   ├── model_v3.pkl
│   ├── vectorizer_v3.pkl
│   ├── model_v4.pkl
│   ├── title_vectorizer_v4.pkl
│   └── article_vectorizer_v4.pkl
│
├── Static/
│   └── style.css
│
├── Templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── train_model_v2.py
├── train_model_v3.py
├── train_model_v4.py
├── prepare_v4.py
├── clean_dataset.py
├── check_dataset.py
├── check_labels.py
├── README.md
└── .gitignore
⚙️ Installation

Clone the repository:

git clone https://github.com/Pulkit0092005/Fake-News-Detector.git

Move into the project directory:

cd Fake-News-Detector

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install the required packages:

pip install flask pandas numpy scikit-learn nltk joblib scipy
▶️ Run the Application

Start the Flask application:

python app.py

Then open the local Flask URL shown in the terminal, usually:

http://127.0.0.1:5000/
🔬 Model Training

The project contains multiple training scripts used during experimentation:

train_model.py
train_model_v2.py
train_model_v3.py
train_model_v4.py

The final application uses the V4 model.

To retrain the V4 model:

python train_model_v4.py
🔍 Limitations

Although the final model achieved 97.16% accuracy on the held-out WELFake test set, the system has several limitations:

The model does not verify news against external trusted sources.
Prediction accuracy depends on the training dataset.
TF-IDF mainly captures statistical word patterns.
Headline–article consistency is based on lexical word overlap.
The model may perform differently on completely new types of news.
A high prediction confidence does not mean that the news is factually verified.

Therefore, the system should be considered a research/academic ML project, not an authoritative fact-checking system.

🔮 Future Improvements

Possible future improvements include:

Transformer-based models such as BERT
Semantic similarity between headline and article
Named Entity Recognition
Source credibility analysis
External fact-checking sources
Multilingual fake news detection
Explainable AI for individual predictions
Real-time news verification
Improved handling of sarcasm and misleading headlines
🎯 Project Objective

The main objective of this project is to demonstrate how Machine Learning and Natural Language Processing can be applied to the problem of fake news classification.

The project also demonstrates the complete ML workflow:

Dataset → Preprocessing → Feature Engineering → Model Training → Evaluation → Flask Deployment

👨‍💻 Author

Pulkit Snehi

B.Tech Computer Science & Engineering
ABES Engineering College, Ghaziabad

⚠️ Disclaimer

This application provides a machine-learning-based classification of news content.

The result should not be treated as a definitive statement about whether a news story is factually true or false. Users should verify important information using reliable and independent sources.
