import pandas as pd
import re

# Dataset load karo
df = pd.read_csv("datasets/WELFake_Dataset.csv")

print("Original dataset shape:")
print(df.shape)

# Missing text ko empty string se replace karo
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

# Title + article text combine karo
df["content"] = df["title"] + " " + df["text"]


# Text cleaning function
def clean_text(text):
    text = text.lower()

    # URLs remove
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # HTML tags remove
    text = re.sub(r"<.*?>", "", text)

    # Special characters remove
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Extra spaces remove
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# NLP cleaning apply karo
df["clean_content"] = df["content"].apply(clean_text)

# Empty content remove karo
df = df[df["clean_content"].str.len() > 0]

# Sirf required columns rakho
df = df[["clean_content", "label"]]

print("\nCleaned dataset shape:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFirst 5 cleaned articles:")
print(df.head())

# Clean dataset save karo
df.to_csv("datasets/cleaned_news.csv", index=False)

print("\nCleaned dataset saved successfully!")