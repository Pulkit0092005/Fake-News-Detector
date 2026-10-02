import pandas as pd
import re

# Original dataset load
df = pd.read_csv("datasets/WELFake_Dataset.csv")

# Missing values handle
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")


# Text cleaning function
def clean_text(text):
    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# Clean title and article separately
df["clean_title"] = df["title"].apply(clean_text)
df["clean_text"] = df["text"].apply(clean_text)


# Remove rows where both are empty
df = df[
    (df["clean_title"].str.len() > 0) |
    (df["clean_text"].str.len() > 0)
]


# Keep required columns
df = df[
    ["clean_title", "clean_text", "label"]
]


# Save V4 dataset
df.to_csv(
    "datasets/v4_news.csv",
    index=False
)


print("V4 dataset created successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3))