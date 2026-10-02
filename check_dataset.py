import pandas as pd

# Dataset load karo
df = pd.read_csv("datasets/WELFake_Dataset.csv")

# Basic information
print("Dataset loaded successfully!")

print("\nShape of dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())