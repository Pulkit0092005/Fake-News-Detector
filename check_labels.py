import pandas as pd

df = pd.read_csv("datasets/WELFake_Dataset.csv")

print("Label 0 examples:")
print(df[df["label"] == 0][["title", "text"]].head(3).to_string(index=False))

print("\n" + "=" * 80)

print("Label 1 examples:")
print(df[df["label"] == 1][["title", "text"]].head(3).to_string(index=False))