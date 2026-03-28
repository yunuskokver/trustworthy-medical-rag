from datasets import load_dataset
import pandas as pd
import os

SEED = 42
TEST_SIZE = 1000

os.makedirs("data", exist_ok=True)

# MedQuAD yükle
ds = load_dataset("abacha/medquad", split="train")
df = pd.DataFrame(ds)

# Gerekli kolonlar
df = df.dropna(subset=["question", "answer"]).copy()
df["question"] = df["question"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()

# 1000 örnek ayır
test_df = df.sample(n=TEST_SIZE, random_state=SEED).copy()

# Sadece evaluation için gerekli kolonları kaydet
test_df[["question", "answer"]].to_csv("data/test_set.csv", index=False)

print("Saved:", "data/test_set.csv")
print("Shape:", test_df.shape)
print(test_df.head())
