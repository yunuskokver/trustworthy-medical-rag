from datasets import load_dataset
import pandas as pd
import os

SEED = 42
TEST_SIZE = 1000

os.makedirs("data", exist_ok=True)

# MedQuAD load
ds = load_dataset("abacha/medquad", split="train")
df = pd.DataFrame(ds)

# Required columns
df = df.dropna(subset=["question", "answer"]).copy()
df["question"] = df["question"].astype(str).str.strip()
df["answer"] = df["answer"].astype(str).str.strip()

# 1000 sample
test_df = df.sample(n=TEST_SIZE, random_state=SEED).copy()

# save for evaluation
test_df[["question", "answer"]].to_csv("data/test_set.csv", index=False)

print("Saved:", "data/test_set.csv")
print("Shape:", test_df.shape)
print(test_df.head())
