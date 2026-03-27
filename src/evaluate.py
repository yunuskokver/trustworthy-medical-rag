import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

from build_index import load_index
from preprocess import normalize_text

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# -----------------------------

# CONFIG

# -----------------------------

MODEL_NAME = "gpt-4"
TOP_K_VALUES = [1, 3, 5]

# -----------------------------

# OPENAI

# -----------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------

# PROMPT

# -----------------------------

def build_prompt(question, contexts):
context_block = "\n".join([f"Passage {i+1}: {c}" for i, c in enumerate(contexts)])

```
return f"""
```

Question:
{question}

Context:
{context_block}

Instruction:
Based on the above context, provide a clear, accurate, and concise medical answer. Do not include information that is not supported by the provided context.
"""

def generate_answer(messages):
response = client.chat.completions.create(
model=MODEL_NAME,
messages=messages,
temperature=0.0,
max_tokens=512
)
return response.choices[0].message.content.strip()

# -----------------------------

# METRICS

# -----------------------------

def compute_metrics(pred, ref):
smoothie = SmoothingFunction().method1

```
bleu1 = sentence_bleu([ref.split()], pred.split(), weights=(1, 0, 0, 0), smoothing_function=smoothie)
bleu2 = sentence_bleu([ref.split()], pred.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
bleu3 = sentence_bleu([ref.split()], pred.split(), weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
bleu4 = sentence_bleu([ref.split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

meteor = meteor_score([ref.split()], pred.split())

pred_norm = normalize_text(pred)
ref_norm = normalize_text(ref)

# F1
pred_tokens = pred_norm.split()
ref_tokens = ref_norm.split()

common = set(pred_tokens) & set(ref_tokens)
if len(common) == 0:
    f1 = 0
else:
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

em = int(pred_norm == ref_norm)

return bleu1, bleu2, bleu3, bleu4, rouge_l, meteor, f1, em
```

# -----------------------------

# MAIN EVALUATION

# -----------------------------

def evaluate():
print("Loading index...")
index, corpus, embed_model = load_index()

```
print("Loading SBERT model...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading test set...")
test_df = pd.read_csv("data/test_set.csv")

results = []

for k in TOP_K_VALUES:
    print(f"\nEvaluating for k={k}")

    preds_baseline = []
    preds_rag = []
    refs = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row["question"]
        reference = row["answer"]

        # -----------------
        # BASELINE
        # -----------------
        baseline_messages = [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": question}
        ]
        baseline_output = generate_answer(baseline_messages)

        # -----------------
        # RETRIEVAL
        # -----------------
        query_vec = embed_model.encode([question])
        D, I = index.search(np.array(query_vec).astype("float32"), k)

        contexts = [corpus[i] for i in I[0]]

        # -----------------
        # RAG
        # -----------------
        rag_prompt = build_prompt(question, contexts)

        rag_messages = [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": rag_prompt}
        ]

        rag_output = generate_answer(rag_messages)

        preds_baseline.append(baseline_output)
        preds_rag.append(rag_output)
        refs.append(reference)

    # -----------------
    # METRICS
    # -----------------
    print("Computing metrics...")

    bleu1_b, bleu2_b, bleu3_b, bleu4_b = [], [], [], []
    bleu1_r, bleu2_r, bleu3_r, bleu4_r = [], [], [], []
    rouge_b, rouge_r = [], []
    meteor_b, meteor_r = [], []
    f1_b, f1_r = [], []
    em_b, em_r = [], []

    for pb, pr, ref in zip(preds_baseline, preds_rag, refs):
        b1, b2, b3, b4, r, m, f, e = compute_metrics(pb, ref)
        bleu1_b.append(b1); bleu2_b.append(b2); bleu3_b.append(b3); bleu4_b.append(b4)
        rouge_b.append(r); meteor_b.append(m); f1_b.append(f); em_b.append(e)

        b1, b2, b3, b4, r, m, f, e = compute_metrics(pr, ref)
        bleu1_r.append(b1); bleu2_r.append(b2); bleu3_r.append(b3); bleu4_r.append(b4)
        rouge_r.append(r); meteor_r.append(m); f1_r.append(f); em_r.append(e)

    # -----------------
    # BERTScore
    # -----------------
    _, _, bert_f1_b = bertscore(preds_baseline, refs, lang="en", rescale_with_baseline=True)
    _, _, bert_f1_r = bertscore(preds_rag, refs, lang="en", rescale_with_baseline=True)

    # -----------------
    # SBERT
    # -----------------
    emb_b = sbert_model.encode(preds_baseline, normalize_embeddings=True)
    emb_r = sbert_model.encode(preds_rag, normalize_embeddings=True)
    emb_ref = sbert_model.encode(refs, normalize_embeddings=True)

    sbert_b = np.sum(emb_b * emb_ref, axis=1)
    sbert_r = np.sum(emb_r * emb_ref, axis=1)

    # -----------------
    # SUMMARY
    # -----------------
    results.append({
        "k": k,
        "BLEU-1_baseline": np.mean(bleu1_b),
        "BLEU-1_rag": np.mean(bleu1_r),
        "BLEU-2_baseline": np.mean(bleu2_b),
        "BLEU-2_rag": np.mean(bleu2_r),
        "BLEU-3_baseline": np.mean(bleu3_b),
        "BLEU-3_rag": np.mean(bleu3_r),
        "BLEU-4_baseline": np.mean(bleu4_b),
        "BLEU-4_rag": np.mean(bleu4_r),
        "ROUGE-L_baseline": np.mean(rouge_b),
        "ROUGE-L_rag": np.mean(rouge_r),
        "METEOR_baseline": np.mean(meteor_b),
        "METEOR_rag": np.mean(meteor_r),
        "F1_baseline": np.mean(f1_b),
        "F1_rag": np.mean(f1_r),
        "EM_baseline": np.mean(em_b),
        "EM_rag": np.mean(em_r),
        "BERTScore_baseline": np.mean(bert_f1_b.numpy()),
        "BERTScore_rag": np.mean(bert_f1_r.numpy()),
        "SBERT_baseline": np.mean(sbert_b),
        "SBERT_rag": np.mean(sbert_r)
    })

df = pd.DataFrame(results)
df.to_csv("results/summary.csv", index=False)

print("\nFINAL RESULTS:")
print(df)
```

if **name** == "**main**":
evaluate()
