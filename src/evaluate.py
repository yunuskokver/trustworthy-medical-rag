import os
import json
import argparse
from collections import Counter

import yaml
import faiss
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


# -----------------------------
# Utilities
# -----------------------------
def ensure_nltk() -> None:
    for pkg in ["punkt", "wordnet", "omw-1.4"]:
        try:
            if pkg == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    # remove punctuation
    import string
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = " ".join(text.split())
    return text


def text_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def compute_bleu(pred: str, gold: str):
    smoothie = SmoothingFunction().method1
    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0, 0.0, 0.0, 0.0

    ref = [gold_tokens]

    bleu1 = sentence_bleu(ref, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu(ref, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu(ref, pred_tokens, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(ref, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return bleu1, bleu2, bleu3, bleu4


def compute_metrics(pred: str, gold: str, rouge_scorer_obj):
    bleu1, bleu2, bleu3, bleu4 = compute_bleu(pred, gold)
    rouge_l = rouge_scorer_obj.score(gold, pred)["rougeL"].fmeasure

    try:
        meteor = meteor_score([nltk.word_tokenize(gold)], nltk.word_tokenize(pred))
    except Exception:
        meteor = 0.0

    f1 = text_f1(pred, gold)
    em = exact_match(pred, gold)

    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "ROUGE-L": rouge_l,
        "METEOR": meteor,
        "F1": f1,
        "EM": em,
    }


# -----------------------------
# Prompting
# -----------------------------
SYSTEM_BASELINE = (
    "You are a medical question answering assistant. "
    "Provide a clear, accurate, and concise medical answer to the question."
)

SYSTEM_RAG = (
    "You are a medical question answering assistant. "
    "Use the provided context to generate a clear, accurate, and concise medical answer. "
    "Do not include information that is not supported by the provided context."
)


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n".join([f"Passage {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Instruction:\n"
        f"Based on the above context, provide a clear, accurate, and concise medical answer to the question. "
        f"Do not include information that is not supported by the provided context."
    )


def call_chat(
    client: OpenAI,
    messages: list[dict],
    model_name: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


# -----------------------------
# Retrieval
# -----------------------------
class Retriever:
    def __init__(self, index_path: str, train_answers_path: str, embedding_model_name: str):
        self.index = faiss.read_index(index_path)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.corpus = [row["answer"] for row in load_jsonl(train_answers_path)]

    def retrieve(self, query: str, k: int) -> list[dict]:
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_emb, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            results.append({
                "rank": rank + 1,
                "text": self.corpus[int(idx)],
                "distance": float(distances[0][rank]),
            })
        return results


# -----------------------------
# Evaluation
# -----------------------------
def run_eval_for_k(
    cfg: dict,
    client: OpenAI,
    retriever: Retriever,
    test_questions: list[str],
    test_answers: list[str],
    k: int,
    max_examples: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = cfg["models"]["openai_model"]
    temperature = cfg["evaluation"]["temperature"]
    max_output_tokens = cfg["evaluation"]["max_output_tokens"]

    rouge_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows = []

    total = len(test_questions) if max_examples is None else min(max_examples, len(test_questions))
    for i in tqdm(range(total), desc=f"Evaluating k={k}"):
        question = test_questions[i]
        gold = test_answers[i]

        # Baseline
        baseline_messages = [
            {"role": "system", "content": SYSTEM_BASELINE},
            {"role": "user", "content": question},
        ]
        pred_baseline = call_chat(
            client=client,
            messages=baseline_messages,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # Retrieval
        hits = retriever.retrieve(question, k=k)
        contexts = [h["text"] for h in hits]

        # RAG
        rag_messages = [
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": build_rag_prompt(question, contexts)},
        ]
        pred_rag = call_chat(
            client=client,
            messages=rag_messages,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # Lexical/exact metrics
        base_metrics = compute_metrics(pred_baseline, gold, rouge_obj)
        rag_metrics = compute_metrics(pred_rag, gold, rouge_obj)

        rows.append({
            "question": question,
            "gold": gold,
            "pred_baseline": pred_baseline,
            "pred_rag": pred_rag,
            "retrieved_contexts": json.dumps(contexts, ensure_ascii=False),

            "BLEU-1_baseline": base_metrics["BLEU-1"],
            "BLEU-2_baseline": base_metrics["BLEU-2"],
            "BLEU-3_baseline": base_metrics["BLEU-3"],
            "BLEU-4_baseline": base_metrics["BLEU-4"],
            "ROUGE-L_baseline": base_metrics["ROUGE-L"],
            "METEOR_baseline": base_metrics["METEOR"],
            "F1_baseline": base_metrics["F1"],
            "EM_baseline": base_metrics["EM"],

            "BLEU-1_rag": rag_metrics["BLEU-1"],
            "BLEU-2_rag": rag_metrics["BLEU-2"],
            "BLEU-3_rag": rag_metrics["BLEU-3"],
            "BLEU-4_rag": rag_metrics["BLEU-4"],
            "ROUGE-L_rag": rag_metrics["ROUGE-L"],
            "METEOR_rag": rag_metrics["METEOR"],
            "F1_rag": rag_metrics["F1"],
            "EM_rag": rag_metrics["EM"],
        })

    df = pd.DataFrame(rows)

    # Semantic metrics
    refs = df["gold"].astype(str).tolist()
    preds_baseline = df["pred_baseline"].astype(str).tolist()
    preds_rag = df["pred_rag"].astype(str).tolist()

    # BERTScore (F1)
    _, _, f1_base = bertscore(preds_baseline, refs, lang="en", rescale_with_baseline=True)
    _, _, f1_rag = bertscore(preds_rag, refs, lang="en", rescale_with_baseline=True)

    df["BERTScore_baseline"] = f1_base.cpu().numpy()
    df["BERTScore_rag"] = f1_rag.cpu().numpy()

    # SBERT cosine similarity
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_base = sbert_model.encode(preds_baseline, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    emb_rag = sbert_model.encode(preds_rag, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    emb_ref = sbert_model.encode(refs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    df["SBERT_baseline"] = np.sum(emb_base * emb_ref, axis=1)
    df["SBERT_rag"] = np.sum(emb_rag * emb_ref, axis=1)

    summary = pd.DataFrame([{
        "k": k,
        "BLEU-1_baseline": df["BLEU-1_baseline"].mean(),
        "BLEU-1_rag": df["BLEU-1_rag"].mean(),
        "BLEU-2_baseline": df["BLEU-2_baseline"].mean(),
        "BLEU-2_rag": df["BLEU-2_rag"].mean(),
        "BLEU-3_baseline": df["BLEU-3_baseline"].mean(),
        "BLEU-3_rag": df["BLEU-3_rag"].mean(),
        "BLEU-4_baseline": df["BLEU-4_baseline"].mean(),
        "BLEU-4_rag": df["BLEU-4_rag"].mean(),
        "ROUGE-L_baseline": df["ROUGE-L_baseline"].mean(),
        "ROUGE-L_rag": df["ROUGE-L_rag"].mean(),
        "METEOR_baseline": df["METEOR_baseline"].mean(),
        "METEOR_rag": df["METEOR_rag"].mean(),
        "F1_baseline": df["F1_baseline"].mean(),
        "F1_rag": df["F1_rag"].mean(),
        "EM_baseline": df["EM_baseline"].mean(),
        "EM_rag": df["EM_rag"].mean(),
        "BERTScore_baseline": df["BERTScore_baseline"].mean(),
        "BERTScore_rag": df["BERTScore_rag"].mean(),
        "SBERT_baseline": df["SBERT_baseline"].mean(),
        "SBERT_rag": df["SBERT_rag"].mean(),
    }])

    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--analyze_k", action="store_true", help="Run k=1,3,5 analysis")
    parser.add_argument("--max_examples", type=int, default=None, help="Optional debug limit")
    args = parser.parse_args()

    ensure_nltk()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("results", exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    retriever = Retriever(
        index_path=cfg["paths"]["index_path"],
        train_answers_path=cfg["paths"]["train_answers_path"],
        embedding_model_name=cfg["models"]["embedding_model"],
    )

    test_questions = [row["question"] for row in load_jsonl(cfg["paths"]["test_questions_path"])]
    test_answers = [row["answer"] for row in load_jsonl(cfg["paths"]["test_answers_path"])]

    if len(test_questions) != len(test_answers):
        raise ValueError("Mismatch between test questions and test answers.")

    k_values = [1, 3, 5] if args.analyze_k else [cfg["retrieval"]["k"]]

    all_summaries = []
    for k in k_values:
        df, summary = run_eval_for_k(
            cfg=cfg,
            client=client,
            retriever=retriever,
            test_questions=test_questions,
            test_answers=test_answers,
            k=k,
            max_examples=args.max_examples,
        )

        df.to_csv(f"results/eval_medquad_k{k}.csv", index=False)
        all_summaries.append(summary)

    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_summary.to_csv("results/summary_medquad.csv", index=False)

    print("\nFinal summary:")
    print(final_summary.round(4))


if __name__ == "__main__":
    main()
