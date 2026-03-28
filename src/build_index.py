import os
import json
import argparse
import random
from typing import List, Dict

import faiss
import numpy as np
import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from preprocess import normalize_text


def dump_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(cfg: dict) -> None:
    data_dir = cfg["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    embedding_model_name = cfg["models"]["embedding_model"]
    index_path = cfg["paths"]["index_path"]
    train_answers_path = cfg["paths"]["train_answers_path"]
    test_questions_path = cfg["paths"]["test_questions_path"]
    test_answers_path = cfg["paths"]["test_answers_path"]

    test_size = int(cfg["evaluation"]["test_size"])
    split_seed = int(cfg.get("evaluation", {}).get("split_seed", 42))

    print("Loading MedQuAD dataset...")
    # IMPORTANT: dataset id must be lowercase
    ds = load_dataset("abacha/medquad", split="train")

    print("Normalizing dataset...")
    questions = [normalize_text(x["question"]) for x in ds if x["question"] and x["answer"]]
    answers = [normalize_text(x["answer"]) for x in ds if x["question"] and x["answer"]]

    if len(questions) != len(answers):
        raise ValueError("Question-answer length mismatch after preprocessing.")

    n_total = len(questions)
    if test_size >= n_total:
        raise ValueError(
            f"test_size ({test_size}) must be smaller than dataset size ({n_total})."
        )

    print(f"Total usable QA pairs: {n_total}")

    print(f"Creating deterministic held-out split (test_size={test_size}, seed={split_seed})...")
    random.seed(split_seed)
    test_idx = set(random.sample(range(n_total), test_size))

    train_answers = [answers[i] for i in range(n_total) if i not in test_idx]
    test_questions = [questions[i] for i in range(n_total) if i in test_idx]
    test_answers = [answers[i] for i in range(n_total) if i in test_idx]

    print(f"Train answers for retrieval index: {len(train_answers)}")
    print(f"Held-out test questions: {len(test_questions)}")

    print("Saving split files...")
    dump_jsonl(train_answers_path, [{"answer": a} for a in train_answers])
    dump_jsonl(test_questions_path, [{"question": q} for q in test_questions])
    dump_jsonl(test_answers_path, [{"answer": a} for a in test_answers])

    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    print("Encoding train answers...")
    emb = model.encode(
        train_answers,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {emb.shape}")

    print("Building FAISS index (IndexFlatL2)...")
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)  # exact search using L2 distance
    index.add(emb)

    faiss.write_index(index, index_path)

    print("\nBuild complete.")
    print(f"Index path         : {index_path}")
    print(f"Index vectors      : {index.ntotal}")
    print(f"Embedding dimension: {dim}")
    print(f"Saved train answers: {train_answers_path}")
    print(f"Saved test queries : {test_questions_path}")
    print(f"Saved test answers : {test_answers_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    main(config)
