import os
import json
import argparse
import yaml

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

from preprocess import normalize_text


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(question, passages):
    context_block = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)])

    return (
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Instruction:\n"
        f"Based on the above context, provide a clear, accurate, and concise medical answer "
        f"to the question."
    )


def main(cfg, query):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    # Load resources
    index = faiss.read_index(cfg["paths"]["index_path"])
    train_answers = [x["answer"] for x in load_jsonl(cfg["paths"]["train_answers_path"])]
    model = SentenceTransformer(cfg["models"]["embedding_model"])

    # Normalize + encode query
    q = normalize_text(query)
    q_vec = model.encode([q], convert_to_numpy=True).astype("float32")

    # Retrieval
    k = cfg["retrieval"]["k"]
    distances, indices = index.search(q_vec, k)
    retrieved = [train_answers[i] for i in indices[0]]

    # Prompt
    prompt = build_prompt(q, retrieved)

    response = client.chat.completions.create(
        model=cfg["models"]["openai_model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a medical question answering assistant. "
                    "Use the provided context to generate a clear, accurate, and concise medical answer. "
                    
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=cfg["evaluation"]["temperature"],
        max_tokens=cfg["evaluation"]["max_output_tokens"],
    )

    answer = response.choices[0].message.content.strip()

    print("=" * 80)
    print("Question:")
    print(query)
    print("\nRetrieved top-k:", k)
    print("\nRetrieved Passages:")
    for i, (passage, dist) in enumerate(zip(retrieved, distances[0]), start=1):
        print(f"\n[{i}] (L2 distance: {dist:.4f})")
        print(passage[:500] + ("..." if len(passage) > 500 else ""))

    print("\nGenerated Answer:")
    print(answer)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--query", required=True, help="Medical question to ask")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.query)
