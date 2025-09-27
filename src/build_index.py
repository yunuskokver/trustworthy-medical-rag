import json, os, argparse, random
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from preprocess import normalize_text
import yaml

def dump_jsonl(path, arr):
    with open(path, "w", encoding="utf-8") as f:
        for x in arr:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

def main(cfg):
    data_dir = cfg["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    # 1) Load & normalize MedQuAD
    ds = load_dataset("Abacha/medquad")["train"]
    questions = [normalize_text(x["question"]) for x in ds]
    answers   = [normalize_text(x["answer"])   for x in ds]

    # 2) Split: hold-out 1000 QAs for evaluation
    random.seed(42)
    N = len(questions)
    test_size = cfg["evaluation"]["test_size"]
    test_idx = set(random.sample(range(N), test_size))
    train_answers = [answers[i] for i in range(N) if i not in test_idx]
    test_questions = [questions[i] for i in range(N) if i in test_idx]
    test_answers   = [answers[i]   for i in range(N) if i in test_idx]

    # 3) Save split to disk (reproducibility)
    dump_jsonl(cfg["paths"]["train_answers_path"], [{"answer": a} for a in train_answers])
    dump_jsonl(cfg["paths"]["test_questions_path"], [{"question": q} for q in test_questions])
    dump_jsonl(cfg["paths"]["test_answers_path"],   [{"answer": a} for a in test_answers])

    # 4) Embeddings for train answers (single semantic unit per answer)
    model = SentenceTransformer(cfg["models"]["embedding_model"])
    emb = model.encode(train_answers, convert_to_numpy=True)

    # 5) Build FAISS index (FlatL2)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    faiss.write_index(index, cfg["paths"]["index_path"])

    print(f"Index built: {cfg['paths']['index_path']} | vectors={emb.shape[0]}, dim={dim}")
    print("Saved:", cfg["paths"]["train_answers_path"], cfg["paths"]["test_questions_path"], cfg["paths"]["test_answers_path"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    main(cfg)
