import os, json, argparse, yaml
import faiss
from sentence_transformers import SentenceTransformer
from preprocess import normalize_text
from dotenv import load_dotenv
from openai import OpenAI

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def build_prompt(question, passages):
    context = "\n\n".join(passages)
    return f"""You are a medical assistant. Use the following context to answer the question.
If the context is insufficient, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

def main(cfg, query):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load resources
    index = faiss.read_index(cfg["paths"]["index_path"])
    train_answers = [x["answer"] for x in load_jsonl(cfg["paths"]["train_answers_path"])]
    model = SentenceTransformer(cfg["models"]["embedding_model"])

    q = normalize_text(query)
    q_vec = model.encode([q])
    k = cfg["retrieval"]["k"]
    D, I = index.search(q_vec, k)
    retrieved = [train_answers[i] for i in I[0]]

    prompt = build_prompt(q, retrieved)
    resp = client.chat.completions.create(
        model=cfg["models"]["openai_model"],
        messages=[{"role":"user","content":prompt}],
        temperature=cfg["evaluation"]["temperature"],
        max_tokens=cfg["evaluation"]["max_output_tokens"]
    )
    print("Question:", query)
    print("Retrieved top-k:", k)
    print("Answer:\n", resp.choices[0].message.content)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    main(cfg, args.query)
