import os, json, argparse, yaml
import random
import nltk, faiss, evaluate
from sentence_transformers import SentenceTransformer
from preprocess import normalize_text
from dotenv import load_dotenv
from openai import OpenAI

# Ensure tokenizers
nltk.download("punkt", quiet=True)

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def token_f1(pred, ref):
    pt = nltk.word_tokenize(pred.lower()); rt = nltk.word_tokenize(ref.lower())
    if not pt or not rt: return 0.0
    pt_set, rt_set = set(pt), set(rt)
    inter = len(pt_set & rt_set)
    if inter == 0: return 0.0
    precision = inter / len(pt)
    recall    = inter / len(rt)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred, ref):
    return int(pred.strip().lower() == ref.strip().lower())

def build_prompt(q, passages):
    ctx = "\n\n".join(passages)
    return f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"

def generate_answer(client, model_name, question, passages, temperature, max_tokens):
    prompt = build_prompt(question, passages)
    r = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return r.choices[0].message.content

def main(cfg):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load data & index
    index = faiss.read_index(cfg["paths"]["index_path"])
    train_answers = [x["answer"] for x in load_jsonl(cfg["paths"]["train_answers_path"])]
    test_questions = [x["question"] for x in load_jsonl(cfg["paths"]["test_questions_path"])]
    test_answers   = [x["answer"] for x in load_jsonl(cfg["paths"]["test_answers_path"])]

    # Models & params
    model = SentenceTransformer(cfg["models"]["embedding_model"])
    k = cfg["retrieval"]["k"]
    temperature = cfg["evaluation"]["temperature"]
    max_tokens  = cfg["evaluation"]["max_output_tokens"]
    lm_name     = cfg["models"]["openai_model"]

    preds, refs = [], []
    indices = list(range(len(test_questions)))

    for i in indices:
        q, ref = test_questions[i], test_answers[i]
        q_vec = model.encode([q])
        D, I = index.search(q_vec, k)
        passages = [train_answers[j] for j in I[0]]
        pred = generate_answer(client, lm_name, q, passages, temperature, max_tokens)
        preds.append(pred); refs.append(ref)

    # Metrics
    bleu_metric   = evaluate.load("bleu")
    rouge_metric  = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    bleu = bleu_metric.compute(predictions=preds, references=refs, max_order=4, smooth=True)
    print("BLEU-1:", bleu["precisions"][0])
    print("BLEU-2:", bleu["precisions"][1])
    print("BLEU-3:", bleu["precisions"][2])
    print("BLEU-4:", bleu["precisions"][3])

    rouge = rouge_metric.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    print("ROUGE-L:", rouge["rougeL"])

    meteor = meteor_metric.compute(predictions=preds, references=refs)
    print("METEOR:", meteor["meteor"])

    f1_scores = [token_f1(p, r) for p, r in zip(preds, refs)]
    em_scores = [exact_match(p, r) for p, r in zip(preds, refs)]
    print("Token-level F1 (avg):", sum(f1_scores)/len(f1_scores))
    print("Exact Match (avg):", sum(em_scores)/len(em_scores))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    main(cfg)
