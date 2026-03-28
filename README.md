# Toward Trustworthy Medical QA: Multi-Metric Evidence for RAG-Enhanced Large Language Models

## 📖 Description
This repository provides the implementation of a GPT-4-powered Retrieval-Augmented Generation (RAG) framework for medical question answering (MedQA).
The system integrates FAISS-based dense retrieval with GPT-4 generation and is evaluated using a comprehensive set of widely adopted computational evaluation metrics.

The framework is designed to improve factual reliability, reduce hallucinations, and enhance semantic fidelity in high-stakes domains such as healthcare.

---

## 📊 Dataset Information
MedQuAD (Primary Dataset)
Name: MedQuAD (Medical Question Answering Dataset)
Size: ~47,000 question–answer pairs
Source: U.S. National Library of Medicine (NLM) / NIH
Access: Loaded automatically via Hugging Face (abacha/medquad)
Structure: Each entry contains a medical question and a curated answer
Coverage: Symptoms, causes, treatments, prevention, and other clinical information

Usage in this project:

1,000 randomly sampled Q–A pairs are used as a held-out test set
Remaining data is used as the retrieval corpus for FAISS indexing
PubMedQA (Cross-Dataset Evaluation)
Role: Evaluates generalization under domain shift
Structure: Expert-annotated biomedical QA pairs derived from PubMed abstracts
Characteristics: Short, evidence-based answers (more concise than MedQuAD)

Usage in this project:

Used only for evaluation
No indexing or training is performed on PubMedQA

---

## 💻 Code Information
Preprocessing: Lowercasing, whitespace cleanup, HTML artifact removal, and basic punctuation normalization
Embedding Model: BAAI/bge-small-en
Indexing: FAISS IndexFlatL2 (exact nearest neighbor search)
Retrieval: Top-k passages (default: k = 5) selected based on L2 distance
Generation: Structured prompt combining query and retrieved passages fed into GPT-4
Evaluation: Multi-metric evaluation including lexical and semantic metrics

---

## 🚀 Running Experiments

The experimental pipeline consists of two main steps:

---

### 1. Build Retrieval Index and Create Evaluation Split

First, construct the FAISS index and automatically generate the held-out evaluation set:

```bash
python src/build_index.py --config config.yaml
```

This step performs the following operations:

* Downloads the MedQuAD dataset from Hugging Face
* Applies text preprocessing and normalization
* Randomly selects a held-out test set (default: 1000 samples)
* Excludes test samples from the retrieval corpus (to prevent data leakage)
* Builds a FAISS index using dense embeddings
* Saves the following files:

```
data/train_answers.jsonl
data/test_questions.jsonl
data/test_answers.jsonl
data/medquad_faiss.index
```

---

### 2. Run Evaluation (Baseline vs RAG)

After building the index, run the evaluation script:

```bash
python src/evaluate.py --config config.yaml
```

This will:

* Generate answers using:

  * Baseline (GPT-4 without retrieval)
  * RAG (GPT-4 + retrieved context)
* Compute evaluation metrics:

  * BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4
  * ROUGE-L
  * METEOR
  * Text-level F1
  * Exact Match (EM)
  * BERTScore (semantic similarity)
  * SBERT cosine similarity
* Save results to:

```
results/eval_medquad_k5.csv
results/summary_medquad.csv
```

---

### 3. Retrieval Depth Analysis (k = 1, 3, 5)

To evaluate the impact of retrieval depth:

```bash
python src/evaluate.py --config config.yaml --analyze_k
```

This runs experiments for:

* k = 1
* k = 3
* k = 5

and saves:

```
results/eval_medquad_k1.csv
results/eval_medquad_k3.csv
results/eval_medquad_k5.csv
results/summary_medquad.csv
```

---

### 4. Debug Mode

To test the pipeline with fewer samples:

```bash
python src/evaluate.py --config config.yaml --analyze_k --max_examples 10
```
🔑 OpenAI API Key

Set your OpenAI API key before running the scripts.

Linux / Mac
export OPENAI_API_KEY="your_api_key_here"
Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"

---

## 🔬 Reproducibility

The experimental setup is designed to ensure full reproducibility:

* A fixed random seed is used for dataset splitting (`split_seed` in config.yaml)
* The test set is automatically excluded from the retrieval index
* Retrieval is performed using FAISS IndexFlatL2 (deterministic behavior)
* All evaluation metrics are computed consistently across runs
* The entire pipeline (indexing + evaluation) is fully script-based

Users do not need to manually prepare any dataset splits.

---

## ⚠️ Notes

* Running the full experiment (1000 samples) may incur OpenAI API costs
* It is recommended to first run in debug mode (`--max_examples 10`)
* BERTScore computation may be slower due to transformer-based scoring


---

## 📦 Requirements
Python 3.9+
Dependencies include:
faiss-cpu
sentence-transformers
transformers
bert-score
rouge-score
nltk
openai
python-dotenv
pyyaml

See requirements.txt for full details.

---

## 🧪 Methodology
Create a held-out evaluation split from MedQuAD
Build a FAISS index over training answers
Encode queries using dense embeddings
Retrieve top-k relevant passages
Construct a structured prompt
Generate answers using GPT-4
Evaluate using lexical and semantic metrics

