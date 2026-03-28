# Toward Trustworthy Medical QA: Multi-Metric Evidence for RAG-Enhanced Large Language Models

## 📖 Description
This repository provides the implementation of a GPT-4-powered Retrieval-Augmented Generation (RAG) framework for **medical question answering**.  
The system integrates **FAISS-based dense retrieval** with **GPT-4 generation** and is evaluated on the **MedQuAD dataset** using widely adopted computational metrics.

---

## 📊 Dataset Information
- **Name:** MedQuAD (Medical Question Answering Dataset)  
- **Size:** ~47,000 question–answer pairs  
- **Source:** [U.S. National Library of Medicine / NIH](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research)  
- **Structure:** Each entry includes a medical question and its expert-curated answer. Categories include symptoms, causes, treatments, and preventive measures.  
- **Usage in this project:**  
  - 1,000 randomly sampled Q–A pairs held out as test set.  
  - Remaining pairs serve as the retrieval corpus for FAISS indexing.  

---

## 💻 Code Information
- **Preprocessing:** Normalizes all text (lowercasing, whitespace cleanup, removal of HTML/non-ASCII characters, punctuation standardization).  
- **Embedding:** Uses `BAAI/bge-small-en` sentence embedding model; each answer embedded as a single semantic unit.  
- **Indexing:** FAISS `IndexFlatL2` is used to build a dense retrieval index.  
- **Retrieval:** Given a user question, top-k relevant answers (default *k* = 5) are retrieved.  
- **Generation:** Retrieved passages + user query are provided as a structured prompt to GPT-4.  
- **Evaluation:** System performance measured on 1,000 held-out questions.

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

### 4. Debug Mode (Recommended)

To test the pipeline with fewer samples:

```bash
python src/evaluate.py --config config.yaml --analyze_k --max_examples 10
```

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
- Python 3.9+  
- Dependencies:  
  - `faiss-cpu`, `sentence-transformers`, `transformers`, `evaluate`, `nltk`, `openai`, `python-dotenv`, `pyyaml`  
- See `requirements.txt` for full list.

---

## 🧪 Methodology
1. **Data Preparation:** Normalize and split MedQuAD into train/test subsets.  
2. **Embedding & Indexing:** Embed answers with BGE model and build FAISS index.  
3. **Retrieval:** Encode query and retrieve top-k passages.  
4. **Generation:** Construct prompt and feed into GPT-4.  
5. **Evaluation:** Report BLEU-1/2/3/4, ROUGE-L, METEOR, token-level F1, and Exact Match.

