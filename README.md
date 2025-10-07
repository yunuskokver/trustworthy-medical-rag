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

## 🚀 Usage Instructions
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build FAISS index
```bash
python src/build_index.py --config config.yaml
```

### 3. Run a demo query
```bash
python src/run_rag.py --config config.yaml --query "What are the symptoms of childhood leukemia?"
```

### 4. Evaluate the system
```bash
python src/evaluate.py --config config.yaml
```

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

---

## 📚 Citations
If you use this code or dataset, please cite:

```bibtex
@article{yourname2025ragmedicalqa,
  title={Toward Trustworthy Medical QA: Multi-Metric Evidence for RAG-Enhanced Large Language Models},
  author={Your Name},
  journal={PeerJ Computer Science},
  year={2025}
}
```

Dataset:
```
Abacha, Asma Ben, and Dina Demner-Fushman. "MedQuAD: Medical Question Answering Dataset." National Library of Medicine, 2019.
```

---

## ⚖️ License & Contribution Guidelines
- **License:** MIT License (feel free to modify/add the LICENSE file).  
- **Contributions:** Pull requests and issues are welcome. Please ensure code style consistency and proper documentation in any contribution.  
