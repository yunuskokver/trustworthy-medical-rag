# Toward Trustworthy Medical QA: Multi-Metric Evidence for RAG-Enhanced Large Language Models

This repository contains the implementation of a GPT-4-powered Retrieval-Augmented Generation (RAG) framework for medical question answering. The system integrates **FAISS-based dense retrieval** with **GPT-4 generation** and is evaluated on the **MedQuAD dataset** using multiple widely adopted computational metrics.

## Dataset
- **MedQuAD**: Public benchmark for medical QA (~47k Q–A pairs).
- Source: https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research

## Methodology Overview
1. **Preprocessing**: lowercasing, whitespace cleanup, HTML & non-ASCII stripping, punctuation standardization.
2. **Embedding**: `BAAI/bge-small-en`; each answer treated as a single semantic unit.
3. **Indexing**: FAISS FlatL2 over answer embeddings.
4. **Retrieval**: query → embedding → top-*k* passages (default *k*=5).
5. **Generation**: retrieved passages + query → optimized prompt → GPT-4 (temperature=0.0).
6. **Evaluation**: held-out 1,000 questions; BLEU-1/2/3/4, ROUGE-L, METEOR, token-level F1, Exact Match.

## Quickstart
```bash
bash run.sh
```

See `config.yaml` to adjust models/paths/parameters (e.g., retrieval *k*).

## Ethical Statement
This study uses publicly available, de-identified data (MedQuAD). No experiments with human participants, animals, or personal health records were conducted.

