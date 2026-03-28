#!/usr/bin/env bash
set -e

# 1) Install dependencies
pip install -r requirements.txt

# 2) Build FAISS index and create the held-out split
python src/build_index.py --config config.yaml

# 3) Quick RAG demo
python src/run_rag.py --config config.yaml --query "What are the symptoms of childhood leukemia?"

# 4) Full evaluation on the held-out set (default k from config.yaml)
python src/evaluate.py --config config.yaml

# 5) Optional: retrieval depth analysis (k = 1, 3, 5)
# python src/evaluate.py --config config.yaml --analyze_k
