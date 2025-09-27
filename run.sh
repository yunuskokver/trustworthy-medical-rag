#!/usr/bin/env bash
set -e

# 1) Install dependencies
pip install -r requirements.txt

# 2) Build FAISS index & split data
python src/build_index.py --config config.yaml

# 3) Quick RAG demo
python src/run_rag.py --config config.yaml --query "What are the symptoms of childhood leukemia?"

# 4) Full evaluation on the held-out set
python src/evaluate.py --config config.yaml
