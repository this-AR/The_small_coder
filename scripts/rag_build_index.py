"""Build the solved-problem corpus + index from the DECONTAMINATED train pool.

  python scripts/rag_build_index.py --pool data/train/pool.jsonl --embedder Qwen/Qwen3-Embedding-0.6B \
      --out data/rag/index
Use --embedder models/retriever_ft for the fine-tuned retriever, or omit it for BM25 only.
"""
import argparse

from smallcoder.problems import load_problems
from smallcoder.rag import Retriever, make_doc

ap = argparse.ArgumentParser()
ap.add_argument("--pool", nargs="+", required=True)
ap.add_argument("--embedder", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

docs = [d for f in a.pool for p in load_problems(f) if (d := make_doc(p))]
print(f"{len(docs)} documents")
Retriever(docs, a.embedder).save(a.out)
print("saved ->", a.out)
