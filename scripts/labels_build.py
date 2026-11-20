"""Collapse harness rows (conditions base + rag, all budgets) into a per-problem label table.

  python scripts/labels_build.py results/labels/vibethinker*.jsonl --rag-contexts data/rag/train_contexts.jsonl \
      --out data/labels/vibethinker_train.json
"""
import argparse

import numpy as np

from smallcoder.io import read_jsonl, read_rows
from smallcoder.labels import build_label_table, complete_arms, difficulty, save_table, to_arrays

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--rag-contexts", default=None, help="contexts file with draft_tokens (retrieval overhead)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

overhead = {r["pid"]: r.get("draft_tokens", 0) for r in read_jsonl(a.rag_contexts)} if a.rag_contexts else None
table = build_label_table(read_rows(a.results), overhead)
arms = complete_arms(table)
pids = sorted(table)
R, N, T = to_arrays({p: table[p] for p in pids if all(x in table[p]["rate"] for x in arms)},
                    [p for p in pids if all(x in table[p]["rate"] for x in arms)], arms)
save_table(table, a.out)
print(f"{len(table)} problems, arms complete for {len(R)}: {arms}")
print("mean pass rate per arm:", dict(zip(arms, np.round(R.mean(0), 3))))
print("mean tokens per arm:", dict(zip(arms, np.round(T.mean(0)))))
print(f"never solved: {(R.max(1) == 0).mean():.1%}; always solved at best arm: {(R.max(1) == 1).mean():.1%}; "
      f"mean difficulty {difficulty(R).mean():.3f}")
