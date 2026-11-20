"""Merge training problem pools, decontaminate against every eval set, and write a removal report.

  python scripts/build_train_pool.py --eval data/eval/*.jsonl --cf-tests-dir $DATA/cf_generated_tests \
      --out data/train/pool.jsonl [--embed-check]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from smallcoder.decontam import Decontaminator, embedding_flags
from smallcoder.problems import (load_deepcoder, load_open_r1_codeforces, load_problems, load_taco_verified,
                                 save_problems)

ap = argparse.ArgumentParser()
ap.add_argument("--eval", nargs="+", required=True)
ap.add_argument("--cf-tests-dir", default=None)
ap.add_argument("--out", default="data/train/pool.jsonl")
ap.add_argument("--min-tests", type=int, default=5)
ap.add_argument("--embed-check", action="store_true", help="stage 3: flag near-paraphrases for review")
a = ap.parse_args()

evals = [p for f in a.eval for p in load_problems(f)]
dec = Decontaminator(evals)
print(f"{len(evals)} eval problems; ignored {dec.dropped_boilerplate} boilerplate n-grams")

sources = {
    "deepcoder-primeintellect": lambda: load_deepcoder("primeintellect"),
    "deepcoder-taco": lambda: load_deepcoder("taco"),
    "deepcoder-lcbv5": lambda: load_deepcoder("lcbv5"),
    "codeforces": lambda: load_open_r1_codeforces("train", "verifiable", a.cf_tests_dir),
    "taco-verified": lambda: load_taco_verified(),
}
kept, removed, seen_text = [], [], set()
stats = Counter()
for name, loader in sources.items():
    for p in loader():
        stats[f"{name}:in"] += 1
        if len(p.tests) < a.min_tests:
            stats[f"{name}:few_tests"] += 1
            continue
        hit = dec.check(p)
        if hit:
            stats[f"{name}:removed_{hit['reason']}"] += 1
            removed.append({"pid": p.pid, **hit})
            continue
        key = " ".join(p.statement.split())[:500]
        if key in seen_text:  # exact duplicates across pools (DeepCoder-taco vs TACO-verified, ...)
            stats[f"{name}:dup"] += 1
            continue
        seen_text.add(key)
        kept.append(p)
        stats[f"{name}:kept"] += 1

n = save_problems(kept, a.out)
report = {"kept": n, "stats": dict(stats), "removed": removed}
if a.embed_check:
    report["embedding_flags_for_review"] = embedding_flags(kept, evals)
Path(a.out).with_suffix(".decontam_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps(dict(stats), indent=1))
print(f"kept {n} -> {a.out}")
