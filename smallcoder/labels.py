"""Track A2: turn harness rows into per-problem controller labels.

For every problem and every arm a = (condition, budget) we keep
  rate[a]   = fraction of the k samples that pass all tests (soft target for P(pass | a))
  tokens[a] = mean generated tokens (thinking + answer), plus retrieval overhead for RAG arms
From these: difficulty = 1 - best rate, and the oracle arm under any token price.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

# ~10 coarse families over Codeforces / TACO tags (unmapped tags are ignored).
TOPIC_FAMILIES = {
    "dp": ["dp", "dynamic programming", "bitmasks"],
    "greedy": ["greedy", "sortings", "sorting", "constructive algorithms"],
    "math": ["math", "number theory", "combinatorics", "probabilities", "matrices", "fft", "chinese remainder theorem"],
    "graphs": ["graphs", "dfs and similar", "shortest paths", "trees", "dsu", "graph matchings", "flows",
               "graph", "tree", "breadth-first search", "depth-first search"],
    "search": ["binary search", "two pointers", "ternary search", "meet-in-the-middle"],
    "data_structures": ["data structures", "segment tree", "range queries", "hashing", "heap"],
    "strings": ["strings", "string suffix structures", "string algorithms"],
    "geometry": ["geometry"],
    "games": ["games", "interactive"],
    "implementation": ["implementation", "brute force", "simulation"],
}
_TAG2FAM = {t: f for f, ts in TOPIC_FAMILIES.items() for t in ts}
FAMILIES = list(TOPIC_FAMILIES)


def tag_families(tags: list[str]) -> list[str]:
    return sorted({_TAG2FAM[t.lower()] for t in tags if t.lower() in _TAG2FAM})


def arm_name(condition: str, budget: int) -> str:
    return f"{condition}@{budget}"


def build_label_table(rows, retrieval_overhead: dict[str, float] | None = None) -> dict[str, dict]:
    """rows: harness rows. retrieval_overhead: pid -> extra tokens spent on the RAG draft
    (added to every arm whose condition is 'rag'). Returns pid -> {"rate": {arm: r}, "n": {arm: n},
    "tokens": {arm: t}}."""
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))  # pid -> arm -> [n, passes, tokens]
    for r in rows:
        a = acc[r["pid"]][arm_name(r["condition"], r["budget"])]
        a[0] += 1
        a[1] += r["verdict"] == "AC"
        a[2] += r["total_tokens"]
    out = {}
    for pid, arms in acc.items():
        rec = {"rate": {}, "n": {}, "tokens": {}}
        for arm, (n, c, t) in arms.items():
            extra = (retrieval_overhead or {}).get(pid, 0.0) if arm.startswith("rag@") else 0.0
            rec["rate"][arm], rec["n"][arm], rec["tokens"][arm] = c / n, n, t / n + extra
        out[pid] = rec
    return out


def complete_arms(table: dict[str, dict]) -> list[str]:
    """Arms present for every problem (the controller's action space)."""
    sets = [set(v["rate"]) for v in table.values()]
    arms = sorted(set.intersection(*sets)) if sets else []
    return sorted(arms, key=lambda a: (a.split("@")[0], int(a.split("@")[1])))


def to_arrays(table: dict[str, dict], pids: list[str], arms: list[str]):
    """rates (N, A), ns (N, A), tokens (N, A) aligned with pids/arms."""
    R = np.array([[table[p]["rate"][a] for a in arms] for p in pids], dtype=np.float32)
    Nn = np.array([[table[p]["n"][a] for a in arms] for p in pids], dtype=np.float32)
    T = np.array([[table[p]["tokens"][a] for a in arms] for p in pids], dtype=np.float32)
    return R, Nn, T


def difficulty(R: np.ndarray) -> np.ndarray:
    """Model-relative difficulty in [0,1]: 1 - best pass rate over all arms."""
    return 1.0 - R.max(axis=1)


def save_table(table: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(table, f)


def load_table(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
