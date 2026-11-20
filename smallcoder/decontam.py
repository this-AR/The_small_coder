"""Decontamination: (1) problem-ID joins, (2) word n-gram overlap, (3) optional embedding similarity.
Every removal is recorded with its reason so the paper can report counts per stage."""
from __future__ import annotations

import re
from collections import defaultdict

from .problems import Problem

_CF_URL = [re.compile(r"codeforces\.com/(?:contest|gym)/(\d+)/problem/([A-Za-z]\d?)"),
           re.compile(r"codeforces\.com/problemset/problem/(\d+)/([A-Za-z]\d?)")]
_WORD = re.compile(r"[a-z0-9]+")


def cf_ids(p: Problem) -> set[str]:
    ids = set()
    if p.source == "codeforces":
        ids.add(p.pid.split("/", 1)[1].upper())
        ids |= {a.upper() for a in p.meta.get("aliases", [])}
    for rx in _CF_URL:
        for m in rx.finditer(p.url or ""):
            ids.add(f"{m.group(1)}/{m.group(2)}".upper())
    return ids


def normalize(text: str) -> list[str]:
    text = text.lower().replace("$$$", " ").replace("$", " ")
    return _WORD.findall(text)


def ngrams(words: list[str], n: int) -> set[tuple]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


class Decontaminator:
    def __init__(self, eval_problems: list[Problem], n: int = 13, max_df: int = 2):
        self.n = n
        self.ids: dict[str, str] = {}
        index: dict[tuple, set[str]] = defaultdict(set)
        for p in eval_problems:
            for i in cf_ids(p):
                self.ids[i] = p.pid
            for g in ngrams(normalize(p.statement), n):
                index[g].add(p.pid)
        # n-grams shared by many eval problems are boilerplate ("the first line contains ... test cases").
        self.index = {g: s for g, s in index.items() if len(s) <= max_df}
        self.dropped_boilerplate = len(index) - len(self.index)

    def check(self, p: Problem, texts: list[str] | None = None) -> dict | None:
        """Return {"reason", "eval_pid"} if p overlaps an eval problem. `texts` adds fields such as
        rStar-Coder's seed_question, whose overlap also counts."""
        for i in cf_ids(p):
            if i in self.ids:
                return {"reason": "id", "eval_pid": self.ids[i], "key": i}
        for t in [p.statement, *(texts or [])]:
            for g in ngrams(normalize(t), self.n):
                hit = self.index.get(g)
                if hit:
                    return {"reason": f"{self.n}gram", "eval_pid": sorted(hit)[0], "key": " ".join(g)}
        return None


def embedding_flags(train: list[Problem], evals: list[Problem], model_name: str = "Qwen/Qwen3-Embedding-0.6B",
                    threshold: float = 0.9, batch_size: int = 64) -> list[dict]:
    """Stage 3: near-paraphrases that n-grams miss. Returns pairs for human/LLM review (not auto-removed)."""
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    e = m.encode([p.statement for p in evals], batch_size=batch_size, normalize_embeddings=True)
    flags = []
    for start in range(0, len(train), 4096):
        chunk = train[start:start + 4096]
        t = m.encode([p.statement for p in chunk], batch_size=batch_size, normalize_embeddings=True)
        sims = t @ e.T
        best = sims.argmax(axis=1)
        for k, j in enumerate(best):
            if sims[k, j] >= threshold:
                flags.append({"train_pid": chunk[k].pid, "eval_pid": evals[int(j)].pid, "cos": float(sims[k, j])})
    return flags
