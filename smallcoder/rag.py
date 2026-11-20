"""Track A4: retrieval over SOLVED problems (statement + editorial/idea + accepted Python code).

Query = statement + a short draft plan written by the coder; BM25 and a dense embedder are fused by
reciprocal rank; documents of the same problem (pid, Codeforces id, or near-identical statement) are
excluded so training-time retrieval cannot return the answer to itself.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .decontam import cf_ids, ngrams, normalize
from .extract import extract_code
from .problems import Problem

_WORD = re.compile(r"[a-z0-9_]+")

DRAFT_PROMPT = ("Read the problem below. In at most 5 sentences, describe the approach you would take "
                "(the key idea and the algorithm). Do not write code.\n\n{statement}")


def make_doc(p: Problem, max_chars: int = 3200) -> dict | None:
    """~800 tokens: statement head, idea (editorial) and one accepted Python solution."""
    sols = [extract_code(s) or s for s in p.solutions]  # PrimeIntellect solutions arrive fenced
    code = next((s for s in sols if "input(" in s or "sys.stdin" in s or "def " in s), None)
    if not (p.editorial or code):
        return None
    parts = ["### Similar solved problem", p.statement.strip()[:1200]]
    if p.editorial:
        parts.append("### Solution idea\n" + p.editorial.strip()[:1000])
    if code:
        parts.append("### Accepted Python solution\n```python\n" + code.strip()[:1500] + "\n```")
    text = "\n\n".join(parts)[:max_chars]
    return {"doc_id": p.pid, "pid": p.pid, "cf_ids": sorted(cf_ids(p)), "statement": p.statement, "text": text}


def _tok(s: str) -> list[str]:
    return _WORD.findall(s.lower())


class Retriever:
    def __init__(self, docs: list[dict], embedder: str | None = None, emb: np.ndarray | None = None):
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.bm25 = BM25Okapi([_tok(d["text"]) for d in docs])
        self.embedder_name = embedder
        self._model = None
        self.emb = emb
        if embedder and emb is None:
            self.emb = self._encode([d["text"] for d in docs])
        self._stmt_grams = [ngrams(normalize(d["statement"]), 8) for d in docs]

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedder_name)
        return np.asarray(self._model.encode(texts, batch_size=64, normalize_embeddings=True), dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        if self.emb is not None:
            np.save(path / "emb.npy", self.emb.astype(np.float16))
        (path / "meta.json").write_text(json.dumps({"embedder": self.embedder_name}))

    @staticmethod
    def load(path: str | Path) -> "Retriever":
        path = Path(path)
        docs = [json.loads(l) for l in (path / "docs.jsonl").open(encoding="utf-8")]
        meta = json.loads((path / "meta.json").read_text())
        emb = np.load(path / "emb.npy").astype(np.float32) if (path / "emb.npy").exists() else None
        return Retriever(docs, meta.get("embedder"), emb)

    def _is_self(self, j: int, p: Problem, grams: set) -> bool:
        d = self.docs[j]
        if d["pid"] == p.pid or (set(d["cf_ids"]) & cf_ids(p)):
            return True
        g = self._stmt_grams[j]
        return bool(g and grams) and len(g & grams) / max(1, min(len(g), len(grams))) > 0.5

    def search(self, p: Problem, draft: str = "", k: int = 2, pool: int = 50, rrf_k: int = 60) -> list[tuple[int, float]]:
        query = p.statement + "\n" + draft
        bm = self.bm25.get_scores(_tok(query))
        ranks = {}
        for r, j in enumerate(np.argsort(-bm)[:pool]):
            ranks[j] = ranks.get(j, 0.0) + 1.0 / (rrf_k + r + 1)
        if self.emb is not None:
            q = self._encode([query])[0]
            for r, j in enumerate(np.argsort(-(self.emb @ q))[:pool]):
                ranks[j] = ranks.get(j, 0.0) + 1.0 / (rrf_k + r + 1)
        grams = ngrams(normalize(p.statement), 8)
        out = []
        for j, s in sorted(ranks.items(), key=lambda x: -x[1]):
            if not self._is_self(int(j), p, grams):
                out.append((int(j), s))
            if len(out) == k:
                break
        return out

    def context(self, hits: list[tuple[int, float]]) -> str:
        return "\n\n".join(self.docs[j]["text"] for j, _ in hits)
