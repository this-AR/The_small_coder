"""Track A1: the hint-granularity ladder.

Rungs (each is a context string placed before the instructions; stopping rules identical):
  none < pred_tags < oracle_tags < technique < sketch < editorial < editorial_pseudo
"""
from __future__ import annotations

import re

from .problems import Problem

RUNGS = ["none", "pred_tags", "oracle_tags", "technique", "sketch", "editorial", "editorial_pseudo"]

TECHNIQUE_PROMPT = """Below is a competitive programming problem and its official editorial.
State the key algorithmic technique needed, in ONE line of at most 25 words
(e.g. "binary search on the answer with a greedy feasibility check").
Do not write code. Do not restate the problem.

### Problem
{statement}

### Editorial
{editorial}

### One-line technique"""

SKETCH_PROMPT = """Below is a competitive programming problem and its official editorial.
Write an algorithm sketch that a programmer could implement, using exactly these fields:
[IDEA] one or two sentences
[COMPLEXITY] time and memory
[CORNER CASES] a short bulleted list
[PSEUDOCODE] at most 15 lines of language-neutral pseudocode
Do NOT write Python, C++ or any real programming language. Do not use code fences.

### Problem
{statement}

### Editorial
{editorial}

### Algorithm sketch"""

PSEUDO_PROMPT = """Below is a competitive programming problem and its official editorial.
Write language-neutral pseudocode (at most 30 lines) implementing the editorial's solution.
Do NOT write Python, C++ or any real programming language. Do not use code fences.

### Problem
{statement}

### Editorial
{editorial}

### Pseudocode"""

# Python-looking or C/C++-looking lines (editorials are mostly C++).
_PY_LINE = re.compile(r"^\s*(def |import |from \w+ import |print\(|class \w+[:(]|.*\binput\(\)|.*\brange\(|"
                      r"#include|using namespace|int main|.*;\s*$|.*\bfor\s*\(.*;.*;)", re.M)
_FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_TOK = re.compile(r"\w+|[^\w\s]")


def longest_common_run(a: str, b: str) -> int:
    """Length (tokens) of the longest common contiguous token run between two texts."""
    x, y = _TOK.findall(a), _TOK.findall(b)
    if not x or not y:
        return 0
    best, prev = 0, [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        xi = x[i - 1]
        for j in range(1, len(y) + 1):
            if xi == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def leak_reason(hint: str, gold_codes: list[str], max_run: int = 12, max_py_lines: int = 2) -> str | None:
    """Reject teacher hints that contain code: a fence, several Python-looking lines, or a long
    verbatim token run shared with any gold solution."""
    if "```" in hint:
        return "code_fence"
    if len(_PY_LINE.findall(hint)) > max_py_lines:
        return "python_lines"
    for g in gold_codes[:5]:
        if longest_common_run(hint, g) >= max_run:
            return "gold_overlap"
    return None


def gold_codes(p: Problem) -> list[str]:
    """Reference code a hint must not copy: accepted solutions plus code blocks inside the editorial
    (Codeforces eval problems carry no solutions, so the editorial's code is the only gold there)."""
    return list(p.solutions) + _FENCED.findall(p.editorial or "")


def clean_teacher_output(text: str, think_end: str = "</think>") -> str:
    if think_end in text:
        text = text.split(think_end, 1)[1]
    return text.strip()


def tags_context(tags: list[str]) -> str:
    return "Algorithm tags for this problem: " + ", ".join(tags) + "." if tags else ""


class TagPredictor:
    """Statement-only multi-label tag baseline (TF-IDF + linear SVM), the classic method that
    reaches ~0.4 macro-F1 on Codeforces tags. Used for the pred_tags rung."""
    IGNORE = {"*special", "implementation", "brute force"}

    def __init__(self, top_k: int = 2):
        self.top_k = top_k

    def fit(self, problems: list[Problem]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.svm import LinearSVC

        data = [(p.statement, [t for t in p.tags if t not in self.IGNORE]) for p in problems]
        data = [d for d in data if d[1]]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
        X = self.vec.fit_transform([d[0] for d in data])
        self.mlb = MultiLabelBinarizer()
        Y = self.mlb.fit_transform([d[1] for d in data])
        self.clf = OneVsRestClassifier(LinearSVC(C=0.5), n_jobs=-1).fit(X, Y)
        return self

    def predict(self, statements: list[str]) -> list[list[str]]:
        scores = self.clf.decision_function(self.vec.transform(statements))
        return [[self.mlb.classes_[j] for j in row.argsort()[::-1][:self.top_k]] for row in scores]
