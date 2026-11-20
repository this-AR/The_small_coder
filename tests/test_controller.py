"""CPU-only tests for labels, policy, controller, self-routing and retrieval (Track A2-A4)."""
import numpy as np
import pytest

from smallcoder.labels import build_label_table, complete_arms, tag_families, to_arrays
from smallcoder.policy import area_under, choose, oracle_curve, pass_at_matched_tokens, upper_frontier
from smallcoder.problems import Problem
from smallcoder.selfroute import draft_agreement


# ---------------------------------------------------------------- labels + policy
def _rows():
    rows = []
    # p1: easy (passes at budget 0); p2: needs budget 4096; tokens grow with budget
    for pid, need in [("p1", 0), ("p2", 4096)]:
        for b in [0, 1024, 4096]:
            for s in range(4):
                rows.append({"pid": pid, "condition": "base", "budget": b, "sample": s,
                             "verdict": "AC" if b >= need else "WA", "total_tokens": 300 + b})
    return rows


def test_label_table_and_oracle():
    t = build_label_table(_rows())
    arms = complete_arms(t)
    assert arms == ["base@0", "base@1024", "base@4096"]
    R, N, T = to_arrays(t, ["p1", "p2"], arms)
    assert R.tolist() == [[1, 1, 1], [0, 0, 1]] and N.min() == 4
    # oracle at a small price picks base@0 for p1 and base@4096 for p2
    assert choose(R, T, lam=0.01).tolist() == [0, 2]
    pts = [(tok, p) for _, tok, p in oracle_curve(R, T)]
    assert pass_at_matched_tokens(pts, (300 + 4396) / 2 + 1) == 1.0
    assert 0 < area_under(pts, 8000) <= 1


def test_rag_overhead_and_families():
    rows = [dict(r, condition="rag") for r in _rows()]
    t = build_label_table(rows, retrieval_overhead={"p1": 100})
    assert t["p1"]["tokens"]["rag@0"] == 400 and t["p2"]["tokens"]["rag@0"] == 300
    assert tag_families(["dp", "Graphs", "unknown"]) == ["dp", "graphs"]


def test_frontier():
    assert upper_frontier([(1, 0.1), (2, 0.05), (3, 0.3)]) == [(1, 0.1), (3, 0.3)]


# ---------------------------------------------------------------- controller (torch, tiny)
def test_controller_learns_and_numpy_matches():
    torch = pytest.importorskip("torch")
    from smallcoder.controller import calibrate, numpy_forward, train_heads
    rng = np.random.default_rng(0)
    n, d = 600, 16
    X = rng.normal(size=(n, d)).astype(np.float32)
    easy = (X[:, 0] > 0).astype(np.float32)          # feature 0 decides difficulty
    R = np.stack([easy, easy, np.ones(n)], 1).astype(np.float32)   # hard problems need the big arm
    N = np.full_like(R, 8)
    T = np.tile(np.array([300, 1300, 4300], np.float32), (n, 1))
    Y = np.zeros((n, 3), np.float32)
    m = train_heads(X[:500], R[:500], N[:500], T[:500], Y[:500], X[500:], R[500:], N[500:], T[500:], Y[500:],
                    hidden=32, epochs=150)
    m = calibrate(m, X[500:], R[500:], N[500:], Y[500:])
    P, That, _ = m.predict(X[500:])
    acc = ((P[:, 0] > 0.5) == (easy[500:] > 0.5)).mean()
    assert acc > 0.9
    assert np.allclose(That.mean(0), T[0], rtol=0.15)
    w = {k: v.numpy() for k, v in m.state_dict().items()}
    from smallcoder.heads_np import numpy_forward as phone_forward  # torch-free copy used on the phone
    assert phone_forward is numpy_forward
    Pn, Tn = numpy_forward(w, X[500:])
    assert np.allclose(Pn, P, atol=1e-4) and np.allclose(Tn, That, rtol=1e-3)
    # utility choice: cheap arm for easy problems, big arm for hard ones
    from smallcoder.policy import choose as ch
    picks = ch(P, That, lam=0.05)
    assert (picks[easy[500:] > 0.5] == 0).mean() > 0.9 and (picks[easy[500:] < 0.5] == 2).mean() > 0.9


# ---------------------------------------------------------------- draft agreement baseline
def test_draft_agreement():
    table = build_label_table(_rows())
    rows = [{"pid": "p1", "condition": "base", "budget": 0, "sample": s, "public_verdict": "AC", "verdict": "AC",
             "total_tokens": 300} for s in (0, 1)]
    rows += [{"pid": "p2", "condition": "base", "budget": 0, "sample": s, "public_verdict": "WA", "verdict": "WA",
              "total_tokens": 300} for s in (0, 1)]
    acc, tok, n = draft_agreement(rows, ["p1", "p2"], 4096, table)
    assert n == 2 and acc == 1.0 and tok == pytest.approx((600 + 600 + 4396) / 2)


# ---------------------------------------------------------------- rag
def test_rag_excludes_self():
    pytest.importorskip("rank_bm25")
    from smallcoder.rag import Retriever, make_doc
    a = Problem(pid="codeforces/1/A", source="codeforces", statement="find the shortest path in a weighted graph "
                "between city one and city n using roads", tests=[], editorial="run dijkstra from node one")
    b = Problem(pid="taco/9", source="taco", statement="count the shortest paths in a weighted road graph from s",
                tests=[], solutions=["import heapq\nn=int(input())\nprint(n)"])
    c = Problem(pid="taco/10", source="taco", statement="reverse a string and print it", tests=[],
                solutions=["print(input()[::-1])"])
    extra = [Problem(pid=f"taco/{i}", source="taco", statement=s, tests=[], solutions=["print(sum(map(int,input().split())))"])
             for i, s in [(11, "add two integers"), (12, "sort an array of numbers"), (13, "print the maximum value")]]
    docs = [make_doc(x) for x in (a, b, c, *extra)]  # enough documents for positive BM25 idf
    r = Retriever(docs)
    hits = r.search(a, draft="dijkstra shortest path", k=1)
    assert r.docs[hits[0][0]]["pid"] == "taco/9"   # never returns problem a itself
