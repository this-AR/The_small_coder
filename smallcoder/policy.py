"""Offline, exact evaluation of compute-allocation policies.

Because every test problem has measured pass rates and token costs for EVERY arm, a policy that picks
one arm per problem can be scored exactly in expectation: E[pass] = rate[chosen], E[tokens] = tokens[chosen].
Sweeping the token price lambda traces each policy's pass@1-vs-tokens curve.
"""
from __future__ import annotations

import numpy as np


def choose(P_hat: np.ndarray, T_hat: np.ndarray, lam: float) -> np.ndarray:
    """argmax_a P_hat[a] - lam * T_hat[a] / 1000 per problem (lam = pass-probability per 1K tokens)."""
    return np.argmax(P_hat - lam * T_hat / 1000.0, axis=1)


def score(choice: np.ndarray, R: np.ndarray, T: np.ndarray, extra_tokens: np.ndarray | float = 0.0):
    idx = np.arange(len(choice))
    return float(R[idx, choice].mean()), float((T[idx, choice] + extra_tokens).mean())


def curve(P_hat, T_hat, R, T, lams=None, extra_tokens=0.0) -> list[tuple[float, float, float]]:
    """[(lambda, mean_tokens, pass@1)] for a predictor (P_hat, T_hat)."""
    lams = np.concatenate([[0.0], np.geomspace(1e-4, 10, 60)]) if lams is None else lams
    pts = []
    for lam in lams:
        acc, tok = score(choose(P_hat, T_hat, lam), R, T, extra_tokens)
        pts.append((float(lam), tok, acc))
    return pts


def oracle_curve(R, T, lams=None):
    return curve(R, T, R, T, lams)


def fixed_points(R, T, arms) -> list[tuple[str, float, float]]:
    return [(a, float(T[:, j].mean()), float(R[:, j].mean())) for j, a in enumerate(arms)]


def upper_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pareto frontier of (tokens, pass): increasing tokens, strictly increasing pass."""
    pts = sorted(points)
    out, best = [], -1.0
    for t, p in pts:
        if p > best:
            out.append((t, p))
            best = p
    return out


def area_under(points: list[tuple[float, float]], t_max: float) -> float:
    """Normalised area under the frontier step function on [0, t_max] (pass@1 averaged over budget)."""
    fr = [(t, p) for t, p in upper_frontier(points) if t <= t_max]
    if not fr:
        return 0.0
    area, prev_t, prev_p = 0.0, 0.0, 0.0
    for t, p in fr:
        area += (t - prev_t) * prev_p
        prev_t, prev_p = t, p
    area += (t_max - prev_t) * prev_p
    return area / t_max


def pass_at_matched_tokens(points: list[tuple[float, float]], target: float) -> float:
    """Best pass@1 reachable with mean tokens <= target."""
    ok = [p for t, p in points if t <= target]
    return max(ok) if ok else 0.0


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> list[tuple[float, float]]:
    """Selective prediction: answer only the most confident fraction; returns (coverage, accuracy)."""
    order = np.argsort(-conf)
    c = np.asarray(correct, float)[order]
    k = np.arange(1, len(c) + 1)
    return list(zip((k / len(c)).tolist(), (np.cumsum(c) / k).tolist()))
