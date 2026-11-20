"""pass@k, paired statistics and calibration metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of Chen et al. (2021): 1 - C(n-c, k) / C(n, k)."""
    if n < k:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def bootstrap_ci(per_problem: np.ndarray, iters: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile CI, resampling problems."""
    x = np.asarray(per_problem, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(iters, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 10000, seed: int = 0) -> dict:
    """a, b: per-problem scores on the SAME problems in the same order. Returns mean(b-a), CI, p (two-sided)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired comparison needs identical problem sets"
    d = b - a
    mean, lo, hi = bootstrap_ci(d, iters=iters, seed=seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boot = d[idx].mean(axis=1)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"delta": mean, "ci_low": lo, "ci_high": hi, "p_boot": float(min(1.0, p))}


def mcnemar_exact(a_pass: np.ndarray, b_pass: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes (e.g. greedy/sample-0 pass per problem)."""
    a, b = np.asarray(a_pass, bool), np.asarray(b_pass, bool)
    n01 = int((~a & b).sum())   # a fails, b passes
    n10 = int((a & ~b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(n01, n, 0.5).pvalue
    return {"b_only": n01, "a_only": n10, "p": float(p)}


def brier(probs: np.ndarray, targets: np.ndarray) -> float:
    """Multi-class Brier score; probs (N, K), targets (N, K) soft or one-hot."""
    return float(np.mean(np.sum((np.asarray(probs) - np.asarray(targets)) ** 2, axis=-1)))


def ece_binary(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error for a probability of a binary event (y may be soft in [0,1])."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]), "count": int((which == b).sum()),
             "mean_pred": float(p[which == b].mean()) if (which == b).any() else None,
             "mean_obs": float(y[which == b].mean()) if (which == b).any() else None} for b in range(bins)]
