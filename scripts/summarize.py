"""Pass@k-vs-budget table with bootstrap CIs, plus paired comparisons.

  python scripts/summarize.py results/vibe_lcb*.jsonl [--problems data/eval/cf_test.jsonl --by-tier]
  python scripts/summarize.py results/*.jsonl --compare "r1distill:base" "vibethinker:base"
"""
import argparse

import numpy as np
import pandas as pd

from smallcoder.io import read_rows
from smallcoder.metrics import bootstrap_ci, mcnemar_exact, paired_bootstrap, pass_at_k
from smallcoder.problems import load_problems

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--problems", default=None, help="needed for --by-tier (Codeforces rating)")
ap.add_argument("--by-tier", action="store_true")
ap.add_argument("--compare", nargs=2, metavar=("MODEL:COND", "MODEL:COND"))
ap.add_argument("--k", type=int, nargs="+", default=[1, 4])
ap.add_argument("--out", default=None, help="CSV path")
a = ap.parse_args()

rows = read_rows(a.results)
df = pd.DataFrame(rows)
df["ok"] = df["verdict"] == "AC"
TIERS = [(0, 1200, "<=1200"), (1201, 1600, "1300-1600"), (1601, 2000, "1700-2000"), (2001, 9999, ">2000")]
if a.by_tier:
    rating = {p.pid: p.rating for p in load_problems(a.problems)}
    def tier(pid):
        r = rating.get(pid)
        return next((t for lo, hi, t in TIERS if r is not None and lo <= r <= hi), "unrated")
    df["tier"] = df["pid"].map(tier)
keys = ["model", "condition", "budget"] + (["tier"] if a.by_tier else [])

table = []
for key, g in df.groupby(keys):
    per = g.groupby("pid").agg(n=("ok", "size"), c=("ok", "sum"))
    rec = dict(zip(keys, key))
    for k in [k for k in a.k if k <= per.n.min()]:  # pass@k needs at least k samples per problem
        scores = np.array([pass_at_k(n, c, k) for n, c in zip(per.n, per.c)])
        m, lo, hi = bootstrap_ci(scores)
        rec[f"pass@{k}"], rec[f"pass@{k}_lo"], rec[f"pass@{k}_hi"] = 100 * m, 100 * lo, 100 * hi
    rec.update(problems=len(per), samples_per_problem=float(per.n.mean()),
               mean_tokens=float(g.total_tokens.mean()), truncated_pct=100 * float(g.forced.mean()),
               no_code_pct=100 * float((~g.has_code).mean()),
               answer_cut_pct=100 * float((~g.answer_finished).mean()))
    rec.update({f"{v}_pct": 100 * float((g.verdict == v).mean()) for v in ["WA", "TLE", "RE", "MLE", "CE"]})
    table.append(rec)
t = pd.DataFrame(table)
with pd.option_context("display.max_rows", 500, "display.width", 250, "display.float_format", "{:.1f}".format):
    print(t)
if a.out:
    t.to_csv(a.out, index=False)

if a.compare:
    (ma, ca), (mb, cb) = (s.split(":", 1) for s in a.compare)
    for b in sorted(df.budget.unique()):
        A = df[(df.model == ma) & (df.condition == ca) & (df.budget == b)]
        Bd = df[(df.model == mb) & (df.condition == cb) & (df.budget == b)]
        common = sorted(set(A.pid) & set(Bd.pid))
        if not common:
            continue
        pa = A.groupby("pid").ok.mean().loc[common].values
        pb = Bd.groupby("pid").ok.mean().loc[common].values
        s0a = A[A["sample"] == 0].set_index("pid").ok.reindex(common).fillna(False).values
        s0b = Bd[Bd["sample"] == 0].set_index("pid").ok.reindex(common).fillna(False).values
        pbz, mc = paired_bootstrap(pa, pb), mcnemar_exact(s0a, s0b)
        print(f"budget {b}: n={len(common)} delta pass@1 = {100 * pbz['delta']:+.1f} "
              f"[{100 * pbz['ci_low']:+.1f}, {100 * pbz['ci_high']:+.1f}] p_boot={pbz['p_boot']:.3g}; "
              f"McNemar(sample 0) p={mc['p']:.3g} ({mc['b_only']} vs {mc['a_only']})")
