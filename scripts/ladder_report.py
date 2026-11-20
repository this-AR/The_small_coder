"""Figure 1 data: pass@1 per rung x difficulty tier, and each rung's gain over `none` (paired).

  python scripts/ladder_report.py results/ladder/vibethinker.jsonl --problems data/ladder/problems.jsonl
"""
import argparse

import pandas as pd

from smallcoder.io import read_rows
from smallcoder.ladder import RUNGS
from smallcoder.metrics import bootstrap_ci, paired_bootstrap
from smallcoder.problems import load_problems

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--problems", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()

TIERS = [(0, 1200, "<=1200"), (1201, 1600, "1300-1600"), (1601, 2000, "1700-2000"), (2001, 9999, ">2000")]
rating = {p.pid: p.rating for p in load_problems(a.problems)}
df = pd.DataFrame(read_rows(a.results))
df["ok"] = df.verdict == "AC"
df["tier"] = df.pid.map(lambda pid: next((t for lo, hi, t in TIERS if rating.get(pid) and lo <= rating[pid] <= hi),
                                         "unrated"))
per = df.groupby(["model", "condition", "tier", "pid"]).ok.mean().reset_index()

out = []
for (model, tier), g in per.groupby(["model", "tier"]):
    base = g[g.condition == "none"].set_index("pid").ok
    for rung in [r for r in RUNGS if r in set(g.condition)]:
        s = g[g.condition == rung].set_index("pid").ok
        common = base.index.intersection(s.index)
        m, lo, hi = bootstrap_ci(s.loc[common].values)
        d = paired_bootstrap(base.loc[common].values, s.loc[common].values)
        out.append({"model": model, "tier": tier, "rung": rung, "n": len(common), "pass@1": 100 * m,
                    "lo": 100 * lo, "hi": 100 * hi, "gain_vs_none": 100 * d["delta"], "p": d["p_boot"]})
t = pd.DataFrame(out)
# Share of the editorial gain already reached by each rung: how much is "selection" vs "implementation".
for (model, tier), g in t.groupby(["model", "tier"]):
    top = g.loc[g.rung == "editorial", "gain_vs_none"]
    if len(top) and top.iloc[0] > 0:
        t.loc[g.index, "share_of_editorial_gain"] = g.gain_vs_none / top.iloc[0]
with pd.option_context("display.max_rows", 500, "display.width", 200, "display.float_format", "{:.2f}".format):
    print(t)
if a.out:
    t.to_csv(a.out, index=False)
