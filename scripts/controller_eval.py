"""Figure 3/4 data: every controller arm on the TEST label table (exact offline expectation).

  python scripts/controller_eval.py --model models/controller_vibethinker \
      --features data/features/vibethinker_lcb.npz --labels data/labels/vibethinker_lcb.json \
      --rows results/track0/vibethinker_lcb_v5.jsonl --out results/controller/vibethinker_lcb
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from smallcoder.controller import ControllerHeads
from smallcoder.io import read_rows
from smallcoder.labels import load_table, to_arrays
from smallcoder.metrics import ece_binary, reliability_table
from smallcoder.policy import (area_under, curve, fixed_points, oracle_curve, pass_at_matched_tokens, risk_coverage)
from smallcoder.selfroute import draft_agreement, option_policy_inputs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--rows", nargs="*", default=[], help="harness rows (for the draft-agreement baseline)")
ap.add_argument("--t-max", type=float, default=8000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ck = torch.load(Path(a.model) / "controller.pt", weights_only=False)
meta = ck["meta"]
model = ControllerHeads(meta["d_in"], len(meta["arms"]), len(meta["families"]), meta["hidden"])
model.load_state_dict(ck["state_dict"])

f = np.load(a.features)
table = load_table(a.labels)
arms = meta["arms"]
keep = [i for i, pid in enumerate(f["pids"]) if pid in table and all(x in table[pid]["rate"] for x in arms)]
pids = [str(f["pids"][i]) for i in keep]
R, N, T = to_arrays(table, pids, arms)
P_hat, T_hat, _ = model.predict(f["X"][keep])

curves = {"controller": curve(P_hat, T_hat, R, T), "oracle": oracle_curve(R, T)}
# controller with true token costs isolates the value of the pass-probability predictions
curves["controller_true_tokens"] = curve(P_hat, T, R, T)
if "option_probs" in f.files:
    Po, To = option_policy_inputs(f["option_probs"][keep], arms, R.shape)
    curves["option_logits"] = curve(np.log(Po + 1e-9), To, R, T, lams=np.geomspace(1e-3, 1e3, 60))

summary = []
for name, pts in curves.items():
    xy = [(t, p) for _, t, p in pts]
    summary.append({"arm": name, "auc@t_max": area_under(xy, a.t_max),
                    **{f"pass@{k}tok": pass_at_matched_tokens(xy, k) for k in (1000, 2000, 4000, 8000)}})
fixed = fixed_points(R, T, arms)
summary.append({"arm": "fixed_best_single", "auc@t_max": area_under([(t, p) for _, t, p in fixed], a.t_max),
                **{f"pass@{k}tok": pass_at_matched_tokens([(t, p) for _, t, p in fixed], k)
                   for k in (1000, 2000, 4000, 8000)}})
rows = read_rows(a.rows)
if rows:
    for b in sorted({int(x.split("@")[1]) for x in arms if x.startswith("base@") and x != "base@0"}):
        acc, tok, n = draft_agreement(rows, pids, b, table)
        summary.append({"arm": f"draft_agreement->{b}", "mean_tokens": tok, "pass@1": acc, "n": n})

# oracle-gap recovered at matched tokens: (controller - best fixed) / (oracle - best fixed)
s = {r["arm"]: r for r in summary}
for k in (1000, 2000, 4000, 8000):
    lo, hi = s["fixed_best_single"][f"pass@{k}tok"], s["oracle"][f"pass@{k}tok"]
    s["controller"][f"oracle_gap_recovered@{k}tok"] = ((s["controller"][f"pass@{k}tok"] - lo) / (hi - lo)
                                                       if hi > lo else None)

calib = {"ece_pass": ece_binary(P_hat.ravel(), R.ravel()),
         "reliability": reliability_table(P_hat.ravel(), R.ravel()),
         # abstention: confidence = best predicted pass prob; correctness = expected pass of chosen arm
         "risk_coverage": risk_coverage(P_hat.max(1), R[np.arange(len(R)), P_hat.argmax(1)])[::max(1, len(R) // 50)]}

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
pd.DataFrame([{"arm": n, "lambda": l, "mean_tokens": t, "pass@1": p} for n, pts in curves.items()
              for l, t, p in pts]).to_csv(out / "curves.csv", index=False)
pd.DataFrame([{"arm": n, "mean_tokens": t, "pass@1": p} for n, t, p in fixed]).to_csv(out / "fixed.csv", index=False)
(out / "calibration.json").write_text(json.dumps(calib, indent=1))
with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
    print(pd.DataFrame(summary))
print("ECE (pass, test):", round(calib["ece_pass"], 4))
