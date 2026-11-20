"""Train + calibrate the controller heads on the TRAIN label table.

  python scripts/controller_train.py --features data/features/vibethinker_train.npz \
      --labels data/labels/vibethinker_train.json --problems data/train/labels_pool.jsonl \
      --out models/controller_vibethinker
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from smallcoder.controller import calibrate, export_numpy, train_heads
from smallcoder.labels import FAMILIES, complete_arms, load_table, tag_families, to_arrays
from smallcoder.metrics import ece_binary
from smallcoder.problems import load_problems

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--problems", required=True, help="for topic tags")
ap.add_argument("--out", required=True)
ap.add_argument("--hidden", type=int, default=256, help="0 = linear probe")
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

f = np.load(a.features)
table = load_table(a.labels)
arms = complete_arms(table)
tags = {p.pid: tag_families(p.tags) for p in load_problems(a.problems)}
keep = [i for i, pid in enumerate(f["pids"]) if pid in table and all(x in table[pid]["rate"] for x in arms)]
pids = [str(f["pids"][i]) for i in keep]
X = f["X"][keep]
R, N, T = to_arrays(table, pids, arms)
Y = np.array([[fam in tags.get(pid, []) for fam in FAMILIES] for pid in pids], dtype=np.float32)

# deterministic split by problem id hash (stable across reruns and machines)
is_val = np.array([int(hashlib.md5(p.encode()).hexdigest(), 16) % 1000 < a.val_frac * 1000 for p in pids])
tr, va = ~is_val, is_val
model = train_heads(X[tr], R[tr], N[tr], T[tr], Y[tr], X[va], R[va], N[va], T[va], Y[va],
                    hidden=a.hidden, seed=a.seed)
P_before, _, _ = model.predict(X[va])
model = calibrate(model, X[va], R[va], N[va], Y[va])
P_after, T_hat, _ = model.predict(X[va])

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
meta = {"arms": arms, "families": FAMILIES, "hidden": a.hidden, "d_in": int(X.shape[1]),
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "val_ece_before": ece_binary(P_before.ravel(), R[va].ravel()),
        "val_ece_after": ece_binary(P_after.ravel(), R[va].ravel()),
        "t_pass": float(model.t_pass), "val_token_mape": float(np.mean(np.abs(T_hat - T[va]) / np.maximum(T[va], 1)))}
torch.save({"state_dict": model.state_dict(), "meta": meta}, out / "controller.pt")
export_numpy(model, str(out / "controller.npz"), meta)
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
