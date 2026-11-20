"""Mine on-policy repair SFT pairs from label-generation rows on the TRAIN pool.

A pair = (failed attempt + its verdict feedback) -> a passing program the SAME model wrote for the same
problem. Verdicts are balanced by capping each verdict class. Output: chat-format jsonl for sft_train.py.

  python scripts/repair_mine_pairs.py results/labels/vibethinker*.jsonl --problems data/train/labels_pool.jsonl \
      --config configs/eval_vibethinker.yaml --out data/repair/sft.jsonl
"""
import argparse
import json
import random
from collections import Counter, defaultdict

import yaml

from smallcoder.gen import ModelSpec
from smallcoder.io import read_rows
from smallcoder.problems import load_problems
from smallcoder.repair import repair_messages

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--problems", required=True)
ap.add_argument("--config", required=True, help="for think tags of the model being trained")
ap.add_argument("--out", required=True)
ap.add_argument("--per-verdict", type=int, default=40000)
ap.add_argument("--per-problem", type=int, default=4)
ap.add_argument("--allow-hidden", action="store_true", help="also use hidden-test failures as feedback")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
spec = ModelSpec(**yaml.safe_load(open(a.config))["model"])
rng = random.Random(a.seed)
probs = {p.pid: p for p in load_problems(a.problems)}

ok, bad = defaultdict(list), defaultdict(list)
for r in read_rows(a.results):
    if r["pid"] not in probs:
        continue
    if r["verdict"] == "AC" and r.get("code"):
        ok[r["pid"]].append(r)
    elif r.get("public_verdict") not in (None, "AC"):           # failure visible on the device
        bad[r["pid"]].append((r, r["public_verdict"], r.get("public_fail")))
    elif a.allow_hidden and r["verdict"] != "AC":                 # passes samples, fails hidden tests
        bad[r["pid"]].append((r, r["verdict"], r.get("first_fail")))

cands = []
for pid, fails in bad.items():
    if not ok[pid]:
        continue
    rng.shuffle(fails)
    seen_code = set()
    for r, v, fail in fails:
        if r.get("code") in seen_code:
            continue
        seen_code.add(r.get("code"))
        tgt = rng.choice(ok[pid])
        # target answer with an empty thinking block: repair is trained and run in no-think mode
        answer = ("\n" + spec.think_end + "\n\n" if spec.thinking else "") + f"```python\n{tgt['code']}\n```"
        cands.append((v, {"messages": repair_messages(probs[pid], r.get("code"), v, fail) +
                          [{"role": "assistant", "content": answer}], "pid": pid, "verdict": v}))
        if len(seen_code) >= a.per_problem:
            break

rng.shuffle(cands)
count, out = Counter(), []
for v, ex in cands:
    if count[v] < a.per_verdict:
        count[v] += 1
        out.append(ex)
with open(a.out, "w", encoding="utf-8") as fo:
    for ex in out:
        fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"{len(out)} pairs from {len(bad)} problems; per verdict: {dict(count)}")
