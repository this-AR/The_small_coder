"""Repair vs resample at EQUAL generated tokens, as a phone would run it.

Per test problem, attempts proceed until one passes the public tests or the token cap is hit; the
submitted program is the first public-passing attempt (else the last). Success = it passes ALL tests.
- resample: reuses the harness's own samples 0..n-1 at `--budget` (already generated, real tokens)
- repair:   sample 0, then up to --rounds repair generations conditioned on the public-test verdict

  python scripts/repair_eval.py configs/repair_eval.yaml
"""
import argparse
from collections import defaultdict

import numpy as np
import yaml

from smallcoder.extract import extract_code
from smallcoder.gen import ModelSpec, generate_budgeted, load_tokenizer, make_backend
from smallcoder.io import append_jsonl, read_rows
from smallcoder.judge import judge_many
from smallcoder.metrics import paired_bootstrap
from smallcoder.problems import load_problems
from smallcoder.repair import repair_messages

ap = argparse.ArgumentParser()
ap.add_argument("config")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
spec = ModelSpec(**cfg["model"])
probs = {p.pid: p for p in load_problems(cfg["problems"])}
B, caps = cfg["budget"], cfg["token_caps"]

by = defaultdict(dict)
for r in read_rows(cfg["rows"]):
    if r["condition"] == cfg.get("condition", "base") and r["budget"] == B and r["pid"] in probs:
        by[r["pid"]][r["sample"]] = r
pids = sorted(p for p in by if 0 in by[p])


def run_sequence(attempts, cap):
    """attempts: list of (tokens, public_ok, full_ok). Returns success under the token cap."""
    spent, last = 0, None
    for t, pub, full in attempts:
        if spent + t > cap and last is not None:
            break
        spent += t
        last = full
        if pub:
            return full, spent
    return bool(last), spent


# ---- resample strategy (no new generation)
resample = {pid: [(r["total_tokens"], r.get("public_verdict") in ("AC", None), r["verdict"] == "AC")
                  for _, r in sorted(by[pid].items())] for pid in pids}

# ---- repair strategy: generate repair rounds for problems whose first attempt fails public tests
tok = load_tokenizer(spec)
backend = make_backend(spec)
state = {pid: {"code": by[pid][0].get("code"), "verdict": by[pid][0].get("public_verdict") or "WA",
               "fail": by[pid][0].get("public_fail")} for pid in pids}
repair = {pid: [resample[pid][0]] for pid in pids}
active = [pid for pid in pids if not resample[pid][0][1]]
log = []
for rnd in range(cfg.get("rounds", 3)):
    if not active:
        break
    prompts = [tok.apply_chat_template(repair_messages(probs[pid], state[pid]["code"], state[pid]["verdict"],
                                                        state[pid]["fail"]), add_generation_prompt=True, tokenize=False)
               for pid in active]
    gens = generate_budgeted(backend, tok, spec, prompts, [cfg.get("repair_budget", 0)], 1, cfg["answer_tokens"],
                             cfg.get("temperature", 0.6), cfg.get("top_p", 0.95), seed=cfg.get("seed"))
    gens = sorted(gens, key=lambda g: g.prompt_idx)  # align with `active`
    codes = [extract_code(g.answer) for g in gens]
    pub = judge_many([(c, probs[pid], probs[pid].public_tests) for c, pid in zip(codes, active)],
                     workers=cfg.get("workers", 32), time_multiplier=cfg.get("time_multiplier", 1.0))
    full = judge_many([(c, probs[pid]) for c, pid in zip(codes, active)], workers=cfg.get("workers", 32),
                      time_multiplier=cfg.get("time_multiplier", 1.0))
    nxt = []
    for pid, g, c, pr, fr in zip(active, gens, codes, pub, full):
        pub_ok = pr.verdict == "AC"
        repair[pid].append((g.total_tokens, pub_ok, fr.verdict == "AC"))
        log.append({"pid": pid, "round": rnd + 1, "tokens": g.total_tokens, "public": pr.verdict,
                    "full": fr.verdict, "code": c})
        if not pub_ok:
            state[pid] = {"code": c, "verdict": pr.verdict, "fail": pr.first_fail}
            nxt.append(pid)
    active = nxt
    print(f"round {rnd + 1}: {len(active)} still failing public tests")

append_jsonl(cfg["out"].replace(".jsonl", ".log.jsonl"), log)
rows = []
for cap in caps:
    rs = np.array([run_sequence(resample[p], cap)[0] for p in pids], float)
    rp = np.array([run_sequence(repair[p], cap)[0] for p in pids], float)
    d = paired_bootstrap(rs, rp)
    rows.append({"token_cap": cap, "resample": rs.mean(), "repair": rp.mean(), "delta": d["delta"],
                 "ci_low": d["ci_low"], "ci_high": d["ci_high"], "p": d["p_boot"], "n": len(pids)})
    print(rows[-1])
# fix rate per first-attempt public verdict
fix = defaultdict(list)
for p in pids:
    if not resample[p][0][1]:
        fix[by[p][0].get("public_verdict") or "WA"].append(any(full for _, _, full in repair[p][1:]))
print("repair fix rate by first public verdict:", {v: (float(np.mean(x)), len(x)) for v, x in fix.items()})
append_jsonl(cfg["out"], rows)
