"""Track 0 harness (also used by the ladder, label generation, RAG and Track C runs).

  python scripts/run_eval.py configs/eval_vibethinker.yaml [--shard 0 --num-shards 8]

Writes one row per (problem, condition, sample, budget). Re-running resumes: complete
(pid, condition) pairs already in the output file are skipped, partial ones are regenerated.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import yaml

from smallcoder.extract import extract_code
from smallcoder.gen import ModelSpec, generate_budgeted, load_tokenizer, make_backend
from smallcoder.io import append_jsonl, load_contexts, read_jsonl
from smallcoder.judge import judge_many
from smallcoder.problems import load_problems
from smallcoder.prompts import chat_prompt

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))

spec = ModelSpec(**cfg["model"])
tok = load_tokenizer(spec)
backend = make_backend(spec)
S, J = cfg["sampling"], cfg.get("judge", {})
out_path = cfg["out"] if a.num_shards == 1 else cfg["out"].replace(".jsonl", f".shard{a.shard}.jsonl")

problems = [p for i, p in enumerate(load_problems(cfg["problems"])) if i % a.num_shards == a.shard]
conditions = cfg.get("conditions") or [{"name": "base"}]
# A pair counts as done only when all its rows are present (a preempted write can leave a partial pair;
# its rows are dropped and regenerated so the file never holds mixed sample counts).
expected = S["n"] * (len(set(cfg["budgets"])) if spec.thinking else 1)
have = Counter((r["pid"], r["condition"]) for r in read_jsonl(out_path))
done = {k for k, v in have.items() if v >= expected}
if len(done) < len(have):
    kept = [r for r in read_jsonl(out_path) if (r["pid"], r["condition"]) in done]
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
    print(f"dropped {len(have) - len(done)} partial (pid, condition) pairs")
ctx = {c["name"]: load_contexts(c.get("contexts"), c.get("context_key")) for c in conditions}

jobs = [(p, c) for c in conditions for p in problems if (p.pid, c["name"]) not in done]
# A condition with a contexts file applies only to problems that have a context (e.g. editorial rungs).
jobs = [(p, c) for p, c in jobs if not c.get("contexts") or p.pid in ctx[c["name"]]]
print(f"{len(jobs)} (problem, condition) jobs on shard {a.shard}; {len(done)} already done")


def judge_and_write(batch, samples):
    rows, todo = [], []
    for s in samples:
        p, c = batch[s.prompt_idx]
        code = extract_code(s.answer)  # s.answer is already the text after the end-of-thinking tag
        todo.append((code, p))
        rows.append({"pid": p.pid, "model": spec.name, "condition": c["name"], "sample": s.sample,
                     "budget": s.budget, "think_tokens": s.think_tokens, "answer_tokens": s.answer_tokens,
                     "total_tokens": s.total_tokens, "forced": s.forced, "answer_finished": s.answer_finished,
                     "has_code": code is not None, "code": code,
                     **({"answer": s.answer} if cfg.get("store_answer") else {}),
                     **({"thinking": s.thinking} if cfg.get("store_thinking") else {})})
    kw = dict(workers=J.get("workers", 32), time_multiplier=J.get("time_multiplier", 1.0),
              mem_mb=J.get("mem_mb"))
    full = judge_many(todo, stop_on_fail=J.get("stop_on_fail", True), **kw)
    for r, res in zip(rows, full):
        r.update(verdict=res.verdict, passed=res.passed, total=res.total, first_fail=res.first_fail)
    if J.get("judge_public", True):  # what an on-device sample-test filter would see
        pub = judge_many([(code, p, p.public_tests) for code, p in todo], **kw)
        for r, res in zip(rows, pub):
            r["public_verdict"] = res.verdict if res.total else None
            r["public_fail"] = res.first_fail
    append_jsonl(out_path, rows)


B = cfg.get("batch_problems", 64)
pending = None
with ThreadPoolExecutor(max_workers=1) as judge_pool:  # judge batch k while the GPU generates batch k+1
    for start in range(0, len(jobs), B):
        batch = jobs[start:start + B]
        prompts = [chat_prompt(tok, p, ctx[c["name"]].get(p.pid), c.get("style", "default"),
                               cfg.get("system"), c.get("context_header", "Guidance")) for p, c in batch]
        t0 = time.time()
        samples = generate_budgeted(backend, tok, spec, prompts, cfg["budgets"], S["n"], cfg["answer_tokens"],
                                    S["temperature"], S["top_p"], S.get("seed"),
                                    keep_thinking=cfg.get("store_thinking", False))
        print(f"batch {start // B}: generated {len(samples)} samples in {time.time() - t0:.0f}s")
        if pending:
            pending.result()
        pending = judge_pool.submit(judge_and_write, batch, samples)
    if pending:
        pending.result()
print("done ->", out_path)
