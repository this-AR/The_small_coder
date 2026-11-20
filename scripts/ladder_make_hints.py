"""Build every ladder rung for problems that have an editorial; write contexts + a paired problem subset.

  python scripts/ladder_make_hints.py configs/ladder_hints.yaml
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from smallcoder.gen import ModelSpec, load_tokenizer, make_backend
from smallcoder.io import append_jsonl
from smallcoder.ladder import (PSEUDO_PROMPT, SKETCH_PROMPT, TECHNIQUE_PROMPT, TagPredictor, clean_teacher_output,
                               gold_codes, leak_reason, tags_context)
from smallcoder.problems import load_open_r1_codeforces, load_problems, save_problems

ap = argparse.ArgumentParser()
ap.add_argument("config")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))

problems = [p for f in cfg["problems"] for p in load_problems(f) if p.editorial and len(p.editorial) > 50]
print(f"{len(problems)} problems with editorials")

# pred_tags rung: trained on the Codeforces TRAIN split only.
pred = TagPredictor(top_k=cfg.get("pred_top_k", 2)).fit(list(load_open_r1_codeforces("train", "default")))
predicted = dict(zip([p.pid for p in problems], pred.predict([p.statement for p in problems])))

spec = ModelSpec(**cfg["teacher"])
tok = load_tokenizer(spec)
backend = make_backend(spec)
tkw = cfg.get("template_kwargs", {})  # e.g. {enable_thinking: false} for Qwen3 teachers


def ask(template, n_tries):
    prompts = [tok.apply_chat_template([{"role": "user", "content": template.format(
        statement=p.statement, editorial=p.editorial)}], add_generation_prompt=True, tokenize=False, **tkw)
        for p in problems]
    outs = backend.generate(prompts, n_tries, cfg.get("max_tokens", 1024), cfg.get("temperature", 0.3), 0.95,
                            seed=cfg.get("seed"))
    return [[clean_teacher_output(c.text, spec.think_end) for c in samples] for samples in outs]


rejects = Counter()
generated = {}
for rung, template in [("technique", TECHNIQUE_PROMPT), ("sketch", SKETCH_PROMPT), ("pseudo", PSEUDO_PROMPT)]:
    tries = ask(template, cfg.get("n_tries", 3))
    for p, cands in zip(problems, tries):
        ok = None
        for c in cands:  # first candidate that passes the leakage filter
            why = leak_reason(c, gold_codes(p))
            if why is None and c:
                ok = c
                break
            rejects[f"{rung}:{why or 'empty'}"] += 1
        generated[(p.pid, rung)] = ok

rows, subset = [], []
for p in problems:
    ctx = {
        "pred_tags": tags_context(predicted[p.pid]),
        "oracle_tags": tags_context(p.tags),
        "technique": generated[(p.pid, "technique")],
        "sketch": generated[(p.pid, "sketch")],
        "editorial": p.editorial,
        "editorial_pseudo": (p.editorial + "\n\nPseudocode:\n" + generated[(p.pid, "pseudo")])
        if generated[(p.pid, "pseudo")] else None,
    }
    if all(ctx.values()):  # paired design: keep only problems where every rung exists
        rows.append({"pid": p.pid, "contexts": ctx, "hint_tokens": {k: len(tok(v)["input_ids"]) for k, v in ctx.items()}})
        subset.append(p)

out = Path(cfg["out_dir"])
(out / "contexts.jsonl").unlink(missing_ok=True)
append_jsonl(out / "contexts.jsonl", rows)
save_problems(subset, out / "problems.jsonl")
(out / "rejects.json").write_text(json.dumps(dict(rejects), indent=1))
print(f"{len(subset)} problems have all rungs -> {out}; rejects: {dict(rejects)}")
