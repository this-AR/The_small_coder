"""Prefill features (arm A) and option-letter probabilities (arm B) for a problems file.
The prompt is exactly the harness's `base` prompt, opened inside <think>, so the state matches what
the coder sees right before generating.

  python scripts/controller_features.py configs/eval_vibethinker.yaml data/train/labels_pool.jsonl \
      --out data/features/vibethinker_train.npz [--option-probs]
"""
import argparse

import numpy as np
import yaml

from smallcoder.controller import prefill_features
from smallcoder.gen import ModelSpec, load_tokenizer, open_thinking
from smallcoder.problems import load_problems
from smallcoder.prompts import chat_prompt

ap = argparse.ArgumentParser()
ap.add_argument("config", help="an eval config; only its model block and `system` are used")
ap.add_argument("problems")
ap.add_argument("--out", required=True)
ap.add_argument("--option-probs", action="store_true")
ap.add_argument("--batch-size", type=int, default=16)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
spec = ModelSpec(**cfg["model"])
tok = load_tokenizer(spec)

problems = load_problems(a.problems)
prompts = [open_thinking(chat_prompt(tok, p, system=cfg.get("system")), spec) for p in problems]
X = prefill_features(spec.path, prompts, batch_size=a.batch_size)
extra = {}
if a.option_probs:
    from smallcoder.selfroute import option_probs
    extra["option_probs"] = option_probs(spec.path, prompts, batch_size=a.batch_size)
np.savez(a.out, pids=np.array([p.pid for p in problems]), X=X, **extra)
print(f"{X.shape} -> {a.out}")
