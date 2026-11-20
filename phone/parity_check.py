"""Check that llama.cpp's pooled-LAST embedding equals the HF post-norm last hidden state the probe was
trained on. Run on the cluster or a desktop BEFORE trusting the probe on the phone.

  python phone/parity_check.py WeiboAI/VibeThinker-1.5B models/gguf/vibe-f16.gguf phone_bundle/ data/eval/cf_test.jsonl
Expect cosine > 0.99 at f16 (lower at 4 bits is the quantization effect itself; report it).
"""
import json
import sys
from pathlib import Path

import numpy as np

from pipeline import last_state, make_embedder  # the exact code path the phone uses
from smallcoder.controller import prefill_features
from smallcoder.problems import load_problems
from smallcoder.prompts import user_message

hf_model, gguf, bundle_dir, probs_path = sys.argv[1:5]
cfg = json.loads((Path(bundle_dir) / "bundle.json").read_text())
problems = load_problems(probs_path)[:16]
prompts = [cfg["prefix"] + user_message(p) + cfg["suffix_opened"] for p in problems]

emb = make_embedder(gguf, verbose=False)
L = np.array([last_state(emb, x) for x in prompts])
# HF side: the bundle prefix had the textual BOS removed (llama.cpp adds BOS), so put it back for HF.
from transformers import AutoTokenizer  # noqa: E402
bos = AutoTokenizer.from_pretrained(hf_model).bos_token or ""
H = prefill_features(hf_model, [bos + x for x in prompts])
cos = (L * H).sum(1) / (np.linalg.norm(L, axis=1) * np.linalg.norm(H, axis=1))
print("cosine per prompt:", np.round(cos, 4))
print("mean cosine:", float(cos.mean()), "| relative L2:", float(np.mean(np.linalg.norm(L - H, axis=1) / np.linalg.norm(H, axis=1))))
