"""Write retrieval contexts for a problems file: draft plan with the coder (no-think), then retrieve.
Also writes a `random` control (same length, random documents) for the RAG ablation.

  python scripts/rag_retrieve.py configs/eval_vibethinker.yaml data/eval/lcb_v5_2408_2502.jsonl \
      --index data/rag/index --out data/rag/lcb_contexts.jsonl
"""
import argparse
import random

import yaml

from smallcoder.decontam import ngrams, normalize
from smallcoder.gen import ModelSpec, load_tokenizer, make_backend, open_thinking
from smallcoder.io import append_jsonl
from smallcoder.problems import load_problems
from smallcoder.rag import DRAFT_PROMPT, Retriever

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("problems")
ap.add_argument("--index", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--draft-tokens", type=int, default=256)
ap.add_argument("--no-draft", action="store_true", help="ablation: statement-only query")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
spec = ModelSpec(**cfg["model"])
problems = load_problems(a.problems)
ret = Retriever.load(a.index)

drafts, dtoks = [""] * len(problems), [0] * len(problems)
if not a.no_draft:
    tok = load_tokenizer(spec)
    backend = make_backend(spec)
    prompts = [tok.apply_chat_template([{"role": "user", "content": DRAFT_PROMPT.format(statement=p.statement)}],
                                       add_generation_prompt=True, tokenize=False) for p in problems]
    if spec.thinking:  # draft without thinking: empty think block
        prompts = [open_thinking(x, spec) + "\n" + spec.think_end + "\n\n" for x in prompts]
    outs = backend.generate(prompts, 1, a.draft_tokens, 0.0, 1.0)
    drafts = [o[0].text for o in outs]
    dtoks = [len(o[0].token_ids) if o[0].token_ids is not None else len(tok(o[0].text)["input_ids"]) for o in outs]

rng = random.Random(a.seed)
rows = []
for p, d, t in zip(problems, drafts, dtoks):
    hits = ret.search(p, d, k=a.k)
    grams = ngrams(normalize(p.statement), 8)
    rand = []
    while len(rand) < a.k:  # random control must obey the same self-exclusion as real retrieval
        j = rng.randrange(len(ret.docs))
        if j not in rand and not ret._is_self(j, p, grams):
            rand.append(j)
    rows.append({"pid": p.pid, "draft_tokens": t, "draft": d, "hits": [ret.docs[j]["doc_id"] for j, _ in hits],
                 "contexts": {"rag": ret.context(hits), "rag_random": ret.context([(j, 0) for j in rand])}})
open(a.out, "w").close()
append_jsonl(a.out, rows)
print(f"{len(rows)} contexts -> {a.out}")
