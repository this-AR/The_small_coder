"""Controller arms that need no trained heads.

Arm B, option logits: the coder reads a short menu inside its open thinking block and we read the
next-token probabilities of the option letters (one forward pass; on a phone the KV cache is shared).
Arm D, draft agreement: a DART-inspired heuristic using two cheap no-think drafts and the public tests.
  It is NOT the official DART method; for the paper, run https://github.com/js-lee-AI/DART as the baseline.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

MENU = ("\nBefore solving, I estimate how much thinking this problem needs.\n"
        "A) none, I can write the code directly\nB) about 1000 tokens\nC) about 2000 tokens\n"
        "D) about 4000 tokens\nE) about 8000 tokens\nMy choice:")  # no trailing space: next token is " X"
LETTERS = ["A", "B", "C", "D", "E"]
LETTER_BUDGETS = [0, 1024, 2048, 4096, 8192]


def option_ids(tok, letters=LETTERS) -> list[int]:
    """First token of " X" for each letter: what a BPE tokenizer emits right after 'My choice:'."""
    ids = []
    for L in letters:
        enc = tok(" " + L, add_special_tokens=False)["input_ids"]
        ids.append(enc[0])
    assert len(set(ids)) == len(ids), "option letters must map to distinct first tokens"
    return ids


def option_probs(model_path: str, prompts_opened: list[str], batch_size: int = 16, device: str = "cuda"):
    """P(letter) per prompt (renormalised over the options). prompts_opened end inside <think>."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
    ids = option_ids(tok)
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts_opened), batch_size):
            batch = [p + MENU for p in prompts_opened[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            logits = model(**enc).logits[:, -1, :].float()
            out.append(torch.softmax(logits[:, ids], dim=-1).cpu().numpy())
    return np.concatenate(out)


def option_policy_inputs(probs: np.ndarray, arms: list[str], R_shape) -> tuple[np.ndarray, np.ndarray]:
    """Map letter probabilities onto the base@budget arms so the generic policy code can sweep lambda.
    Arms without a letter (e.g. rag@*) get probability 0 (this arm never retrieves)."""
    P = np.zeros(R_shape, dtype=np.float32)
    T = np.zeros(R_shape, dtype=np.float32)
    for j, a in enumerate(arms):
        cond, b = a.split("@")
        if cond == "base" and int(b) in LETTER_BUDGETS:
            P[:, j] = probs[:, LETTER_BUDGETS.index(int(b))]
            T[:, j] = int(b) + 512  # nominal cost used only to order arms under lambda
        else:
            T[:, j] = 1e9
    return P, T


def draft_agreement(rows, pids: list[str], escalate_budget: int, table: dict, partial_budget: int | None = None):
    """Per problem: two no-think drafts (samples 0 and 1 at base@0).
    - both drafts pass the public tests -> submit draft 0           (cost: both drafts)
    - exactly one passes                -> submit the passing draft  (cost: both drafts)
    - neither passes                    -> think at escalate_budget  (cost: drafts + that arm's mean tokens)
    Returns (mean pass, mean tokens, problems scored). Uses real verdicts for drafts and expected rates
    for escalation."""
    drafts = defaultdict(dict)
    for r in rows:
        if r["condition"] == "base" and r["budget"] == 0 and r["sample"] in (0, 1):
            drafts[r["pid"]][r["sample"]] = r
    acc, tok = [], []
    for pid in pids:
        d = drafts.get(pid, {})
        if len(d) < 2:
            continue
        cost = d[0]["total_tokens"] + d[1]["total_tokens"]
        pub = [d[s].get("public_verdict") == "AC" for s in (0, 1)]
        if any(pub):
            s = 0 if pub[0] else 1
            acc.append(float(d[s]["verdict"] == "AC"))
            tok.append(cost)
        else:
            arm = f"base@{escalate_budget}"
            acc.append(table[pid]["rate"][arm])
            tok.append(cost + table[pid]["tokens"][arm])
    return float(np.mean(acc)), float(np.mean(tok)), len(acc)
