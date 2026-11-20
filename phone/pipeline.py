"""On-device System-One / System-Two solver (llama-cpp-python; runs in Termux on Android or on a desktop).

  solver = OnDeviceSolver("vibe-q4_k_m.gguf", "phone_bundle/", rag_index="rag_index/")
  result = solver.solve(problem, lam=0.05)

Memory note: two Llama objects map the SAME GGUF file (use_mmap=True), so weight pages are shared by
the OS; only the KV caches are separate. One handles generation, the other returns the post-norm
last-token hidden state (pooling LAST) that the controller heads read.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from smallcoder.extract import extract_code
from smallcoder.heads_np import numpy_forward
from smallcoder.judge import judge
from smallcoder.problems import Problem
from smallcoder.prompts import user_message
from smallcoder.rag import DRAFT_PROMPT
from smallcoder.repair import repair_messages


EMB_CTX = 4096


def make_embedder(gguf: str, **kw):
    """Llama object returning the post-norm last-token state. Pooled embeddings need the whole prompt in
    one micro-batch, so n_batch = n_ubatch = n_ctx (the 512 default rejects ~1K-token prompts)."""
    from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama
    return Llama(model_path=gguf, n_ctx=EMB_CTX, n_batch=EMB_CTX, n_ubatch=EMB_CTX, embedding=True,
                 pooling_type=LLAMA_POOLING_TYPE_LAST, **kw)


def last_state(emb, text: str) -> np.ndarray:
    """embed() with chat-template tokens parsed as SPECIAL tokens, as the HF tokenizer does.
    Some llama-cpp-python versions tokenize embed() input with special=False, which would split
    <|im_start|> etc. into plain text and break parity with the trained probe."""
    orig = emb.tokenize
    emb.tokenize = lambda b, add_bos=True, special=False: orig(b, add_bos=add_bos, special=True)
    try:
        h = np.asarray(emb.embed(text, normalize=False), dtype=np.float32)
    finally:
        del emb.tokenize  # drop the instance override, restoring the class method
    return h.reshape(-1, emb.n_embd())[-1]  # some versions return per-token rows; take the last token


class OnDeviceSolver:
    def __init__(self, gguf: str, bundle_dir: str, rag_index: str | None = None, n_ctx: int = 12288,
                 n_threads: int | None = None, seed: int = 0):
        from llama_cpp import Llama

        b = Path(bundle_dir)
        self.cfg = json.loads((b / "bundle.json").read_text())
        w = np.load(b / "controller.npz")
        self.w = {k: w[k] for k in w.files if k != "meta"}
        self.meta = json.loads(str(w["meta"]))
        self.arms = self.meta["arms"]
        kw = dict(model_path=gguf, use_mmap=True, verbose=False, seed=seed, **({"n_threads": n_threads} if n_threads else {}))
        self.gen = Llama(n_ctx=n_ctx, **kw)
        self.emb = make_embedder(gguf, **kw)
        self.ret = None
        if rag_index:
            from smallcoder.rag import Retriever
            self.ret = Retriever.load(rag_index)
            self.ret.emb = None  # shortcut: BM25-only on the phone; add the 0.6B embedder GGUF if RAM allows
        self.seed = seed

    # ---------------------------------------------------------------- primitives
    def _prompt(self, content: str) -> str:
        return self.cfg["prefix"] + content + self.cfg["suffix_opened"]

    def _complete(self, prompt, max_tokens, stop=None, temperature=0.6, top_p=0.95):
        t0 = time.perf_counter()
        r = self.gen.create_completion(prompt, max_tokens=max_tokens, stop=stop or [], temperature=temperature,
                                       top_p=top_p, seed=self.seed)
        self.seed += 1
        ch = r["choices"][0]
        return ch["text"], r["usage"]["completion_tokens"], ch["finish_reason"], time.perf_counter() - t0

    def decide(self, opened_prompt: str, lam: float):
        t0 = time.perf_counter()
        h = last_state(self.emb, opened_prompt)
        p, t = numpy_forward(self.w, h[None, :])
        p, t = p[0], t[0]
        allowed = [j for j, a in enumerate(self.arms) if self.ret or not a.startswith("rag@")]
        util = {j: p[j] - lam * t[j] / 1000 for j in allowed}
        j = max(util, key=util.get)
        return self.arms[j], float(p[j]), {"p": p.tolist(), "t": t.tolist(), "probe_s": time.perf_counter() - t0}

    def think_then_answer(self, opened: str, budget: int, answer_tokens: int):
        te, stats = self.cfg["think_end"], {"think_tokens": 0, "answer_tokens": 0, "seconds": 0.0}
        if budget == 0:
            cont = opened + "\n" + te + "\n\n"
        else:
            text, n, finish, dt = self._complete(opened, budget, stop=[te])
            stats.update(think_tokens=n, seconds=dt)
            cont = opened + text + (self.cfg["force_text"] if finish == "length" else "") + te + "\n\n"
        ans, n, _, dt = self._complete(cont, answer_tokens)
        stats["answer_tokens"], stats["seconds"] = n, stats["seconds"] + dt
        return ans, stats

    # ---------------------------------------------------------------- full solve
    def solve(self, p: Problem, lam: float = 0.05, answer_tokens: int = 2048, repair_rounds: int = 2,
              time_multiplier: float = 1.0) -> dict:
        log = {"pid": p.pid, "steps": []}
        t_start = time.perf_counter()
        base_content = user_message(p)
        arm, conf, dinfo = self.decide(self._prompt(base_content), lam)
        cond, budget = arm.split("@")[0], int(arm.split("@")[1])
        log.update(arm=arm, confidence=conf, decide=dinfo)

        content = base_content
        if cond == "rag" and self.ret:
            dprompt = self._prompt(DRAFT_PROMPT.format(statement=p.statement)) + "\n" + self.cfg["think_end"] + "\n\n"
            draft, n, _, dt = self._complete(dprompt, 256, temperature=0.0)
            hits = self.ret.search(p, draft, k=2)
            content = user_message(p, self.ret.context(hits))
            log["steps"].append({"step": "draft+retrieve", "tokens": n, "seconds": dt})

        ans, st = self.think_then_answer(self._prompt(content), budget, answer_tokens)
        code = extract_code(ans)
        pub = judge(code, p, p.public_tests, time_multiplier=time_multiplier)
        log["steps"].append({"step": "solve", **st, "public": pub.verdict})

        for r in range(repair_rounds):
            if pub.verdict == "AC" or not p.public_tests:
                break
            msg = repair_messages(p, code, pub.verdict, pub.first_fail)[0]["content"]
            ans, st = self.think_then_answer(self._prompt(msg), 0, answer_tokens)
            code = extract_code(ans) or code
            pub = judge(code, p, p.public_tests, time_multiplier=time_multiplier)
            log["steps"].append({"step": f"repair{r + 1}", **st, "public": pub.verdict})

        log.update(code=code, public_verdict=pub.verdict, seconds=time.perf_counter() - t_start,
                   tokens=sum(s.get("think_tokens", 0) + s.get("answer_tokens", 0) + s.get("tokens", 0)
                              for s in log["steps"]))
        return log
