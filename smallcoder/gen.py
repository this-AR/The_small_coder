"""Generation backends and the thinking-budget engine.

Backends share one call: generate(prompts, n, max_tokens, temperature, top_p, stop, seed)
-> list (per prompt) of list (per sample) of Completion.
- VLLMBackend: offline vLLM on the cluster (main path).
- OpenAIBackend: any OpenAI-compatible /v1/completions server (llama-server for GGUF/Track C, vllm serve).
- HFBackend: plain transformers, for small debugging runs only.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Completion:
    text: str
    token_ids: list[int] | None
    finish_reason: str  # "stop" | "length"


@dataclass
class ModelSpec:
    name: str
    path: str
    backend: str = "vllm"
    tokenizer: str | None = None
    thinking: bool = True
    think_start: str = "<think>"
    think_end: str = "</think>"
    force_text: str = "\n\nTime is up. I will now write the final solution.\n"
    base_url: str | None = None
    max_model_len: int = 16384
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    extra: dict = field(default_factory=dict)


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec.tokenizer or spec.path)


def make_backend(spec: ModelSpec):
    if spec.backend == "vllm":
        return VLLMBackend(spec)
    if spec.backend == "openai":
        return OpenAIBackend(spec)
    if spec.backend == "hf":
        return HFBackend(spec)
    raise ValueError(spec.backend)


class VLLMBackend:
    def __init__(self, spec: ModelSpec):
        from vllm import LLM
        self.llm = LLM(model=spec.path, tokenizer=spec.tokenizer or spec.path, dtype=spec.dtype,
                       max_model_len=spec.max_model_len, tensor_parallel_size=spec.tensor_parallel_size,
                       gpu_memory_utilization=spec.gpu_memory_utilization, enable_prefix_caching=True,
                       **spec.extra)

    def generate(self, prompts, n, max_tokens, temperature, top_p, stop=None, seed=None):
        from vllm import SamplingParams
        # keep special tokens in the text: if </think> is a special token, stop-string matching needs it
        sp = SamplingParams(n=n, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                            stop=stop or None, seed=seed, skip_special_tokens=False)
        outs = self.llm.generate(prompts, sp, use_tqdm=True)
        return [[Completion(o.text, list(o.token_ids), "stop" if o.finish_reason == "stop" else "length")
                 for o in out.outputs] for out in outs]


class OpenAIBackend:
    def __init__(self, spec: ModelSpec):
        self.url = (spec.base_url or "http://127.0.0.1:8080").rstrip("/") + "/v1/completions"
        self.model = spec.name
        self.concurrency = int(spec.extra.get("concurrency", 8))  # match llama-server -np
        # llama-server adds BOS itself; drop the template's textual BOS to avoid a double BOS.
        self.strip_prefix = spec.extra.get("strip_prefix", "")

    def _one(self, p, max_tokens, temperature, top_p, stop, seed):
        if self.strip_prefix and p.startswith(self.strip_prefix):
            p = p[len(self.strip_prefix):]
        body = {"model": self.model, "prompt": p, "max_tokens": max_tokens, "temperature": temperature,
                "top_p": top_p, "stop": stop or []}
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            ch = json.loads(r.read())["choices"][0]
        return Completion(ch["text"], None, "length" if ch.get("finish_reason") == "length" else "stop")

    def generate(self, prompts, n, max_tokens, temperature, top_p, stop=None, seed=None):
        from concurrent.futures import ThreadPoolExecutor
        # llama-server ignores n>1, so every sample is its own request (distinct seeds when seeded).
        jobs = [(i, j) for i in range(len(prompts)) for j in range(n)]
        with ThreadPoolExecutor(self.concurrency) as ex:
            outs = list(ex.map(lambda ij: self._one(prompts[ij[0]], max_tokens, temperature, top_p, stop,
                                                    None if seed is None else seed + 1000 * ij[0] + ij[1]), jobs))
        res = [[] for _ in prompts]
        for (i, _), c in zip(jobs, outs):
            res[i].append(c)
        return res


class HFBackend:
    def __init__(self, spec: ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        self.tok = load_tokenizer(spec)
        self.model = AutoModelForCausalLM.from_pretrained(spec.path, torch_dtype="auto")
        self.model.eval()

    def generate(self, prompts, n, max_tokens, temperature, top_p, stop=None, seed=None):
        if seed is not None:
            self.torch.manual_seed(seed)
        res = []
        for p in prompts:
            ids = self.tok(p, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**ids, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                          top_p=top_p, max_new_tokens=max_tokens, num_return_sequences=n,
                                          pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            samples = []
            for row in out[:, ids["input_ids"].shape[1]:]:
                toks = [t for t in row.tolist() if t not in (self.tok.pad_token_id, self.tok.eos_token_id)]
                text = self.tok.decode(toks, skip_special_tokens=False)  # same as vLLM: keep </think> visible
                finish = "length" if len(toks) >= max_tokens else "stop"
                for s in stop or []:
                    if s in text:
                        text, finish = text[:text.index(s)], "stop"
                        toks = None  # ids no longer match the cut text
                        break
                samples.append(Completion(text, toks, finish))
            res.append(samples)
        return res


# ------------------------------------------------------------------ thinking budgets

@dataclass
class BudgetedSample:
    prompt_idx: int
    sample: int
    budget: int            # thinking-token cap; 0 = no-think
    think_tokens: int
    answer_tokens: int
    forced: bool           # True if thinking was cut at the budget
    answer: str            # text after the end-of-thinking tag
    answer_finished: bool  # False if the answer itself hit answer_tokens
    thinking: str = ""

    @property
    def total_tokens(self) -> int:
        return self.think_tokens + self.answer_tokens


def open_thinking(prompt: str, spec: ModelSpec) -> str:
    """Make sure the prompt ends inside an open thinking block (some templates already add <think>)."""
    return prompt if prompt.rstrip().endswith(spec.think_start) else prompt + spec.think_start + "\n"


def _count(tok, c: Completion) -> int:
    return len(c.token_ids) if c.token_ids is not None else len(tok(c.text, add_special_tokens=False)["input_ids"])


def generate_budgeted(backend, tok, spec: ModelSpec, prompts: list[str], budgets: list[int], n: int,
                      answer_tokens: int, temperature: float, top_p: float, seed: int | None = None,
                      keep_thinking: bool = False) -> list[BudgetedSample]:
    """Sample n thinking traces per prompt ONCE at the largest budget, then fork each trace at every
    smaller budget (s1-style budget forcing). A trace that ends naturally within a budget is reused
    unchanged, so each distinct continuation is answered only once."""
    out: list[BudgetedSample] = []
    if not spec.thinking:  # plain instruct model: one "budget" = answer only
        gens = backend.generate(prompts, n, answer_tokens, temperature, top_p, seed=seed)
        for i, samples in enumerate(gens):
            for j, c in enumerate(samples):
                out.append(BudgetedSample(i, j, 0, 0, _count(tok, c), False, c.text, c.finish_reason == "stop"))
        return out

    opened = [open_thinking(p, spec) for p in prompts]
    think_budgets = sorted({b for b in budgets if b > 0})
    cont_prompts: list[str] = []
    cont_meta: list[tuple] = []  # (prompt_idx, sample, [budgets], think_tokens, forced, thinking_text)

    if 0 in budgets:  # no-think: n real samples of the answer behind an empty thinking block
        nothink = [p + "\n" + spec.think_end + "\n\n" for p in opened]
        for i, samples in enumerate(backend.generate(nothink, n, answer_tokens, temperature, top_p, seed=seed)):
            for j, a in enumerate(samples):
                out.append(BudgetedSample(i, j, 0, 0, _count(tok, a), False, a.text, a.finish_reason == "stop"))

    if think_budgets:
        bmax = think_budgets[-1]
        traces = backend.generate(opened, n, bmax, temperature, top_p, stop=[spec.think_end], seed=seed)
        for i, samples in enumerate(traces):
            for j, c in enumerate(samples):
                natural = c.finish_reason == "stop"
                # vLLM's token_ids include the stop string's tokens while the text excludes them, so a
                # naturally ended trace is re-tokenized from its text to keep cuts and counts text-exact.
                ids = (c.token_ids if c.token_ids is not None and not natural
                       else tok(c.text, add_special_tokens=False)["input_ids"])
                L = len(ids)
                shared = []  # budgets where the natural trace fits
                for b in think_budgets:
                    if natural and L <= b:
                        shared.append(b)
                        continue
                    cut = tok.decode(ids[:b], skip_special_tokens=False)
                    cont_prompts.append(opened[i] + cut + spec.force_text + spec.think_end + "\n\n")
                    cont_meta.append((i, j, [b], b, True, cut))
                if shared:
                    cont_prompts.append(opened[i] + c.text + spec.think_end + "\n\n")
                    cont_meta.append((i, j, shared, L, False, c.text))

    if not cont_prompts:
        return out
    answers = backend.generate(cont_prompts, 1, answer_tokens, temperature, top_p,
                               seed=None if seed is None else seed + 1)
    for (i, j, bs, tt, forced, thinking), [a] in zip(cont_meta, answers):
        at = _count(tok, a)
        for b in bs:
            out.append(BudgetedSample(i, j, b, tt, at, forced, a.text, a.finish_reason == "stop",
                                      thinking if keep_thinking else ""))
    return out
