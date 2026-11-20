# Chosen plans: datasets, sources, and what the project looks like

*Written 2026-11-20. This builds on [`Small LM for competitive programming.md`](Small%20LM%20for%20competitive%20programming.md), "the main report" below. That report listed 23 ideas and an 8-stage plan. This document picks the ones worth doing, lists the data and code for each one, and describes the finished project: the system, the repository, the experiments, the timeline and the paper. Facts marked "unverified" came from search snippets or project pages I did not open in full, so check them before you rely on them.*

---

## 1. The decision in one page

**Do one paper with three tracks on a shared foundation.** Each track supports one claim in the paper, and all of them reuse the same evaluation harness and the same pool of model generations.

| | Track | Ideas from the main report | Question it answers | Why it was chosen |
|---|---|---|---|---|
| **0** | **Foundation**: fixed evaluation and a strong base coder | Stage 0; base-model choice | What does the coder actually score, at each token budget? | Every later number depends on it. The current 8% may be below the untouched base. |
| **A** | **"System One controller, System Two coder"**: hint ladder, then a calibrated controller with gated retrieval | 1, 2, 3, 23 (+ retrieval redesign) | How much of a small coder's failure is choosing the algorithm, and how much compute does a tiny controller save at phone budgets? | Highest novelty and impact. It is your original router idea, reshaped by the evidence and the Jev framing. It is mostly inference, so it is cheap. |
| **B** | **Verdict-aware repair, on the phone** | 6, 7 | When a program fails, does targeted repair beat simply sampling again at equal tokens and energy? | It targets the dominant failure (Wrong Answer). Track A's generations supply the training data for free. |
| **C** | **Short traces survive quantization** (a side result) | 4, 5 | Do short structured plans ("blueprints") lose less accuracy at 4 bits than 19K-token chains? | It is the cheapest novel result: about one week of evaluation, and it reuses your existing blueprint work. |

**Deliberately dropped from the headline.** These stay as ablations or are cut entirely:
- Topic LoRAs (idea 11): keep only as an oracle-bounded ablation.
- GEPA and LLM-JEPA (ideas 8–10): optional ablations.
- TRM-style recursion, diffusion repair, hypernetwork LoRAs, looped bases (ideas 15, 17, 18, 20): high risk, no code evidence.
- Pretraining-scale weight transfer: only worth it for a custom sub-1B model, which this paper does not need.

**The base-model decision.** This changes how much compute the project needs.
- **Main coder: VibeThinker-1.5B** (MIT). It has the same Qwen2 architecture and tokenizer as your current base and self-reports LiveCodeBench v6 51.1. Starting from it saves the thousands of GPU-hours of SFT, distillation and RL that the main report's Stages 1–4 would cost. The base is a *means*, not the contribution.
- **Control model: DeepSeek-R1-Distill-Qwen-1.5B**, your current base, plus your existing Stage 1, Stage 2 and M3 checkpoints. This keeps continuity with your current paper and gives a weak-coder point.
- **Transfer check: MiniCPM5-2B** (Apache-2.0). One table only, to show the findings hold on a 2026 model.
- **Optional Track 0b, only if VibeThinker fails the harness check:** full SFT on about 0.5M verified traces, then on-policy distillation from a same-tokenizer teacher (Section 3.3).

**What the project becomes.** A phone app and a paper. The app runs a 1.5B coder at 4 bits. Before generating, its own first pass over the problem makes four typed decisions with calibrated probabilities: thinking budget, retrieve or not, difficulty, abstain. It then writes code, tests it on the sample tests in an embedded Python sandbox, and repairs it if the verdict says so. The paper reports problems solved **per token, per second and per joule** at phone budgets, which no 2026 vendor publishes.

---

## 2. Track 0: the foundation (do this first, about 2 weeks)

### 2.1 Evaluation sets

| Role | Dataset | Size | Licence | Notes |
|---|---|---|---|---|
| **Headline benchmark** | `livecodebench/code_generation_lite`, release v6, windows **2408–2502** (v5 window, 279 problems) and **2502–2505** | 279 + ~175 | "cc" tag on HF | Standard for ≤2B models, so numbers compare with rStar-Coder, DeepCoder and VibeThinker. **No release after v6 could be confirmed as of mid-2026**, so v6 is still the latest ([LCB GitHub](https://github.com/LiveCodeBench/LiveCodeBench); [secondary report, unverified](https://systems-analysis.ru/eng/LiveCodeBench)). |
| **Held-out set with checkers, ratings and editorials** | `open-r1/codeforces`, `test` split (`verifiable` config) | 468 (422 verifiable) | CC-BY-4.0 | Has `rating`, `tags`, `editorial` (about 60% of problems), generated checkers and tests. **This is also the Track A hint-ladder set.** ([HF](https://huggingface.co/datasets/open-r1/codeforces)) |
| **Post-cutoff set** | Codeforces rounds after VibeThinker's and MiniCPM5's training cutoffs, collected yourselves, or LCB Pro easy problems | ~100–200 | Problem statements © Codeforces; use for evaluation only | Guards against contamination from the base models' training data. |
| **Sanity checks** | `evalplus/humanevalplus`, `evalplus/mbppplus` | 164 / 378 | Apache-2.0 | Secondary only. |
| **Your old set** | `test_suite_200_FINAL.jsonl` | 200 | Derived from verifiable-coding-problems (no HF licence tag) | Keep for continuity only after auditing it: run gold solutions, drop broken problems, flag problems needing checkers. |

### 2.2 Harness rules

These are taken from the main report's protocol table. In short:
- Use the model's own chat template.
- Use n = 16 samples, temperature 0.6, top-p 0.95.
- Report Pass@1 at **1K, 2K, 4K, 8K and 32K** tokens, plus truncation rate and parse-failure rate.
- Grade with checkers or LCB-style comparison, run in a sandbox with no network.
- Use paired bootstrap CIs and McNemar tests, and release the raw generations.

**Tools**

| Need | Tool | Licence |
|---|---|---|
| Fast sampling | vLLM (or SGLang) | Apache-2.0 |
| LCB grading logic | [LiveCodeBench `testing_util.py`](https://github.com/LiveCodeBench/LiveCodeBench) | MIT |
| Sandbox at scale | [SandboxFusion](https://github.com/bytedance/SandboxFusion), or nsjail/firejail around a Python subprocess | Apache-2.0 |
| Decontamination | [open-r1 `decontaminate.py`](https://github.com/huggingface/open-r1/blob/main/scripts/decontaminate.py) (n-gram), plus ID/URL joins and embedding similarity | Apache-2.0 |
| Thinking-budget caps | Budget forcing: stop thinking at N tokens, append the end-of-thinking tag, then let the model answer (s1-style). The llama.cpp reasoning-budget work allows the same on the phone. | – |

**Gate.** You need the baseline table for R1-Distill-1.5B, VibeThinker-1.5B and your old checkpoints at every budget before anything else is reported. If your Stage 2 checkpoint scores below the untouched base, say so in the paper: it becomes the motivating negative result.

### 2.3 Base models

| Model | HF id | Licence | Role |
|---|---|---|---|
| VibeThinker-1.5B | [`WeiboAI/VibeThinker-1.5B`](https://huggingface.co/WeiboAI/VibeThinker-1.5B) | MIT | Main coder. Weights and evaluation code are released; the **training code is not** (a public HF question about it is unanswered), and part of the training data is proprietary ([GitHub](https://github.com/WeiboAI/VibeThinker); [paper](https://arxiv.org/abs/2511.06221)). |
| DeepSeek-R1-Distill-Qwen-1.5B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | MIT | Control model; continuity with your paper |
| rStar-Coder-1.5B / DeepCoder-1.5B | `microsoft/…`, `agentica-org/DeepCoder-1.5B-Preview` | MIT | Extra baselines in the Track 0 table |
| MiniCPM5-2B | [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) | Apache-2.0 | Transfer check |

---

## 3. Track A: the System One controller (paper core, weeks 2–10)

### 3.1 Step A1: the hint ladder (diagnosis, weeks 2–4)

**The question.** Give the coder progressively more help, from nothing up to a gold editorial. Where does accuracy jump? If it jumps at "oracle plan", the coder is *selection-bound*, and a planner or retriever can help. If it barely moves even with a gold editorial, it is *implementation-bound*, and repair (Track B) is the lever.

**The rungs.** Each one is a prompt prefix, with identical stopping rules across rungs:
1. **None:** the statement only.
2. **Predicted tags:** from the controller's soft topic head, or a TF-IDF/SVM baseline.
3. **Oracle tags:** the Codeforces tags.
4. **Oracle technique:** one line, such as "binary search on the answer + greedy check", extracted from the editorial by a teacher.
5. **Oracle blueprint:** your Stage 2 blueprint format, written by a teacher *from the editorial*, with no code allowed.
6. **Gold editorial:** the full text.
7. *(Optional)* **Gold editorial plus pseudocode.**

**Leakage control.** Reject any hint with long n-gram overlap with the gold solution code. Report hint length in tokens for every rung, so a jump cannot be explained by length alone.

**Data**

| Use | Dataset | Size | Licence | What it gives |
|---|---|---|---|---|
| Main ladder set | `open-r1/codeforces` test split, filtered to problems with editorials | ≈ 280 of 468 (60% have editorials) | CC-BY-4.0 | Statement, `editorial`, `tags`, `rating`, checkers, tests ([HF](https://huggingface.co/datasets/open-r1/codeforces)) |
| Second ladder set (ICPC style) | [`samahadhoud/idea-first-code-later-cp`](https://huggingface.co/datasets/samahadhoud/idea-first-code-later-cp) | 83 ICPC Asia (Jakarta 2017–2019) problems | MIT | Gold editorials and tests. This is the set from "Idea First, Code Later" (ACL 2026 Findings), so you can compare directly with its 19 large LLMs ([paper](https://arxiv.org/abs/2601.11332); [project page](https://samahadhoud.github.io/idea-first-code-later-cp-page)). |
| Optional third set | USACO-bench (307 problems with official analyses) | 307 | Licence unverified | Bronze and silver tiers are within reach of 1.5B models ([paper](https://arxiv.org/abs/2404.10952)). |
| Difficulty tiers | Codeforces `rating` buckets: ≤1200, 1300–1600, 1700–2000, >2000 | – | – | The ladder is reported per tier. |

**Models.**
- **Coders:** VibeThinker-1.5B and R1-Distill-1.5B, plus MiniCPM5-2B in the transfer table.
- **Hint writer (teacher):** any strong open model, such as Qwen3-32B or QwQ-32B. Hints are plain text, so the tokenizer does not matter here.

**Compute.** About 400 problems × 7 rungs × 16 samples ≈ 45K generations, roughly 0.2B tokens. That is **a few GPU-hours** (estimate).

**Expected outcome.** These are hypotheses from prior work, not promises:
- Topic rungs (2–3) give about **+0**. GPT-3.5's "used algorithm" hint left solve@10 at 6.1%, the same as no hint ([Explaining CP solutions](https://ar5iv.labs.arxiv.org/html/2307.05337)).
- Plan and editorial rungs give a large jump at easy and medium tiers. The same study saw 42.4% with step-by-step hints.
- The gap shrinks at hard tiers, where implementation dominates ("Idea First, Code Later").

This is **figure 1 of the paper**.

### 3.2 Step A2: label generation (weeks 3–5)

**The key idea.** The controller's labels are *your coder's own measured outcomes*, not human topic tags. This is the open, RL-free analogue of Jev's "calibrated decisions". In the main report's words: "train on your coder's own measured pass rates as soft labels, then calibrate."

**Procedure.** For each training problem:
- Sample the coder **8 times** at each thinking budget in {no-think, 1K, 2K, 4K, 8K}, each with and without retrieval.
- Use **prefix sharing** to cut cost: run one 8K-thinking generation and fork it at 1K, 2K and 4K by injecting the end-of-thinking tag. This makes the cost about 2 × 8 long runs per problem, not 80.
- Execute every program and store the **verdict** (AC, WA, TLE, RE) and the first failing test.

**The labels derived from this:**

| Label | Definition | Head type (Jev-style) |
|---|---|---|
| `budget` | Soft distribution over buckets: pass rate at each budget, then the expected-value-optimal bucket | Choice (5-way) |
| `retrieve` | Did retrieval raise the pass rate by at least δ? | Yes/no ("Noul") |
| `difficulty` | 1 − pass rate at 8K (model-relative, as in Skywork's 0–16 score) | Score |
| `abstain` | Pass rate at 8K below τ | Yes/no |
| `topic` (soft, top-k) | Codeforces/TACO tags mapped to about 10 families | Choice (multi-label), used only for rung 2 and as an auxiliary loss |

**Training problems.** Decontaminate all of these against every evaluation set (problem-ID join, then 13-gram overlap, then embeddings):

| Dataset | Size | Licence | Why |
|---|---|---|---|
| [`agentica-org/DeepCoder-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) | 24K problems, ≥5 tests each | MIT | Already verified tests (TACO-verified, SYNTHETIC-1, LCB v5 train) |
| `open-r1/codeforces` `verifiable` train | 8,338 | CC-BY-4.0 | Checkers, ratings, tags, editorials |
| [`likaixin/TACO-verified`](https://huggingface.co/datasets/likaixin/TACO-verified) | 12,898 | MIT tag | Tags, skill types, expected complexity |
| [`ByteDance-Seed/Code-Contests-Plus`](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus) | 11,690 | CC-BY-4.0 | Generators, validators and checkers with 100% validation; correct and incorrect submissions |
| [`Skywork/Skywork-OR1-RL-Data`](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) (code part) | ~14K | **No licence tag found** | Ready-made 0–16 difficulty *measured on R1-Distill-Qwen-1.5B*: a free sanity check for your own difficulty labels |

Use about **10–15K problems** after deduplication.

**Compute.** These are estimates, not measurements:
- **Generation:** about 10K problems × 2 retrieval settings × 8 samples × about 8K tokens ≈ **1.3B tokens**, which is about **1–3 H100-days** for a 1.5B model on vLLM.
- **Execution:** about 800K programs × about 20 tests ≈ **16M test runs**, which is about a day on a 64-core CPU box.

**By-product.** This step produces roughly **hundreds of thousands of labelled failed programs (WA, TLE, RE) from your own coder**. That is exactly Track B's training data (Section 4).

### 3.3 Step A3: the controller (weeks 5–9)

**Arms compared.** They share one set of labels and are evaluated on one harness.

| Arm | What it is | Phone cost | Source |
|---|---|---|---|
| **A. Self-probe (proposed default)** | Typed heads (softmax, sigmoid, regression) on the coder's own **last-layer prefill hidden state**: one forward pass, no extra model | ≈ 0 MB, under 1 ms | Sonata design (Apple, ICLR 2026, 20–60% fewer thinking tokens; [paper](https://www.iclr.cc/virtual/2026/poster/10011708); **no public code found**). Difficulty probes reach ρ = 0.81 at 70B ([arXiv 2510.18147](https://arxiv.org/html/2510.18147v1)); idea 3 tests whether this holds at 1.5B. |
| **B. Self-routing by option logits** | The coder reads a short menu (`A = no-think`, `B = 1K`, …) and picks by the log-probabilities of the letters, sharing the KV cache with the later generation | A few tokens | [Rizzo Flow](https://github.com/Rizzo-AI-Academy/rizzo-flow) pattern (Apache-2.0) |
| **C. Separate small decision model** | **NanoJev** (Qwen3-0.6B with decision heads; ships a training pipeline), fine-tuned on the A2 labels | About 0.4 GB at 4 bits | [NanoJev](https://github.com/TianyuCodings/NanoJev) (MIT) |
| **D. Training-free baseline (baseline to beat)** | **DART**: two cheap no-think drafts; answer directly if they agree, otherwise set the budget from draft entropy | Two short drafts | [Code](https://github.com/js-lee-AI/DART); [EMNLP 2026 Findings](https://aclanthology.org/2026.findings-emnlp.1347/). Reports up to +22.5 points on code with 51–63% fewer thinking tokens. |
| **E. Fixed budgets** | Always 1K, 2K, 4K or 8K | – | – |
| **F. Oracle** | Picks the best bucket using the labels | – | Upper bound |
| **G. External decision models (optional)** | Jev (closed API) and Kev-4B (open, Jev-API-compatible), so the *same request* goes to both | Cloud | [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev); [Kev](https://github.com/jaredpalmer/kev). **Check Jev's terms before sending benchmark problems.** |

**Loss and calibration.** Train with a proper scoring rule (log loss or Brier score) against the *soft* pass-rate targets. Then fit a temperature on a held-out split. Report ECE and reliability diagrams. Calibration is what makes "abstain" and expected-value budget choice principled.

**Decision rule.** Pick the bucket that maximises *P̂(pass | b) − λ · tokens(b)*. Sweeping λ traces a Pass@1-versus-tokens curve for each arm.

**Primary metric.** The **area under the Pass@1-versus-mean-tokens curve**, plus Pass@1 at matched mean tokens (1K, 2K, 4K). Label accuracy is secondary.

### 3.4 Step A4: retrieval that the controller gates (weeks 5–8)

**What changes from M3.**

| | M3 today | New design |
|---|---|---|
| Corpus | 7 hand-written topic blueprints | About 40–60K **solved problems**: statement, editorial or explanation, accepted Python code |
| Query | Statement only | Statement **plus a 256-token draft plan** (USACO-bench: statement-only queries kept only about 3% of the gain) |
| Retriever | General MiniLM | BM25 plus a CP-trained embedder |
| Use | Always | Only when the controller's `retrieve` probability is above a threshold; k = 1–2 documents, about 800 tokens each |

**Corpus sources.** Deduplicate all of them against the evaluation sets:

| Dataset | Size | Licence | Content |
|---|---|---|---|
| `open-r1/codeforces` train + [`open-r1/codeforces-cots`](https://huggingface.co/datasets/open-r1/codeforces-cots) `solutions_w_editorials_py` | ~9.5K problems; 11.7K Python rows with editorials | CC-BY-4.0 | Statement, editorial, R1 trace and code |
| `likaixin/TACO-verified` | 12,898 problems / 1.04M verified solutions | MIT tag | Statement and verified code |
| `deepmind/code_contests` (train) | ~13.3K | CC-BY-4.0 | Statement and solutions |
| [CPRet data](https://huggingface.co/collections/coldchair16/cpret) | 38.8K problems / 2.93M solutions | **Check the `CPRet-data` card.** The `CPRet-Embeddings` repo is **CC-BY-NC-4.0**, which is fine for research but not for a commercial app. | Duplicate and similar-problem pairs |

**Retriever models**

| Model | Size | Score on CPRet | Use |
|---|---|---|---|
| `coldchair16/CPRetriever-Prob-Qwen3-4B-2510` | 4B | Best (84.6 NDCG@10 for the earlier 4B) | **Offline** teacher and upper bound only ([GitHub](https://github.com/coldchair/CPRet)) |
| `Qwen3-Embedding-0.6B` fine-tuned on CPRet pairs, distilled from the 4B model | 0.6B | 56.8 before fine-tuning | **Phone retriever.** The index is about 50K × 1024-dim int8 ≈ 50 MB. |
| BM25 over statement and code | – | +11.8% over plain embeddings when matching attempted solutions ([AlgoSimBench](https://arxiv.org/abs/2507.15378)) | Always fused in |

**Reuse.** Your M3 ChromaDB code stays. Change only the corpus, the query and the embedder.

**Required control.** For the same prompts and stopping rules, compare an empty or random retrieved document against the real one. Also measure how retrieval changes response length.

### 3.5 What Track A contributes to the paper

- **Figure 1:** the hint ladder by difficulty tier.
- **Figure 2:** Pass@1 versus token budget for every coder; no vendor publishes this curve.
- **Figure 3:** Pass@1 per token for controller arms A–G, against oracle and DART.
- **Figure 4:** reliability diagrams, plus the share of the oracle gap recovered.
- **Table:** retrieval ablation (corpus × query × retriever × gating).

---

## 4. Track B: verdict-aware repair (weeks 6–12)

**The question.** Given a failed program and its judge verdict, can a 1.5B coder trained on verdict-specific repair beat **resampling at equal tokens**? Self-repair is often not worth its cost unless the model is trained for it ([Olausson et al.](https://arxiv.org/abs/2306.09896); [CoCoS](https://arxiv.org/abs/2505.23060)).

**Repair policy by verdict.** This comes from the main report's idea 6:
- **WA:** show the first failing *sample* test. The model writes a counterexample hypothesis and updates its corner cases.
- **TLE:** state the input limits; the model must propose a lower-complexity approach.
- **RE:** show the error class (recursion depth, index, parse), with targeted fixes.
- **Phone constraint:** only sample tests and model-generated tests are available, never hidden tests.

**Data**

| Use | Dataset | Size | Licence | Notes |
|---|---|---|---|---|
| **On-policy failures (best source)** | Your own Track A2 generations | Hundreds of thousands of WA/TLE/RE programs (estimate) | Yours | The *coder's own* mistakes. Fixes come from (a) its own passing samples on the same problem, or (b) a teacher repair, verified by execution. |
| Human failure–fix pairs (Python) | [`open-r1/codeforces-submissions`](https://huggingface.co/datasets/open-r1/codeforces-submissions): `selected_incorrect` (failed, but passed at least one public test) + `selected_accepted` | 10M–100M rows in total | CC-BY-4.0 | Has `verdict` (11 values), `passedTestCount`, `timeConsumedMillis`, `programmingLanguage`. Pair a user's failed and accepted submissions on the same problem. |
| Correct and incorrect submissions with strong tests | `ByteDance-Seed/Code-Contests-Plus` | More than 13M submissions | CC-BY-4.0 | Its checkers make verdicts trustworthy. |
| Large verdict-labelled corpus | [IBM Project CodeNet](https://github.com/IBM/Project_CodeNet) | 13.9M submissions over 4,053 problems; 53.6% accepted, 29.5% WA | Data reportedly **CDLA-Permissive-2.0** (check the LICENSE file) | AtCoder and Aizu; statuses include WA, TLE, RE, CE, MLE |
| Repair benchmark with tests | [FixEval](https://github.com/mahimanzum/FixEval) | About 700 problems, Python and Java, ~25 tests each | Per the repository | Built on CodeNet; execution-based ([paper](https://arxiv.org/abs/2206.07796)) |
| 2026 repair benchmark | UOJ-Bench (ICML 2026): generation, hacking and repair tracks | – | Unverified | A good external repair evaluation if released ([arXiv 2606.12864](https://arxiv.org/abs/2606.12864)) |
| Counterexample tests | Codehacks (Codeforces hacks) | – | Unverified | Useful for checking whether a "fix" is real ([arXiv 2503.23466](https://www.arxiv.org/pdf/2503.23466)) |

**Training**
1. **SFT** on (statement, failed code, verdict, failing sample) → (diagnosis line, fixed code) pairs. Keep only fixes that pass the full tests. Balance the verdicts.
2. **CoCoS-style multi-turn RL**, using an accumulated reward over the trajectory and partial credit for tests passed. Code: [github.com/jeonghun3572/CoCoS](https://github.com/jeonghun3572/CoCoS) (contents not checked). Use verl or TRL for the RL loop, with the Track 0 sandbox as the reward.

**Evaluation.** Compare repair against resampling against generate-then-filter-by-sample-tests, each at **equal total tokens** and, on the phone, at **equal joules**. Report the fix rate per verdict. Prior work found TLE about 20% and WA about 35% fixable for large models ([alphaXiv 2606.17514](https://www.alphaxiv.org/abs/2606.17514)).

**An honest negative result is still useful.** If repair does not beat resampling at 1.5B, that is a clean, publishable finding, and it tells phone apps to spend the tokens on more samples instead.

---

## 5. Track C: quantization robustness (one week, any time after Track 0)

- **Test:** the same coders at FP16, Q8_0, Q4_K_M and Q3_K (llama.cpp GGUF), plus a 4-bit QAT build if time allows.
- **Compare three output formats:**
  1. a long chain of thought;
  2. a token-matched truncated chain;
  3. your **blueprint** format (a short structured plan, then code).
- **Measure:** Pass@1, the change in output length (quantization tends to lengthen chains), and the truncation rate at fixed budgets.
- **Data:** the LCB window from Track 0. No new data is needed.
- **Why it matters:** no study was found that measures quantization damage on *short structured* traces; existing studies cover only long-CoT distills ([Lotfi et al.](https://arxiv.org/html/2606.00206v1)). Rename "blueprint" to "algorithm sketch" in the paper to avoid a name clash with Blueprint2Code.

---

## 6. What the finished project looks like

### 6.1 The system on the phone

```
            ┌──────────────────────────────────────────────────────────────┐
problem ──▶ │  Coder 1.5B (4-bit, llama.cpp / LiteRT)                      │
statement   │   1. PREFILL the statement once (KV cache kept)              │
            │   2. SYSTEM ONE: typed heads on the last prefill hidden state│
            │        budget  ∈ {no-think, 1K, 2K, 4K, 8K}  (Choice + probs)│
            │        retrieve? (yes/no prob)   difficulty (Score)          │
            │        abstain?  (yes/no prob)   → one pass, < 1 ms          │
            └──────────┬───────────────────────────────────────────────────┘
                       │ if retrieve: draft 256-token plan → BM25 + 0.6B
                       │ embedder over a ~50 MB on-device index → 1–2 solved
                       │ problems appended (KV cache reused)
                       ▼
            ┌──────────────────────────────────────────────────────────────┐
            │  SYSTEM TWO: think up to the chosen budget → write code      │
            └──────────┬───────────────────────────────────────────────────┘
                       ▼
            ┌──────────────────────────────────────────────────────────────┐
            │  Sandbox: embedded CPython (Android: Chaquopy, separate      │
            │  process + watchdog; iOS: Pyodide in a WKWebView worker)     │
            │  run sample tests → verdict AC / WA / TLE / RE               │
            └──────────┬───────────────────────────────────────────────────┘
                       │ not AC and budget left → verdict-aware REPAIR
                       │ (or resample, whichever won in Track B)
                       ▼
                 code + confidence  (or "low confidence" if abstain)
```

**Memory estimate.** About 1.0–1.1 GB for the coder at 4 bits, about 0.35 GB for the 0.6B embedder at 4 bits, about 50 MB for the index, plus the KV cache. That fits an 8 GB phone.

**Speed estimate (unverified).** A 1.5B model at Q4 likely decodes at about **20–40 tokens/s** on a recent Snapdragon. No source measured exactly that configuration ([Ertas](https://www.ertas.ai/blog/llm-android-benchmarks-snapdragon-tensor); [Qualcomm Genie, Llama-3.2-1B ≈ 34 tok/s](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct)). That puts a 2K-token answer at roughly **1–2 minutes** and a 19K-token chain at **8–16 minutes**. Measuring this yourselves on two phones is part of the paper.

### 6.2 Offline training and research pipeline

```
 problem pools (DeepCoder, open-r1/codeforces, TACO-verified, CC+)
        │  decontaminate vs LCB / CF-test / post-cutoff
        ▼
 Track A2: sample coder × budgets × retrieval ── execute (sandbox) ──▶ labels.parquet
        │                                                    │
        │                                                    └──▶ failed programs ─▶ Track B SFT → CoCoS RL
        ▼
 Track A3: probe / NanoJev / option-logit training → calibrate (temperature) → controller.pt
        ▼
 Track 0 harness: every arm × every budget × n=16 → results/*.jsonl → paired stats → figures
        ▼
 Track C + phone: GGUF/QAT builds → on-device runs (tok/s, TTFT, joules, Pass@1)
```

### 6.3 Repository layout

The current repo is a set of notebooks. Turn it into a package; the old notebooks move to `legacy/`.

```
The_small_coder/
├── README.md                    # how to reproduce every table
├── configs/                     # one YAML per experiment (model, budget grid, n, seeds)
├── data/
│   ├── build_eval_sets.py       # LCB windows, CF test, post-cutoff, audited old-200
│   ├── build_train_pool.py      # merge DeepCoder / open-r1 / TACO-verified / CC+
│   ├── decontaminate.py         # ID join → 13-gram → embedding + LLM review; writes a report
│   └── build_rag_corpus.py      # solved-problem corpus + BM25 + vector index
├── eval/
│   ├── harness.py               # sample (vLLM) → extract → judge → pass@k, budgets
│   ├── judge/                   # sandbox runner, checkers, float tolerance, time limits
│   └── stats.py                 # paired bootstrap, McNemar, reliability/ECE
├── ladder/                      # Track A1: hint writers, leakage filter, rung prompts
├── controller/                  # Track A2–A3
│   ├── make_labels.py           # budget forcing w/ prefix sharing; stores verdicts
│   ├── probe.py                 # typed heads on prefill hidden state
│   ├── nanojev_finetune.py      # separate-model arm
│   ├── option_logits.py         # self-routing arm
│   ├── dart_baseline.py         # wraps js-lee-AI/DART
│   └── calibrate.py             # temperature scaling, decision rule, λ sweep
├── rag/                         # retriever fine-tune, query = statement + draft
├── repair/                      # Track B: pair mining, SFT, CoCoS-style RL, eval vs resample
├── quant/                       # Track C: GGUF builds, QAT, format comparison
├── phone/
│   ├── android/                 # llama.cpp JNI or LiteRT-LM + Chaquopy sandbox app
│   └── bench/                   # tok/s, TTFT, battery/energy logging scripts
├── results/                     # raw generations (released), tables, figures
├── paper/                       # LaTeX (move small_coder.tex here)
└── legacy/                      # SLM_notebook_2..9.ipynb, old M1–M3 reports
```

### 6.4 Compute budget

All figures are estimates extrapolated from published runs, assuming VibeThinker-1.5B is used as-is.

| Item | GPU | CPU |
|---|---|---|
| Track 0 baselines (5 models × ~900 problems × 16 samples × 5 budgets, with prefix sharing) | ~1–2 H100-days | ~1 day (64 cores) |
| A1 hint ladder | A few H100-hours | Hours |
| A2 labels (10K problems) | ~1–3 H100-days | ~1 day |
| A3 controllers (probe: minutes; NanoJev fine-tune: hours) | < 1 H100-day | – |
| A4 retriever fine-tune + index | < 1 H100-day | – |
| B repair SFT + CoCoS RL | ~1–2 weeks of one 8×H100 node (RL dominates) | Sandbox during RL |
| C quantization | < 1 day | – |
| **Optional Track 0b** (own SFT + on-policy distillation instead of VibeThinker) | Hundreds to low thousands of H100-hours (main report, Stages 2–3) | – |

**On-policy distillation, if you do Track 0b.** Use TRL's `GKDTrainer` with `lmbda=1` (fully on-policy). In recent TRL versions it lives under `trl.experimental.gkd` ([docs](https://huggingface.co/docs/trl/gkd_trainer)). It needs a teacher with the **same vocabulary**: check that `len(tokenizer)` and `config.vocab_size` match. Same-tokenizer candidates include R1-Distill-Qwen-7B/14B/32B, QwQ-32B and OpenReasoning-Nemotron-7B/14B/32B. For a mismatched teacher such as Gemma, TRL's `GOLDTrainer` does cross-tokenizer distillation.

### 6.5 Timeline and roles (team of four, about 16 weeks)

| Weeks | Person 1: eval and data | Person 2: controller | Person 3: repair | Person 4: phone and quantization |
|---|---|---|---|---|
| 1–2 | Harness, sandbox, decontamination, eval sets | Hint writer prompts, leakage filter | Mine codeforces-submissions pairs; study CoCoS code | Android app skeleton: llama.cpp + Chaquopy sandbox |
| 3–4 | Track 0 baseline table (**gate**) | **A1 ladder results** | Repair prompts on base models (zero-shot) | Track C quantization runs |
| 5–7 | A2 label generation at scale | Probe, option-logit and NanoJev arms; DART baseline | Repair SFT on on-policy failures | Retriever fine-tune + on-device index |
| 8–10 | Retrieval ablations; stats | Calibration, λ sweeps, oracle-gap analysis | CoCoS-style RL | Integrate controller + retrieval + sandbox on device |
| 11–12 | MiniCPM5-2B transfer table | External arms (Kev; Jev if terms allow) | Repair vs resample at equal tokens and joules | Measurements on two phones: tok/s, TTFT, joules, Pass@1 |
| 13–16 | Everyone: paper writing, figures, release of code, generations and the decontamination report |||

**Decision gates**
- **Week 2:** if VibeThinker underperforms its card badly in your harness, either debug the harness or switch to Track 0b.
- **Week 4:** if the ladder shows no gap even with a gold editorial, the paper's weight shifts to Track B.
- **Week 9:** if no controller arm beats DART, report it honestly and make the calibration and oracle-gap analysis the contribution.

### 6.6 The paper

**Working title.** *"Selection or Implementation? A System-One Controller for a Phone-Sized Competitive-Programming Coder."*

**Contributions**
1. **Diagnosis.** A hint ladder by difficulty tier on two coders, showing what share of failure is algorithm selection and what share is implementation. Topic labels alone buy nothing.
2. **Control.** A calibrated, single-pass, typed-output controller trained on the coder's own pass rates, with gated retrieval and abstention. It is compared against DART, fixed budgets, a separate decision model (NanoJev) and an oracle, and reported as Pass@1 per token at phone budgets.
3. **Repair.** Verdict-aware repair against resampling at equal tokens and joules.
4. **Deployment.** The first Pass@1-versus-budget curves for 2026-era small coders, with quantization robustness of short structured traces and measurements on real phones.

**Main table.** LCB v6 windows plus the Codeforces test split, at 1K, 2K, 4K, 8K and 32K tokens. Columns: R1-Distill (old pipeline), VibeThinker, + controller, + retrieval, + repair, and the full phone system at 4 bits.

**Honest expectations.**
- The controller should shift the accuracy-versus-tokens curve left by a margin comparable to Sonata or DART (20–60% fewer tokens at equal accuracy). Beating DART is uncertain.
- Absolute LCB numbers will be about VibeThinker's level, and they are **not** the contribution.

**Venue fit.** Efficient-reasoning and on-device ML workshops, or a main-conference findings track. Check deadlines yourselves; I did not.

---

## 7. Licence checklist

| Item | Status | Use |
|---|---|---|
| open-r1/codeforces, codeforces-cots, codeforces-submissions; CodeContests; CodeContests+; rStar-Coder; Nemotron/OCR | CC-BY-4.0 | Free to train on and evaluate; attribution required |
| TACO (Apache-2.0); TACO-verified, DeepCoder set, Idea-First-Code-Later (MIT) | Permissive | Free |
| VibeThinker-1.5B, R1-Distill-1.5B (MIT); MiniCPM5-2B, Qwen3-Embedding (Apache-2.0); NanoJev (MIT); Kev, Rizzo Flow (Apache-2.0) | Permissive | Free |
| CPRet-Embeddings | **CC-BY-NC-4.0** | Research only; check `CPRet-data` separately |
| Project CodeNet | Reportedly CDLA-Permissive-2.0 | Confirm the LICENSE file |
| Skywork-OR1-RL-Data, HardTests, verifiable-coding-problems, Mixture-of-Thoughts | **No licence tag** | Use only for internal analysis until resolved |
| KodCode | CC-BY-NC-4.0 | Avoid for a releasable model |
| Jev | Closed API | Baseline only, after a terms check |
| LFM3 / d1 models | LFM Open v1.0 (free under $10M revenue) | Optional comparison |

---

## 8. Open checks before committing

1. Confirm the `CPRet-data` licence and the Project CodeNet data licence.
2. Load `open-r1/codeforces-submissions` and list the 11 `verdict` values. I confirmed the field exists, not its values.
3. Open the DART and CoCoS repositories and confirm they run.
4. Check whether Sonata's code has been released. If not, arm A is your own implementation, which is fine because it is a small adapter.
5. Re-check for a LiveCodeBench release after v6 just before the experiments are frozen.
6. Revoke the leaked Google API key from the old notebooks, if not done yet.

---

## Sources

New in this document:
- [Project CodeNet (IBM GitHub)](https://github.com/IBM/Project_CodeNet), [CodeNet paper](https://arxiv.org/abs/2105.12655v2), [CDLA licence coverage](https://analyticsindiamag.com/ai-features/the-linux-foundation-launches-new-license-agreement-for-open-source-datasets), [HF mirror schema](https://huggingface.co/datasets/systemk/codenet/blob/main/README.md)
- [FixEval paper](https://arxiv.org/abs/2206.07796), [FixEval code](https://github.com/mahimanzum/FixEval)
- [CPRet collection](https://huggingface.co/collections/coldchair16/cpret), [CPRet GitHub](https://github.com/coldchair/CPRet), [CPRet-Embeddings card](https://huggingface.co/datasets/coldchair16/CPRet-Embeddings/blob/main/README.md), [CPRet paper](https://arxiv.org/abs/2505.12925)
- [CoCoS paper](https://arxiv.org/abs/2505.23060), [CoCoS ACL Anthology](https://preview.aclanthology.org/new-sigs/2025.findings-emnlp.127/)
- [Idea First, Code Later (arXiv)](https://arxiv.org/abs/2601.11332), [dataset](https://huggingface.co/datasets/samahadhoud/idea-first-code-later-cp), [project page](https://samahadhoud.github.io/idea-first-code-later-cp-page), [ACL 2026 Findings PDF](https://aclanthology.org/2026.findings-acl.1778.pdf)
- [DART (ACL Anthology)](https://aclanthology.org/2026.findings-emnlp.1347/), [DART arXiv](https://arxiv.org/pdf/2606.23181)
- [open-r1/codeforces-submissions card](https://www.huggingface.co/datasets/open-r1/codeforces-submissions/blob/main/README.md), [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces), [open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots)
- [Sonata / Adaptive Thinking (ICLR 2026)](https://www.iclr.cc/virtual/2026/poster/10011708), [Apple research page](https://machinelearning.apple.com/research/adaptive-thinking)
- [TRL GKD trainer docs](https://huggingface.co/docs/trl/gkd_trainer), [HF on-policy distillation write-up](https://huggingface.co/docs/trl/gkd_trainer.md)
- [VibeThinker-1.5B card](https://huggingface.co/WeiboAI/VibeThinker-1.5B), [VibeThinker paper](https://arxiv.org/abs/2511.06221), [VibeThinker GitHub mirror](https://github.com/JumpSpider/VibeThinker)
- [UOJ-Bench](https://arxiv.org/abs/2606.12864), [DePro](https://arxiv.org/pdf/2603.19399), [Codehacks](https://www.arxiv.org/pdf/2503.23466)
- [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data)
- [LiveCodeBench GitHub](https://github.com/LiveCodeBench/LiveCodeBench), [LCB release status (secondary)](https://systems-analysis.ru/eng/LiveCodeBench)
- [Ertas Android LLM benchmarks](https://www.ertas.ai/blog/llm-android-benchmarks-snapdragon-tensor), [Qualcomm Llama-3.2-1B on-device card](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct), [Qualcomm CPU acceleration blog](https://www.qualcomm.com/developer/blog/2026/04/llama-models-acceleration-on-cpu-qmx)

Carried over from the main report and its research notes (datasets, baselines, Jev alternatives, phone sandboxing): see the sources listed there, especially `research_notes/Small LM for competitive programming/datasets_and_evaluation.md`, `router_adaptive_reasoning_retrieval.md` and `compression_and_on_device_inference.md`.
