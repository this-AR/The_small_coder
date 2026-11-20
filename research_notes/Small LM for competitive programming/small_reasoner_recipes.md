# Training recipes behind small (≤4B, esp. ~1.5B) reasoning/code models with strong competitive-programming scores (status as of 2026-11-19)

Conventions used below: "self-reported" = numbers from the releasing team's own model card/paper; "cross-lab" = the same model evaluated by a *different* lab in its own paper/card (not a neutral third party); "independent" = neutral third party (e.g., Artificial Analysis). LCB = LiveCodeBench. Always check the LCB window. They differ: DeepSeek-R1 used 2024-08–2025-01; "v5" is usually 2024-08-01–2025-02-01 (279 problems); "v6" is usually 2025-02-01–2025-05-01 (131 problems). Some labs report a "v6" covering 2024-08–2025-05 (454 problems), which "generally yields higher scores" ([VibeThinker paper](https://arxiv.org/html/2511.06221); [AceReason-Nemotron-14B card](https://huggingface.co/nvidia/AceReason-Nemotron-14B)). Numbers I derived myself (e.g., GPU-hour products) are marked "my estimate".

## Q1. How exactly were the named models trained, and what did they score?

### Takeaway
Two recipe families dominate below 4B. (a) Large-scale distillation SFT from a strong teacher: OpenReasoning-Nemotron-1.5B used about 5M DeepSeek-R1-0528 traces and reached LCB v6 28.6. Qwen3-0.6B/1.7B/4B came from strong-to-weak on-/off-policy distillation. (b) Long, carefully stabilised RLVR on an already-distilled 1.5B base. DeepCoder, ProRL/BroRL and JustRL add roughly 6–11 LCB points to DeepSeek-R1-Distill-Qwen-1.5B, which starts at 16.9. The outlier is VibeThinker-1.5B (Qwen2.5-Math-1.5B base). It combines "diversity-exploring" SFT (Pass@K-selected domain specialists merged by weight averaging) with entropy-weighted RL (MGPO). It reports LCB v5 55.9 / v6 51.1 for about $7.8K of compute. Its SFT data sizes, teachers and trace lengths are undisclosed, so the recipe cannot be fully replicated.

### Cited Findings

**VibeThinker-1.5B (Weibo AI; arXiv 2511.06221, Nov 2025). Superseded within ≤4B by VibeThinker-3B, Nanbeige4.1-3B and MiniCPM5-2B (see Q2).**
- Base model is Qwen2.5-Math-1.5B. The base scores LCB v5 0.0, LCB v6 0.0, AIME24 6.7, AIME25 4.3 and MATH-500 58.5 — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- The framework is the "Spectrum-to-Signal Principle". The SFT "Spectrum phase" uses Two-Stage Diversity-Exploring Distillation. Domain-aware diversity probing splits math into N=4 subdomains (algebra, geometry, calculus, statistics). An LLM builds a probing set for each subdomain. Checkpoints saved every k steps are scored by **Pass@K** on each probing set, and the best checkpoint per subdomain is kept. "Expert model fusion" then merges the specialists by unweighted parameter averaging (w_i = 1/N) — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- SFT data is "mostly public open-source datasets plus a minor portion of proprietary synthetic data". The paper does **not** state teacher models, sample counts, trace lengths, max sequence length or epochs — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- The RL "Signal phase" uses **MaxEnt-Guided Policy Optimization (MGPO)**. For a question with group accuracy p_c, the deviation from maximum uncertainty is D_ME = p_c·log(p_c/0.5) + (1−p_c)·log((1−p_c)/0.5). The advantage weight is w_ME = exp(−λ·D_ME), applied as A' = w_ME·A. The effect is to up-weight prompts near 50% pass rate; λ=0 recovers GRPO. The paper does not state group size, clip ε, KL coefficient or λ — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- RL stage order is math at a 16K context, then math at 32K, then a code-generation stage. Decontamination used text normalisation plus **10-gram matching**, applied to both SFT and RL data — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- Compute was about **3,900 H800 GPU-hours** for SFT+RL. At $2/GPU-h that is about **$7.8K**. For comparison, the paper puts DeepScaleR at 3.8K A100-h ($4.5K) and DeepSeek-R1 at 147K H800-h — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- Evaluation used vLLM with code temperature 0.6, math temperature 1.0, top-p 0.95 and a max response of 40K tokens. Pass@1 was averaged over 8 samples for code and 64 for math — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- Results (self-reported): AIME24 80.3, AIME25 74.4, HMMT25 50.4, MATH-500 95.0, **LCB v5 55.9** (279 problems, Aug 2024–Feb 2025), **LCB v6 51.1** (131 problems, Feb–May 2025) and GPQA-D 46.7. No Codeforces rating is reported. Comparators in the same table: Magistral Medium LCB v6 50.3; DeepSeek-R1-0120 LCB v5 65.9; o3-mini-medium LCB v5 66.3; Qwen3-1.7B LCB v6 26.9; SmolLM3-3B LCB v5 27.6 — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- The model card recommends temperature 0.6 or 1.0, top-p 0.95 and **max tokens 40,960**. It is intended for competitive math and algorithm coding only, and English prompts are advised. License is MIT — [HF WeiboAI/VibeThinker-1.5B](https://huggingface.co/WeiboAI/VibeThinker-1.5B)
- The authors admit that code trails larger models "due to a math-heavy base", and call the model "not a deployable solution" — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)

**DeepCoder-14B-Preview and DeepCoder-1.5B-Preview (Agentica + Together AI, Apr 2025). The 1.5B is superseded on code by ProRL v2, OpenReasoning-Nemotron-1.5B and VibeThinker.**
- Base models are DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Qwen-1.5B (the 1.5B card says "fine-tuned from DeepSeek-R1-Distilled-Qwen-1.5B") — [Together blog](https://www.together.ai/blog/deepcoder); [HF DeepCoder-1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- Data is about 24K verified problems: 7.5K TACO-Verified, 16K PrimeIntellect SYNTHETIC-1 (verified), and 600 LCB problems from 2023-05-01 to 2024-07-31. Each problem's official solution must pass all tests, and each needs **≥5 tests**, because fewer tests encouraged reward hacking. Data was deduplicated and checked for contamination against LCB 2024-08-01–2025-02-01 and 57 Codeforces contests. APPS, CodeContests, KodCode and LeetCode were rejected as noisy, unverifiable or too easy — [Together blog](https://www.together.ai/blog/deepcoder)
- The reward is a sparse outcome reward. It is 1 only if all sampled tests pass; for problems with many tests, the 15 hardest (by input length) are used. It is 0 for any failure or a missing code block. Tests time out after 6–12 s. Sandboxes were Together Code Interpreter and a local sandbox that mirrors LCB's evaluation code — [Together blog](https://www.together.ai/blog/deepcoder)
- **GRPO+** has no entropy loss (entropy loss caused exponential entropy growth and collapse) and no KL loss. It uses **overlong filtering**, which masks the loss for truncated sequences, and "clip-high" (no value given). It does offline difficulty filtering instead of DAPO's online dynamic sampling — [Together blog](https://www.together.ai/blog/deepcoder); [HF DeepCoder-1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- **Iterative context lengthening (14B)** went from 16K to 32K at step 180, and the best 32K checkpoint was evaluated at 64K with no extra training. LCB was 45.6 at 16K, 57.9 at 32K and 60.6 at 64K. The R1-Distill-14B base scored 50.2, 53.0 and 53.0. The schedule for the 1.5B is not given — [Together blog](https://www.together.ai/blog/deepcoder); [HF DeepCoder-1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- Compute was **32 H100s for about 2.5 weeks** (14B), which is about 13.4K H100-h (my estimate). The verl-pipeline "one-off pipelining" gave up to about 2× end-to-end speed for coding — [Together blog](https://www.together.ai/blog/deepcoder)
- Results, 14B (self-reported, LCB 8/1/24–2/1/25): LCB 60.6, Codeforces 1936 (95.3 percentile), HumanEval+ 92.6, AIME24 73.8. The R1-Distill-14B base scored 53.0, 1791, 92.0 and 69.7; o3-mini-low scored 60.9 and 1918 — [Together blog](https://www.together.ai/blog/deepcoder)
- Results, 1.5B (self-reported): **LCB 25.1** (base 16.9), **Codeforces 963 / 28.5 percentile** (base 615 / 1.9), **HumanEval+ 73.0** (base 58.3). No AIME score is given — [HF DeepCoder-1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- Codeforces method: Qwen CodeElo, 408 problems from 57 contests (Div. 4 to Div. 1), 8 samples per problem, Elo estimate per contest. Percentiles use 89,352 rated users from 2024 — [Together blog](https://www.together.ai/blog/deepcoder)

**DeepScaleR-1.5B-Preview (Agentica, Feb 2025; math-only RL). Superseded on math by ProRL, BroRL and JustRL.**
- Base model is DeepSeek-R1-Distill-Qwen-1.5B. Data is about 40K problem-answer pairs from AIME 1984–2023, AMC (pre-2023), Omni-MATH and STILL. Training used GRPO with KL and a binary LaTeX/SymPy answer reward — [HF DeepScaleR](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- Context schedule: 8K for steps 0–1040 (8×A100, AIME24 22.9→33%), then 16K for steps 1040–1520 (32×A100, 33→43%), then 24K (32×A100, 38→43%) — [HF DeepScaleR](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- Results: AIME24 43.1 (base 28.8), MATH-500 87.8, AMC23 73.6, average 57.0 (base 48.9). **No LCB was reported.** License is MIT — [HF DeepScaleR](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)

**Nemotron-Research-Reasoning-Qwen-1.5B: ProRL v1 (May 2025), v2 (Jul 2025), BroRL (Oct/Nov 2025)**
- Base model is DeepSeek-R1-Distill-Qwen-1.5B. ProRL data is 136K problems: math 40K (DeepScaleR), **code 24K (Eurus-2-RL/PRIME, continuous reward)**, STEM 25K (SCP-116K), logic 37K (Reasoning Gym, 96 tasks) and instruction-following 10K — [ProRL arXiv 2505.24864](https://arxiv.org/html/2505.24864)
- ProRL v1 used GRPO with DAPO decoupled clip (**ε_low 0.2, ε_high 0.4**) and dynamic sampling (drops prompts with accuracy 0 or 1). It adds a KL penalty with **periodic hard reset of the reference policy and optimizer**. Other settings: 16 rollouts per prompt (32 in some runs), **rollout temperature 1.2**, batch 256, mini-batch 64, constant LR 2e-6. Context was **8K for most of training and 16K for the last ~200 steps**. Training ran more than 2K steps on 32 H100s, about **16K GPU-hours** — [ProRL arXiv](https://arxiv.org/html/2505.24864)
- ProRL v2 extended training to more than 3K steps, with the "3K model trained with 8K context". It uses a REINFORCE++-style baseline (group-mean-subtracted, batch-normalised advantages) and clip ε 0.20/0.28. The KL uses a k2 estimator. The reference is reset every 200–500 steps or on KL spikes. A **scheduled cosine length penalty** is switched on periodically (e.g., 100 updates on, 500 off) — [NVIDIA ProRL v2 blog](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/)
- BroRL started from the ProRLv2 checkpoint (3K steps at 8K) and expanded context to 16K. It raised rollouts per prompt from **N=16 to 512** (128 prompts per step) on 64 H100s. It surpassed ProRL's final score after 98.1 h wall-clock, versus 133.8 h for ProRL. The dynamic-sampling pass rate rose from 41% to 62%, and generation throughput from 36.5 to 72.4 samples/s. The paired-test gain is statistically significant but small (ΔPass@1 0.0033) — [BroRL arXiv 2510.01180](https://arxiv.org/html/2510.01180v1)
- Code scores (self-reported, NVIDIA card; LCB version not stated):

  | Model | APPS | CodeContests | Codeforces | TACO | HumanEval+ | LCB |
  |---|---|---|---|---|---|---|
  | R1-Distill-1.5B | 20.95 | 16.79 | 14.13 | 8.03 | 61.77 | 16.80 |
  | DeepCoder-1.5B | 30.37 | 23.76 | 21.70 | 13.76 | 73.40 | 22.76 |
  | v1 | 41.99 | 31.80 | 34.50 | 20.81 | 72.05 | 23.81 |
  | v2 | 46.39 | 35.59 | 40.75 | 22.89 | 72.89 | 27.69 |
  | BroRL | 50.61 | 38.71 | 45.88 | 25.90 | – | – |

  The "Codeforces" column here is pass@1 on a problem set, not a rating — [HF Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- Math (card): AIME24/AIME25 were 48.13/33.33 for v1, 49.58/36.04 for v2 and 60.42/35.63 for BroRL. The base scored 28.54/22.71 — [HF card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)

**OpenReasoning-Nemotron-1.5B and OpenCodeReasoning-Nemotron(-1.1) (NVIDIA, 2025; SFT-only distillation)**
- OpenReasoning-Nemotron-1.5B is derived from **Qwen2.5-1.5B** by **SFT only** on about **5M DeepSeek-R1-0528 responses** (OpenCodeReasoning, OpenCodeReasoning-II, OpenMathReasoning and synthetic science). It was trained for and evaluated with **up to 64K output tokens** — [HF OpenReasoning-Nemotron-1.5B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)
- 1.5B results (self-reported): **LCB v6 (2408–2505) 28.6**, AIME24 55.5, AIME25 45.6, HMMT-Feb25 31.5, GPQA 31.6, SciCode 1.0. GenSelect (parallel sampling plus selection) lifts AIME25 to 70.0, but is reported for math only. The 7B/14B/32B models score LCB 63.3/67.8/70.2. License is CC-BY-4.0 (commercial use allowed) — [HF OpenReasoning-Nemotron-1.5B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)
- OpenCodeReasoning-Nemotron-1.1 comes in 7B/14B/32B only (**no 1.5B**). It was trained on 1.165M R1-0528 competitive-programming samples at 64K tokens. LCB v5 (2408–2501) is 55.5/65.9/69.9 for 1.1, up from 51.3/59.4/61.7 for 1.0. License is the NVIDIA Open Model License — [HF OCR-Nemotron-1.1-7B](https://huggingface.co/nvidia/OpenCodeReasoning-Nemotron-1.1-7B)

**AceReason-Nemotron 1.0/1.1 (NVIDIA, May/Jun 2025; 7B/14B only, no 1.5B found)**
- Version 1.0 does RL only, starting from R1-Distill-Qwen-7B/14B: **math-only RL first, then code-only RL**. The card reports that "math-only RL also improves code reasoning". 14B results: AIME24 78.6, AIME25 67.4, LCB v5 61.1, LCB v6 54.9, Codeforces 2024 (+543 over base). The base R1-Distill-14B scored 69.7, 50.2, 53.1 and 47.9 — [HF AceReason-Nemotron-14B](https://huggingface.co/nvidia/AceReason-Nemotron-14B)
- Version 1.1-7B builds on Qwen2.5-Math-7B with SFT then RL. Findings: scaling the *number of prompts* in SFT helps more than scaling responses per prompt. RL sampling temperature should keep **"temperature-adjusted entropy around 0.3"**. The gap between SFT starting points narrows during RL — [arXiv 2506.13284](https://arxiv.org/abs/2506.13284)
- 1.1-7B results: AIME24 72.6, AIME25 64.8, LCB v5 57.2, LCB v6 52.1, evaluated at temperature 0.6, top-p 0.95, 32K max tokens. License is the NVIDIA Open Model License — [HF AceReason-Nemotron-1.1-7B](https://huggingface.co/nvidia/AceReason-Nemotron-1.1-7B)

**Skywork-OR1 (Skywork, May 2025; 7B/32B; included for entropy-collapse insights)**
- Base models are R1-Distill-7B/32B. The algorithm is **MAGIC**: GRPO with offline and online filtering, rejection sampling of zero-advantage groups, token-level loss, **adaptive entropy control** (target entropy 0.2, step 0.005) and **no KL**. Training is multi-stage (8K→16K→longer); switching from 8K to 16K at step 540 matched from-scratch 16K training while saving about 100 training-hours — [arXiv 2505.22312](https://arxiv.org/html/2505.22312)
- Results: 7B scores AIME24 70.2, AIME25 54.6 and LCB (2024-08–2025-02) 47.6. 32B scores 82.2, 73.3 and 63.0 — [arXiv 2505.22312](https://arxiv.org/html/2505.22312)

**JustRL (Tsinghua et al., Dec 2025; ICLR 2026; math-only)**
- Base models are R1-Distill-Qwen-1.5B and OpenMath-Nemotron-1.5B. Data is DAPO-Math-17k with **no** difficulty filtering or dynamic sampling. Training is **single-stage GRPO with fixed hyperparameters**: clip [0.8, 1.28], no KL, no entropy bonus, no length penalty, no curriculum. Other settings: 16K context (15K response), batch 256, mini-batch 64, 8 rollouts, LR 1e-6 constant, temperature 1.0, and 4,380 steps for the DeepSeek-based run — [arXiv 2512.16649](https://arxiv.org/html/2512.16649v1)
- Compute was 32 A800-80GB for about 15 days per model, about 11.5K GPU-h (my estimate). The token budget was about 1.4×10⁸, versus ProRL-V2 2.8×10⁸ and BroRL 6.8×10⁸. The nine-benchmark math average was **54.87 (vs ProRL-V2 53.08)**. No code evaluation was reported — [arXiv 2512.16649](https://arxiv.org/html/2512.16649v1)

**Open-RS and STILL-3 (early-2025 low-budget math RL on R1-Distill-1.5B; superseded)**
- Open-RS used GRPO on about 7K math samples, on 4×A40 for under 24 h at a cost of **$42**. AMC23 rose from 63% to 80% and AIME24 reached 46.7%. The authors report "optimization instability and length constraints" with prolonged training — [arXiv 2503.16219](https://arxiv.org/abs/2503.16219)
- STILL-3-1.5B-preview moved from its "Backbone" to AIME 28.67→39.33, MATH 84.04→85.48 and average 42.91→49.33. The card does not name the base or the RL details — [HF STILL-3-1.5B-preview](https://huggingface.co/RUC-AIBOX/STILL-3-1.5B-preview)

**POLARIS (HKU NLP + ByteDance Seed, Jun 2025; math-only; Polaris-4B-Preview from Qwen3-4B).** The details below come from a search-result summary of the blog, not a direct fetch.
- Difficulty is calibrated from pass rates: R1-Distill-1.5B/7B run 8 attempts on each DeepScaleR-40K problem, and the data is refreshed as problems become easy. The **initial rollout temperature is 1.4 for Qwen3-4B**, because the blog calls t=0.6 and 1.0 "too low". Training ran 3 stages at 40K/48K/52K over 700 steps, taking **10 days on 32 H800** (about 7.7K GPU-h, my estimate). Results: AIME24 81.2, AIME25 79.4 (self-reported) — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris); [HF Polaris-4B-Preview](https://huggingface.co/POLARIS-Project/Polaris-4B-Preview)

**Phi-4-mini-reasoning and Phi-4-mini-flash-reasoning (Microsoft, 3.8B; math-only, no code evaluated)**
- Phi-4-mini-reasoning has four stages: mid-training on distilled long CoT, SFT, Rollout DPO, and RL with verifiable reward — [arXiv 2504.21233](https://arxiv.org/abs/2504.21233)
- The Phi-4-mini-reasoning card lists over 1M synthetic math problems with 8 R1 rollouts each; only verified-correct solutions were kept, about 30B tokens. Reasoning training used 150B tokens on 128 H100 for about 2 days (about 6.1K GPU-h, my estimate). Results: AIME24 57.5, MATH-500 94.6, GPQA-D 52.0. License is MIT, and the card says "designed and tested for math reasoning only" — [HF Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning)
- The flash model uses the SambaY hybrid architecture (SSM + Gated Memory Unit + differential attention) with 64K context. The card reports reasoning results achieved "without any reinforcement learning". Results: AIME24 52.29, AIME25 33.59. It claims **up to 10× throughput** versus Phi-4-mini-reasoning, with near-linear latency growth up to 32K generated tokens. The same card's re-evaluation gives Phi-4-mini-reasoning **AIME24 48.13**, against the 57.5 on its own card — [HF Phi-4-mini-flash-reasoning](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)

**Qwen3-0.6B/1.7B/4B (Apr/May 2025) and Qwen3-4B-Thinking-2507 (2025). Superseded by Qwen3.5 small models (2026).**
- Small Qwen3 models get **no RL of their own**. They are trained by "strong-to-weak distillation": off-policy distillation from teacher outputs in think and no-think modes, then **on-policy distillation** that matches student logits to a Qwen3-32B or 235B-A22B teacher. On Qwen3-8B, on-policy distillation reached AIME24 74.4 and LCB v5 60.3 using **1,800 GPU-h**. RL from the same starting point reached 67.6 and 52.9 using 17,920 GPU-h. Pass@64 improved after distillation but not after RL — [Qwen3 report arXiv 2505.09388](https://arxiv.org/html/2505.09388)
- Thinking-mode results (LCB v5 = 2024.10–2025.02): Qwen3-0.6B LCB 12.3, AIME24/25 10.7/15.1. Qwen3-1.7B LCB 33.2, AIME 48.3/36.8. **Qwen3-4B LCB 54.2, Codeforces 1671 (92.8%), AIME 73.8/65.6.** Base-model EvalPlus is 36.23, 52.70 and 63.53 for 0.6B, 1.7B and 4B — [Qwen3 report](https://arxiv.org/html/2505.09388)
- **Qwen3-4B-Thinking-2507**: LCB v6 (25.02–25.05) **55.2** (original Qwen3-4B: 48.4), CFEval 1852 (1671), OJBench 17.9, AIME25 81.3, HMMT25 55.5, GPQA 65.8. The card recommends 32,768 output tokens, or 81,920 for reasoning and coding, at temperature 0.6, top-p 0.95, top-k 20. License is Apache-2.0 — [HF Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)

**MiMo-7B (Xiaomi, May 2025; recipe ideas only)**
- Pretraining used about 25T tokens with reasoning-dense data plus multi-token prediction (MTP; about 90% speculative acceptance). RL used **130K rule-verified math and code problems** with these tricks: a **test-difficulty-driven code reward** for sparse code rewards (partial credit by test difficulty tier), **re-sampling of easy problems** to stabilise late-stage RL, and a "Seamless Rollout Engine" (2.29× faster training). The RL window grew from 32K to 48K — [HF MiMo-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)
- Results (LCB v5 = 20240801–20250201, v6 = 20250201–20250501). LCB v5 and v6 by variant: Base 32.9 and 29.1; RL-Zero 49.1 and 42.9; SFT 52.3 and 45.5; RL 57.8 and 49.3. AIME24/25 for the RL model: 68.2/55.4. License is MIT — [HF MiMo-7B-RL](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL)

**Data-scale reference point for a 1.5B stdin/stdout coder**
- A large verified competitive-code SFT corpus (418K problems, 580K long-reasoning solutions with generated and mutually verified tests) lifted Qwen2.5-7B from **17.4 to 57.3 on LCB** and Qwen2.5-14B from **23.3 to 62.5**, using full SFT on the whole corpus. The abstract gives no 1.5B number — [arXiv 2505.21297](https://arxiv.org/abs/2505.21297)

### Inferences
- An R1-Distill-Qwen-1.5B base plus RL alone appears to plateau around LCB 23–28, even with about 16K H100-h (ProRL). The two ≤1.5B models far above that changed the *base and SFT*: VibeThinker (55.9 v5) used Qwen2.5-Math-1.5B plus large diversity-selected SFT; OpenReasoning-Nemotron (28.6 v6) used 5M distilled traces. The data volume used in those SFT stages is orders of magnitude above 12K LoRA samples. LoRA versus full fine-tuning is also a likely confound, but none of these works used LoRA.
- The most transferable code-RL ingredients at ≤4B are these. Verified problems with ≥5 tests and an all-tests-pass binary reward (DeepCoder). Removing too-easy and impossible prompts (DeepCoder offline; ProRL/Nanbeige online or 8-rollout filters). Clip-higher, no entropy bonus, and either no KL or KL with resets. Overlong filtering when training below the target context. Up-weighting prompts near 50% pass rate (MGPO) or using more rollouts (BroRL). Temperatures of about 1.0–1.4 to avoid entropy collapse.
- On-policy distillation from a stronger teacher (Qwen3; MiniCPM5's OPD, Q2) is reported to be about 10× cheaper than RL for comparable gains. For a team with "ample compute" and access to a strong open teacher, it is a strong alternative or complement to GRPO on a 1.5B model.

### Gaps
- VibeThinker-1.5B: SFT data sizes, teacher models, trace lengths, probing-set sizes, λ, group size and RL datasets are undisclosed, and no training code was found (see Q6).
- DeepCoder-1.5B: context schedule, steps and compute are not given. DeepScaleR: LCB was never reported by its authors.
- MBPP+ is essentially unreported for these models, and HumanEval+ appears only for DeepCoder and the Nemotron-1.5B series.
- POLARIS details come from a search-result summary; the blog was not fetched.
- I did not find a ≤4B AceReason or OpenCodeReasoning-Nemotron checkpoint.

## Q2. Which newer (2026) small reasoning/coding models went viral for beating much larger models, and how were they trained?

### Takeaway
The 2026 frontier for ≤4B competitive coding has moved far beyond 2025 levels, on self-reported numbers only:
- VibeThinker-3B (Jun 2026): LCB v6 80.2.
- Nanbeige4.1-3B (Feb 2026): LCB v6 76.9.
- MiniCPM5-2B (Sep 2026): LCB v6 69.1.
- Qwen3.5-4B (Feb/Mar 2026): LCB v6 55.8.

The common ingredients are very large reasoning SFT (up to 400B tokens), long-context RL (64K–128K) run per domain, specialist experts merged by weight averaging or on-policy distillation, critic-based or entropy-aware RL, and explicit length control (Long2Short, truncation-reducing RL). None states an LCB v6 date window, and none has been independently reproduced on code.

### Cited Findings

**VibeThinker-3B (Weibo AI; arXiv 2606.16140, Jun 15 2026)**
- Base model: the paper says Qwen2.5-Coder-3B; the HF model tree lists Qwen2.5-3B → Qwen2.5-Coder-3B — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1); [HF VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B)
- SFT covers math, code, STEM, chat and instruction following. Seeds were limited to verifiable math and code with unit tests, then expanded by rewriting. Traces come from unnamed "strong teacher models", with multiple traces kept per query. Filters: n-gram (degeneracy and benchmark overlap), LLM quality checks, and correctness via sandbox or majority vote. The data is stratified by length and difficulty — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1)
- SFT stage 1: packed sequences, batch 128, LR 5e-5 cosine decay to 8e-8, **5 epochs**. SFT stage 2 **drops traces under 5K tokens** and keeps only problems where the VibeThinker-1.5B reference fails at least 75% of the time (8 rollouts), training 2 more epochs. Diversity-exploring distillation (Pass@K-selected specialists merged) is retained — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1)
- RL uses MGPO, fully **on-policy**, in a **single 64K window with no progressive context warm-up**. Stage order: Math RL, Code RL, STEM RL, then **Long2Short Math RL**, where correct rollouts get a zero-sum brevity bonus ∝1/length with λ=0.2. Prompts at 0% or 100% accuracy are removed. Next comes offline self-distillation, selecting verified RL trajectories by length-normalised student NLL within length buckets, then instruct RL. Compute is **not disclosed** — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1)
- Results (self-reported; temperature 1.0, top-p 0.95; code averaged over 8 samples): **LCB v6 80.2** (window not stated), **OJBench 38.6**, AIME25 91.4, AIME26 94.3, HMMT25 89.3, GPQA-D 70.2, IFEval 93.4. On recent LeetCode contests (Apr 25–May 31 2026, 8 contests, one-shot Python, first attempt) it scored **123/128 (96.1%)**; the paper lists GPT-5.2 at 95.3% and Claude Opus 4.6 at 86.7% — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1)
- The card's sample code uses max_new_tokens 102,400, with 60K–100K suggested for very hard math. It is not trained for tool calling or agents. License is MIT — [HF VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B)

**Nanbeige4.1-3B (Nanbeige; arXiv 2602.13367, Feb 13 2026)**
- Base model is Nanbeige4-3B-Base. SFT context was extended 32K → 64K → 256K. The final mix is code 27%, deep-search 26%, STEM 23%, tool-use 13% and general 10%. Moving from the 4-3B-SFT to the 4.1-3B-SFT took **LCB v6 from 45.5 to 62.0** — [arXiv 2602.13367](https://arxiv.org/html/2602.13367)
- RL stages: (1) point-wise GRPO with a general reward model and 8 rollouts. This **cut overlong truncation on LCB v6 from 5.27% to 0.38%**. (2) Pair-wise RL. (3) **Code RL**: stage 1 uses a pass-rate reward; stage 2 adds a **time-complexity reward only when pass rate = 1**, judged by sandbox plus an instruct model against a reference complexity bound. Problems are kept only with 1–5 successes out of 8 rollouts. (4) Agentic RL with up to 600 tool-call turns — [arXiv 2602.13367](https://arxiv.org/html/2602.13367)
- Results (self-reported; window not stated): **LCB v6 76.9** (Nanbeige4-3B-2511: 46.0; Qwen3-4B-2507: 57.4; Qwen3-32B: 55.7; Qwen3-30B-A3B-2507: 66.0). LCB-Pro Easy 81.4, Medium 28.1. AIME 2026-I 87.40 — [arXiv 2602.13367](https://arxiv.org/html/2602.13367)
- License is Apache-2.0 per a model listing (search-result summary) — [featherless listing](https://featherless.ai/models/Nanbeige/Nanbeige4.1-3B)

**MiniCPM5-2B (OpenBMB, released Sep 7 2026; on-device oriented)**
- Architecture: 2.52B dense (1.98B non-embedding), Llama architecture, 42 layers, GQA 16/2, 131K context. Post-training starts with **400B tokens of deep-thinking SFT**. Specialised **RL teachers** follow for math, code, agentic and writing, using a **critic-based RL algorithm from "JustRL II"**. **On-Policy Distillation (OPD)** then merges **16 expert RL models**, using full-vocabulary reverse KL as the advantage. RL+OPD adds +10.96 points on reasoning/general and +6.96 on agentic tasks — [HF MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
- Results (self-reported; window not stated): **LCB v6 69.1** (vs OpenBMB's runs of Qwen3.5-4B 56.4 and Qwen3.5-2B 20.2). LCB-Pro 25Q2 Easy 68.0, Medium 17.5. OJBench 32.5. SWE-bench Verified 46.4. AIME25 86.5. Deployment formats are GGUF, MLX 4-bit, GPTQ 4-bit and **LiteRT-LM for Android/iOS**. License is Apache-2.0 — [HF MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)
- Release date: Artificial Analysis scored it 15 (Index v4.2) and 14 (v4.3), against OpenBMB's claimed 23. AA still calls it "the highest Intelligence Index of any open weights model under 4B". AA measured about **19K output tokens per task (11K reasoning)**, jointly the lowest in its comparison set; Ling 3.0 Tiny used about 56K — [RITS/NYU Shanghai summary of AA](https://rits.shanghai.nyu.edu/ai/minicpm5-2b-tops-open-models-under-4b/)
- **JustRL II** (MiniCPM RL team, 2026; the blog did not render, details from the HF card). The value model is initialised from the policy, with bias calibrated to the pool pass rate. It uses token-level GAE with a **length-adaptive λ**, keeps GRPO groups and dynamic sampling, and applies **tail-only overlong control**. Data is 32,412 math problems after a 3-stage audit of about 100K (DAPO-Math, DeepScaleR, DeepMath), with problems the base solves 8/8 removed. Rollouts: 8 per prompt at T=1.0 with a **128K generation budget**. AIME25 went from 61% to **81% in about 300 steps, versus a critic-free GRPO plateau of about 74%** — [HF openbmb/JustRL-II-base-model](https://huggingface.co/openbmb/JustRL-II-base-model)

**Qwen3.5 small dense models (0.8B/2B/4B/9B; Alibaba, early Mar 2026)**
- Architecture is a hybrid Gated DeltaNet (linear attention) plus Gated Attention, with a 3:1 layer pattern, MTP, and 262K native context. Thinking is on by default for 4B; 2B defaults to non-thinking. RL was "scaled across million-agent environments". **Qwen3.5-4B: LCB v6 55.8, OJBench 24.1** (9B: 65.6 and 29.2). It recommends 32K output tokens, or 81,920 for hard programming. License is Apache-2.0 — [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- Qwen3.5-2B: the card shows no LCB score. It warns that in thinking mode the 2B is "more prone to entering thinking loops" — [HF Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- Artificial Analysis Intelligence Index (reasoning) scores: 9B 32, 4B 27, 2B 16, 0.8B 9 (search snippet; the page could not be fetched) — [Artificial Analysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)

**Other 2026 small or edge releases (lower coding relevance or no LCB)**
- NVIDIA Nemotron-Labs Dory 1.0 4B (Oct 16 2026; secondary sources only). It is a **looped** model: an 8-layer input block, a 20-layer shared block run 1–4 times per token, and an 8-layer output block. Terminal-Bench 2.1 rose from 12.5% to 22.2% going from 1 to 4 loops. One reviewer measured 4.36 tok/s on consumer hardware. License is OpenMDW-1.1. I found no LCB score — [MindStudio](https://www.mindstudio.ai/blog/nvidia-dory-1-0-4b-benchmarks) (search-result summary)
- X-Coder (Jan 2026; **7B**, not ≤4B). Trained on fully synthetic tasks, solutions and tests (the "SynthSmith" pipeline) for SFT and RL. It reports LCB v5 62.9 avg@8 and v6 55.8; v2 of the paper may report 64.0/56.5 — [arXiv 2601.06953](https://arxiv.org/abs/2601.06953v1) (search-result summary)
- LFM2.5-1.2B-Thinking (Liquid AI): 1.17B hybrid, 32K context, about 900MB on phones, MATH-500 87.96, AIME25 31.73. **No code benchmark was published** — [Liquid AI blog](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb) (search-result summary)
- MobileLLM-R1-950M (Meta, Sep 2025). Trained on about 4.2T total tokens (about 2T high-quality), it scores **LCB 19.9 vs Qwen3-0.6B 14.9**; an author's blog labels this v6 — [arXiv 2509.24945](https://arxiv.org/pdf/2509.24945) (search-result summary)

### Inferences
- If phone deployment matters, MiniCPM5-2B (Apache-2.0, LiteRT-LM build, about 19K output tokens per task on AA) and VibeThinker-1.5B/3B (MIT) are the strongest open starting points on paper. They could serve either as a base for further RL or as teachers or targets for on-policy distillation into a 1.5B student. Their 40K–100K-token generation budgets are a serious latency and memory problem on phones. Length-control stages (Long2Short, truncation-reducing RL) are therefore not optional for that goal.
- The 2026 leaders train domain or task *specialists* and then fuse them: VibeThinker merges by parameter averaging, MiniCPM5 by OPD across 16 experts. This is an alternative to an inference-time router; the specialisation is baked into one model.
- Nanbeige's second-stage *time-complexity* reward is directly relevant to stdin/stdout CP, where TLE is a common failure mode. It is cheap to add to a pass-rate reward.

### Gaps
- None of the 2026 cards or papers states the LCB v6 problem window or count, so 80.2, 76.9, 69.1 and 55.8 are not guaranteed to be comparable with each other or with 2025 numbers.
- No Codeforces Elo is reported for any 2026 ≤4B model found. VibeThinker-3B's compute and SFT data volume are undisclosed.
- The JustRL II blog did not render, and no code-RL numbers were found for it. I found no LCB numbers for Dory, Ternary Bonsai 2 or LFM2.5.
- I could not verify LinkedIn/X virality metrics or Hugging Face trending ranks directly. Virality is inferred from press and aggregator coverage (e.g., [aiweekly on VibeThinker-3B](https://aiweekly.co/alerts/vibethinker-3b-scores-943-on-aime26-matching-671b-deepseek-v32), [VentureBeat on Qwen3.5 small](https://venturebeat.com/technology/alibabas-small-open-source-qwen3-5-9b-beats-openais-gpt-oss-120b-and-can-run)).

## Q3. Which recipes start from DeepSeek-R1-Distill-Qwen-1.5B, and how much did code scores improve over that base?

### Takeaway
The base's official numbers are LCB 16.9 (window 2024-08–2025-01) and Codeforces rating 954. Code-focused or multi-domain RL from this base adds +6 to +11 LCB points: DeepCoder-1.5B reaches 25.1 self-reported (22.76 on NVIDIA's evaluation), ProRL v1 23.81 and ProRL v2 27.69. The other R1-Distill-1.5B recipes (DeepScaleR, JustRL, Open-RS, STILL-3, Thinkless, AdaptThink, DLER) are math-only or efficiency work and report no code scores. Much larger code gains at 1.5B came only from switching base and SFT: VibeThinker reports LCB v5 55.9; OpenReasoning-Nemotron-1.5B reports LCB v6 28.6.

### Cited Findings
- **Official base numbers** for DeepSeek-R1-Distill-Qwen-1.5B: AIME24 28.9 (cons@64 52.7), MATH-500 83.9, GPQA-D 33.8, **LiveCodeBench 16.9**, **Codeforces rating 954**. It is distilled from Qwen2.5-Math-1.5B on 800K R1 samples. Evaluation used **32,768 max tokens**, temperature 0.6, top-p 0.95 and 64 samples. License is MIT — [HF DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- The DeepSeek-R1 paper defines its LCB window as **2024-08 to 2025-01** — [arXiv 2501.12948](https://arxiv.org/html/2501.12948)
- Cross-lab re-evaluations of the same base agree on LCB (16.9 in Agentica's evaluation, 16.80 in NVIDIA's) but not on Codeforces. Agentica's CodeElo-style evaluation gives the base **615 (1.9 percentile)**, against DeepSeek's 954. HumanEval+ is 58.3 in Agentica's evaluation and 61.77 in NVIDIA's — [HF DeepCoder-1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview); [HF Nemotron-Research-Reasoning-Qwen-1.5B](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- **DeepCoder-1.5B** (code RL, 24K problems, GRPO+): LCB 16.9 → **25.1 (+8.2)**, Codeforces 615 → 963, HumanEval+ 58.3 → 73.0 (self-reported). NVIDIA's evaluation gives LCB 22.76 (+5.96 over 16.80) — [HF DeepCoder-1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview); [HF Nemotron card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- **ProRL v1** (2K steps, multi-domain including 24K code): LCB 16.80 → 23.81 (+7.0), APPS 20.95 → 41.99, CodeContests 16.79 → 31.80, Codeforces pass@1 14.13 → 34.50. **ProRL v2** (3K steps): LCB **27.69 (+10.9)**, Codeforces pass@1 40.75. **BroRL**: Codeforces pass@1 45.88, no LCB reported — [HF Nemotron card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- Math-only or efficiency recipes from this base, with no code scores:
  - DeepScaleR (AIME24 28.8 → 43.1) — [HF DeepScaleR](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
  - JustRL-DeepSeek-1.5B (nine-benchmark math average 54.87) — [arXiv 2512.16649](https://arxiv.org/html/2512.16649v1)
  - Open-RS (AIME24 46.7) — [arXiv 2503.16219](https://arxiv.org/abs/2503.16219)
  - Thinkless, AdaptThink, AutoThink and DLER-R1-1.5B (see Q4) — [Thinkless HF](https://huggingface.co/Vinnnf/Thinkless-1.5B-RL-DeepScaleR); [AdaptThink arXiv](https://arxiv.org/abs/2505.13417); [DLER arXiv](https://arxiv.org/html/2510.15110v1)
- STILL-3-1.5B-preview's card calls its start point "Backbone" (AIME 28.67), without naming it — [HF STILL-3](https://huggingface.co/RUC-AIBOX/STILL-3-1.5B-preview)
- For contrast, 1.5B models from **other** bases: VibeThinker-1.5B (Qwen2.5-Math-1.5B) reports LCB v5 55.9 (Aug 2024–Feb 2025), a window close to R1's, which is about 39 points above 16.9 — [arXiv 2511.06221](https://arxiv.org/html/2511.06221). OpenReasoning-Nemotron-1.5B (Qwen2.5-1.5B, SFT-only) reports LCB v6 (2408–2505) 28.6 — [HF OpenReasoning-Nemotron-1.5B](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)

### Inferences
- STILL-3's backbone is very likely R1-Distill-Qwen-1.5B: its 28.67 AIME matches the official 28.9. This is an inference, not stated on the card.
- At 1.5B, RL from R1-Distill gives a reliable but modest code gain of about 6–11 LCB points, and costs thousands to about 16K H100-hours. By comparison, the team's pipeline goes from 8% to 11% Pass@1 on a non-LCB set, which cannot be placed on this scale. To make claims comparable, report LCB v5 (2024-08-01–2025-02-01) and/or v6 (2025-02-01–2025-05-01) avg@8 at temperature 0.6, top-p 0.95, ≥32K tokens, plus a CodeElo-style Codeforces rating, alongside the 16.9/954 base reference.
- Because the Codeforces rating of the same base varies between 954 and 615 depending on the evaluator, only within-paper Codeforces deltas are meaningful.

### Gaps
- DeepSeek's Codeforces method for distilled models was not visible in the fetched text.
- NVIDIA's LCB version for its 16.80–27.69 numbers is unstated.
- No published code-RL result from R1-Distill-1.5B beyond ProRL v2/BroRL was found for 2026.

## Q4. What do these works report about verbose reasoning, response-length growth or shrinkage, truncation at max length, and entropy collapse in small models (plus evidence for thinking-budget routing)?

### Takeaway
Small models fail in two opposite ways. They learn poorly from very long teacher CoT and tend to loop or ramble. Under RL, entropy and length either collapse (when there are length penalties or low temperature) or blow up (when there is an entropy bonus). Recipes that work keep length unpenalised early: overlong filtering, no length penalty, temperature ≥1.0, clip-higher. They add length control late: Long2Short, scheduled cosine penalties, truncation penalties. Learned "think or not" gates on R1-Distill-1.5B cut tokens by about 50% while matching or improving accuracy, though only on math.

### Cited Findings
- **Learnability gap**: models with ≤3B parameters do not reliably benefit from long-CoT data or large teachers. They do better with **shorter CoT or smaller teachers**. "Mix-Long" (long and short CoT at a fixed ratio) improved Qwen2.5-3B-Instruct by more than 8 points on MATH/AMC compared with long-CoT-only training — [arXiv 2502.12143](https://arxiv.org/abs/2502.12143)
- **DeepCoder**: mean response length grew from about 8K to 17.5K tokens under GRPO+. Without overlong filtering, short-context training "hurt initial performance and shortened responses". An entropy loss caused exponential entropy growth and collapse, while clip-high with no entropy loss kept entropy stable. The 14B evaluated at 16K scores only 45.6 LCB because long responses are truncated — [Together blog](https://www.together.ai/blog/deepcoder)
- **ProRL**: raising temperature to 1.2 "only delays entropy collapse". DAPO clipping plus KL plus reference resets avoided it. Response length first drops briefly, then rises, and "length is not decisive" — [ProRL arXiv](https://arxiv.org/html/2505.24864)
- **ProRL v2**: describes conventional RL as producing "verbose, lengthy outputs" and uses a periodically scheduled cosine length penalty. It reports a 17.54% shorter output on math; the blog is inconsistent about which checkpoint this refers to — [NVIDIA blog](https://developer.nvidia.com/blog/scaling-llm-reinforcement-learning-with-prolonged-training-using-prorl-v2/)
- **JustRL**: with no length penalty, length *fell* naturally from about 7–8K to 4–5K tokens by about step 1,000 and stayed there, with stable entropy (about 1.2–1.4). **Adding DAPO's overlong penalty dropped AIME24 from about 55% to about 50% and entropy to about 0.5–0.6.** Adding a robust verifier as well dropped AIME24 to about 45% — [arXiv 2512.16649](https://arxiv.org/html/2512.16649v1)
- **Skywork-OR1**: off-policy updates (more SGD steps per batch or data reuse) accelerate entropy collapse and hurt test accuracy. Temperature 0.6 caused early collapse, while 1.0 kept entropy higher. Advantage-masking of truncated responses gave **no gain** at 32K — [arXiv 2505.22312](https://arxiv.org/html/2505.22312)
- **AceReason-1.1**: tune sampling temperature so that temperature-adjusted entropy stays at about 0.3 — [arXiv 2506.13284](https://arxiv.org/abs/2506.13284). **POLARIS**: Qwen3-4B needs an initial temperature of 1.4 (search-result summary) — [POLARIS blog](https://hkunlp.github.io/blog/2025/Polaris). **MGPO** (VibeThinker) builds an entropy-like prior into the advantage weights — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- **Nanbeige4.1-3B**: a point-wise RL stage cut **overlong truncation on LCB v6 from 5.27% to 0.38%** — [arXiv 2602.13367](https://arxiv.org/html/2602.13367). **VibeThinker-3B** adds a final Long2Short math RL stage that gives a brevity bonus to correct rollouts only (λ=0.2) — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1). **JustRL II** uses "tail-only overlong control" with a 128K budget — [HF JustRL-II](https://huggingface.co/openbmb/JustRL-II-base-model)
- **Looping and rambling**: Qwen3.5-2B's card warns it is "more prone to entering thinking loops" in thinking mode — [HF Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B). HN users report VibeThinker-1.5B going into endless repetition on stories and games, "rambling thinking" and "thinking loops" (anecdotal) — [Hacker News](https://news.ycombinator.com/item?id=45910410)
- **DLER** (NVIDIA): batch-wise reward normalisation, higher clipping, dynamic sampling and a simple **truncation penalty** (zero reward above 4,000 tokens) on the DeepScaleR data with 16 rollouts. It cut DeepSeek-R1-1.5B's average length by about 80%, to about 2,466 tokens, with accuracy maintained. Difficulty-aware DLER cut a further 15%. It is research-only — [arXiv 2510.15110](https://arxiv.org/html/2510.15110v1); [GitHub NVlabs/DLER](https://github.com/NVlabs/DLER)
- **Think/no-think routing at 1.5B** (all math, all from R1-Distill-Qwen-1.5B):
  - **Thinkless** uses a control token trained with "Decoupled GRPO". On Minerva Algebra it used long thinking on only 25.88% of samples. Code and a checkpoint are released — [Thinkless HF](https://huggingface.co/Vinnnf/Thinkless-1.5B-RL-DeepScaleR); [EmergentMind summary](https://www.emergentmind.com/articles/2505.13379)
  - **AdaptThink**: −53% average length and +2.4% accuracy on three math sets — [arXiv 2505.13417](https://arxiv.org/abs/2505.13417)
  - **AutoThink**: −52% tokens and +6.4% relative accuracy (search-result summary) — [sotaverified 2505.10832](https://sotaverified.org/papers/2505-10832)
- **Qwen3 thinking budget**: when a user-set token threshold is reached, the system injects "Considering the limited time by the user, I have to give the solution based on the thinking directly now". The behaviour emerges from thinking-mode fusion and was not explicitly trained. Accuracy rises smoothly with budget — [Qwen3 report](https://arxiv.org/html/2505.09388)
- **Token cost in practice**: AA measured MiniCPM5-2B at about 19K output tokens per task, versus Ling 3.0 Tiny at about 56K — [RITS summary of AA](https://rits.shanghai.nyu.edu/ai/minicpm5-2b-tops-open-models-under-4b/)

### Inferences
- The team's Stage-1 result (about 0–1% Pass@1 after long-CoT SFT, improved by short structured traces) matches the published ≤3B "learnability gap". The literature fix is *mixing* long and short traces (Mix-Long), not choosing one. RL with overlong filtering then lets length grow only as far as reward justifies.
- For a phone target, the evidence favours shaping length late. First train for accuracy without length penalties (JustRL and DeepCoder show early penalties collapse exploration). Then apply a Long2Short, DLER-style truncation penalty or cosine-schedule stage, and/or a learned think/no-think gate (Thinkless/AdaptThink). This is closer to proven practice than a separate topic router.
- A separate "router" that predicts algorithm topic and difficulty to set a thinking budget is plausible: Qwen3 shows budget cut-offs degrade gracefully, and AdaptThink/Thinkless show difficulty-gated thinking works at 1.5B. However, no source found tests topic-routing for competitive programming.

### Gaps
- No source found measures truncation rates or loop frequency specifically for R1-Distill-1.5B on LCB, or for LoRA-SFT'd 1.5B coders.
- All think/no-think routing results found are on math, not code.
- I found no study of algorithm-topic classifiers or retrieval routers for CP at ≤4B. This is a genuine gap, and it could be a contribution.

## Q5. Which weights, datasets and training code are open (licenses, HF ids, GitHub), and what hardware / GPU-hours were used?

### Takeaway
Most 1.5B RL recipes are fully open: DeepCoder/DeepScaleR (MIT, rllm), JustRL (thunlp/JustRL), Thinkless, AdaptThink and DLER. NVIDIA's ProRL checkpoints are CC-BY-NC (non-commercial), while OpenReasoning-Nemotron is CC-BY-4.0. The strongest small models (VibeThinker-1.5B/3B MIT, MiniCPM5-2B Apache-2.0, Nanbeige4.1-3B Apache-2.0, Qwen3/3.5 Apache-2.0) release weights but not full training data or pipelines. Disclosed compute for 1.5B RL runs ranges from $42 (Open-RS) to about 16K H100-h (ProRL).

### Cited Findings

| Model | Weights (HF id) / license | Data / code released | Hardware / compute | Source |
|---|---|---|---|---|
| VibeThinker-1.5B | WeiboAI/VibeThinker-1.5B, MIT | GitHub WeiboAI/VibeThinker; training data and code not described as released | ~3,900 H800-h (~$7.8K) | [HF](https://huggingface.co/WeiboAI/VibeThinker-1.5B); [arXiv](https://arxiv.org/html/2511.06221) |
| VibeThinker-3B | WeiboAI/VibeThinker-3B, MIT | same GitHub | undisclosed | [HF](https://huggingface.co/WeiboAI/VibeThinker-3B); [arXiv](https://arxiv.org/html/2606.16140v1) |
| DeepCoder-14B/1.5B | agentica-org/DeepCoder-14B-Preview, -1.5B-Preview, MIT | dataset agentica-org/DeepCoder-Preview-Dataset; code agentica-project/rllm + verl-pipeline; W&B logs | 14B: 32 H100 × ~2.5 weeks | [Together blog](https://www.together.ai/blog/deepcoder); [HF 1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview) |
| DeepScaleR-1.5B | agentica-org/DeepScaleR-1.5B-Preview, MIT | DeepScaleR-Preview-Dataset (~40K) | 8→32 A100-80GB; ~3.8K A100-h ($4.5K, per VibeThinker's table) | [HF](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview); [arXiv 2511.06221](https://arxiv.org/html/2511.06221) |
| Nemotron-Research-Reasoning-Qwen-1.5B (v1/v2/BroRL) | nvidia/Nemotron-Research-Reasoning-Qwen-1.5B, **CC-BY-NC-4.0**, research only | public datasets (DeepScaleR, Eurus-2-RL, Reasoning-gym, SCP-116K, IFEval) | v1: 32 H100, ~16K GPU-h; BroRL: 64 H100 | [HF](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B); [ProRL](https://arxiv.org/html/2505.24864); [BroRL](https://arxiv.org/html/2510.01180v1) |
| OpenReasoning-Nemotron-1.5B | nvidia/OpenReasoning-Nemotron-1.5B, CC-BY-4.0 | built on OpenCodeReasoning(-II), OpenMathReasoning | evaluated on H100 (training compute not stated) | [HF](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B) |
| OCR-Nemotron-1.1 (7/14/32B) | NVIDIA Open Model License | OpenCodeReasoning (1.165M R1-0528 samples) | not stated | [HF](https://huggingface.co/nvidia/OpenCodeReasoning-Nemotron-1.1-7B) |
| AceReason-Nemotron(-1.1) | NVIDIA Open Model License | — | not stated | [HF 14B](https://huggingface.co/nvidia/AceReason-Nemotron-14B); [HF 1.1-7B](https://huggingface.co/nvidia/AceReason-Nemotron-1.1-7B) |
| Skywork-OR1 | open weights; report says code, RL data and weights open-sourced (license not captured) | yes | ablations on 32 H800 | [arXiv](https://arxiv.org/html/2505.22312) |
| JustRL | HF collection hbx/justrl; GitHub thunlp/JustRL (license not stated) | DAPO-Math-17k | 32 A800 × ~15 days per model | [arXiv](https://arxiv.org/html/2512.16649v1) |
| JustRL II | openbmb/JustRL-II-base-model (RL-init checkpoint) | reference code claimed (search summary) | not stated | [HF](https://huggingface.co/openbmb/JustRL-II-base-model) |
| Open-RS | GitHub knoveleng/open-rs (code and data) | ~7K math samples | 4×A40, <24 h, $42 | [arXiv](https://arxiv.org/abs/2503.16219) |
| POLARIS-4B | POLARIS-Project/Polaris-4B-Preview, Apache-2.0 (README front-matter, search summary) | recipe and data per blog | 32 H800 × 10 days | [HF](https://huggingface.co/POLARIS-Project/Polaris-4B-Preview); [blog](https://hkunlp.github.io/blog/2025/Polaris) |
| Phi-4-mini-reasoning / flash | microsoft/…, MIT | — | reasoning: 128 H100 × ~2 days; flash pretraining: 1024 A100 × ~14 days | [HF reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning); [HF flash](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning) |
| Qwen3-4B-Thinking-2507 / Qwen3.5-2B/4B | Qwen/…, Apache-2.0 | — | Qwen3-8B distillation 1,800 GPU-h vs RL 17,920 | [HF](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507); [HF](https://huggingface.co/Qwen/Qwen3.5-4B); [report](https://arxiv.org/html/2505.09388) |
| MiniCPM5-2B | openbmb/MiniCPM5-2B, Apache-2.0; GGUF/MLX/GPTQ/LiteRT-LM | agent SFT data released separately | not stated | [HF](https://huggingface.co/openbmb/MiniCPM5-2B) |
| Nanbeige4.1-3B | Nanbeige/Nanbeige4.1-3B, Apache-2.0 (listing) | — | not stated | [arXiv](https://arxiv.org/html/2602.13367); [listing](https://featherless.ai/models/Nanbeige/Nanbeige4.1-3B) |
| MiMo-7B-RL | XiaomiMiMo/MiMo-7B-RL, MIT | — | not stated | [HF](https://huggingface.co/XiaomiMiMo/MiMo-7B-RL) |
| DLER-R1-1.5B | nvidia/DLER-R1-1.5B-Research, research only | GitHub NVlabs/DLER | not stated | [GitHub](https://github.com/NVlabs/DLER) |
| Thinkless / AdaptThink | Vinnnf/Thinkless-1.5B-RL-DeepScaleR; THU-KEG/AdaptThink | code released | not stated | [HF](https://huggingface.co/Vinnnf/Thinkless-1.5B-RL-DeepScaleR); [arXiv](https://arxiv.org/abs/2505.13417) |

### Inferences
- For a university paper aiming at a deployable phone model, license matters. The ProRL/BroRL 1.5B checkpoints are CC-BY-NC research-only, as is DLER. VibeThinker (MIT), DeepCoder (MIT), OpenReasoning-Nemotron-1.5B (CC-BY-4.0), MiniCPM5-2B and Qwen3/3.5 (Apache-2.0) are permissive.
- Rough GPU-hour scale for one serious 1.5B RL run: about 4K A100-h (DeepScaleR), about 11–16K H100/A800-h (ProRL, BroRL, JustRL; my products of GPU count × wall-clock), and about 3.9K H800-h for the whole VibeThinker SFT+RL.

### Gaps
- Licenses for JustRL, Skywork-OR1 and the VibeThinker GitHub repo were not confirmed.
- Training compute is not disclosed for VibeThinker-3B, MiniCPM5-2B, Nanbeige4.1-3B, OpenReasoning-Nemotron or the Qwen3.5 small models.
- Whether VibeThinker training data or code is released was not verified from the GitHub repo itself.

## Q6. Are there independent replications or credible critiques (contamination, benchmark overfitting, cherry-picked evaluations) of headline claims such as VibeThinker's?

### Takeaway
No independent reproduction of VibeThinker-1.5B's or VibeThinker-3B's LCB/AIME numbers was found. The evidence against taking them at face value is the following:
- Credible contamination findings for its Qwen2.5-Math base family, which bear on math benchmarks more than on time-windowed LCB.
- Anecdotal user reports of looping and weak general coding.
- Undisclosed SFT data, and internal number inconsistencies.

The authors' defence rests on 10-gram decontamination, LCB time windows (the base scores 0.0 on LCB), and self-run post-release LeetCode contests. More broadly, cross-lab re-evaluations routinely move small-model scores by several points. The one neutral evaluation found (Artificial Analysis on MiniCPM5-2B) scored it well below its vendor claim.

### Cited Findings
- **No independent reproduction found.** The main VibeThinker results are the authors' own. A low-reliability wiki states that no independent lab had reproduced them at release. A third-party review notes the repository ships neither training code nor the post-training method — [search-result summary of Miraheze wiki / Hysen Labs](https://hysenlabs.com/projects/weiboai-vibethinker) (not fetched; treat as weak evidence)
- **Contamination of the base family.** Wu et al. show that Qwen2.5-Math-7B can regenerate the remaining 40% of MATH-500 problems from a 60% prefix with **54.6% exact match**. Spurious or random-reward RL boosts Qwen2.5-Math but not Llama, and on a post-release clean set (RandomCalculation) only correct rewards help. VibeThinker-1.5B is built on Qwen2.5-Math-1.5B, so math-benchmark gains (MATH-500, AIME) inherit this concern — [arXiv 2507.10532](https://arxiv.org/html/2507.10532v2)
- **Authors' defences.** VibeThinker-1.5B applied 10-gram decontamination to SFT and RL data, and its base scores 0.0 on LCB v5/v6 — [arXiv 2511.06221](https://arxiv.org/html/2511.06221). VibeThinker-3B reports 123/128 on 8 LeetCode contests held Apr 25–May 31 2026. This is self-run (one-shot Python, 4 rollouts per problem), and its paper describes n-gram decontamination — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1)
- **Internal inconsistencies.** VibeThinker-1.5B's text gives LCB v5 as 55.1 in one place and 55.9 in the tables and abstract — [arXiv 2511.06221](https://arxiv.org/html/2511.06221). VibeThinker-3B's test-time-scaling (CLR) row puts 72.9 in the LCB column while the text assigns 72.9 to GPQA — [arXiv 2606.16140](https://arxiv.org/html/2606.16140v1). DeepCoder's text gives LCB 54%→58% at 16K/32K, while its table shows 45.6/57.9 — [Together blog](https://www.together.ai/blog/deepcoder)
- **Comparability caveat stated by VibeThinker itself.** Its LCB v6 uses 131 problems (Feb–May 2025). Some comparators use 454 problems (Aug 2024–May 2025), "which generally yields higher scores". Cross-paper v6 comparisons are therefore not apples-to-apples — [arXiv 2511.06221](https://arxiv.org/html/2511.06221)
- **Anecdotal independent use** (HN, Nov 2025):
  - "Both of those resulted in an endless loop" on story and game prompts.
  - SpringBoot coding answers "don't compare so favorably to their benchmarks".
  - "Implausibly good for a 1.5B model … probably just rote learning".
  - On the other hand, "weirdly good at some simple python questions", running at about 100 tok/s on a GTX 1650.

  No commenter ran a systematic contamination test — [Hacker News](https://news.ycombinator.com/item?id=45910410)
- **Cross-lab score drift for the same models.**
  - DeepCoder-1.5B LCB: 25.1 self-reported vs 22.76 by NVIDIA.
  - R1-Distill-1.5B Codeforces: 954 by DeepSeek vs 615 by Agentica.
  - Qwen3-4B-2507 LCB v6: 55.2 by Qwen vs 57.4 in Nanbeige's table.
  - Qwen3.5-4B LCB v6: 55.8 by Qwen vs 56.4 by OpenBMB.
  - Phi-4-mini-reasoning AIME24: 57.5 on its own card vs 48.13 in Microsoft's flash-model card.

  Sources: [HF DeepCoder-1.5B](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview); [HF Nemotron](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B); [HF R1-Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B); [HF Qwen3-4B-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507); [Nanbeige arXiv](https://arxiv.org/html/2602.13367); [HF Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B); [HF MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B); [HF Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning); [HF Phi-4-mini-flash](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)
- **Same benchmark name, different numbers.** BroRL's paper reports code scores (e.g., APPS 61.59, Codeforces 60.86) far above those on NVIDIA's own model card for the same model (APPS 50.61, Codeforces 45.88). This implies a different evaluation protocol — [BroRL arXiv](https://arxiv.org/html/2510.01180v1); [HF Nemotron card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- **Neutral evaluation of a 2026 claim.** Artificial Analysis scored MiniCPM5-2B at 15 (v4.2) / 14 (v4.3) on its Intelligence Index, against OpenBMB's claimed 23, while still ranking it #1 among open models under 4B — [RITS summary of AA](https://rits.shanghai.nyu.edu/ai/minicpm5-2b-tops-open-models-under-4b/)
- **Narrowness admitted by the releasers.** VibeThinker-1.5B is "not a deployable solution" and its GPQA is 20–40 points behind leaders — [arXiv 2511.06221](https://arxiv.org/html/2511.06221). VibeThinker-3B is not for tool or agent use — [HF VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B). Phi-4-mini-reasoning and flash are "math only" — [HF Phi-4-mini-flash](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)

### Inferences
- VibeThinker's *code* claims are harder to dismiss as contamination than its math claims. LCB is time-windowed, the base scores 0.0, and the 3B's LeetCode test is post-dated. The more likely failure modes are narrow specialisation (competition-format only), evaluation-setting sensitivity (40K tokens, 8 samples, 131-problem v6), and poor robustness outside the format. A team adapting the recipe should run its own LCB v5/v6 evaluation, plus a fresh post-cutoff contest set (e.g., the latest Codeforces/LeetCode rounds), under identical settings for every baseline.
- Treat any ≤4B LCB v6 number in the 69–80 range (VibeThinker-3B, Nanbeige4.1-3B, MiniCPM5-2B) as provisional until independently re-run. A prudent paper would cite them as "self-reported".

### Gaps
- I found no third-party LCB or Codeforces re-run of VibeThinker-1.5B/3B, Nanbeige4.1-3B or MiniCPM5-2B. Artificial Analysis pages could not be fetched directly (certificate error), so their per-benchmark coding numbers were not checked.
- The Wu et al. contamination critique targets Qwen2.5-Math on math benchmarks. No study was found testing VibeThinker's own training data for LCB overlap.
