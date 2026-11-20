# RLVR and Test-Time Methods for ~1.5B Competitive-Programming Code Models (state of the art as of 2026-11-19)

Tags used below: **[SR]** = self-reported by the authors; **[3P]** = measured or re-evaluated by a third party; **[NR]** = taken from the paper's abstract or main text as known before this session and *not re-opened in this session* (well-known numbers, but spot-check before quoting in a paper); **[2nd]** = only seen via a secondary summary or search snippet, not the primary text. Benchmarks: LCB = LiveCodeBench; "R1-Distill-1.5B" = DeepSeek-R1-Distill-Qwen-1.5B (the brief's base model).

---

## 1. RLVR algorithms and training tricks for code (2025–2026): which matter at 1.5B, typical hyperparameters, GPU-hours

### Takeaway
At 1.5B, the strongest published code-RL results all start from R1-Distill-1.5B. They use a GRPO/DAPO-family critic-free objective with clip-higher, dynamic (zero-variance) sampling, overlong filtering and a short-then-longer context schedule. They raise LCB pass@1 from ~17% to ~23–30%. The choice among GRPO, DAPO, GSPO and CISPO seems second-order next to data and test quality, exploration/entropy control, and context budget. Reported 1.5B runs cost roughly 10^3–10^4 GPU-hours. Two cheaper alternatives deserve a look. On-policy distillation from a strong teacher was ~10x cheaper than RL at 8B in Qwen3's study, and LoRA appears adequate for RL.

### Cited Findings
**Baselines for R1-Distill-1.5B (important for calibrating the team's 8–11%)**
- R1-Distill-1.5B scores LCB pass@1 16.9, Codeforces rating 615 (1.9th percentile) and HumanEval+ 58.3, evaluated with long contexts [SR by Agentica] — [Together AI / Agentica DeepCoder blog](https://www.together.ai/blog/deepcoder)
- NVIDIA's independent evaluation of the same base gives APPS 20.95, CodeContests 16.79, Codeforces 14.13, TACO 8.03, HumanEval+ 61.77, LCB 16.80 (avg 23.08) [3P] — [ProRL, arXiv 2505.24864](https://arxiv.org/html/2505.24864)

**1.5B code-RL results (chronological; earlier ones are superseded)**
- *DeepCoder-1.5B-Preview (Apr 2025; superseded at 1.5B)* was RL'd from R1-Distill-1.5B on the same 24K-problem set as DeepCoder-14B. It reached LCB 25.1 (vs 16.9), Codeforces rating 963 (28.5th percentile, vs 615) and HumanEval+ 73.0 (vs 58.3) [SR]. ProRL's re-evaluation gives DeepCoder-1.5B LCB 22.76 and HumanEval+ 73.40 [3P]. Archer's re-evaluation gives LCB v5 23.3 (avg@8) [3P]. — [DeepCoder blog](https://www.together.ai/blog/deepcoder); [ProRL](https://arxiv.org/html/2505.24864); [Archer-Code-1.5B card](https://huggingface.co/Fate-Zero/Archer-Code-1.5B/blob/main/README.md)
- *ProRL / Nemotron-Research-Reasoning-Qwen-1.5B (May 2025)* reports these code pass@1 scores: APPS 41.99, CodeContests 31.80, Codeforces 34.50, TACO 20.81, HumanEval+ 72.05, LCB 23.81; average 37.49 vs 23.08 for the base (+14.4 points) [SR] — [ProRL](https://arxiv.org/html/2505.24864)
- *Archer-Code-1.5B (Jul 2025)* reports LCB v5 (avg@8) 29.4, vs 26.0 for the authors' own DAPO baseline, 26.1 for Nemotron-1.5B and 23.3 for DeepCoder-1.5B. On LCB v6 (avg@16) it reports 30.2, vs 27.6 for DAPO and 29.5 for Nemotron-1.5B. The method is "dual-token constraints": weaker KL and higher clip thresholds on high-entropy "reasoning" tokens, stronger constraints on low-entropy "knowledge" tokens [SR]. A third-party aggregator lists different LCB figures for the same work (e.g., 42.1 v5 avg@8), which suggests different evaluation setups. — [Archer card](https://huggingface.co/Fate-Zero/Archer-Code-1.5B/blob/main/README.md); [arXiv 2507.15778](https://arxiv.org/html/2507.15778v1); [WizWand listing](https://www.wizwand.com/paper/68fa7a2c1e209131cea81848)
- *Note:* the three groups use different LCB versions, date windows and avg@k, so cross-paper LCB numbers at 1.5B differ by 2–3 points for the same checkpoint (above).

**Recipes in detail**
- *DeepCoder "GRPO+"* [SR]:
  - Changes to GRPO: entropy loss removed (it made entropy blow up and training collapse); KL loss removed, following DAPO, which also saves reference log-prob compute; clip-high (value not given in the blog); overlong filtering, i.e. masking the loss on truncated sequences.
  - Context schedule: 16K→32K, switching at step 180; evaluation at 64K. Mean response length grew from 8K to 17.5K tokens.
  - 14B context trade-off: LCB 45.6 at 16K vs the base's 50.2 at 16K, but 60.6 at 64K vs the base's 53.0.
  - Compute (14B): 32 H100s for ~2.5 weeks, 1,200–2,500 s per step.
  - Systems: verl-pipeline gives "up to 2.5x" speedup over verl; on 8xA100 with the 1.5B model, one-off pipelining halved end-to-end coding training time.
  - Not reported in the blog: learning rate, batch size and samples per prompt.
  - Source: [DeepCoder blog](https://www.together.ai/blog/deepcoder)
- *DeepScaleR-1.5B (Feb 2025, math, same base)*: GRPO with KL and a binary reward on ~40K problems, with iterative context lengthening [SR]:
  - 8K for steps 0–1040 on 8 A100-80GB, batch 128x8: AIME24 22.9→33%.
  - 16K for steps 1040–1520 on 32 A100s, batch 128x16: 33→43%.
  - Then 24K.
  - Final AIME24 43.1 vs 28.8 for the base.
  - Source: [DeepScaleR model card](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview)
- *ProRL hyperparameters (1.5B)* [SR]:
  - Objective: GRPO plus a KL penalty, with periodic hard resets of the reference policy and optimizer state; DAPO decoupled clip with ε_low=0.2 and ε_high=0.4; dynamic sampling that drops prompts with accuracy 0 or 1.
  - Sampling: rollout temperature 1.2, which the authors say only delays entropy collapse; n=16 rollouts per prompt.
  - Optimisation: batch 256, mini-batch 64 (4 updates per rollout step), AdamW at a constant learning rate of 2e-6.
  - Schedule: 8K context for most of training and 16K for the last ~200 steps; more than 2K steps over 8 runs.
  - Evaluation: temperature 0.6, top-p 0.95, 32K tokens.
  - Data: 136K problems, of which 24K are code (Eurus-2-RL), the rest math, STEM, logic and instruction following.
  - Compute: 4x8 H100-80GB, **~16K GPU-hours**, verl.
  - Source: [ProRL](https://arxiv.org/html/2505.24864)
- *JustRL (Dec 2025, math only; ICLR 2026)*: a single-stage recipe with fixed hyperparameters on two 1.5B reasoning models. It reports 54.9% and 64.3% averages on 9 math benchmarks with "2x less compute than sophisticated approaches". The abstract says "explicit length penalties and robust verifiers may degrade performance by collapsing exploration" [SR]. Secondary summaries give: 4,380 steps; 32 A800-80GB for ~15 days (≈11.5K A800-hours); a token budget of ~1.4x10^8 k tokens vs ProRL-V2's 2.8x10^8 k and BroRL's 6.8x10^8 k; 54.87% vs ProRL-V2's 53.08% [2nd]. — [arXiv 2512.16649](https://arxiv.org/abs/2512.16649); [ICLR 2026 slides](https://iclr.cc/media/iclr-2026/Slides/10012134_I8OS4k1.pdf); [Moonlight review](https://www.themoonlight.io/fr/review/justrl-scaling-a-15b-llm-with-a-simple-rl-recipe)
- *DAPO (Mar 2025)* [NR]:
  - Four techniques: Clip-Higher (ε_low=0.2, ε_high=0.28), Dynamic Sampling, token-level policy-gradient loss, and Overlong Reward Shaping (soft penalty, 16,384 expected length plus a 4,096 "cache", 20,480 maximum).
  - Optimisation: learning rate 1e-6, 512 prompts x 16 responses.
  - Result: AIME24 50 on Qwen2.5-32B base, vs 47 for DeepSeek-R1-Zero-Qwen-32B, in 50% of the steps.
  - Source: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476). The same 0.2/0.28 clip values are used by NousCoder (below).
- *Dr. GRPO*: removes GRPO's per-response length normalisation and per-group std normalisation, which bias training toward long wrong answers; Oat-Zero-7B reaches AIME24 43.3% [NR] — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- *GSPO (Qwen, Jul 2025)*: a sequence-level importance ratio with sequence-level clipping; stabilises MoE RL and was used for later Qwen3 models [NR] — [arXiv 2507.18071](https://arxiv.org/abs/2507.18071)
- *CISPO (MiniMax-M1, Jun 2025)*: clips importance-sampling weights instead of dropping token gradients. MiniMax-M1's full RL took 3 weeks on 512 H800s (~$534.7K rental) [NR] — [arXiv 2506.13585](https://arxiv.org/abs/2506.13585)
- *VAPO*: value-based (PPO with a critic); 60.4 on AIME24 with Qwen2.5-32B [NR] — [arXiv 2504.05118](https://arxiv.org/abs/2504.05118). *REINFORCE++*: critic-free REINFORCE with PPO tricks and global (batch) advantage normalisation [NR] — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262). *RLOO*: leave-one-out baseline [NR] — [arXiv 2402.14740](https://arxiv.org/abs/2402.14740)
- *ScaleRL (Meta et al., Oct 2025; ICLR 2026 oral)*: a >400K GPU-hour study that fits sigmoidal compute-performance curves. The recipe combines asynchronous PipelineRL, forced length interruptions, CISPO loss, prompt-level loss averaging, batch-level advantage normalisation and FP32 logits. CISPO reached a higher asymptote than DAPO, and a 100K GPU-hour run followed the predicted curve [SR/2nd] — [arXiv 2510.13786](https://arxiv.org/pdf/2510.13786); [ICLR 2026](https://iclr.cc/virtual/2026/oral/10010608)
- *"Tricks or Traps" (Alibaba, Aug 2025)* [2nd]:
  - "Lite PPO" (group-mean / batch-std advantage normalisation plus token-level loss aggregation) beats GRPO and DAPO on non-aligned base models.
  - Overlong filtering helps short and medium reasoning tasks but gives limited benefit on long-tail tasks.
  - For aligned models, sequence-level aggregation can be better.
  - The authors warn that conclusions vary by model family.
  - Source: [arXiv 2508.08221](https://arxiv.org/html/2508.08221v2); [HF page](https://huggingface.co/papers/2508.08221)
- *NousCoder-14B (Jan 2026; Qwen3-14B base)* [SR]:
  - Objective comparison: DAPO, GSPO and GSPO+ differed by only ~0.3 LCB points at 40K (63.35 / 63.13 / 63.06).
  - Settings: KL 0; DAPO clip 0.2/0.28; overlong filtering (zero advantage when truncated); context 32K, then 40K from step 50; group size 8; 1,024 sequences per batch; learning rate 1e-6; training temperature 1.0; PipelineRL-k=3.
  - Data: 24K problems (TACO-Verified, SYNTHETIC-1, LCB before Aug 2024); ~10K always-solved problems removed offline, and problems dropped online once at 100%.
  - Compute: 48 B200 for 4 days.
  - Results: LCB v6 60.79→67.87 (80K context with YaRN); pass@G (G=8) 75.11→79.52.
  - Source: [Nous Research](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model)
- *AceReason-Nemotron (NVIDIA, May 2025)*: math-only RL first also raised LCB by +6.8 and +5.8 for 7B and 14B; code-only RL afterwards added more code gains with little math loss [NR] — [arXiv 2505.16400](https://arxiv.org/abs/2505.16400)
- *Entropy control*: policy entropy collapses early in RLVR, and performance tracks R ≈ −a·e^H + b. The paper proposes Clip-Cov and KL-Cov [NR] — [arXiv 2505.22617](https://arxiv.org/abs/2505.22617)
- *Length control / anti-"yapping" RL at 1.5B*:
  - ThinkPrune reports halving R1-Distill-1.5B's AIME24 reasoning length with ~2% accuracy loss by RL under a token limit [NR] — [arXiv 2504.01296](https://arxiv.org/abs/2504.01296)
  - L1/LCPO trains 1.5B models to follow a target length [NR] — [arXiv 2503.04697](https://arxiv.org/abs/2503.04697)
  - Contrary evidence: JustRL found explicit length penalties can collapse exploration (above).
- *Distillation vs RL for small models*: Qwen3 compared RL with on-policy distillation from the same Qwen3-8B checkpoint. RL took 17,920 GPU-hours; on-policy distillation took 1,800. AIME'24 was 67.6 (RL) vs 74.4 (distillation), and AIME'25 55.5 vs 65.5. The report says RL did not improve pass@64 [SR] — [Qwen3 Technical Report, arXiv 2505.09388](https://arxiv.org/pdf/2505.09388)
- *Learnability gap*: small models (≤3B) learn better from shorter, simpler CoT than from long teacher traces [NR] — [arXiv 2502.12143](https://arxiv.org/abs/2502.12143). This bears on why Stage 1, LoRA SFT on long traces, failed.
- *LoRA for RL (Thinking Machines, "LoRA Without Regret")* [SR]:
  - LoRA "fully matches" full fine-tuning for policy-gradient RL "even with ranks as low as 1"; the argument is that RL gives O(1) bits per episode.
  - The optimal LoRA learning rate is ~10x the full-FT learning rate.
  - MLP-only LoRA clearly beats attention-only.
  - LoRA is less tolerant of large batch sizes.
  - Experiments: Llama-3.1-8B on MATH and GSM8K, and Qwen3-8B-base on DeepMath with an 8,192-token cap.
  - Source: [thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)

### Inferences
- **Priority ordering at 1.5B for code** (my synthesis of the above):
  1. Verified training problems with strong tests (Section 2).
  2. Remove zero-variance groups through dynamic sampling and offline difficulty filtering. A ~10%-pass@1 model produces many all-fail groups that carry no GRPO signal.
  3. Exploration control: clip-high 0.28–0.4 or token-entropy-aware constraints (Archer: +3.4 LCB v5 over DAPO at exactly 1.5B), no entropy bonus, training temperature 1.0–1.2.
  4. A context schedule that starts short (8K–16K).
  5. The objective itself (GRPO vs DAPO vs GSPO vs CISPO), which matters least at this scale. CISPO/GSPO mainly buy stability in long, asynchronous or MoE runs.
- **KL**: drop it for runs of a few hundred steps (DAPO, DeepCoder, NousCoder). For runs of thousands of steps, consider a small KL with periodic reference resets (ProRL).
- **A starting configuration for a 1.5B code run** (synthesised from DeepCoder, ProRL, NousCoder and DAPO; not a published recipe):
  - Objective: token-level GRPO/DAPO loss, ε_low/ε_high 0.2/0.28 (up to 0.4), dynamic sampling, overlong filtering.
  - Sampling: 8–16 rollouts per prompt, temperature 1.0.
  - Optimisation: 64–256 prompts per step, constant learning rate 1e-6 to 2e-6 for full fine-tuning (≈1e-5 to 2e-5 for LoRA, per the ~10x rule, with LoRA on MLP layers too).
  - Context: 8K, then 16K.
  - Reward: binary, on ≥5 (ideally 15+) verified tests.
  - Evaluation: temperature 0.6, top-p 0.95.
- **Expected gain**: published 1.5B code RL moves LCB from ~17 to ~23–30 (+6 to +13 points) and average code-benchmark pass@1 by ~+14 (ProRL). Other things equal, the team should expect gains of a similar order on their stdin/stdout set, not a multiple.
- **Compute**: published 1.5B runs are ProRL ~16K H100-hours (multi-domain, >2K steps) and JustRL ~11.5K A800-hours (math). DeepScaleR's first 8K stage ran on only 8 A100s. My unverified extrapolation for a code-only run of ~500–1,000 steps at 8–16K context: ~1–3K H100-hours.
- **Deployment budget**: DeepCoder-14B was *worse* than its base at 16K after training at 32K with overlong filtering (45.6 vs 50.2). If the deployment budget is short, as on a phone, train at or near that budget, or use length-control RL. Accept JustRL's warning that hard length penalties can hurt exploration.
- **Baselines and confidence intervals**: the base R1-Distill-1.5B scores ~17% on LCB and ~21% on APPS when given long budgets. The team's 8% and 11% (on different 200- and 100-problem sets) should therefore be compared against the *untouched base model under the same harness and token budget*; the SFT stages may be *below* base. Wilson 95% intervals are wide: 8% of 200 → ~5.0–12.6%; 11% of 100 → ~6.3–18.6%. The 8% vs 11% difference is not statistically distinguishable.
- **Alternatives given ample compute**: on-policy distillation from a strong reasoning-coder teacher, followed by a short RLVR polish, is a plausible cheaper path than long RL from a weak base. This is extrapolated from the Qwen3 8B evidence, not shown at 1.5B for code.

### Gaps
- DeepCoder-1.5B's exact RL hyperparameters (learning rate, batch, rollouts, steps, GPU-hours) are not in the blog. They may be in the rllm/verl-pipeline scripts, which were not retrieved.
- DeepScaleR's total GPU-hours (often quoted as ~3,800 A100-hours in Agentica's blog) could not be verified; the model card does not list them.
- No head-to-head comparison of GRPO, DAPO, GSPO and CISPO on *code at 1.5B* was found. NousCoder's comparison is at 14B; ScaleRL's is at 8B and MoE scale.
- No 2026 paper claiming a better 1.5B code-RL model than Archer-Code-1.5B was found. RLVE (ICML 2026) trains ProRL-1.5B-v2 and OpenThinker3-1.5B on 400 verifiable environments and evaluates on LCB v6, but its numbers were not retrieved — [RLVE repo](https://github.com/Zhiyuan-Zeng/RLVE).
- The LoRA-for-RL evidence is from math at 8B. It is untested in the sources for long-CoT code RL at 1.5B.

---

## 2. Reward design for code, test-suite quality and false positives, sandboxes, and checkers

### Takeaway
The dominant 2025–26 practice is a binary all-tests-pass reward on problems with verified, sufficiently many tests. The required tests are ≥5, and ideally the 15 longest or hardest, with time and memory limits. Partial credit can help sparse-reward regimes but must be calibrated against "easy-test" bias and hacking. Test quality is the biggest lever. Weak tests (TACO-style) can make RL *hurt* pass@1, and ~25% of competitive-programming problems need a special checker. The open-r1 verifiable-coding-problems set documents neither how its tests were made nor how they were validated, so the team's evaluation set should be audited.

### Cited Findings
- **Binary reward rationale (DeepCoder)** [SR]: reward 1 only if all sampled tests pass; 0 for any failure or a malformed answer such as a missing code block. There is no separate format reward. 15 tests are sampled per problem, chosen as the longest inputs, with a 6–12 s timeout per test. Partial rewards were rejected because the model could hack them by memorising public test outputs or handling only simple edge cases. Problems need ≥5 tests, because hacking was seen with fewer. — [DeepCoder blog](https://www.together.ai/blog/deepcoder)
- **Data filtering (DeepCoder)**: 24K problems (7.5K TACO-Verified, 16K PrimeIntellect SYNTHETIC-1, 600 LCB from May 2023 to Jul 2024). APPS, CodeContests, KodCode and LeetCode were rejected as too easy or as having noisy or flawed tests. The official solution must pass all tests; data were deduplicated and decontaminated against the LCB test window. — [DeepCoder blog](https://www.together.ai/blog/deepcoder)
- **NousCoder** uses a binary ±1 reward: −1 for a wrong answer, exceeding 15 s, or exceeding 4 GB on any test [SR] — [Nous Research](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model)
- **Partial credit, used successfully at 1.5B (ProRL)**: the code reward is the *fraction of tests passed*; compile or syntax errors and runs over a 5 s total timeout score 0 [SR]. No binary-vs-partial ablation is reported. — [ProRL](https://arxiv.org/html/2505.24864). open-r1's code reward also returns "the overall success rate" — [open-r1 GitHub](https://github.com/huggingface/open-r1)
- **Difficulty-weighted partial credit (MiMo-7B, Xiaomi)**: a "test difficulty driven reward" buckets tests by pass rate across rollouts (IOI-style). A "strict" variant gives the reward only if a group and all easier groups pass; a "soft" variant spreads each group's score across its tests. A summary says the soft variant worked better [2nd]. — [arXiv 2505.07608](https://arxiv.org/abs/2505.07608); [EmergentMind summary](https://www.emergentmind.com/papers/2505.07608)
- **VeRPO (Jan 2026)**: partial-success rewards suffer a "cardinality bias" that over-rewards easy-test gains. VeRPO adds a "dynamic, density-calibrated local reward" combined with the global outcome. It reports up to **+8.83 pass@1** over outcome-driven and reward-model baselines at "<0.02%" time cost [SR]. Model sizes are not on the abstract page. — [arXiv 2601.03525](https://arxiv.org/abs/2601.03525)
- **RobustTests (Aug 2026, EMNLP 2026)**: insufficient test coverage leads to "reward hacking and policy degradation". The method synthesises tests from "near-correct" faulty programs, filters them with validator agents and behavioural clustering, and adds a stepwise dense pass-rate reward. It reports a +3% absolute LCB gain for Qwen3-32B on an augmented CodeContests+ [SR]. — [arXiv 2608.24135](https://arxiv.org/abs/2608.24135)
- **HardTests (May 2025)** [SR]:
  - Data: 26.6K problems from 13 online judges.
  - Test precision vs existing suites, for programs written by Qwen2.5-Coder-7B (precision/recall): TACO 61/78.97, CodeContests 51.64/85.98, HardTests 79.21/91.69. On the hardest bucket TACO precision is **17.83 vs 55.88**, meaning most "passing" programs under TACO tests are actually wrong.
  - Input generation: LLM-written inputs, random generators, category generators and "hack" generators, plus an LLM-written input validator.
  - Outputs: up to 8 human oracle programs, kept only if two oracles agree on >90% of inputs.
  - **25.4% of problems need a special judge**; GPT-4o writes these at ~$0.23 per problem.
  - Downstream (Qwen3-4B, LCB-105 in C++): base pass@1/pass@10 is 38.48/56.19. GRPO with TACO tests gives 36.95/57.14 (pass@1 *drops*); with HardTests, 39.42/64.76.
  - Source: [arXiv 2505.24098](https://arxiv.org/html/2505.24098)
- **CodeContests+ (ByteDance Seed, Jun 2025)**: an LLM-agent pipeline for test generation, validated against **1.72M labelled submissions**. It reports "significantly higher accuracy than CodeContests", with a notably higher true-positive rate, and "considerable advantages for RL" [SR]. Exact figures and checker details were not retrieved. — [arXiv 2506.05817](https://arxiv.org/abs/2506.05817)
- **TestCase-Eval (ACL 2025)**: 500 Codeforces problems and 100K human solutions; measures fault coverage (Coverage@N) and fault exposure for 19 LLMs. Reasoning models are much better at fault exposure [SR/2nd]. — [arXiv 2506.12278](https://arxiv.org/abs/2506.12278); [ACL Anthology](https://aclanthology.org/2025.acl-short.82)
- **The team's evaluation-set source**:
  - open-r1/verifiable-coding-problems-python is "all Python problems from PrimeIntellect's verifiable-coding-problems" (~35.7K rows), reformatted only.
  - Tests are in `verification_info.test_cases` (input, output, fn_name, type such as `stdin_stdout`); metadata include difficulty and time/memory limits.
  - The card says nothing about how tests were generated or validated, or about multi-answer problems.
  - The upstream card (144,169 rows; "a subset of the task data used to construct SYNTHETIC-1") is also silent, and `gold_standard_solution` is nullable.
  - Sources: [open-r1 dataset card](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python); [PrimeIntellect card](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems)
- **Sandboxes**:
  - *Together Code Interpreter*: 3¢ per problem, 100+ concurrent sandboxes, 1K+ executions per minute. DeepCoder also used a "local sandbox": a guard-railed Python subprocess on stdin/stdout running the official LCB evaluation code. Each RL iteration evaluated 1,024 problems with ≥5 tests each [SR]. — [DeepCoder blog](https://www.together.ai/blog/deepcoder)
  - *Modal*: one container per rollout, with inference and verification pipelined — [Nous Research](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model)
  - *open-r1*: E2B (Python), Morph (Python/JS/C++/Rust) and Piston (Codeforces/IOI), with CPU "router" jobs to avoid sandbox rate limits. Its GRPO demo trains Qwen2.5-1.5B-Instruct with 1 vLLM GPU and 7 training GPUs. The repo is now "no longer maintained" and points to TRL. — [open-r1 GitHub](https://github.com/huggingface/open-r1)
  - *SandboxFusion*: ByteDance's open-source multi-language execution and judging sandbox [NR] — [GitHub](https://github.com/bytedance/SandboxFusion)

### Inferences
- **Reward recommendation**: use a binary all-pass reward over gold-verified tests (≥5, ideally 10–15 including large and edge inputs, with a per-test time limit close to the problem's), with 0 for runtime errors, timeouts or missing code.
  - Because the model starts around 10% pass@1, many groups will be all-zero. Pair binary rewards with dynamic sampling and difficulty filtering first.
  - Try calibrated partial credit (MiMo soft or VeRPO-style) only as an ablation. Raw pass-fraction rewards carry an easy-test bias.
  - Do not add a standalone format reward. Section 3 shows format-only rewards can move Qwen-lineage models spuriously.
- **Fix the tests before the algorithm.** The HardTests result (TACO tests reduce pass@1; good tests raise pass@10 by ~8.6 points) suggests test quality is worth more than any GRPO variant. One route is to have the team's Gemma-3-27B write input generators, validators and checkers (HardTests-style), with the dataset's gold solutions as oracles.
- **Audit the evaluation set** (this affects the paper's headline numbers):
  1. Run every gold solution on every test and drop problems where gold fails or times out.
  2. Flag problems whose statements allow multiple answers ("print any…", floating tolerance, permutations) and add checkers. Otherwise correct outputs are scored "Wrong Answer"; that is HardTests' 25.4% special-judge share.
  3. Report how many tests each problem has. The 69% "wrong answer" share in the failure analysis may be inflated by missing checkers.
- **Throughput sizing**: e.g., 128 prompts x 8 rollouts x 15 tests ≈ 15K executions per step. At the 1K+/min cloud figure that is ~15 min per step. A local, resource-limited subprocess sandbox parallelised over many CPU cores, as DeepCoder did, is the practical choice for Python stdin/stdout.

### Gaps
- How PrimeIntellect's verifiable-coding-problems tests were produced (which model, whether gold solutions verify them) is undocumented on both cards. The two dataset discussions on the hub were not read.
- CodeContests+'s exact true- and false-positive rates and checker coverage were not retrieved.
- No controlled *binary vs partial* reward ablation for code at 1.5B was found. DeepCoder asserts hacking without published numbers; ProRL used partial credit without an ablation.
- TestCase-Eval's headline gap between best model and human was not retrieved from the primary text.
- Documentation for the sandboxes in rllm, NeMo-RL and OpenRLHF, and verl's SandboxFusion integration, was not retrieved in this session.

---

## 3. Co-evolving test generators, self-play, label-free RL, and small-model caveats

### Takeaway
Co-evolving coder and tester (CURE/ReasonFlux-Coder) and self-play (AZR) work at ≥4B and ≥3B respectively, with gains of a few points. Label-free RL (TTRL, INTUITOR, one-shot RLVR) is mostly demonstrated on math and on Qwen models. Those are exactly the models where spurious rewards also "work", so a Qwen-Math-lineage 1.5B like R1-Distill needs random-reward and format-only controls and contamination-free evaluation. The "RL doesn't expand reasoning" debate leans the same way for weak bases: RL mostly converts pass@k into pass@1. Prolonged RL (ProRL) and high-quality code tests (HardTests) show some pass@k expansion.

### Cited Findings
- **CURE / ReasonFlux-Coder (Jun 2025)**: co-evolves a coder and a unit tester from their interaction outcomes, *without ground-truth code*. It reports +5.3% code-generation accuracy and +9.0% Best-of-N accuracy over Qwen2.5-Instruct, and +8.1% on downstream test-time scaling and agentic coding. ReasonFlux-Coder-4B (long-CoT) "consistently outperforms Qwen3-4B" and shows 64.8% inference efficiency in unit-test generation. The model works as a reward model for RL on base models [SR]. The smallest size is 4B. — [arXiv 2506.03136](https://arxiv.org/abs/2506.03136)
- **CodeT (2022)**: generates tests with the same model and ranks solutions by dual execution agreement. It lifted code-davinci-002 HumanEval pass@1 to 65.8% (+18.8 absolute) [NR] — [arXiv 2207.10397](https://arxiv.org/abs/2207.10397)
- **UTGen / UTDebug (2025)**: trains LLMs to produce error-revealing unit tests with correct expected outputs. It names a trade-off between generating failure-exposing inputs and predicting correct outputs without the gold solution [NR] — [arXiv 2502.01619](https://arxiv.org/abs/2502.01619)
- **Agentic verifier (Feb 2026; Qwen-team authors)**: an execution-based agent searches for "highly discriminative test inputs" across candidate solutions. It is trained by data synthesis, then rejection fine-tuning, then agentic RL, and reports "+10–15% absolute gains in Best@K" on five competitive-programming benchmarks [SR] — [arXiv 2602.04254](https://arxiv.org/abs/2602.04254)
- **Absolute Zero Reasoner (May 2025; NeurIPS 2025)**: self-play in which one model proposes and solves code-reasoning tasks (deduction, abduction, induction) verified by a Python executor, with zero external data. It claims SOTA coding among zero-data models, and +10.9 / +15.2 math-average gains for AZR-Base-7B / AZR-Coder-7B [SR] — [arXiv 2505.03335](https://arxiv.org/abs/2505.03335); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116121). The paper also reports gains growing with size (Coder 3B/7B/14B ≈ +5.7/+10.2/+13.2) [NR; verify].
- **One-shot RLVR (Apr 2025)**: one training example raised Qwen2.5-Math-1.5B MATH500 from 36.0% to 73.6% [NR] — [arXiv 2504.20571](https://arxiv.org/abs/2504.20571)
- **TTRL (Apr 2025)**: RL on unlabeled test data using majority-vote pseudo-labels. It reports a >150% relative AIME24 pass@1 gain for Qwen2.5-Math-7B (the exact figure differs across versions) and claims to exceed the initial maj@n ceiling [NR] — [arXiv 2504.16084](https://arxiv.org/abs/2504.16084)
- **INTUITOR (May 2025)**: replaces the GRPO reward with self-certainty, the KL divergence from uniform, computed online by the current policy. Trained on MATH from Qwen2.5-3B, it reports a **65% relative LCB gain vs no gain for GRPO**, and CRUXEval-O +76% vs +44%. A summary says Qwen2.5-1.5B went from 0% to 9.9% on LCB from MATH-only training [SR/2nd]. Prompt sensitivity is noted. — [arXiv 2505.19590](https://arxiv.org/abs/2505.19590); [code](https://github.com/sunblaze-ucb/Intuitor)
- **SDPO, "RL via Self-Distillation" (Jan 2026)**: the model, conditioned on rich feedback such as runtime errors, serves as its own teacher, and its feedback-informed predictions are distilled back into the policy. The abstract reports better sample efficiency and final accuracy than RLVR baselines on LCB v6, and at test time "the same discovery probability as best-of-k sampling or multi-turn conversations with 3x fewer attempts" [SR]. A search snippet gave 48.8% vs 41.2% vs GRPO on LCB v6, model size unknown [2nd]. — [arXiv 2601.20802](https://arxiv.org/abs/2601.20802)
- **Spurious rewards (Jun 2025)** [SR]:
  - On Qwen2.5-Math-7B, MATH-500 improves by **+21.4** with *random* rewards, +13.8 with format-only rewards, +24.1 with incorrect labels, and +29.1 with ground truth.
  - The effect also appears for Qwen2.5-Math-1.5B, but largely fails on Llama3 and OLMo2.
  - Mechanism: "code reasoning" frequency rises from 65% to over 90%. With clipping disabled, the random-reward gains disappear.
  - The authors urge validating RLVR on diverse models.
  - Sources: [arXiv 2506.10947](https://arxiv.org/abs/2506.10947); [project blog](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f)
- **Contamination explanation**: Qwen2.5 appears contaminated on common math benchmarks. On a clean synthetic set, only accurate rewards help [NR] — [arXiv 2507.10532](https://arxiv.org/abs/2507.10532)
- **"Does RL really incentivize reasoning beyond the base model?" (NeurIPS 2025 oral)**: RLVR beats the base at small k, but the base wins at large k, across math, coding and visual reasoning. Six RLVR algorithms "perform similarly". Distillation, unlike RL, can introduce new reasoning patterns [SR]. — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837)
- **ProRL counter-evidence (1.5B)**: on some tasks the base fails at any number of samples while the RL model reaches 100%. RL gains correlate *negatively* with the base's pass@128, so gains are largest where the base is weak. Pass@128 still *declines* on some math tasks, while code keeps improving with prolonged training [SR] — [ProRL](https://arxiv.org/html/2505.24864)
- **Code-specific pass@k evidence**: with high-quality tests, GRPO on Qwen3-4B raised LCB pass@10 from 56.19 to 64.76 — [HardTests](https://arxiv.org/html/2505.24098). NousCoder raised pass@8 from 75.11 to 79.52 — [Nous Research](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model). Qwen3's report says RL did not improve pass@64 at 8B — [Qwen3 report](https://arxiv.org/pdf/2505.09388).

### Inferences
- **Mandatory controls for this base model**: R1-Distill-1.5B descends from Qwen2.5-Math-1.5B, the family where random and format rewards produce gains. Any RL result in the paper should include:
  - a random-reward or format-only baseline;
  - evaluation on post-cutoff problems, e.g., LCB v6 (Aug 2024 to May 2025), plus the team's audited set.
  Otherwise reviewers can attribute gains to elicitation or contamination.
- **Headroom check**: measure the model's pass@k (k = 8–64) on the evaluation set before RL. Under the Yue et al. view, RL will mostly close the pass@1–pass@k gap. ProRL suggests sustained runs can add a bit beyond it for weak bases.
- **Co-evolution and test generation**: evidence exists only at ≥4B (CURE) and ≥7B (CodeRM, Section 5). At 1.5B, a cheaper and safer split is to generate tests offline with a larger model (the team's Gemma-27B or a reasoning model) using the HardTests pipeline. Optionally, also train the 1.5B to emit a few *inputs* (no expected outputs needed) for differential testing at inference time (Section 4).
- **Label-free RL (TTRL, INTUITOR, AZR) at 1.5B on code is risky**:
  - Majority-vote pseudo-labels need the model to be right more often than wrong. At ~10% pass@1 that fails, and for code "voting" must use execution agreement on generated inputs (CodeT-style), not string equality.
  - INTUITOR's self-certainty may reinforce confident "yapping".
  - AZR's own scaling trend suggests small models gain least.
- **SDPO-style self-distillation from execution feedback** is the most relevant 2026 idea for the team's runtime-error share (25% of failures), since tracebacks are rich feedback. It is unverified at 1.5B.

### Gaps
- No co-evolving coder/tester result below 4B, and no TTRL result on competitive programming, was found.
- AZR results for models below 3B were not retrieved.
- SDPO's model sizes and exact LCB numbers were not retrieved from the primary text.
- RLVE's 1.5B results on LCB v6 were not retrieved.

---

## 4. Test-time scaling and self-repair for small code models (gains per compute; phone feasibility)

### Takeaway
For small models, the largest test-time gains come from *parallel sampling plus execution-based filtering and selection*, not from self-repair alone. S* lifts Qwen2.5-Coder-1.5B on LCB from 7.0% to 27.6%. Majority vote alone gives 11.0%, and self-debug alone gives 9.4%. Self-repair is limited by feedback quality and is often no better than i.i.d. sampling at equal budget, unless the model is *trained* for multi-turn repair (RLEF, μCode). On a phone, a 1.5B 4-bit model decodes at roughly 10–25 tokens/s. Best-of-N over multi-thousand-token reasoning traces is therefore impractical. On-device test-time scaling is viable only with short outputs, small N and early stopping.

### Cited Findings
- **S* (Feb 2025)**: N=16 samples at temperature 0.7, then 2 rounds of debugging on public tests, then "adaptive input synthesis" selection with GPT-4o-mini as the selector LLM. LCB (v2) pass@1, zero-shot → majority vote / self-debug / S* [SR]:

  | Model | Zero-shot | Majority vote | Self-debug | S* |
  |---|---|---|---|---|
  | Qwen2.5-Coder-0.5B | 1.2 | 2.5 | 2.4 | 10.9 |
  | Qwen2.5-Coder-1.5B | 7.0 | 11.0 | 9.4 | 27.6 |
  | Qwen2.5-Coder-3B | 18.4 | 25.2 | 27.8 | 42.7 |
  | Qwen2.5-Coder-7B | 29.4 | 40.5 | 39.9 | 54.4 |
  | R1-Distill-7B | 48.4 | 58.7 | 58.4 | 73.2 |
  | R1-Distill-32B | 69.1 | 75.8 | 70.1 | 85.7 |

  More from the paper [SR]:
  - Debugging gains plateau after ~2–3 rounds.
  - Temperature 0.95 did not improve pass@N.
  - Qwen2.5-Coder-7B gains >35 points in oracle pass@N at N=64.
  - On CodeContests, S* lifts Qwen-Coder-7B from 1.8 to 10.9, and 12.1 with an oracle selector.
  - Selection-policy averages (LCB v4, N=8): public tests only 53.8, generated tests 53.1, LLM judge 55.6, adaptive input synthesis 57.5. At Qwen-Coder-7B the first three all scored 42.3, and adaptive scored 44.5.
  - R1-Distill models were tested only at 7B and above.
  - Source: [arXiv 2502.14382](https://arxiv.org/html/2502.14382)
- **Repeated sampling**: coverage grows log-linearly with samples. On SWE-bench Lite, DeepSeek-Coder-V2 went from 15.9% with 1 sample to 56% with 250. Majority vote and reward models plateau after a few hundred samples, so verification is the bottleneck [NR] — [Large Language Monkeys, arXiv 2407.21787](https://arxiv.org/abs/2407.21787)
- **AlphaCode**: massive sampling, filtering on example tests (removing ~99% of samples), clustering by behaviour on model-generated inputs, and 10 submissions. It ranked in the top 54.3% on average in contests [NR] — [arXiv 2203.07814](https://arxiv.org/abs/2203.07814). **CodeMonkeys** (SWE-bench, not competitive programming) combines serial and parallel scaling with model-written tests for selection: 57.4% on SWE-bench Verified for ~$2.3K [NR] — [arXiv 2501.14723](https://arxiv.org/abs/2501.14723)
- **Self-repair at equal budget (Olausson et al., ICLR 2024)**:
  - Counting repair cost, self-repair is "not a silver bullet". GPT-3.5 with repair was at or below the no-repair baseline at every budget.
  - GPT-4 gained only from about 66% to 71% at a 7K-token budget.
  - Spending the budget on diverse initial samples beats extensive repair.
  - Better feedback is the main lever: human feedback raised GPT-4's repaired-pass share from 33.3% to 52.6%.
  - Sources: [arXiv 2306.09896](https://arxiv.org/abs/2306.09896); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9ddc141bdbf9d1db510cefff56c586ad-Abstract-Conference.html); [EmergentMind summary](https://www.emergentmind.com/papers/2306.09896)
- **Older prompting-based self-repair (2023–24, superseded for reasoning models)**:
  - Self-Debug: up to +12% on TransCoder and MBPP, matching baselines that sample >10x more [NR] — [arXiv 2304.05128](https://arxiv.org/abs/2304.05128)
  - Reflexion: 91% HumanEval pass@1 with GPT-4 vs 80% [NR] — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)
  - LDB (runtime block-level debugging): up to +9.8% [NR] — [arXiv 2402.16906](https://arxiv.org/abs/2402.16906)
- **Multi-turn RL with execution feedback**:
  - *RLEF (Meta, Oct 2024)*: PPO over a multi-turn loop in which public-test feedback is given between attempts and the reward comes from private tests. Per a secondary summary, Llama-3.1-8B CodeContests 1@3 rose from 4.1% to 12.5% and 70B from 25.9% to 37.5% (validation). The paper reports about an order of magnitude fewer samples needed [2nd]. — [arXiv 2410.02089](https://arxiv.org/abs/2410.02089); [deep-paper summary](https://deep-paper.org/en/paper/2410.02089); [EmergentMind](https://www.emergentmind.com/papers/2410.02089)
  - *μCode (ICML 2025)*: treats code generation as "one-step recoverable" and trains a learned verifier and a generator by expert iteration using single-step rewards. At inference it runs multi-turn Best-of-N (N=5) with public tests plus the learned verifier, and it beats multi-turn baselines on MBPP and HumanEval [SR]. — [arXiv 2502.20380](https://arxiv.org/abs/2502.20380); [project page](https://portal.cs.cornell.edu/muCode)
  - *SDPO (2026)*: at test time, matches best-of-k or multi-turn discovery with 3x fewer attempts [SR] — [arXiv 2601.20802](https://arxiv.org/abs/2601.20802)
- **Compute-optimal allocation**: allocating test-time compute by problem difficulty is >4x more efficient than best-of-N [NR] — [Snell et al., arXiv 2408.03314](https://arxiv.org/abs/2408.03314). CodeRM scales the number of unit tests per problem by difficulty, and harder problems gain more [SR] — [arXiv 2501.01054](https://arxiv.org/abs/2501.01054)
- **On-device speed (independent measurement, Jun 2026)**: Qwen2.5-1.5B at 4 bits, warm, greedy decoding [3P] — [arXiv 2603.23640](https://arxiv.org/html/2603.23640):
  - *iPhone 16 Pro (MLX, Metal GPU)*: decode peaks at 40.5 tok/s and plateaus at **23.7 tok/s** under sustained load (−41.5% from thermal throttling); prompt throughput ~606 tok/s.
  - *Galaxy S24 Ultra (Snapdragon 8 Gen 3, MLC-LLM, OpenCL)*: decode **~10.4–10.8 tok/s**; ~1.59 W; **~146 mJ per token**. These figures are at a 2,048-token context.
- **Other on-device data points**: a community llama.cpp table lists Qwen2.5-1.5B Q5 at 4.2 tok/s on a Snapdragon 865 [unverified] — [HF dataset](https://huggingface.co/datasets/dispatchAI/per-chip-benchmark-matrix). PhoneLM reports ~600+ tok/s NPU prefill for 1.5B models on Snapdragon 8 Gen 3 — [arXiv 2411.05046](https://arxiv.org/pdf/2411.05046)
- **Trace lengths after RL**: DeepCoder-14B's mean response grew from 8K to 17.5K tokens during RL; ProRL trains 1.5B at 8–16K context — [DeepCoder blog](https://www.together.ai/blog/deepcoder); [ProRL](https://arxiv.org/html/2505.24864)

### Inferences
- **Per-compute gains for ~1.5B**:
  - Selection without execution (majority vote) gives ~+4 points (7.0→11.0 at N=16).
  - Self-debug alone gives ~+2.4.
  - Sampling, public-test filtering, debugging and execution-based selection together give ~+20 points (≈4x relative). That costs up to ~48 generations plus selector calls, and the selector in S* was GPT-4o-mini, not the small model itself.
  - Oracle pass@N bounds what any selector can reach. S* closes most of that gap only for larger models.
- **Recommended inference pipeline for stdin/stdout problems**:
  1. Sample N small (4–16) at temperature ≈0.7.
  2. Run each candidate on the statement's example tests and drop failures. This is AlphaCode's ~99% filter and costs only CPU.
  3. Cluster the survivors by their outputs on model- or random-generated inputs, which needs no expected outputs, and submit from the largest cluster.
  4. For runtime errors only, do 1–2 repair rounds conditioned on the traceback. Tracebacks are the high-quality feedback Olausson et al. found necessary; wrong-answer feedback is weak.
  5. Stop at the first candidate that passes examples and agrees with the others.
- **Phone budget** (worked numbers; my arithmetic, with battery capacity assumed ~15–20 Wh for a flagship):
  - One 8K-token R1-style trace takes ~13 min on an S24 Ultra or ~6 min on an iPhone 16 Pro.
  - Best-of-16 at 4K tokens each (64K tokens) takes ~1.7 h or ~45 min and ~9.3 kJ (~2.6 Wh, ~13–17% of a battery) on the S24 Ultra.
  - The short "blueprint" format (~600 tokens) with N=8 (~4.8K tokens) takes ~8 min or ~3.4 min and ~0.7 kJ. A single blueprint sample takes ~1 min or ~25 s.
  - Long-context decode is also slower, and the KV cache larger, than the 2K-context measurements.
  - So on-device test-time scaling needs short traces, small N, early exit, and RL done at a short budget (Section 1).
- **Router idea**: a small difficulty or topic router fits the compute-optimal-allocation evidence (Snell; CodeRM dynamic scaling). Spend N=1 with no thinking on predicted-easy problems and larger N or more debug rounds on predicted-hard ones. Its benefit hinges on difficulty-prediction accuracy, which is not established for competitive programming at this scale.
- **Multi-turn training**: if self-repair is wanted on device, it should be *trained*, via RLEF-style multi-turn RL, μCode-style expert iteration with single-step rewards, or SDPO-style self-distillation. Prompting a 1.5B model to self-repair is unlikely to beat i.i.d. resampling at equal tokens.

### Gaps
- No S*-style test-time scaling numbers were found for R1-Distill-1.5B or other ≤3B *reasoning* models. S* covers only non-reasoning Qwen2.5-Coder at ≤3B.
- μCode's model sizes and exact numbers, and RLEF's primary-table numbers, were not retrieved (only secondary summaries).
- No peer-reviewed llama.cpp decode numbers for a 1.5B model on a current Snapdragon were found. No measurements of long-context (8–32K) decode or of on-device Python sandboxing on phones were found.
- No published estimate of how accurately a tiny router predicts competitive-programming topic or difficulty was found in this research scope.

---

## 5. Verifiers for code (outcome/process reward models, generative verifiers, LLM-as-judge): useful at ~1.5B?

### Takeaway
For stdin/stdout competitive programming, execution-grounded verification dominates in the evidence: public tests, generated inputs for differential testing, and test-generating verifiers. Learned judges add little at small scale. S* found an LLM judge gave no gain over public tests at 7B. The successful learned verifiers are ≥4–8B test *generators* (CodeRM-8B, CURE, the 2026 agentic verifier), not scalar judges. No evidence was found that a 1.5B ORM, PRM or generative verifier for code is useful.

### Cited Findings
- **CodeRM-8B (ACL 2025)**: Llama-3.1-8B fine-tuned as a unit-test generator used for Best-of-N. It reports +18.43% for Llama3-8B on HumanEval+, a Best-of-N result "on par with Llama3.1-70B", and that scaling the number of tests helps more on harder problems [SR] — [arXiv 2501.01054](https://arxiv.org/abs/2501.01054); [ACL Anthology](https://preview.aclanthology.org/setup/2025.acl-long.343)
- **S* selection ablation (LCB v4, N=8)**: averages are public tests only 53.8, generated tests 53.1, LLM judge 55.6, adaptive (execution-grounded pairwise) 57.5. For Qwen-Coder-7B, the first three were identical at 42.3 [SR] — [arXiv 2502.14382](https://arxiv.org/html/2502.14382)
- **Agentic verifier (2026)**: RL-trained to generate discriminative inputs; +10–15% absolute Best@K over execution-based baselines [SR] — [arXiv 2602.04254](https://arxiv.org/abs/2602.04254)
- **CURE**: the co-evolved unit tester serves as a reward model for RL on base models and improves Best-of-N by +9.0% [SR] — [arXiv 2506.03136](https://arxiv.org/abs/2506.03136)
- **μCode**: a learned verifier helps train a better generator, and BoN uses public tests plus the verifier [SR] — [arXiv 2502.20380](https://arxiv.org/abs/2502.20380)
- **AceCoder**: large-scale synthesized tests used to train code reward models and for RL. RL from Qwen2.5-Coder-7B-base with test-pass rewards reports about +25% on HumanEval+ and +6% on MBPP+ in 80 steps [NR] — [arXiv 2502.01718](https://arxiv.org/abs/2502.01718)
- **PRIME**: implicit process rewards learned online from outcome labels. Its Eurus-2-RL code data was reused by ProRL at 1.5B [NR] — [arXiv 2502.01456](https://arxiv.org/abs/2502.01456); [ProRL](https://arxiv.org/html/2505.24864)
- **Test quality as the "verifier"**: weak dataset tests reach only 17.83% precision on hard problems (TACO), and RL against them lowered pass@1 [SR] — [HardTests](https://arxiv.org/html/2505.24098)
- **Verifier plateau**: reward models and majority vote plateau beyond several hundred samples [NR] — [Large Language Monkeys](https://arxiv.org/abs/2407.21787)
- **Robust verifiers can hurt**: JustRL's ablation at 1.5B (math) found "robust verifiers may degrade performance by collapsing exploration" [SR] — [arXiv 2512.16649](https://arxiv.org/abs/2512.16649)

### Inferences
- **At 1.5B, invest in execution, not judgment**:
  - Public examples, plus generated inputs for output-clustering or differential testing, plus checkers.
  - If a learned component is added, make it an *input generator* (agentic-verifier or CodeRM style). It needs no expected outputs, and its errors are cheaper than a judge's.
  - Train it either jointly with the coder (CURE style, only shown at ≥4B), or as a larger server-side model when not running on device.
- **Training-time verification**: a larger model's generated tests and checkers (Section 2) are worth more than any small RM. PRMs for code at 1.5B are unproven in the found literature; implicit PRMs (PRIME) are the only route with a small-scale code track record, and only as training data and reward.
- **Phone deployment**: a separate verifier model doubles memory and compute. A single 1.5B model that both writes code and emits a few test inputs, with on-device Python execution, is the most plausible design. That is an untested hypothesis, not an evidenced result.

### Gaps
- No study of ~1.5B code ORMs, PRMs, generative verifiers or LLM-as-judge accuracy on competitive programming was found.
- The agentic verifier's model sizes and selection-baseline numbers were not retrieved.
- No direct measurement of a 1.5B model's self-verification (judging its own code) accuracy on stdin/stdout tasks was found.
