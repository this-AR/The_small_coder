# Scale the data, shorten the thought

The team's 1.5B coder is not stuck because 1.5B parameters are too few, or because it lacks a router. It is stuck for three reasons. The measurement is broken. The training data is two to three orders of magnitude too small, and it was applied with LoRA in exactly the regime where small models are known to collapse. And the output format (chains of about 19,000 tokens) would be unusable on a phone anyway. The same Qwen2 1.5B architecture reaches **LiveCodeBench 40.1** after full fine-tuning on 580K verified traces ([rStar-Coder](https://arxiv.org/html/2505.21297)). It reports **51.1–55.9** with diversity-selected SFT plus entropy-weighted RL ([VibeThinker-1.5B](https://arxiv.org/html/2511.06221)). The team's base scores **16.9** ([DeepSeek-R1-Distill-Qwen-1.5B card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)). Phone-class 2026 models now self-report **69–73 on LCB v6** ([MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B); [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)).

With compute no longer a constraint, the evidence supports a six-step path:
1. Fix the judge and the test set.
2. Run full-parameter SFT on 0.5–1M execution-verified traces, mixed about 80/20 long to structured.
3. Run on-policy distillation from a teacher that shares the student's tokenizer. In Qwen3's study this beat RL by about 7 LiveCodeBench points at one-tenth the GPU-hours ([Qwen3 report](https://arxiv.org/html/2505.09388)).
4. Run RL on well-tested problems, with length control added late.
5. Quantize to 4 bits with QAT plus on-policy distillation.
6. On the device, filter candidate programs against the problem's sample tests.

The router idea survives, but it has to do a different job. Telling a model only the algorithm name produced no measurable gain, while problem-specific step-by-step plans raised solve@10 from 6.1% to 42.4% ([Explaining CP solutions](https://ar5iv.labs.arxiv.org/html/2307.05337)). So the router should set the thinking budget, gate retrieval and abstain when unsure. Retrieval should switch from 7 topic blueprints to solved problems (statement, solution and code), queried with a first draft attempt. The team's "JEV model" turns out to be TypeSafe's **Jev**, a closed, hosted "System One" decision model. It returns calibrated, typed decisions in one parallel query and writes no code. It validates the router's shape: a fast, calibrated System One controller in front of a slow System Two coder. It cannot run on a phone, though, so the deployable version must be an open, single-pass, calibration-trained controller built by the team, with Jev used as an external baseline. Open reimplementations already exist: NanoJev (MIT, Qwen3-0.6B), Kev (Apache-2.0, 0.8B–9B, Jev-API-compatible), Laya and Von (Apache-2.0, ~400M ModernBERT), and Rizzo Flow (Apache-2.0, llama.cpp option-logit reading). The cheapest phone option is to let the coder route itself by reading option logits over a shared KV cache. GEPA (prompt evolution) and LLM-JEPA (an embedding-prediction training loss) are real, open-source and worth an ablation, but they are secondary levers. A new state-of-the-art number is not available to publish, because 2026 vendors have taken that ground. What is publishable is a controlled answer to two questions: how much of a small coder's failure is choosing the algorithm and how much is implementing it, and how much does a tiny controller recover at phone-realistic token budgets? That answer has to be measured on a decontaminated, checker-graded, multi-sample benchmark.

## Known effects explain both the 0% baseline and the "yapping" collapse

The most important finding for the paper is that its reference point is wrong. DeepSeek evaluated this base model with **32,768 generated tokens, temperature 0.6, top-p 0.95 and 64 samples per problem**. Its card warns that temperatures outside 0.5–0.7 cause "endless repetitions or incoherent outputs" ([R1-Distill-Qwen-1.5B card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)). An independent re-measurement puts the model's average chain-of-thought at **19,066 tokens per LiveCodeBench problem** ([Lotfi et al.](https://arxiv.org/html/2606.00206v1)). A harness with a short token cap, repetition penalty 1.2, no chat template and a parser that only reads `<code>` tags is therefore measuring truncation and parsing, not ability. Under proper settings, NVIDIA's evaluation of the same untouched checkpoint gives **APPS 20.95, CodeContests 16.79, TACO 8.03 and LCB 16.80** ([ProRL](https://arxiv.org/html/2505.24864)). A model with that profile should not score 0% on a mixed APPS/CodeContests/TACO-derived set. The Stage 2 result (8%) may well sit *below* the base model evaluated correctly. Until the base is re-run under the same harness, budget and parser, no fine-tuning claim in the current draft can stand.

The Stage 1 collapse also has a name and a literature. "Through the Valley" documents **Long CoT Degradation**: models from 0.5B to 14B (Qwen2.5, LLaMA3, Gemma3) fine-tuned on about 8k long-CoT examples lose **up to 75% of their original performance**. The cause is length inflation that compounds errors, and some very small models fail to recover even with 220k examples ([arXiv 2506.07712](https://arxiv.org/abs/2506.07712)). The "small-model learnability gap" paper measured the same direction at 1.5B: Qwen2.5-1.5B scored **27.0 with long CoT versus 34.2 with short CoT** (−7.1). The sign flips only at 7B and above ([Li et al.](https://arxiv.org/html/2502.12143v3)). Three further factors compound this:
- **LoRA.** In Thinking Machines' experiment, LoRA left a 13% gap to full fine-tuning after SFT. The gap shrank to 6% only after on-policy distillation ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)).
- **Prompt format.** Stage 1 used Llama-3 chat tokens on a Qwen model, and Stage 2 used an unrelated "### Instruction" format. The model never trained in the format it was pretrained and distilled on.
- **Curriculum.** The data was sorted by length, so the last and most influential updates came from the longest traces.

The fix is not "short traces forever". The same literature shows long CoT works at 1.5B once the data is large enough. rStar-Coder-1.5B reached LCB 40.1 from **about 580K QwQ-32B traces that passed tests, with 6 epochs of full fine-tuning** ([rStar-Coder](https://arxiv.org/html/2505.21297)). OpenReasoning-Nemotron-1.5B reached LCB v6 28.6 from **about 5M R1-0528 traces with SFT alone** ([card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)). MobileLLM-R1-950M used **2.16M code-reasoning samples** ([card](https://huggingface.co/facebook/MobileLLM-R1-950M)). Against those, 12k samples under LoRA is under-training, not a format verdict. The cleanest small-data fix is "Mix-Long": long and short traces at **1:4 (α=0.2)**. This gave a 3B student 45.9 on the math average, against 40.3 for long-only and 43.4 for short-only, with average responses of 1,249 tokens instead of 3,385 ([Li et al.](https://arxiv.org/html/2502.12143v3)).

### The M3 gain cannot be distinguished from noise or leakage

Four problems compound in the RAG result:
- **The two numbers are not on the same problems.** 8% of 200 problems has a Wilson 95% interval of roughly 5.0–12.6%. 11% of 100 problems has roughly 6.3–18.6%. Even on a shared set, the two would not differ significantly. Detecting a 10%→15% change with an unpaired test at 80% power needs about 680 problems per arm. Pairing (the same problems for every system) and multiple samples per problem reduce that substantially. The methods are in [Miller's "Adding Error Bars to Evals"](https://arxiv.org/abs/2411.00640).
- **The blueprints were chosen from failures on the same test set.** That is test-set tuning. The ARC Prize's audit of HRM shows how much this can inflate a result: dropping the training tasks and training only on the evaluation tasks' demonstrations still gave 31% against 41% pass@2, so most of the score came from material taken from the evaluation set itself ([ARC Prize](https://arcprize.org/blog/hrm-analysis)).
- **The with-RAG and without-RAG arms used different prompts and stopping rules.** Retrieved context also silently shortens reasoning: Qwen3.5-27B produced about 3.6× fewer reasoning tokens on LiveCodeBench in a multi-turn setting (secondary report) ([Reasoning Shift](https://arxiv.org/pdf/2604.01161)). Any difference may come from the stopping rule rather than from the retrieved knowledge.
- **The blueprints were not grounded in a solution.** Gemma saw only the first 1,800 characters of each trace, so many blueprints describe an approach the trace never finished. In competitive programming, the correctness of the plan dominates: accuracy conditioned on an idea sketch tends toward either 0% or 100% depending on whether the sketch is right ([PlanSearch](https://arxiv.org/html/2409.03733v2)).

The overlap between training and test data is structural, not incidental:
- open-r1/verifiable-coding-problems-python is the Python subset of PrimeIntellect's 144K problems. Those are "curated from Apps, Codecontests, Codeforces and TACO", and neither dataset card documents how the tests were generated or validated ([open-r1 card](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python); [PrimeIntellect card](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems)).
- rStar-Coder's seed problems come from TACO, APPS, CodeContests, open-r1 Codeforces and USACO. Its decontamination covered only HumanEval(+), MBPP(+), LiveCodeBench and USACO 2025, and its synthetic problems are rewrites of the seeds ([rStar-Coder](https://arxiv.org/html/2505.21297v1)).

A model trained on rStar-Coder and tested on verifiable-coding-problems is therefore probably being tested partly on problems it has seen, or on paraphrases of them.

Exact-match judging adds systematic false negatives. About **30% of Codeforces problems** accept multiple correct answers and need a checker ([open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces)). HardTests found that **25.4%** of problems need a special judge ([HardTests](https://arxiv.org/html/2505.24098)). So the repo's 69% "wrong answer" failure share is partly an artifact of the judge.

Finally, a Google API key hardcoded in a public notebook is a live security incident, not a style issue. Revoke and rotate it today, purge it from the repository history, and load secrets from environment variables in every notebook from now on.

## Viral small models got sharp through data scale, RL and fusion, not new architectures

The small models that drew attention in 2025–2026 share one pattern. A strong but conventional transformer base gets far more post-training than earlier recipes used: millions of distilled traces or hundreds of billions of SFT tokens, long and carefully stabilised RL with verifiable rewards, and some way of fusing specialists. LinkedIn or X virality metrics themselves could not be verified. The models below are the ones with the strongest coverage and claims ([aiweekly on VibeThinker-3B](https://aiweekly.co/alerts/vibethinker-3b-scores-943-on-aime26-matching-671b-deepseek-v32); [VentureBeat on Qwen3.5 small](https://venturebeat.com/technology/alibabas-small-open-source-qwen3-5-9b-beats-openais-gpt-oss-120b-and-can-run)).

| Model (date) | Base | Recipe essentials | Code result (self-reported unless noted) | Caveat |
|---|---|---|---|---|
| DeepCoder-1.5B (Apr 2025) | R1-Distill-1.5B | GRPO+ on 24K verified problems (≥5 tests each); no entropy or KL loss; overlong filtering; clip-high | LCB 16.9→**25.1**; CF 615→963 ([card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)) | NVIDIA re-ran it at 22.76 ([card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)) |
| ProRL v2 / BroRL (2025) | R1-Distill-1.5B | Over 3K GRPO steps; DAPO clip 0.2/0.4; KL with reference resets; temperature 1.2; ~16K H100-h | LCB **27.69** ([card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)) | CC-BY-NC license |
| Archer-Code-1.5B (Jul 2025) | R1-Distill-1.5B | Entropy-aware "dual-token" constraints on top of DAPO | LCB v5 **29.4** vs 26.0 for DAPO ([card](https://huggingface.co/Fate-Zero/Archer-Code-1.5B/blob/main/README.md)) | Other evaluations of the same work disagree |
| OpenReasoning-Nemotron-1.5B (Jul 2025) | Qwen2.5-1.5B | SFT only, ~5M R1-0528 traces | LCB v6 **28.6** ([card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)) | 64K-token evaluation |
| rStar-Coder-1.5B (2025) | Qwen2.5-Coder-1.5B | Full SFT on ~580K test-filtered QwQ-32B traces | LCB v5 **40.1** ([paper](https://arxiv.org/html/2505.21297)) | Seeds overlap the team's test sources |
| VibeThinker-1.5B (Nov 2025) | Qwen2.5-Math-1.5B | Pass@K-selected domain specialists merged by averaging; MGPO RL; ~3,900 H800-h (~$7.8K) | LCB v5 **55.9** / v6 **51.1** ([paper](https://arxiv.org/html/2511.06221)) | SFT data undisclosed; 40K-token budget; users report loops |
| Nanbeige4.1-3B (Feb 2026) | Nanbeige4-3B | SFT to 256K context; point-wise RL that cut truncation; code RL with a time-complexity reward | LCB v6 **76.9** ([paper](https://arxiv.org/html/2602.13367)) | Evaluation window not stated |
| VibeThinker-3B (Jun 2026) | Qwen2.5-Coder-3B | Hard-problem SFT stage; MGPO at a single 64K window; Long2Short RL; offline self-distillation | LCB v6 **80.2**; 123/128 on new LeetCode contests ([paper](https://arxiv.org/html/2606.16140v1)) | Compute undisclosed |
| MiniCPM5-2B (Sep 2026) | MiniCPM5-2B-Base | 400B-token reasoning SFT; critic RL (JustRL II); on-policy distillation fusing 16 expert RL models | LCB v6 **69.1** ([card](https://huggingface.co/openbmb/MiniCPM5-2B)) | Artificial Analysis scored it 15 vs a claimed 23 ([RITS](https://rits.shanghai.nyu.edu/ai/minicpm5-2b-tops-open-models-under-4b/)) |
| LFM3-1.7B (Oct 2026) | LFM3-1.7B-Base (conv/attention hybrid) | 22T pretraining tokens; 1.75T SFT tokens; RL with a length penalty; distillation from six specialist RL teachers | LCB v6 **73.46** ([card](https://huggingface.co/LiquidAI/LFM3-1.7B)) | Three weeks old; LFM license |

The decisive pattern is this. RL alone on the team's exact base plateaus around **LCB 23–30**: DeepCoder, ProRL, Archer. The large jumps came from changing the base and the SFT stage.

VibeThinker-1.5B is the most instructive case, because its architecture and tokenizer match the team's base exactly. Its "Spectrum-to-Signal" recipe has two phases ([VibeThinker paper](https://arxiv.org/html/2511.06221)):
- **Spectrum (SFT).** It first maximises diversity rather than accuracy. Checkpoints are scored by **Pass@K** on probing sets for each subdomain, the best checkpoint per subdomain is kept, and these specialists are merged by unweighted averaging.
- **Signal (RL).** MGPO then weights each question's GRPO advantage by w = exp(−λ·D), where D is the KL distance of the group pass rate from 0.5. This concentrates learning on problems the model solves about half the time.

The paper also reports 10-gram decontamination, a context schedule of 16K→32K math followed by code, and evaluation at 40K tokens with 8 samples. Its weaknesses are just as instructive. The SFT data volume and teachers are undisclosed. The authors call it "not a deployable solution". Hacker News users report "endless loops" outside its narrow format ([HN thread](https://news.ycombinator.com/item?id=45910410)).

Three 2026 moves are directly transferable to the team:
- **Fuse specialists instead of routing between them.** MiniCPM5-2B merges 16 domain RL experts by on-policy distillation with full-vocabulary reverse KL ([MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)). VibeThinker averages weights. Both bake specialisation into one model instead of routing at inference time, a strong counterpoint to topic-LoRA routing.
- **Shape length late.** VibeThinker-3B's final Long2Short stage gives correct rollouts a zero-sum brevity bonus proportional to 1/length (λ=0.2) ([VibeThinker-3B](https://arxiv.org/html/2606.16140v1)). Nanbeige's point-wise RL cut overlong truncation on LCB v6 from **5.27% to 0.38%** ([Nanbeige4.1](https://arxiv.org/html/2602.13367)).
- **Reward time complexity.** Nanbeige adds a time-complexity reward only once a solution passes every test. That targets TLE, which is a common Python failure.

Skepticism is warranted:
- No independent lab has reproduced VibeThinker's LCB numbers.
- Qwen2.5-Math, VibeThinker's base, is credibly contaminated on math benchmarks. Qwen2.5-Math-7B regenerates the rest of MATH-500 problems from 60% prefixes with **54.6% exact match** ([Wu et al.](https://arxiv.org/html/2507.10532v2)).
- Cross-lab re-runs of the same checkpoint routinely drift by several points. The team's own base has an official Codeforces rating of **954** from DeepSeek but **615** in Agentica's evaluation ([DeepCoder card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)).
- "LCB v6" means a 131-problem window for some labs and a 454-problem window for others, and the larger one "generally yields higher scores" ([VibeThinker paper](https://arxiv.org/html/2511.06221)).
- Nearly all of these numbers assume 32K–100K-token generation budgets that a phone cannot afford.

Treat every ≤4B LCB v6 number in the 69–80 range as "self-reported", and re-run any model the paper compares against under the paper's own harness.

### The tiny-recursive-model hype does not transfer to code generation

The other viral "small but sharp" story is the 7M-parameter Tiny Recursive Model (TRM). It beats the 27M HRM on Sudoku-Extreme (87.4 vs 55.0) and ARC-AGI-1/2 (44.6/7.8 vs 40.3/5.0) by recursively refining a latent state and an answer under deep supervision ([TRM paper](https://arxiv.org/html/2510.04871)). Independent audits trace the gains to three sources: training-time refinement loops, heavy augmentation with test-time voting, and per-puzzle ID conditioning. Replacing the correct puzzle ID with a blank or random token gives **zero accuracy** ([Roye-Azar et al.](https://arxiv.org/abs/2512.11847)). HRM's verified ARC-AGI-2 semi-private score was 2% ([ARC Prize](https://arcprize.org/blog/hrm-analysis)). When a ~28M autoregressive TRM was trained on natural-language-to-Python, it lost to a depth-matched transformer, and its edge over a parameter-matched one cost about **175×** the compute per step ([CASCON 2026 study](https://arxiv.org/abs/2608.29376)).

What transfers is the training recipe, not the architecture. ARC Prize found that a model *trained* with 16 refinement loops beat one trained with a single loop by more than 15 points even when run with only one loop ([ARC Prize](https://arcprize.org/blog/hrm-analysis)). That points to "supervised self-repair" trajectories in SFT, a learned halting/correctness head, and weight EMA during small-data fine-tuning. EMA alone moved TRM's Sudoku score from 79.9 to 87.4 ([TRM paper](https://arxiv.org/html/2510.04871)).

## Distillation: off-policy SFT installs the format, on-policy KL does the sharpening

Knowledge distillation now covers eleven distinct mechanisms. They differ in three things: what signal moves from teacher to student (text, token distributions, hidden states or weights), whose trajectories the loss is computed on (the teacher's, the student's, or a mix), and whether the two models must share a vocabulary. The table maps each family to its best evidence and a verdict for this project.

| Family | Mechanism | Same tokenizer needed? | Best evidence | Verdict for the team |
|---|---|---|---|---|
| Sequence-level KD (SFT on teacher text) | Student imitates teacher-generated sequences ([Kim & Rush](https://arxiv.org/abs/1606.07947)) | No | rStar-Coder-1.5B LCB 40.1; OR-Nemotron-1.5B 28.6 | **Core stage 1**, scaled to 0.5–1M traces with full fine-tuning |
| Forward-KL logit KD | Match teacher token distributions on fixed data; Gemma 3 samples 256 logits per token ([Gemma 3](https://arxiv.org/html/2503.19786)) | Yes | Minitron, Gemma and Llama 3.2 pretraining recovery | Use only for prune or quantization recovery |
| Reverse-KL / MiniLLM | Mode-seeking divergence via policy gradient ([MiniLLM](https://arxiv.org/abs/2306.08543)); forward and reverse KL converge with enough training ([Wu et al.](https://arxiv.org/abs/2404.02657)) | Yes | Superseded by token-level on-policy KD | Skip as a standalone method |
| On-policy distillation (GKD/OPD) | Student samples; teacher scores every token; advantage = −reverse KL ([GKD](https://arxiv.org/abs/2306.13649); [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)) | Yes (or GOLD) | Qwen3-8B LCB v5 42.0→**60.3** vs RL 52.9, at 1,800 vs 17,920 GPU-h ([Qwen3](https://arxiv.org/html/2505.09388)) | **Core stage 2; the single largest untested lever** |
| Loss refinements | Skew KL and contrastive loss ([DistiLLM-2](https://arxiv.org/html/2503.07067)); interpolated teacher ([TAID](https://arxiv.org/abs/2501.16937)); interleaved drafts ([SKD](https://arxiv.org/abs/2410.11325v2)) | Yes | Qwen2.5-Coder-7B→1.5B average 52.47 vs 51.38 for GKD | Ablation only (+1–2 points) |
| Cross-tokenizer KD | Sorted-distribution matching ([ULD](https://arxiv.org/abs/2402.12030)), cross-model attention ([DSKD](https://arxiv.org/abs/2406.17328)), optimal transport ([MultiLevelOT](https://arxiv.org/abs/2412.14528v1)), likelihood matching ([ALM](https://arxiv.org/html/2503.20083)), on-policy [GOLD](https://huggingfaceh4-on-policy-distillation.hf.space/) | No | ALM+SFT 53.3 vs SFT 51.7 on math; a Qwen→Gemma-1B GOLD run had tokenizer similarity 0.063 and scored only 0.0305 | Gemma stays a *text* generator only |
| Hidden-state / feature KD | MSE or cosine on activations ([DistillKit](https://github.com/arcee-ai/DistillKit)) | Usually | Minitron used logit-only KD; a hidden-state loss slightly hurt in ALM | Low priority |
| Weight-transfer initialisation | Low-Rank Clone ([LRC](https://arxiv.org/html/2505.12781v5)), weight selection ([Xu et al.](https://arxiv.org/abs/2311.18823)), subcloning ([Samragh et al.](https://arxiv.org/abs/2312.09299)), [Inheritune](https://arxiv.org/abs/2404.08634v3), [learngene](https://arxiv.org/abs/2305.02279) | n/a | LRC-1.7B matches Qwen3-1.7B (64.98 vs 63.17) with 20B vs 36T tokens, on commonsense benchmarks only | Only for a custom sub-1B phone model |
| Prune-then-distill | Width/depth pruning of a 1.5–2× parent, then logit KD ([Minitron](https://arxiv.org/abs/2407.14679); [in practice](https://arxiv.org/html/2408.11796); [Sheared-LLaMA](https://arxiv.org/abs/2310.06694)) | Yes | Width-pruned 4B GSM8K 41.2 vs depth-pruned 16.8 | Pretraining-scale cost; avoid depth pruning |
| Hypernetwork-generated LoRA | Text→LoRA ([T2L](https://arxiv.org/abs/2506.06105)), prompts→weights ([DnD](https://arxiv.org/abs/2506.16406)), cloud→edge ([LoRA-Gen](https://arxiv.org/abs/2506.11638)), [Transformer²](https://arxiv.org/abs/2501.06252) | n/a | DnD already covers coding with Qwen2.5-1.5B | Low novelty; depends on topic LoRAs working first |
| Model merging | Soups, [TIES](https://arxiv.org/abs/2306.01708), [DARE](https://arxiv.org/abs/2311.03099), [evolutionary merge](https://arxiv.org/abs/2403.13187), [mergekit](https://arxiv.org/abs/2403.13257) | Same architecture | Long-to-short merging at 1.5B dropped AIME24 from 20.0 to 10.0 ([Wu et al.](https://arxiv.org/html/2503.20641v2)) | Specialist averaging (VibeThinker) only |

The order of the stages matters. The standard 2025–2026 small-model recipe is off-policy SFT on teacher traces followed by on-policy distillation. On-policy distillation "works best when the student's initialization already places reasonable probability on the teacher's behavior" ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)). SFT installs the format and coverage. On-policy reverse KL then corrects the student on its own trajectories. That correction is exactly what a small model needs when it drifts into loops partway through a long trace, because SFT on a fixed dataset never shows the student its own mistakes.

The costs line up the same way. Thinking Machines reproduced a **9–30×** cost advantage of on-policy distillation over scaling SFT ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)). Qwen3 found it cheaper than RL by about 10× and better on pass@64 ([Qwen3](https://arxiv.org/html/2505.09388)). Hugging Face showed it can also undo forgetting caused by SFT: Codeforces SFT dropped Qwen3-4B's IFEval from 83.4 to 79.48, and a following GKD step restored it to 82.8 while keeping LCB at 39.8 ([HF H4](https://huggingfaceh4-on-policy-distillation.hf.space/)). The stage needs only problem prompts, not verified solutions, so the pool of usable training problems is effectively unlimited.

### Choose the tokenizer lineage before choosing teachers

The vocabulary decides which teachers can supply logits:
- **Gemma 3** uses a 262k SentencePiece vocabulary ([Gemma 3](https://arxiv.org/html/2503.19786)).
- **Qwen2.5 and Qwen3** use a byte-level BPE with about 151k entries ([Qwen3](https://arxiv.org/html/2505.09388)). The team's R1-Distill student is in this family.
- **Qwen3.5 and Qwen3.6** moved to a 248,320-entry embedding ([Qwen3.5-2B card](https://huggingface.co/Qwen/Qwen3.5-2B)).

The student therefore shares a tokenizer with QwQ-32B, Qwen3-8B/14B/32B and Qwen3-235B-A22B. Those are the logit and on-policy teachers. Strong cross-family models serve only as text generators:
- DeepSeek-R1-0528/V4;
- gpt-oss-120b, which reaches LCB v6 82.7 in Qwen's table ([Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B)), and whose low/medium/high effort levels yield traces of graded length ([gpt-oss card](https://arxiv.org/html/2508.10925v1));
- Qwen3.6-27B (LCB v6 83.9) ([Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B));
- Gemma.

DeepSeek's distilled checkpoints remap special tokens, so diff the tokenizer files before running plain GKD. GOLD handles "same token, different ID" natively.

"Bigger teacher" is not automatically better:
- OpenThoughts found **QwQ-32B a better teacher than DeepSeek-R1** (+1.9% code), even though R1 scores higher on CodeElo ([OpenThoughts](https://arxiv.org/abs/2506.04178)).
- The learnability-gap paper measured a −0.8 large-teacher penalty at 1.5B ([Li et al.](https://arxiv.org/html/2502.12143v3)).
- Distillation scaling laws formalise a capacity gap in which a stronger teacher yields a worse student ([Busbridge et al.](https://arxiv.org/abs/2502.08606)).
- "Teacher hacking" degrades students trained too long on fixed offline teacher data. Online generation and prompt diversity mitigate it ([Tiapkin et al.](https://arxiv.org/abs/2502.02671)).

The practical rule: pilot a teacher sweep (Qwen3-8B, QwQ-32B, Qwen3-32B) on a small student run before committing compute. Use the frontier models to *generate and verify* data, not as logit targets.

### Using big models to set a small model's weights is real, but it is a pretraining-scale tool

"Use a large model to choose or set the layers and weights of a small one" has three working forms:
- **Structured pruning plus distillation** (Minitron). Pruning Llama 3.1 8B to 4B by width and distilling on 94B tokens recovered GSM8K 41.2 against the teacher's 48.6. "Teacher correction", a brief fine-tune of the teacher on the distillation corpus, mattered ([Minitron in practice](https://arxiv.org/html/2408.11796)).
- **Low-Rank Clone.** It learns projections that compress teacher weights and align activations ([LRC](https://github.com/CURRENTF/LowRankClone)).
- **Inheriting early layers**, then growing the model ([Inheritune](https://arxiv.org/abs/2404.08634v3)).

None reports any reasoning, math or code result, and all are pretraining-scale (tens to hundreds of billions of tokens). Qwen already ships 1.7B and 2B models distilled from far larger teachers on trillions of tokens. So these methods earn their cost only in two cases. One is a custom sub-1B phone coder, for example width-pruning a fine-tuned Qwen3-4B or Qwen3.5-4B to about 2B. The other is a paper whose contribution *is* pruning a reasoning model. The team's realistic use of "big model sets small weights" is the cheap version: a large model generates and verifies the training signal, and on-policy KL sets the student's weights. For topic adapters, the proven pattern is distilled LoRAs swapped at runtime on a shared base. Apple's rank-16 adapters on a ~3B on-device model are tens of MB each ([Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models)). Hypernetwork generation of such adapters already exists for coding at 1.5B ([DnD](https://arxiv.org/abs/2506.16406)).

## Data, RL and length control: fix the tests first, then let length grow, then shrink it

### Large CC-BY corpora exist, and test quality matters more than any RL variant

With compute unconstrained, the team should stop writing its own long traces and reuse large verified corpora, each decontaminated by problem ID, n-grams and embedding similarity before use.

| Dataset | Size and teacher | Tests or labels | License | Best use |
|---|---|---|---|---|
| [rStar-Coder](https://huggingface.co/datasets/microsoft/rStar-Coder) | 418K problems, ~580K QwQ-32B solutions | Mutually verified synthetic tests (96.8% label accuracy) | CC-BY-4.0 | SFT core; test-filtered |
| [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) / [OCR-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2) | 735K / 1.4M Python R1 traces over 28–34K questions | OCR-2 has a QwQ right/wrong `judgement` field | CC-BY-4.0 | SFT volume; correctness filter |
| [Nemotron-Competitive-Programming-v1](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1) | 1.82M Python R1-0528 traces | 34,799 questions | CC-BY-4.0 | SFT volume |
| [open-r1/codeforces](https://huggingface.co/datasets/open-r1/codeforces) | ~10K problems through 2025 | Ratings, tags, editorials, generated checkers | CC-BY-4.0 | Router labels; held-out test split (468) |
| [CodeContests+](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus) | 11,690 problems | Generator, validator and **checker** per problem; validated with 1.72M labelled submissions | CC-BY-4.0 | RL rewards; evaluation |
| [HardTests](https://huggingface.co/datasets/sigcp/hardtests_problems) | 47K problems from 13 judges | Synthetic hard tests (+11.22 precision) | Untagged | RL rewards |
| [DeepCoder set](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) | 24K problems | ≥5 tests each; includes LCB v5 test split | MIT | RL rewards |
| [TACO](https://huggingface.co/datasets/BAAI/TACO) / [TACO-verified](https://huggingface.co/datasets/likaixin/TACO-verified) | 26K / 12.9K problems | Tags, skill types, difficulty, expected complexity | Apache-2.0 / MIT | Router labels |
| [Skywork-OR1-RL-Data](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data) | 14K code problems | Difficulty 0–16 measured *on R1-Distill-1.5B itself* | Untagged | Budget-router labels |
| [codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) | 47.8K R1 traces | ~84% of Python traces pass public tests | CC-BY-4.0 | Editorial-grounded blueprints |

Two findings about these corpora should shape the data recipe.

**First, filtering by execution is not automatically good for SFT.** NVIDIA reports that "execution filtering negatively affected benchmark accuracy, leading us to prioritize instruction diversity over solution correctness" ([OCR paper](https://arxiv.org/abs/2504.01943)). rStar-Coder filtered by tests and still produced the strongest 1.5B result. Correctness versus diversity is therefore an open ablation axis, not a settled rule. The team's specific twist, filtering for traces that are correct *and short*, is untested and worth running.

**Second, RL reward quality is dominated by test quality.** On the hardest bucket, programs that "pass" TACO's tests are actually correct only **17.83%** of the time, against 55.88% under HardTests. GRPO on Qwen3-4B with TACO tests *lowered* pass@1 (38.48→36.95). With HardTests it raised pass@1 to 39.42 and pass@10 from 56.19 to **64.76** ([HardTests](https://arxiv.org/html/2505.24098)). DeepCoder found that problems with fewer than 5 tests invited reward hacking, so it requires ≥5 tests and samples the 15 longest-input tests per problem ([DeepCoder blog](https://www.together.ai/blog/deepcoder)). The team already has Gemma-3-27B and gold solutions, which is everything a HardTests-style pipeline needs: a large model writes input generators, validators and checkers, and gold solutions label the outputs.

The blueprint corpus itself should be rebuilt, not just expanded. Each blueprint should be written from the **complete accepted solution and its editorial**, not from a truncated trace. Every blueprint should be verified by having the student or teacher implement it and pass the tests. Blueprints should come from a same-tokenizer Qwen teacher so they can also be distilled on-policy. OmniThought shows the scaffolding for this: it annotates 2M traces with a Reasoning Verbosity score (0–9) and a Cognitive Difficulty score, explicitly because "overly complicated reasoning steps" do not suit small students ([OmniThought](https://arxiv.org/html/2505.10937v1)). Diverse, problem-specific plans beat repetitions of one plan. One-sentence hints drawn from 10 different human solutions gave solve@10 of 13.9%, against 12.1% for ten explanations of a single solution ([Explaining CP solutions](https://ar5iv.labs.arxiv.org/html/2307.05337)).

### The RL recipe at 1.5B is settled enough to copy

Published 1.5B code-RL runs converge on a critic-free GRPO/DAPO-style objective with a handful of stabilisers:
- **Clip-higher:** ε_low/ε_high of 0.2/0.28 to 0.4.
- **Dynamic sampling:** drop groups where every rollout passes or every rollout fails, plus offline difficulty filtering.
- **Overlong filtering:** mask the loss on truncated sequences.
- **No entropy bonus.** DeepCoder saw entropy blow up and training collapse with one.
- **Training temperature 1.0–1.2.**
- **Either no KL** (short runs) **or KL with periodic reference resets** (long runs). ProRL resets the reference policy and optimiser state ([ProRL](https://arxiv.org/html/2505.24864); [DeepCoder blog](https://www.together.ai/blog/deepcoder); [DAPO](https://arxiv.org/abs/2503.14476)).

The choice of policy-gradient objective is second-order. At 14B, NousCoder found DAPO, GSPO and GSPO+ within **0.3 LCB points** of each other ([NousCoder](https://nousresearch.com/nouscoder-14b-a-competitive-olympiad-programming-model)). ScaleRL's 400K-GPU-hour study found CISPO's asymptote above DAPO's, but that matters mainly for long asynchronous runs ([ScaleRL](https://arxiv.org/pdf/2510.13786)). Archer's entropy-aware token constraints added +3.4 LCB v5 over DAPO at exactly 1.5B ([Archer](https://huggingface.co/Fate-Zero/Archer-Code-1.5B/blob/main/README.md)). That is the most promising exploration-control tweak at this scale.

LoRA is acceptable for RL even though it hurt SFT. Thinking Machines found LoRA "fully matches" full fine-tuning for policy-gradient RL even at rank 1, on 8B math tasks. The LoRA learning rate should be about 10× the full-fine-tuning rate, and adapters should go on the MLP layers ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)). Treat this as untested for long-CoT code at 1.5B.

Expect gains of the same order as prior work, not a multiple. RL from this base moves LCB by roughly **+6 to +13 points** and costs about 10³–10⁴ GPU-hours. ProRL used about 16K H100-hours and JustRL about 11.5K A800-hours ([ProRL](https://arxiv.org/html/2505.24864); [JustRL](https://arxiv.org/html/2512.16649v1)). A code-only run of 500–1,000 steps at an 8–16K context plausibly needs 1–3K H100-hours, an unverified extrapolation from those runs.

Two controls are mandatory for this particular base. R1-Distill-Qwen-1.5B descends from Qwen2.5-Math-1.5B, and on that family **random rewards** raise MATH-500 by +21.4 points at 7B ([Spurious Rewards](https://arxiv.org/abs/2506.10947)). Any RL claim therefore needs a random-reward or format-only baseline, plus evaluation on problems released after the training-data cutoff. The paper should also report pass@k (k=8–64) before and after RL. Yue et al. find that RLVR mostly converts pass@k into pass@1 and that the base wins at large k ([Yue et al.](https://arxiv.org/abs/2504.13837)). ProRL finds sustained code RL keeps improving pass@k for weak bases ([ProRL](https://arxiv.org/html/2505.24864)). Measuring this settles which regime the team's model is in.

### Length control works best when applied late

The central tension with a phone target is that early length penalties kill exploration, yet the deployed model must be short:
- **JustRL** let length fall naturally from 7–8K to 4–5K tokens with *no* penalty. Adding DAPO's overlong penalty dropped AIME24 from about 55% to about 50% and collapsed entropy ([JustRL](https://arxiv.org/html/2512.16649v1)).
- **DeepCoder-14B** trained at 32K and then scored *below its base* when evaluated at 16K (45.6 vs 50.2) ([DeepCoder blog](https://www.together.ai/blog/deepcoder)).

The resolution is a schedule. Train for accuracy first with overlong filtering. Then add a late length stage, chosen from several tested options:
- Long2Short brevity bonus ([VibeThinker-3B](https://arxiv.org/html/2606.16140v1)).
- DLER's truncation penalty. Zero reward above 4,000 tokens cut R1-1.5B's average length by about 80%, to about 2,466 tokens, at maintained math accuracy ([DLER](https://arxiv.org/html/2510.15110v1)).
- ShorterBetter's shortest-correct-sample target: 50–80% shorter at 1.5B and 7B ([ShorterBetter](https://arxiv.org/html/2504.21370v4)).
- L1/LCPO, which trains *this exact base* to obey a length budget in the prompt ([L1](https://arxiv.org/abs/2503.04697)).
- LEASH, a Lagrangian length constraint: −62.7% length and +0.8 accuracy on the same base ([LEASH](https://arxiv.org/html/2512.21540v1)).

Code-specific evidence is thinner but exists:
- **Elastic Reasoning** trains separate thinking and solution budgets of about 1K tokens each. E1-Code-14B matched DeepCoder-14B on LCB (58.4 vs 58.1) with **37.4% fewer tokens**, and degrades gracefully: 37.3% at a 1K thinking budget, 47.0% at 4K. DeepCoder-14B stays below 10% under 4K ([Elastic Reasoning](https://arxiv.org/html/2505.05315v2)).
- **ASAP** prunes at the step level, guided by first-token surprisal. On R1-Distill-7B it cut tokens 23.5% *and* raised LCB pass@1 from 33.58 to 36.19. Token-level pruners such as TokenSkip and LLMLingua-2 degrade code reasoning ([ASAP](https://arxiv.org/html/2508.05988v1)).
- **"Think Less, Locally"** (November 2026) reports 23.4–36.3% shorter outputs on three code benchmarks at maintained or improved accuracy ([arXiv 2611.04814](https://arxiv.org/abs/2611.04814)).

The blueprint format is step-level compression. That is the kind of compression that survives on code, which is a second reason to keep it.

### Spend test-time compute on execution, not on self-judgement

For small coders, the biggest test-time gains come from parallel sampling plus selection grounded in execution. S* lifted Qwen2.5-Coder-1.5B on LCB from **7.0% to 27.6%**. Majority vote alone gave 11.0% and self-debugging alone 9.4% ([S*](https://arxiv.org/html/2502.14382)). The S* selector was GPT-4o-mini, though, and R1-style models were tested only at 7B and above.

Self-repair once its cost is counted is "not a silver bullet". GPT-3.5 with repair was at or below the no-repair baseline at every budget, and feedback quality is the bottleneck ([Olausson et al.](https://arxiv.org/abs/2306.09896)). Models around 1B struggle to revise their own code unless trained with multi-turn RL. CoCoS reports +35.8% on MBPP with accumulated, fine-grained rewards ([CoCoS](https://arxiv.org/abs/2505.23060)).

The on-device pipeline that follows from this evidence:
1. Sample 4–8 candidates.
2. Drop any candidate that fails the statement's sample tests. This is AlphaCode's filter, which removes about 99% of samples and costs only CPU ([AlphaCode](https://arxiv.org/abs/2203.07814)).
3. Cluster the survivors by their outputs on model-generated or random inputs, which needs no expected outputs ([CodeT](https://arxiv.org/abs/2207.10397)).
4. Repair only runtime errors, using the traceback, for one or two rounds.

Learned judges add little at this scale. In S*'s ablation, an LLM judge scored the same as public tests alone at 7B (42.3 vs 42.3) ([S*](https://arxiv.org/html/2502.14382)).

## The router should decide budget and retrieval, not topic labels

### Telling a small coder the topic does almost nothing; telling it the plan does a lot

The router idea should be kept, but its purpose has to change. The hint-granularity evidence is clear. With GPT-3.5 on 165 CodeContests problems, a "used algorithm" hint left **solve@10 at 6.1%**, the same as no hint. A time-complexity hint gave 6.7%, a one-sentence summary 13.9%, an explanation 23.6%, and a **step-by-step description 42.4%**. With GPT-4-written hints, the algorithm-name hint scored 1.2% solve@1, *below* the 1.8% baseline ([Explaining CP solutions](https://ar5iv.labs.arxiv.org/html/2307.05337)). Platform tags are "usually too general (e.g., dynamic programming)" to choose a strategy ([AlgoSimBench](https://arxiv.org/abs/2507.15378)). Even gold editorials leave large LLMs failing at implementation ([Idea First, Code Later](https://arxiv.org/abs/2601.11332)). So a router that predicts a topic and injects a generic topic blueprint reproduces the "used algorithm" condition, the one condition with zero measured gain.

The router's own accuracy compounds the problem:
- On the 20 most frequent Codeforces tags, statement-only macro-F1 is about **0.41–0.43**, and TF-IDF, gradient boosting and BigBird all land there ([AMT/PSG](https://arxiv.org/html/2310.05791)).
- Exact difficulty prediction over 28 rating classes reaches about **11%**, with a mean error of about 4.5 classes ([AMT/PSG](https://arxiv.org/html/2310.05791)).
- On 8 coarse tags, top methods reach about 0.69–0.75 macro-F1 and are statistically tied with TF-IDF plus a linear SVM. In that comparison, fine-tuning ModernBERT gave no gain "despite hours of GPU tuning" ([codeforces-tag-prediction](https://github.com/msobroza/codeforces-tag-prediction)). This is an unreviewed hobby benchmark.

A hard top-1 topic decision over 7 classes will therefore be wrong often, and a wrong blueprint can mislead, since accuracy conditioned on a plan tends toward 0% when the plan is wrong ([PlanSearch](https://arxiv.org/html/2409.03733v2)).

### The thinking budget is the right decision to predict

Thinking is worth 2.5–3.4× at phone scale. Qwen3-1.7B scores **33.2 with thinking versus 11.6 without** on LCB v5 ([Qwen3 report](https://arxiv.org/html/2505.09388v1)). "Skip reasoning" is therefore only safe for confidently easy problems, and the main thing to predict is a budget level (say 1K/2K/4K/8K tokens).

Difficulty is more predictable than topic, especially from the model's own internals. Linear probes on a solver's hidden states recover human Codeforces difficulty with Spearman **ρ = 0.81**, though that was measured on a 70B model ([LLMs Encode How Difficult Problems Are](https://arxiv.org/html/2510.18147v1)). Learned budget controllers have strong evidence:
- **Sonata** reads last-layer hidden states during prefill and sets the thinking budget before reasoning starts, cutting thinking tokens by **20–60%** at equal accuracy with under 1‰ overhead, on benchmarks that include LiveCodeBench ([Sonata, ICLR 2026](https://iclr.cc/virtual/2026/poster/10011708)).
- **"Learning How Hard to Think"** probes predict the marginal value of extra compute per query and match non-adaptive methods with up to 50% less compute on math and code ([Damani et al.](https://arxiv.org/abs/2410.04707)).
- **DART** is training-free. It samples two cheap no-think drafts, answers directly if they agree, and otherwise sets a budget from the drafts' entropy. It reports up to **+22.5 points on code with 32–73% fewer thinking tokens**, and the first-stage signal works from 0.6B upward ([DART](https://arxiv.org/abs/2606.23181)).
- **Thinkless** and **AdaptThink** learn a think/no-think choice on this exact base model and cut length by 50–90% at maintained or improved accuracy, but only on math ([Thinkless](https://arxiv.org/abs/2505.13379); [AdaptThink](https://arxiv.org/abs/2505.13417)). Applying them to code would be a new result.

### Topic-specialised adapters have no supporting evidence for small coders

Routing between topic LoRAs has weak support:
- Self-MoE's routed experts add **3.7 HumanEval points** over the base ([Self-MoE](https://arxiv.org/html/2406.12034v2)).
- Arrow gains about 2% over uniform routing ([Arrow](https://arxiv.org/html/2405.11157v1)).
- SMoRA shows routed LoRAs are functionally equivalent to partitioning the rank of a single LoRA ([SMoRA](https://www.researchgate.net/publication/388422159_Each_Rank_Could_be_an_Expert_Single-Ranked_Mixture_of_Experts_LoRA_for_Multi-Task_Learning)).
- Per-token dynamic adapters slow decoding by **2.5× or more** without fused kernels ([LoRA-Switch](https://arxiv.org/abs/2405.17741)).
- Soft mixtures tend to collapse onto one expert, which argues for hard routing if routing is used at all ([Hard-Routed Reasoning LoRAs](https://arxiv.org/abs/2606.31413)).
- MapCoder-Lite's working 7B system specialises adapters by *role* (retriever, planner, coder, debugger), not by topic ([MapCoder-Lite](https://arxiv.org/abs/2509.17489)).
- The 2026 leaders fuse experts into one model instead of routing between them ([MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B)).

With routing accuracy of 0.4–0.7 and expected expert gains of 2–4 points, misrouting can erase the benefit. Topic LoRAs belong in the paper as an *oracle-bounded ablation*, not as the architecture.

### Keep the RAG, but retrieve whole solved problems with a CP-trained retriever

The current M3 retrieval (top-1 over 7 hand-written topic blueprints, general-text MiniLM embeddings, queried with the statement alone) is the weakest configuration in the evidence on four counts:
- **The corpus is topics, not solved problems.** On USACO, retrieving whole solved problems (statement, English solution and Python code) took GPT-3.5 from **0.59% to 5.49%**, or **8.79%** combined with reflection. Textbook retrieval helped far less (2.04%) ([USACO-bench](https://arxiv.org/html/2404.10952v1)).
- **The query is the statement alone.** In the same study, the query was the problem plus an initial generated solution. Retrieving on problem descriptions alone kept only **3.18%** of the performance ([USACO-bench](https://arxiv.org/html/2404.10952v1)).
- **The embedder is general-text.** General small embedders are poor at telling which problems are *algorithmically* similar. Qwen3-Embedding-0.6B averages **56.83 NDCG@10** on CPRet, against 84.60 for a CP-trained retriever ([CPRet](https://arxiv.org/html/2505.12925v2)). Matching attempted solutions plus BM25 gains 11.8% over embedding models ([AlgoSimBench](https://arxiv.org/abs/2507.15378)).
- **The corpus has 7 documents.**

Retrieval gains also depend on the coder. On LiveCodeBench, retrieved solutions gave DeepSeekCoder-7B only +1.3 to +3.3 points and gave StarCoder2-7B nothing. Retrieval sometimes hurt, through "over-complicated and ungrammatically repetitive generations" and by copying functions out of the context ([CodeRAG-Bench](https://arxiv.org/html/2406.14497v2)).

The replacement design follows from that evidence. The corpus becomes solved problems drawn from TACO, CodeContests and CPRet's 38.8K problems, deduplicated against every test set. The query is the statement plus a draft plan. Retrieval combines BM25 with a small retriever fine-tuned on CPRet-style data. It keeps k=1–2 documents of about 800 tokens each, does no chunk reranking, and is *gated by the router's confidence*. Measure response length with and without retrieval, because added context can silently shorten reasoning. Retrieval keeps one phone-specific advantage: retrieved code can double as free draft tokens for n-gram speculative decoding.

### A new design: the router as a controller over budget, retrieval and sampling

The design the evidence supports is not a topic classifier feeding a template. It is a **controller** that makes four cheap decisions before and during generation. The table states what each decision should be trained on and what evidence backs it.

| Decision | Training label | Cheapest implementation | Evidence |
|---|---|---|---|
| Thinking budget (no-think / 1K / 2K / 4K / 8K) | Shortest budget at which the coder's own k samples pass (Skywork-style difficulty measured on the student) | Linear probe on the coder's prefill hidden state (no extra model) | [Sonata](https://iclr.cc/virtual/2026/poster/10011708); [Damani et al.](https://arxiv.org/abs/2410.04707) |
| Retrieve or not; which solved problems | Whether retrieval flipped fail→pass on training problems | BM25 + small CP retriever, gated by probe confidence | [USACO-bench](https://arxiv.org/html/2404.10952v1); [CPRet](https://arxiv.org/html/2505.12925v2) |
| Number of samples and repair rounds | Marginal pass gain per extra sample, per difficulty bucket | Same probe, regression head | [Snell et al.](https://arxiv.org/abs/2408.03314) |
| Abstain or escalate (fall back to a long budget or generic plan) | Calibrated probability that the predicted label is right | Per-tag thresholds; top-k soft topics | [tag-prediction repo](https://github.com/msobroza/codeforces-tag-prediction) |
| Coarse topic (soft, top-k) | open-r1/codeforces and TACO tags mapped to ~8–12 families | TF-IDF/SVM or MiniLM head (tens of MB) | [AMT/PSG](https://arxiv.org/html/2310.05791) |

This controller is distinct from prior work in one important way. Its decisions are judged by the *Pass@1 they buy per token spent at a phone budget*, not by label accuracy. If a separate model is wanted, Gemma 3 270M is designed for "query routing" fine-tunes, and its INT4 build used 0.75% of battery for 25 conversations on a Pixel 9 Pro ([Gemma 3 270M](https://developers.googleblog.com/en/introducing-gemma-3-270m/)). LFM3's "d1" decision models answer routing and triage questions in a single forward pass with calibrated probabilities, but they are coder-sized ([LFM3 blog](https://www.liquid.ai/blog/lfm3.md)). At ≤0.5B, a planner that writes competitive-programming plans is not supported by the evidence. Even a fine-tuned 7B planner moved DeepSeek-Coder-7B's CodeContests solve@10 only from 1.8% to 4.2% ([Reasoner–Coder](https://arxiv.org/html/2404.08148v1)). The tiny model should control compute, not design algorithms.

## Jev, GEPA and LLM-JEPA: one maps onto the router idea, two are secondary levers

### "JEV model" is TypeSafe's Jev, a "System One" decision model (confirmed by the team's link)

An earlier draft of this report guessed that "JEV" meant LLM-JEPA. The team has since supplied the source: **Jev**, the first public model in TypeSafe AI's "System One" class ([TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). It is the closest existing product to the team's "very small decision model" idea, so it belongs in the router discussion and not with the training losses.

**What TypeSafe says it is.** All of the following is vendor-reported, from an announcement post; there are no public benchmarks, no parameter count and no paper.
- **The model makes decisions; it does not generate text.** It is "a new class of frontier models built to make fast, structured decisions that software can use directly", described as "a frontier-intelligence function call: unstructured state in, typed probabilistic decisions out."
- **Outputs are typed and schema-defined.** Question types are Choice, Score and "Noul" (a probabilistic yes/no). Each output carries a calibrated probability distribution. Cardinality is up to 255 options per question; beyond that it falls back to a slower two-stage scorer ([TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev); [datanorth](https://datanorth.ai/blog/jev-what-it-is-and-why-this-new-model-matters)).
- **Answers come in one parallel query.** A "new model architecture" with a "hardware-aware" parallel sampler returns all outputs at once instead of token by token. The Register summarises it as returning all outputs to a query at once, unlike next-token LLMs ([The Register](https://www.theregister.com/a/5296711)).
- **Training uses calibration RL.** The method is "Reinforcement Learning for Calibrated Decisions" (RLCD), which "optimizes for calibrated decisions: answers with epistemically honest probabilities." Training data and the reason a new algorithm was needed are not disclosed.
- **Claimed speed and cost.** End-to-end latency is 70–500 ms, against 3–329 s for frontier LLMs, i.e. 40–200× faster on "System One-shaped" queries. Input costs $0.042 per million tokens and output is free. The authors concede they "can't prove it isn't subsidized", and that their workflow evals were built by their own team and used GPT/Claude-averaged reference answers.
- **Availability.** It is hosted-API only, with early access through a waitlist; CometAPI lists version 1.13. No weights or licence are published. The only open artefact is a Python wrapper, `system-one-adapter-python`, which constrains ordinary LLMs to output structured decisions ([TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev); [CometAPI](https://www.cometapi.com/en/models/typesafe-ai/jev)).
- **Hard limits.** It cannot write code, prose or chat replies. It can still pick the wrong answer: a type-safe output is not a correct one ([levelup](https://levelup.gitconnected.com/stop-using-generative-llms-for-every-decision-meet-jev-96f2ec183fc4)).

**What this means for the project.** Jev validates the shape of the router idea, but it cannot be the system itself.
1. **The router fits the System One shape; the coder does not.** Jev's own framing (System One from Kahneman's *Thinking, Fast and Slow*) gives the paper a clean narrative: a fast, calibrated **System One controller** makes typed decisions (budget bucket, retrieve or not, difficulty score, abstain) for a slow **System Two coder** that does the reasoning and writes the code. This is exactly the controller in the previous section, with a ready-made vocabulary.
2. **Jev can serve as an external baseline, not as the deployed router.** It is cloud-only, so it cannot run on a phone. Sending benchmark problems to a third-party API also needs a check of its terms. Use it only as an "off-the-shelf decision model" arm in the controller comparison: Choice over budget buckets, Noul for "retrieve?", Score for difficulty.
3. **The open, on-device equivalent can be built from known parts:**
   - **One forward pass with typed heads.** Classification heads (softmax over budget buckets, sigmoid for retrieve or abstain, regression for difficulty) sit on a small encoder or on the coder's own prefill hidden state. This is the probe design above, and it is "parallel, all outputs in one query" by construction.
   - **Option scoring with a small LM.** Gemma 3 270M or FunctionGemma reads the log-probability of each enumerated option. With single-token option labels this is one forward pass. Constrained decoding (Outlines, XGrammar, llama.cpp GBNF grammars) guarantees schema-valid output whenever generation is needed.
   - **An "RLCD-like" objective without RL.** Train against *soft* targets equal to the coder's empirical pass rate at each budget, measured from k samples, with a proper scoring rule (log loss or Brier score). Then apply temperature scaling on a held-out split, and report ECE and reliability diagrams. Calibrated probabilities are what make abstention and expected-value budget selection work.
   - **A reference open-source decision model.** LFM3's "d1" decision models already return calibrated routing and triage answers in a single forward pass ([LFM3 blog](https://www.liquid.ai/blog/lfm3.md)).
4. **Novelty.** No published work applies a calibrated, typed, single-pass decision model to *controlling a code generator's compute* in competitive programming. Jev is general-purpose, closed and unbenchmarked. That makes the "System One controller for a System Two coder" framing **medium-to-high novelty**, provided the paper measures decision quality by Pass@1 per token at phone budgets and not by label accuracy.

### Open-source alternatives to Jev, and which one to use as the router

Jev's release (Sept 2026) triggered a wave of open "System One" reimplementations within days. None is a clone: Jev's architecture, weights and training data are not public. They fall into three designs: **(A)** a bidirectional encoder with a decision head, **(B)** a small decoder LLM with decision heads (no text decoding), and **(C)** training-free reading of option logits from any existing LLM ([DataCamp roundup](https://www.datacamp.com/blog/top-open-source-jev-alternatives)). Licences and creation dates below were checked against the GitHub API on 2026-11-20. Everything else is self-reported by each project and independently unverified. All of them are 1–3 months old.

| Project | Design | Base / size | Output types | How decisions are made / calibrated | Licence | Get it |
|---|---|---|---|---|---|---|
| **Laya** | A: encoder | ModernBERT-large (English) + mmBERT-base (multilingual), ~322–421M, with a built-in router | Choice, Score, yes/no | Decision head; trained with RL against strictly proper scoring rules (an RLCD analogue). ~33–40 ms per question on a T4, 72 ms for a batch of 10; CPU 193–464 ms | Apache-2.0 | [GitHub](https://github.com/NandhaKishorM/laya), [HF](https://huggingface.co/convaiinnovations/laya), `pip install laya` |
| **Von** | A: encoder | Bidirectional ModernBERT, ~395M | Choice, Noul, Score | OptionMarker representations; several questions in one forward pass; post-hoc temperature scaling; claims sub-15 ms | Apache-2.0 | [GitHub](https://github.com/wfzyx/von), HF `wfzyx/von-1.0`, `pip install von-sdk` |
| **NanoJev** | B: decoder + heads | Qwen3-0.6B backbone | Choice, Boolean, Score | Choice = set attention + softmax; Boolean = sigmoid; Score = distribution over ordered levels; batches all questions into one backbone pass; **explicit end-to-end training pipeline** | MIT | [GitHub](https://github.com/TianyuCodings/NanoJev), HF `C-Tianyu/NanoJev` |
| **Kev** | B: decoder + heads | Qwen3.5 + rank-16 LoRA + pointer head; **0.8B / 4B / 9B** | Noul, Choice, Score | Pointer head scores option vs decision representations; shared state encoded once and reused across questions; open training pipeline; **implements Jev's `/v1/systemone` API contract** | Apache-2.0 | [GitHub](https://github.com/jaredpalmer/kev), HF `jaredpalmer/kev-4b` |
| **Rizzo Flow** | C: logit reading | 1.7B–4B models served with **llama.cpp** | Yes/no, choice, score, numeric | Softmax over the logits of answer letters; the state is prefilled once and kept in the KV cache; CUDA / MLX / CPU back ends | Apache-2.0 | [GitHub](https://github.com/Rizzo-AI-Academy/rizzo-flow) |
| **SemIf** | C: logit reading | Any open LLM (e.g. Qwen3.5-4B) | Typed options | No training: converts native option logits into probabilities | Not verified (repo moved) | [GitHub](https://github.com/TheoLeeCJ/SemIf-OpenJev) |
| **Nimble** (Bespoke Labs) | B | Qwen3.5-9B + rank-16 LoRA | Boolean, enum with per-option probabilities | Softmax over candidate logits; model, recipe, evals and data-curation pipeline released | **No licence file**: treat as all rights reserved | [GitHub](https://github.com/bespokelabsai/nimble), HF `bespokelabs/Bespoke-Nimble-9B` |
| **LFM3 d1-2B / d1-Omni-2B** (Liquid AI) | Vendor decision models | ~2B, specialised from LFM3 | Many questions per pass, each with a calibrated probability; typed values | Vendor-reported calibration | LFM Open License v1.0 (free under $10M revenue) | [LFM3 blog](https://www.liquid.ai/blog/lfm3.md) |
| **Clef** (Cloudflare) | Vendor decision models | LLM backbone, size not stated | Choice, Predicate, Score | Scores the allowed answers directly instead of generating them | Downloadable weights; licence not verified | [Cloudflare blog](https://blog.cloudflare.com/clef-decision-models/) |
| **GLiClass** (Knowledgator) | A: zero-shot classifier (not a typed-decision API) | Small/base/large encoders, including an edge model | Multi-label classification in one forward pass | Cross-encoder-like accuracy at lower compute | Apache-2.0 for the variants checked | [HF](https://huggingface.co/knowledgator/gliclass-base-v3.0), [paper](https://www.arxiv.org/pdf/2508.07662) |

An independent Red Hat guardrail benchmark found Jev strongest on one task and open alternatives better on another ([Red Hat](https://developers.redhat.com/articles/2026/10/02/benchmarking-ai-decision-models-against-traditional-guardrails)). That is the only third-party comparison found. Laya's own card shows why calibration has to be measured on the target data rather than trusted: its English checkpoint scores 0.000 on Khmer while reporting 0.952 confidence. Its zero-shot score on a typed-decision benchmark is 0.362, rising to 0.766 only after fine-tuning ([Laya](https://www.opensourcealternatives.to/item/laya)). For competitive-programming decisions, plan to **fine-tune**, not to use any of these zero-shot.

**Recommendation for this project.** Build an open, CP-specific System One controller in one of three ways, from cheapest to most capable:

1. **Self-routing with no extra model (design C), the first choice for the phone.** Use the Rizzo-Flow/SemIf technique on the coder itself. Prefill the problem statement once in llama.cpp, read the logits of single-token option labels for each decision ("budget: A=no-think B=1K C=2K D=4K E=8K", "retrieve: Y/N", "difficulty 1–5"), then **reuse the same KV cache** to start generating the solution. The router then costs a few extra tokens of prefill, needs no second model in phone RAM, and shares the coder's understanding of the problem. Add a small calibration step (temperature scaling per question on held-out problems). For stronger decisions, train a LoRA or linear heads on the frozen coder's hidden states, which is the probe design above.
2. **A separate tiny decision model with the same tokenizer family: NanoJev (MIT, Qwen3-0.6B).** It ships an end-to-end training pipeline and uses Qwen-family tokenization. Retrain its heads on CP decision data. Use it if the router must run *before* the coder is loaded, or when routing between several coders. It fits in about 0.4 GB at 4-bit.
3. **A controlled comparison arm: Kev (Apache-2.0, 0.8B).** Because it implements Jev's `/v1/systemone` API, one client and one schema can call both Jev (closed) and Kev (open, self-trained). That gives a clean open-vs-closed decision-model ablation with identical requests. Laya or Von (Apache-2.0, ~400M ModernBERT) are good encoder-design arms.

**The CP decision schema and how to train it (common to all three):**
- **Questions.** `budget: Choice{0,1K,2K,4K,8K}`, `retrieve: Noul`, `difficulty: Score{1..5}`, `abstain: Noul`, and optionally `topic_family: Choice{~8–12 coarse families}` as soft top-k only.
- **Targets come from the coder, not from human tags.** For each training problem (open-r1/codeforces, TACO, DeepCoder set, after decontamination), sample the coder k=8–16 times at each budget, with and without retrieval. The empirical pass rates become **soft labels**:
  - `budget` = distribution over the smallest budgets that pass;
  - `retrieve` = P(retrieval flips fail → pass);
  - `difficulty` = the coder's own pass@k bucket (Skywork-style).
- **Loss and calibration.** Use a proper scoring rule (log loss or Brier score) against those soft targets. This is the supervised analogue of RLCD. Laya's "RL against strictly proper scoring rules" is the RL version, and can be added as a second stage that rewards the realised Pass@1-per-token of the chosen action. Then temperature-scale each head on a held-out split.
- **Metrics.** ECE and reliability diagrams per head; decision accuracy versus the oracle action; and the metric that matters, **Pass@1 per token (and per joule) at phone budgets** against fixed budgets, DART and Jev/Kev.

Paper-wise, this turns idea #23 into a concrete, fully open contribution: a calibrated System One controller, trained on the coder's own success statistics, that runs on the phone (option 1 costs almost nothing). It is compared against the closed Jev and the open Kev/NanoJev/Laya baselines.

### "G model" is most likely GEPA, but this is unconfirmed

**GEPA** (Genetic-Pareto reflective prompt evolution) remains the most plausible "G", although it is an optimiser rather than a model. Other candidates are Gemma 4, FunctionGemma, gpt-oss, GLM-5 and GRPO (the RL algorithm GEPA is benchmarked against). If the team has a link for the "G model" too, it should replace this guess as the JEV link did. **LLM-JEPA**, covered below, is no longer the "JEV" referent. It stays in the report as an independently useful training loss.

### GEPA evolves text artifacts by reflecting on execution traces

GEPA runs a multi-module LLM system and collects full trajectories: reasoning, tool calls and tool outputs. A reflection model diagnoses failures in natural language and proposes edits to the prompts. GEPA keeps a **Pareto frontier** of candidates, each of which wins on some subset of examples, and combines lessons from different frontier members ([GEPA](https://arxiv.org/abs/2507.19457)). Version 2 reports an average gain over GRPO of **6%** (up to 20%) with up to **35× fewer rollouts**, and more than 10% over MIPROv2, including +12% on AIME-2025 ([GEPA v2](https://arxiv.org/pdf/2507.19457v2)). The GRPO baseline used 24,000 rollouts on Qwen3-8B ([GEPA](https://arxiv.org/html/2507.19457v1)). The 2026 `optimize_anything` API generalises GEPA to any text artifact. Optimised skills raised coding-agent resolve rates from 24% to 93% on one repository (self-reported) ([GEPA blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)).

The independent caveat matters for this team. On FEM-Bench, GEPA "did not yield any broadly applicable, domain-agnostic prompt improvements"; gains appeared only when optimised prompts absorbed domain knowledge ([FEM-Bench](https://arxiv.org/pdf/2512.20732)). Blueprints are exactly that kind of domain knowledge, which makes the blueprint library GEPA's natural target.

**ACE** (Agentic Context Engineering) is the closest open analogue to growing the library from execution feedback. It treats context as an evolving playbook maintained by Generator, Reflector and Curator roles with incremental "delta" updates, which avoids "context collapse". It reports **+10.6%** on AppWorld over ICL, MIPROv2, GEPA and Dynamic Cheatsheet, without labels ([ACE](https://arxiv.org/abs/2510.04618)). The open alternatives are:
- the GEPA library itself;
- DSPy's [MIPROv2](https://arxiv.org/abs/2406.11695), which optimises instructions and demonstrations jointly;
- [TextGrad](https://arxiv.org/abs/2406.07496), which back-propagates textual "gradients";
- [OPRO](https://arxiv.org/abs/2309.03409), which reports up to 8% over human prompts on GSM8K;
- [OpenEvolve](https://github.com/codelion/openevolve), for evolving code artifacts rather than prompts.

The team's concrete GEPA loop has five parts:
- The task model is the local 1.5B coder, whose rollouts are cheap.
- The reflection model is a strong open model.
- The metric is the hidden-test pass rate, with failing-test input/output, tracebacks and TLE signals as feedback text.
- The search runs over the system prompt, the blueprint template and the blueprint library, on a strict train/validation split. Tuning on the test set is exactly the M3 leakage problem.
- Finally, the optimised prompts are distilled into weights by generating SFT data with them, so the phone never pays the prefill cost of a long prompt.

GEPA's published wins are on 8B-and-larger task models, and a 1.5B model may follow long instructions poorly, so the paper must report gain against prompt length.

### LLM-JEPA adds a predict-the-other-view embedding loss to fine-tuning

LLM-JEPA keeps the next-token loss and adds λ·d(Pred(Enc(view A)), Enc(view B)):
- **Enc** is the last-layer hidden state of the last token.
- **Pred** appends k [PRED] tokens that run through the same LLM.
- **d** is cosine distance.
- **The views** are paired texts packed with an attention mask so they cannot see each other, e.g. a natural-language description and a regex, or a question and SQL ([LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)).

On Llama-3.2-1B, NL→regex accuracy rose from **57.29 to 71.46**, and Spider from 47.52 to 50.55. On the team's exact base, GSM8K moved only **13.87→15.04**. Removing the next-token term produced empty outputs ([LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)). Under LoRA, the baseline plateaued and degraded while LLM-JEPA kept improving with more epochs ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac818ec2da1c976b267de32f59709c69-Abstract-Conference.html)). That makes it most interesting in the data-poor blueprint regime.

The reported cost conflicts across sources. The v2 text says about **2×** training compute, reducible to (2−α)× with random per-batch JEPA-loss dropout α ([LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)). Aggregators cite about 3× ([emergentmind](https://www.emergentmind.com/topics/llm-jepa)). Inference cost is zero either way. Code is at [galilai-group/llm-jepa](https://github.com/galilai-group/llm-jepa). DLLM-JEPA brings the idea to diffusion LMs, using two masking rates of the same input as the two views, at 33% lower training FLOPs than LLM-JEPA ([DLLM-JEPA](https://arxiv.org/abs/2606.00091)).

The natural competitive-programming view pairs are statement↔accepted solution, statement↔blueprint and blueprint↔code. Since a problem has many correct solutions, this needs a multi-positive variant of the loss. LLM-JEPA has no code-generation benchmark results yet, so expect small effects and A/B test them. The more interesting use is architectural: the predictor output Pred(Enc(statement)) is trained to land near Enc(solution). If library problems are indexed by solution embedding and queried with the predicted embedding, retrieval works *by predicted solution* rather than by the story-heavy surface text, which is the failure AlgoSimBench documents. A JEPA-style text→code retriever has been reported, but only via a secondary source ([emergentmind](https://www.emergentmind.com/topics/llm-jepa)).

## On a phone, output length is the dominant constraint

### One long reasoning trace costs minutes and percent of a battery; a blueprint costs seconds

On-device speed reduces to arithmetic once the decode rate is known:
- **LiteRT-LM, Galaxy S24 Ultra.** An int8 build of the team's exact base model decodes at **26.35 tok/s** on CPU and 26.98 on GPU, with a 6.41 s time-to-first-token and 2.2 GB peak memory ([litert-community card](https://huggingface.co/litert-community/DeepSeek-R1-Distill-Qwen-1.5B)).
- **MLX, iPhone 16 Pro.** Independent measurements of Qwen2.5-1.5B at 4 bits peak at 40.5 tok/s and settle at **23.7 tok/s** under sustained load ([arXiv 2603.23640](https://arxiv.org/html/2603.23640)).
- **Snapdragon NPUs.** Vendor numbers reach 57–58 tok/s for Qwen3-1.7B ([Qualcomm](https://huggingface.co/qualcomm/Qwen3-1.7B)). But an independent 400-configuration study measured NPU decode of Qwen2.5-1.5B at only **23.0 tok/s**, against 55.7 for llama.cpp on CPU, while the NPU won prefill by more than 15× (1,463.7 vs 115.1 tok/s) ([arXiv 2607.05475](https://www.alphaxiv.org/abs/2607.05475)).

At about 26 tok/s, the base model's 19,066-token average trace takes about **12 minutes** per problem before slowdowns for context length and heat, and uses roughly 3–5% of a phone battery (estimated from measured energy per token). A 500–1,000-token blueprint plus code takes 10–40 seconds.

The KV cache compounds this. The base has 28 layers × 2 KV heads × 128 head dimensions, which is 28 KiB per token in FP16 (computed from the [config](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/config.json)). A 19K-token trace therefore needs about 547 MB of cache on top of 1.12 GB of Q4_K_M weights ([bartowski GGUF](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)).

Shortening the output by 10–30× is worth more than any architecture or kernel change. This is the strongest argument for the team's blueprint direction, and it means the paper should report **Pass@1 as a function of token budget (1K/2K/4K/8K)**, not only at 32K.

### Quantize to 8 bits freely; 4 bits needs QAT, and 3 bits needs on-policy distillation

On this exact model, the bit width sets most of the cost:
- **8 bits is effectively lossless.** W8A8KV8 with FlatQuant averaged 49.2 against 49.0 in BF16 ([Liu et al.](https://arxiv.org/html/2504.04823v2)).
- **4-bit weight-only is not free on code.** AWQ took LCB from 16.4 to 13.9 and GPTQ to 13.4 ([Liu et al.](https://arxiv.org/html/2504.04823v2)). An independent re-measurement found **17.5→12.7** at W4, with traces growing from 19.1K to 20.5–23.7K tokens ([Lotfi et al.](https://arxiv.org/html/2606.00206v1)).
- **3 bits collapses.** AWQ-W3 scored LCB 4.5 with 38.8K-token traces, so it is slower *and* worse.
- **KV cache.** 4-bit KV via KVQuant cost 1.4 points, but QuaRot KV4 collapsed because of outlier channels in the 1.5B model's K cache ([Liu et al.](https://arxiv.org/html/2504.04823v2)).
- **Calibration data matters at 3 bits.** Calibrating 3-bit GPTQ on math data instead of WikiText2 raised the average by 9.8 points.
- **A training-free logit penalty helps.** Penalising about 50 "overthinking markers" ("Wait", "Alternatively", ...) made outputs 12–23% shorter and raised 3-bit MATH-500 accuracy from 29.0 to 35.2 ([Lotfi et al.](https://arxiv.org/abs/2606.00206)).

With compute available, the 2026 recipe for going below 8 bits has three steps:
1. Start from a PTQ checkpoint and run QAT with distillation from the BF16 fine-tuned model ([Reasoning-QAT](https://arxiv.org/abs/2601.14888)).
2. Add on-policy distillation on the *quantized* student's own generations. This raised BF16-accuracy retention from 66% to **91% on HumanEval** at 1.88–2.79 effective bits ([Train Where the Quantized Model Goes](https://arxiv.org/abs/2609.26708)).
3. Ship 4-bit groupwise weights with 8-bit embeddings and LM head.

For comparison, MobileLLM-Pro's int4 QAT lost 1.5% where PTQ lost 34% ([MobileLLM-Pro](https://huggingface.co/facebook/MobileLLM-Pro)), and Apple ships 2-bit QAT decoder weights plus LoRA adapters that recover accuracy ([Apple](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)). Ternary conversion (BitDistill) preserves classification accuracy but has no reasoning or code results ([BitDistill](https://arxiv.org/html/2510.13998v1)).

Pruning is the weakest lever. At 50% unstructured sparsity, R1-Distill-Qwen-7B fell from 65.2 to 50.9 with generic calibration, or 62.0 when calibrated on its own selected traces ([Think Before You Prune](https://arxiv.org/html/2511.18864v1)). Standard structured pruning can collapse reasoning at 20% sparsity ([RESP](https://arxiv.org/abs/2512.02185)). No mobile runtime was found that turns unstructured sparsity into speed.

One cheap, untested compression idea is specific to this model. The untied embedding and LM head (151,936 × 1,536 each) are about **26% of parameters**. Pruning the vocabulary to tokens that actually occur in English and Python competitive programming could cut memory without touching the transformer.

### Pick a runtime, a decoding trick and a sandbox that phones actually support

The lowest-friction deployment path is:
1. Merge the LoRA into the base model.
2. Convert to GGUF with an importance matrix computed on in-domain prompts and blueprint traces.
3. Ship Q4_K_M with Q8_0 embeddings and output layer (1.29 GB), or Q8_0 (1.89 GB) if Pass@1 at 4 bits drops too far.
4. Run llama.cpp on Android (CPU, OpenCL or Hexagon) and Metal or MLX on iOS.

LiteRT-LM is Google's maintained successor to the MediaPipe LLM API ([LiteRT-LM](https://developers.google.com/edge/litert-lm/overview)). llama.cpp also supports runtime LoRA hot-swapping with per-request scales (`--lora-init-without-apply`), but requests with different LoRA configurations are not batched together ([llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

Speculative decoding barely helps on a phone CPU: 0.92–1.13× with retrieval drafting and 0.56–1.18× with an EAGLE-2 draft model. It reaches **3.8×** on an NPU-coordinated path for Qwen2.5-1.5B ([sd.npu](https://arxiv.org/html/2510.15312v4)). llama.cpp's `ngram-mod` mode targets "reasoning models repeating their thinking in the answer" ([llama.cpp speculative](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). That matches code that re-copies a blueprint's pseudocode, and it costs no extra RAM. R-KV keeps only 10–34% of the KV cache in reasoning models at parity with the full cache ([R-KV](https://arxiv.org/html/2505.24133v4)).

Running generated Python on the phone is possible, with limits:
- **Android** can embed CPython; subprocesses are "possible but officially unsupported".
- **iOS** forbids all subprocesses, multiprocessing and IPC ([Python docs](https://docs.python.org/3/library/intro.html)). A killable sandbox therefore means Pyodide in a Web Worker, which has no `resource` limits ([Pyodide](https://pyodide.org/en/stable/usage/wasm-constraints.html)).
- **App Review guideline 2.5.2** restricts executing code that changes app functionality, with an educational exception that requires the code to be fully viewable and editable by the user ([App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)).

Stdin/stdout tests need no subprocess: redirect `sys.stdin` and `sys.stdout` to `io.StringIO` and enforce cooperative timeouts.

### Choosing the base is now a real decision

The table summarises the base-model options most relevant to a phone competitive-programming coder.

| Candidate | License | Code (self-reported unless noted) | Phone fit | Role in the paper |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B (current) | MIT | LCB 16.9; v6 19.9 in Meta's run ([MobileLLM-R1 card](https://huggingface.co/facebook/MobileLLM-R1-950M)) | 28 KiB/token KV; extremely verbose | Controlled baseline |
| VibeThinker-1.5B | MIT | LCB v6 51.1 ([card](https://huggingface.co/WeiboAI/VibeThinker-1.5B)) | Same architecture and tokenizer; 40K budget | Drop-in "strong base" with identical tooling |
| MiniCPM5-2B | Apache-2.0 | LCB v6 69.1; 70.52 in Liquid's run ([card](https://huggingface.co/openbmb/MiniCPM5-2B)) | Llama architecture; 42 KiB/token KV; GGUF/MLX/LiteRT builds | 2026 transfer test; Base→Midtrain→SFT checkpoint ladder |
| LFM3-1.7B | LFM Open v1.0 (free under $10M revenue; research exempt) ([license](https://www.liquid.ai/en/lfm-license)) | LCB v6 73.46 ([card](https://huggingface.co/LiquidAI/LFM3-1.7B)) | ~1 GB at 4 bits; fastest decode (self-reported) | Efficiency-frontier comparison |
| Gemma 4 E2B | Apache-2.0 | LCB v6 44.0 ([report](https://huggingface.co/papers/2607.02770)) | 0.8 GB + 0.05 GB int8 KV at 32K | Deployment-first option |
| Qwen3-1.7B | Apache-2.0 | LCB v5 33.2 thinking ([Qwen3](https://arxiv.org/html/2505.09388v1)) | 112 KiB/token KV (4× the current base) | Native thinking budget; tooling maturity |

The scientifically cleanest choice runs the main experiments on two models. R1-Distill-1.5B is where the failures were observed. VibeThinker-1.5B shares its tokenizer, architecture and same-tokenizer teachers, so the comparison isolates recipe from architecture. The paper should then show that its controller and evaluation findings carry over to one 2026 model, MiniCPM5-2B for license reasons. Qwen3.5-2B should be avoided as a coder: another lab measured it at LCB v6 20.2 ([MiniCPM5-2B card](https://huggingface.co/openbmb/MiniCPM5-2B)), and its own card warns of "thinking loops" ([Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)).

## Every candidate idea, with a prior-art novelty verdict

The table rates each idea's novelty against the closest prior work found. "Novelty" here means how much of the idea, *in the ≤2B competitive-programming regime*, remains unpublished. These ratings rest on targeted searches, not a systematic review, so run a final Semantic Scholar check before submission.

| # | Idea | Closest prior art | Novelty | Feasibility | Expected impact |
|---|---|---|---|---|---|
| 1 | **Hint-granularity oracle-gap ladder** at ≤2B: none → predicted tag → oracle tag → oracle technique → oracle blueprint → gold editorial, by difficulty tier | [Explaining CP solutions](https://ar5iv.labs.arxiv.org/html/2307.05337) (GPT-3.5); [USACO hints](https://arxiv.org/html/2404.10952v1); [Idea First, Code Later](https://arxiv.org/abs/2601.11332); [KareCoder](https://arxiv.org/abs/2401.15940) | Medium (no ≤2B study) | High | High: separates selection failures from implementation failures |
| 2 | **Router as a compute controller** (budget, retrieve-or-not, sample count, abstain), scored by Pass@1 per token at phone budgets | [Sonata](https://iclr.cc/virtual/2026/poster/10011708); [Damani et al.](https://arxiv.org/abs/2410.04707); [DART](https://arxiv.org/abs/2606.23181); [TALE](https://arxiv.org/abs/2412.18547) | Medium (combination and regime) | High | Medium–High |
| 3 | **Self-probe difficulty at 1.5B**: does the coder's own prefill state predict the shortest passing budget for code? | Difficulty probes shown only at 70B ([arXiv 2510.18147](https://arxiv.org/html/2510.18147v1)) | Medium | High | Medium (no extra model on the phone) |
| 4 | **Structure vs length**: blueprint vs token-matched long, short or truncated CoT, with field ablations ([CORNER CASES], [PSEUDOCODE], ...) | [Learnability gap](https://arxiv.org/abs/2502.12143) (math); [CodePlan](https://arxiv.org/abs/2409.12452); [Blueprint2Code](https://www.frontiersin.org/articles/10.3389/frai.2025.1660912) (name clash) | Low–Medium | High (mostly done) | Medium |
| 5 | **Quantization robustness of short structured traces**: do blueprints survive 4-bit better than 19K-token chains? | Quantization studies cover only long-CoT distills ([Lotfi et al.](https://arxiv.org/html/2606.00206v1)) | Medium–High (none found) | High (cheap) | Medium |
| 6 | **Verdict-aware repair curriculum** (WA → counterexample and update [CORNER CASES]; TLE → lower complexity; RE → recursion, bounds and parsing fixes), then multi-turn RL | [LeDex](https://assets.amazon.science/46/bf/3743cf75474290526f1147d9373f/training-llms-to-better-self-debug-and-explain-code.pdf); [CoCoS](https://arxiv.org/abs/2505.23060); [Self-Error-Instruct](https://arxiv.org/pdf/2505.22591) (math) | Medium | High | Medium–High |
| 7 | **On-device, cost-aware generate→test→repair**, measured as problems solved per joule against resampling at equal energy | [Olausson et al.](https://arxiv.org/abs/2306.09896); mobile energy studies | Medium (systems) | Medium | Medium |
| 8 | **Retrieval by predicted solution**: an LLM-JEPA predictor maps statement→solution embedding to index solved problems | [LLM-JEPA](https://arxiv.org/html/2509.14252v2); [CPRet](https://arxiv.org/html/2505.12925v2) | Low–Medium | Medium | Medium |
| 9 | **JEPA-aligned coder as its own router** (linear tag/difficulty probe on the coder's embedding) | LLM-JEPA; difficulty probes | Low–Medium | Medium | Low–Medium |
| 10 | **GEPA/ACE-evolved blueprint playbook**, distilled into weights | [GEPA](https://arxiv.org/abs/2507.19457); [ACE](https://arxiv.org/abs/2510.04618) (no CP ≤2B results) | Medium | High | Medium |
| 11 | **Topic LoRAs vs one generalist LoRA**, at matched parameters, with an oracle-routing bound | [Hard-routed LoRAs](https://arxiv.org/abs/2606.31413); [MapCoder-Lite](https://arxiv.org/abs/2509.17489) | Medium | High | Low–Medium (likely negative, still informative) |
| 12 | **Specialists fused by on-policy distillation** (per-topic RL experts → one student) | [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B); [VibeThinker](https://arxiv.org/html/2511.06221) | Low | Medium | Medium–High (as a recipe, not a claim) |
| 13 | **Think/no-think and elastic budgets on code at 1.5B** (Thinkless/AdaptThink/L1 applied to CP) | [Thinkless](https://arxiv.org/abs/2505.13379); [L1](https://arxiv.org/abs/2503.04697); [Elastic](https://arxiv.org/html/2505.05315v2) | Low–Medium (math only so far) | Medium | Medium |
| 14 | **Time-complexity reward** after all tests pass, to target Python TLE | [Nanbeige4.1](https://arxiv.org/html/2602.13367) | Low | High | Medium |
| 15 | **TRM-style deep supervision over repair steps** plus a learned halting head on the coder | [TRM](https://arxiv.org/html/2510.04871); [ARC Prize](https://arcprize.org/blog/hrm-analysis); [TRM for code](https://arxiv.org/abs/2608.29376) | Medium–High | Low–Medium | Low–Medium |
| 16 | **Offline evolutionary data engine** (OpenEvolve/ThetaEvolve search on failing *training* problems, then distil the verified solutions) | [ThetaEvolve](https://arxiv.org/html/2511.23473v1); [OpenEvolve](https://github.com/codelion/openevolve) | Low–Medium | Medium | Medium |
| 17 | **Diffusion coder for repair by infilling** (CoDA-1.7B masks failing lines) | [CoDA](https://arxiv.org/html/2510.03270v1); [DiffuCoder](https://arxiv.org/html/2506.20639v2) | Medium | Low–Medium | Low (phone speed unproven) |
| 18 | **Looped base** (Ouro-1.4B) for sharper reasoning per parameter | [Ouro](https://arxiv.org/pdf/2510.25741) | Low | Medium | Low–Medium (saves RAM, not latency) |
| 19 | **Retrieved template as free speculative draft** | [REST](https://arxiv.org/abs/2311.08252); [CREST](https://arxiv.org/abs/2408.04678) | Low–Medium | Medium | Low (latency only) |
| 20 | **Hypernetwork-generated topic LoRAs** | [DnD](https://arxiv.org/abs/2506.16406); [LoRA-Gen](https://arxiv.org/abs/2506.11638) | Low | Low–Medium | Low |
| 21 | **Topic/difficulty bandit curriculum** | [Self-Evolving Curriculum](https://arxiv.org/abs/2505.14970); [DRIVE](https://arxiv.org/pdf/2511.06307) | Low | High | Low–Medium |
| 22 | **Vocabulary pruning** of the 26%-of-parameters embedding and head | None found | Medium | High | Low–Medium (memory only) |
| 23 | **"System One" controller for a "System Two" coder**: an open, single-forward-pass, typed-output decision model (budget bucket / retrieve? / difficulty / abstain) trained with proper-scoring-rule losses against the coder's own measured pass rates, so its probabilities are calibrated; compared against TypeSafe's closed Jev as an off-the-shelf decision-model baseline | [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (general-purpose, closed, unbenchmarked); [LFM3 d1](https://www.liquid.ai/blog/lfm3.md); [Sonata](https://iclr.cc/virtual/2026/poster/10011708) | Medium–High (no CP or compute-control application published) | High | High: gives the controller a clear framing and calibration metrics (ECE), and makes abstention principled |

Ideas 1, 2 and 6 belong together, and together they are the strongest paper:
- **Idea 1** is a clean scientific question nobody has answered at this scale. It also explains the team's own observation: M3's gain is real only if retrieved plans carry problem-specific content, and the oracle rungs measure exactly how much such content could ever buy.
- **Idea 2** turns the router into a measurable control problem whose payoff is minutes and joules on a phone.
- **Idea 6** targets the dominant failure. Wrong Answer accounts for over 90% of GPT-4o's failed Codeforces solutions ([arXiv 2606.05228](https://arxiv.org/abs/2606.05228)), and TLE fixes succeed only about 20% of the time ([alphaXiv 2606.17514](https://www.alphaxiv.org/abs/2606.17514)).

Idea 5 is the cheapest novel result on the list: one week of quantized evaluation could produce a clean finding either way. Ideas 8–10 are good ablations but weak headlines, because LLM-JEPA and GEPA already exist and lack code evidence. Ideas 15, 17 and 20 are high-risk side bets that should not carry the paper.

## A staged plan that produces a publishable paper

### Fix the evaluation before training anything

Every later number depends on the protocol below, so Stage 0 must finish before any training run is reported.

| Element | Protocol | Why |
|---|---|---|
| Test sets | (a) **LiveCodeBench v5 window 2024-08-01→2025-02-01 (279 problems)** and v6 (2025-02-01→2025-05-01), for comparability with prior 1.5B work; (b) a post-cutoff set: Codeforces rounds after mid-2025 or LCB Pro easy problems, ID-checked against every training corpus; (c) the open-r1/codeforces test split, with checkers | Time windows avoid contamination; the two windows make the numbers comparable with prior work ([LCB](https://github.com/LiveCodeBench/LiveCodeBench); [DeepCoder set](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)) |
| Retire or audit the 200/100 sets | Run gold solutions on every test; drop problems where gold fails; flag multi-answer problems and add checkers; report test counts | Undocumented tests; ~25–30% need checkers ([HardTests](https://arxiv.org/html/2505.24098)) |
| Decontamination | Problem-ID/URL joins, then 8–16-gram overlap including rStar `seed_question`, then embedding similarity with LLM review of borderline pairs; report removals at each stage | Training and test sources overlap structurally ([TACO overlap](https://arxiv.org/html/2312.14852); [open-r1 script](https://github.com/huggingface/open-r1/blob/main/scripts/decontaminate.py)) |
| Prompting | The model's own chat template via `apply_chat_template`, identical in training and evaluation; no repetition penalty | Removes the Llama-3 / "### Instruction" mismatch |
| Extraction | Accept the last code block, whether a Markdown python fence or `<code>` tags; count missing code as a failure; report parse-failure rate separately | 74/200 baseline outputs were lost to the parser |
| Sampling | n=16 samples per problem, T=0.6, top-p 0.95; report pass@1 as the mean over samples and pass@k with the unbiased estimator ([Chen et al.](https://arxiv.org/abs/2107.03374)) | Matches the model cards' settings; single greedy samples are noisy |
| Budgets | Report Pass@1 at 1K, 2K, 4K, 8K and 32K generated tokens, plus the truncation rate | Phone viability; the gap vendors leave open |
| Judging | Checker or LCB-style token and Decimal comparison, with float tolerance as stated in each problem; per-test time limit = problem limit × a documented Python multiplier (or LCB's 6 s); sandboxed with no network | Exact-match false negatives; timing noise ([LCB testing_util](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py)) |
| Statistics | Paired comparisons on identical problems; bootstrap CIs that resample problems; McNemar or paired Wilcoxon; ≥500 problems in the headline set; release raw generations | [Miller](https://arxiv.org/abs/2411.00640); [Sober Look](https://arxiv.org/abs/2504.07086) |
| Controls | The untouched base under the same harness; for RL, a random-reward baseline; for RAG, identical prompts and stopping rules with an empty or random retrieved document | Spurious-reward and context confounds ([Spurious Rewards](https://arxiv.org/abs/2506.10947)) |
| Hygiene | Revoke the leaked key; move secrets to environment variables; pin seeds and library versions | Security and reproducibility |

### Stages, deliverables and decision gates

| Stage | Work | Deliverable and gate | Compute (estimate) |
|---|---|---|---|
| 0. Measurement | Build the harness above. Re-run the untouched R1-Distill-1.5B, VibeThinker-1.5B, DeepCoder-1.5B and L1 checkpoints; re-score the existing Stage 1/2/M3 checkpoints | A baseline table at every token budget. **Gate:** if Stage 2 falls below the base, say so; it becomes the paper's motivating negative result | Inference only |
| 1. Data engine | Pool rStar-Coder, OCR-2, Nemotron-CP and codeforces-cots Python traces; decontaminate. Rebuild blueprints from full accepted solutions and editorials with a same-tokenizer Qwen teacher; verify each by implementation. HardTests-style tests and checkers for RL problems | 0.5–1M verified long traces; 100–200K verified blueprints; ~20–30K RL problems with ≥5 strong tests | Teacher inference; CPU sandboxes |
| 2. Full-parameter SFT | Mix-Long (~80% long, ~20% blueprint), native chat template, 16–32K context, 3–6 epochs (rStar-Coder used 6 epochs, lr 4e-5); EMA of weights. Ablate 12K vs 100K vs 500K, LoRA vs full, long vs blueprint vs mix | **Gate:** LCB v5 well above 16.9 (public anchors: 28.6–40.1) | ~4.8B tokens/epoch at 600K × 8K; tens of H100-hours per epoch (estimate) |
| 3. On-policy distillation | Reverse-KL distillation (TRL GKD lmbda=1, or the Tinker recipe) from {Qwen3-8B, QwQ-32B, Qwen3-32B} on CP prompts; 150–500 steps × 64 prompts × 4 samples to start ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)). Optional: fuse topic-specialist students by on-policy distillation | Teacher-sweep table. **Gate:** a gain over SFT, and higher pass@k | Hundreds to low thousands of GPU-hours |
| 4. RL with verifiable rewards | DAPO/GRPO with clip-high, dynamic sampling, overlong filtering, temperature 1.0, 8K→16K context, binary all-tests reward; optional time-complexity bonus after a full pass; random-reward control | +6–13 LCB is the prior-work range | ~1–3K H100-h for 500–1,000 steps (extrapolation) |
| 5. Length and budget | Late Long2Short or DLER truncation stage; Elastic-style split thinking/solution budgets; L1-style budget tokens so the controller can set budgets | Pass@1-vs-budget curve shifted left; truncation under 1% at 4K | Short RL phase |
| 6. Controller (paper core) | Hint ladder (idea 1); budget probe on the coder's prefill (idea 3); retrieve gate over solved problems with a CP retriever; abstention. Compare against DART, a self-predicted budget, a fixed budget and oracle routing | Router accuracy → share of oracle gap recovered → Pass@1 per token | Mostly inference |
| 7. Repair | Verdict-aware SFT pairs (WA/TLE/RE), then CoCoS-style multi-turn RL; evaluate repair against resampling at equal tokens | Repair beats resampling at matched budget, or an honest negative | Medium |
| 8. Phone | QAT + on-policy distillation to 4 bits; GGUF and LiteRT builds; on-device sample-test filter; measure tok/s, time-to-first-token, energy and Pass@1 at deployed precision on two phones (one Snapdragon, one A-series) under sustained load; test blueprint-vs-long quantization robustness (idea 5) | Solved problems per minute and per joule | Small |

### The paper this plan produces

A working title that states the question is *"Selection or implementation? What a phone-sized competitive-programming coder actually needs."* It makes four claims, each backed by a stage above:

1. **Diagnosis.** The hint ladder crossed with difficulty tiers shows how much of a 1.5B coder's failure is choosing the algorithm and how much is implementing it. It also shows that topic labels alone buy nothing, against oracle upper bounds.
2. **Control.** A controller that sets the thinking budget, gates retrieval and abstains recovers a measured share of the oracle gap. It is reported as Pass@1 per token at phone budgets, against DART, self-budgeting and fixed budgets.
3. **Format.** A controlled structure-versus-length experiment explains why long-CoT LoRA SFT collapsed. It covers token-matched baselines and field ablations, and contrasts the method explicitly with SCoT, CodePlan and Blueprint2Code. Renaming "blueprint", for example to "algorithm sketch", avoids the name clash.
4. **Deployment.** A quantized, on-device evaluation including the generate-test-repair loop, reported in seconds and joules.

The headline should not be a leaderboard number. Large-scale SFT, on-policy distillation and RL are the *means* of reaching a credible operating point, which plausibly puts LCB v5 at 30–45 given the public 1.5B anchors. That operating point is necessary for reviewers to take the controller results seriously, but it is not the contribution.

The main risks, and their mitigations:
- **The oracle gap may be small** because the 1.5B model is implementation-bound, as "Idea First, Code Later" found even for large models ([arXiv 2601.11332](https://arxiv.org/abs/2601.11332)). That is still a publishable finding, and it would redirect effort toward repair (Stage 7).
- **The controller may not beat DART.** DART is training-free and strong, so the paper should include it as the baseline to beat.
- **The 2026 bases may make the 1.5B results look dated.** The MiniCPM5-2B transfer experiment and the phone-budget curves (which no vendor publishes) protect against that.
- **The base's math-heavy lineage may make RL gains look like elicitation.** The random-reward control and the post-cutoff test set handle it.

## Conclusion

The "yapping" problem is not a quirk of this team's data. It is the same constraint that decides phone viability. Every point of accuracy a 1.5B coder buys with 19,000-token chains costs minutes of decoding and percent of a battery, and quantization makes those chains longer still. Accuracy at a 32K budget, the number every model card reports, is therefore the wrong objective for this project. The right objective is accuracy per token at 1–8K. That reframing turns the team's original instinct, a small decision model that cuts yapping, from a weak topic classifier into a compute controller with a clear payoff, and it exposes a gap the 2026 vendors leave open: none of them publish Pass@1 at phone-realistic budgets.

The sharper insight concerns what "sharp" means at this scale. The evidence says a small coder improves enormously when given a *correct, problem-specific* plan (6.1%→42.4% solve@10), and not at all when given a topic name. The highest-leverage experiment is therefore to measure the oracle gap directly, before building more retrieval or routing machinery. If plan quality is the bottleneck, invest in verified plans: on-policy distillation from a same-tokenizer teacher, and retrieval of solved problems. If implementation is the bottleneck, invest in verdict-aware repair and on-device execution. Either answer is publishable, and the evaluation fixes are prerequisite to both, because no number the team has reported so far can tell the two apart.
