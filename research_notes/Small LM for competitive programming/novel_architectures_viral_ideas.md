# Novel "small but sharp" architectures and viral methods (2025–2026): how they work, what is verified, and how they could apply to a ~1.5B competitive-programming code model

Compiled 2026-11-19. Numbers are author-reported unless marked "independent" or "secondary". Everything under "Inferences" is my own reasoning about how a result could transfer to a ~1.5B Python stdin/stdout code model (R1-Distill-Qwen-1.5B base, blueprint SFT, retrieval, possible router, phone target). None of it is a sourced claim.

---

## 1. Tiny Recursive Model (TRM) and Hierarchical Reasoning Model (HRM): architecture, training, results, independent analyses, 2026 follow-ups, and use for language/code

### Takeaway
TRM is a 7M-parameter, 2-layer network that recursively refines a latent state and an answer under deep supervision. It beats the 27M HRM on Sudoku-Extreme (87.4 vs 55.0), Maze-Hard (85.3 vs 74.5) and ARC-AGI-1/2 (44.6/7.8 vs 40.3/5.0). Independent analyses, however, trace most of these gains to training-time refinement loops, heavy augmentation plus voting, and per-puzzle identity conditioning, which is transductive and close to test-time training. They do not show general reasoning. In 2026, papers that made TRMs autoregressive, or applied them to NL→Python, found no capability advantage over depth-matched transformers. What transfers to a code LLM is the training recipe (deep supervision over refinement steps, a learned halting head, EMA, augmentation), not the architecture.

### Cited Findings

**How HRM works (Sapient Intelligence, June 2025)**
- HRM has two interdependent recurrent modules: a slow high-level module for abstract planning and a fast low-level module for detailed computation. It has 27M parameters and is trained from scratch on about 1,000 examples per task, with no pretraining or chain-of-thought data. — [HRM paper (arXiv 2506.21734)](https://arxiv.org/abs/2506.21734)
- The TRM author describes HRM's implementation as two 4-layer networks trained with a 1-step gradient approximation. Its ACT halting uses a Q-learning "continue" loss that needs an extra forward pass per optimization step. — [TRM paper](https://arxiv.org/html/2510.04871)
- HRM receives only the input grid and a `puzzle_id` (task hash plus augmentation code), with no few-shot demonstration pairs in context. It can therefore only be applied to puzzles whose ids appeared in training. — [ARC Prize HRM analysis](https://arcprize.org/blog/hrm-analysis)

**How TRM works (Samsung SAIL Montréal, Oct 2025; arXiv 2510.04871)**
- TRM is a single 2-layer network with hidden size 512. TRM-Att has 7M parameters (Maze, ARC). TRM-MLP has 5M parameters on Sudoku and 19M on Maze/ARC; it replaces self-attention with an MLP over the sequence dimension. — [TRM paper](https://arxiv.org/html/2510.04871)
- **Recursion.** The latent z is updated n=6 times from (x, y, z), then the answer y is updated once from (y, z). That is 7 network calls per full recursion. Each supervision step runs T=3 full recursions: the first T−1 without gradients, and only the last backpropagated. Effective depth is 42 layers per supervision step. — [TRM paper](https://arxiv.org/html/2510.04871)
- **Deep supervision.** Up to N_sup=16 supervision steps are used. Training on an example stops early once the halting output is positive. — [TRM paper](https://arxiv.org/html/2510.04871)
- **Simplified ACT.** TRM drops the Q-learning continue loss and uses a single BCE head that predicts whether the current answer is correct. This needs one forward pass per step instead of HRM's two, and moved Sudoku from 86.1 to 87.4. — [TRM paper](https://arxiv.org/html/2510.04871)
- **EMA** of weights (decay 0.999) moved Sudoku from 79.9 to 87.4. MoE, weight tying and removing ACT all hurt generalization. — [TRM paper](https://arxiv.org/html/2510.04871)
- **Gradient approximation.** With HRM's 1-step gradient approximation instead of full backprop through the last recursion, Sudoku-Extreme drops to 56.5 (vs 87.4). — [ar5iv TRM](https://ar5iv.labs.arxiv.org/html/2510.04871) (via search summary)
- **Training setup.** AdamW (β 0.9/0.95), batch 768, lr 1e-4 (1e-2 for embeddings on ARC), weight decay 1.0 (Sudoku/Maze) or 0.1 (ARC). Compute: Sudoku under 36 h on 1 L40S; Maze under 24 h on 4 L40S; ARC about 3 days on 4×H100. — [TRM paper](https://arxiv.org/html/2510.04871)
- **Data.** Sudoku-Extreme has 1K training and 423K test examples; Maze-Hard has 1K/1K. ARC-AGI-1 has 800 tasks and ARC-AGI-2 has 1,120, plus 160 ConceptARC tasks.
- **Augmentation.** Sudoku uses 1,000 shuffles per example and Maze uses 8 dihedral transforms. ARC uses 1,000 augmentations (colour permutations, dihedral transforms, translations), with a majority vote across augmentations at test time. — [TRM paper](https://arxiv.org/html/2510.04871)
- **Stated limitations.** TRM and HRM are supervised, not generative: each input gets one deterministic answer. Architecture choice depends on the task. There is no theory for why small, deeply recursed nets beat larger ones. Generative extension is left as future work. — [TRM paper](https://arxiv.org/html/2510.04871)

**Reported results (author tables)**
- **Sudoku-Extreme:** HRM 55.0, TRM-Att 74.7, TRM-MLP 87.4.
- **Maze-Hard:** HRM 74.5, TRM-Att 85.3.
- DeepSeek R1, Claude 3.7 8K and o3-mini-high score 0.0 on both. — [TRM paper](https://arxiv.org/html/2510.04871)
- **ARC-AGI-1 / ARC-AGI-2 (2 attempts):** HRM 40.3/5.0; TRM-Att 44.6/7.8; TRM-MLP 29.6/2.4. LLM rows in the same table: DeepSeek R1 15.8/1.3, o3-mini-high 34.5/3.0, Gemini 2.5 Pro 32K 37.0/4.9, Grok-4-thinking 66.7/16.0, bespoke Grok-4 79.6/29.4. — [TRM paper](https://arxiv.org/html/2510.04871)

**Independent analyses (verified vs hype)**
- **ARC Prize verification of HRM (15 Aug 2025).**
  - ARC-AGI-1 Semi-Private: **32%** (9 h 16 m; $148.50 total, $1.48/task).
  - ARC-AGI-2 Semi-Private: **2%** ($201; $1.68/task), which the team says is not "meaningful progress".
  - The claimed public-eval score was 41%, so the drop is −9 pp.
  - Source: [ARC Prize](https://arcprize.org/blog/hrm-analysis)
- **Hierarchy contributes little.** A same-size (~27M) transformer comes within about 5 pp of HRM without tuning, and changing H/L iterations from L=2/H=2 made results worse. — [ARC Prize](https://arcprize.org/blog/hrm-analysis)
- **The outer refinement loop drives performance.**
  - 1→2 loops added +13 pp, and 1→8 loops roughly doubled the public-eval score.
  - A model *trained* with 16 loops beat a model trained with 1 loop by more than 15 pp even when run with a *single* loop at inference.
  - The team's conclusion: "refinement during training matters more than refinement at inference". — [ARC Prize](https://arcprize.org/blog/hrm-analysis)
- **Augmentation has steep diminishing returns.** 300 augmentations (vs the paper's 1,000) give "near max performance", and 30 come within 4% of max. Training-time augmentation matters more than inference-time augmentation. — [ARC Prize](https://arcprize.org/blog/hrm-analysis)
- **Little cross-task transfer.** Training only on the 400 evaluation tasks (dropping the 400 training and 160 ConceptARC tasks) gave 31% vs 41% pass@2, so most of the performance comes from training on the evaluation tasks' demonstrations.
  - The team calls HRM "fundamentally a zero-pretraining test-time training approach".
  - Because HRM is purely transductive, they hypothesize it "will not generalize". — [ARC Prize](https://arcprize.org/blog/hrm-analysis)
- **Independent TRM analysis (Roye-Azar et al., Dec 2025, revised Jan 2026).**
  - 1000-sample augmentation voting raises Pass@1 by about 11 pp over single-pass inference.
  - Replacing the correct puzzle ID with a blank or random token gives **zero accuracy**.
  - Most final accuracy is reached at the first recursion step ("shallow effective recursion").
  - Conclusion: performance comes from efficiency, task-specific conditioning and aggressive test-time compute rather than deep internal reasoning.
  - TRM does have much higher throughput and lower memory than a naive QLoRA Llama-3-8B baseline.
  - Source: [arXiv 2512.11847](https://arxiv.org/abs/2512.11847)
- **ARC Prize 2025 (Dec 2025).**
  - TRM won the **1st-place paper award ($50k)**. 2nd place was "Self-Improving Language Models for Evolutionary Program Synthesis" (Pourcel, Colas, Oudeyer); 3rd was "ARC-AGI Without Pretraining" (Liao & Gu).
  - The page repeats TRM's 45% / 8% but does not say these scores were verified on the semi-private set.
  - Kaggle ARC-AGI-2 private leaders: NVARC 24.03%, the ARChitects 16.53%, MindsAI 12.64%.
  - The team says "The central theme driving AGI progress in 2025 is refinement loops".
  - CompressARC reaches about 20% on ARC-AGI-1 with 76K parameters and no pretraining.
  - Source: [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis)

**Follow-ups, Dec 2025 – 2026 (supersession flags)**
- **Universal Reasoning Model (URM; Ubiquant, Dec 2025)** supersedes TRM on ARC. It is a Universal-Transformer variant with ConvSwiGLU (a depthwise short conv inside SwiGLU) and truncated backprop through loops.
  - Reported pass@1: ARC-AGI-1 **53.8%** and ARC-AGI-2 **16.0%**, vs TRM 40.0% / 7.8% in its comparison.
  - Ablations: without the short conv, 45.3%; also without truncated backprop, 40.0%.
  - Comparisons exclude test-time scaling and ensembling.
  - Source: [URM (arXiv 2512.14693)](https://arxiv.org/html/2512.14693)
  - Conflict: a later version reportedly lists 57.5% on ARC-AGI-1 — [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/poster/151990) (secondary attribution; not checked).
- **Tiny Autoregressive Recursive Models (Rauba, Fanconi, van der Schaar; Mar 2026; ICLR 2026 workshop spotlight).** The paper adapts TRM's two-step refinement to next-token prediction on character-level algorithmic tasks. The authors "find no reliable performance gains from the full Autoregressive TRM architecture", note that simpler two-level refinement baselines were strong, and advise against investing in the AR-TRM-specific design. — [arXiv 2603.08082](https://arxiv.org/abs/2603.08082)
- **Evaluating Tiny Recursive Models Across Training for Code Generation (Aug 2026; CASCON 2026).** A ~28M autoregressive TRM is trained on NL→Python.
  - **vs a parameter-matched transformer:** TRM fits, generates and generalizes better, recovering about 45% of the validation-loss gap and 57% of the generation-quality gap between the two controls, at **about 175× the per-step cost**.
  - **vs a depth-matched transformer:** the larger transformer fits and generates better at its validation optimum. The authors attribute TRM's edge to resistance to overfitting, not capability.
  - The fit ranking "reverses twice" during training. No pass@k is reported in the abstract.
  - Source: [arXiv 2608.29376](https://arxiv.org/abs/2608.29376)
- Other 2026 recursive-model titles surfaced but were not read: "Energy-guided Recursive Model". — [arXiv 2607.10128](https://arxiv.org/pdf/2607.10128)
- DiffusionGemma's June 2026 release reportedly included a fine-tuned Sudoku variant (single source). — [Gigazine](https://gigazine.net/gsc_news/en/20260611-google-ai-diffusiongemma/)

**Open-source implementations**
- TRM: [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- HRM: [sapientinc/HRM](https://github.com/sapientinc/HRM)
- TRM is cheap to reproduce: ARC takes about 3 days on 4×H100 and Sudoku under 36 h on 1 L40S. — [TRM paper](https://arxiv.org/html/2510.04871)

### Inferences
- **Do not use TRM/HRM as the code generator.** TRM's authors say it is non-generative. The 2026 code and AR studies show no capability gain over depth-matched transformers, and the per-step cost is ~175×. The headline "7M beats LLMs" compares a transductive, per-task-trained model against general LLMs, so it is not evidence that a tiny recursive net can solve unseen Codeforces-style problems.
- **Borrow the training recipe in four ways.**
  - **(a) Deep supervision over refinement, as "supervised self-repair" SFT.** Build K-step trajectories: draft (blueprint+code) → run the sample tests → feedback (traceback, WA diff, TLE) → revised code. Supervise every step toward the accepted solution. The analogy is ARC Prize's finding that training with 16 refinement loops improved even 1-loop inference by more than 15 pp. At inference on a phone, run 1–3 repair steps with a local Python interpreter.
  - **(b) A learned halting / correctness head**, like TRM's BCE "is the answer correct" head, on the 1.5B coder. It would decide whether to stop, fetch another blueprint, or spend more thinking budget. This is a natural part of the planned router.
  - **(c) Cheap augmentation of the ~2k blueprint set**: statement paraphrases, variable renaming, input-format permutations, constraint rescaling. ARC Prize found 30 augmentations reach within 4% of max for HRM, so a handful per sample may capture most of the gain. This is unverified for code.
  - **(d) EMA of LoRA weights** during small-data SFT. It helped TRM substantially (79.9→87.4) and costs almost nothing.
- **Router.** A TRM-style tiny recursive classifier (5–20M parameters, deep supervision, ACT confidence) over frozen sentence embeddings is a plausible design for fixed-size outputs (topic, difficulty bucket, thinking budget). Start with a linear or MLP probe on a strong small embedder, and add recursion only if the probe underfits. TRM's gains were shown on combinatorial exact-answer tasks, not text classification.
- **Leakage warning.** HRM's ARC score depended on training on the evaluation tasks' demonstrations, and TRM scores 0% with a wrong puzzle ID. Make sure the blueprint library and router training data exclude test problems (deduplicate by statement and solution similarity). Otherwise M3-style retrieval gains may partly measure memorization.
- **Link to the observed Stage 1 → Stage 2 jump.** TRM/HRM show small models do best with many short, supervised refinement steps rather than one long pass. This matches the observation that short structured blueprints beat long "yapping" CoT at 1.5B.

### Gaps
- I found no ARC Prize semi-private verification of TRM (only HRM was verified). The 45%/8% figures remain author-reported.
- I found no TRM/HRM-family result on standard code benchmarks (HumanEval, MBPP, LiveCodeBench). The 2026 code study reports only relative fit and generation quality in its abstract.
- I did not read "Energy-guided Recursive Model" (arXiv 2607.10128), or a "probabilistic TRM" (arXiv 2605.19943) that other papers reportedly cite.
- Whether recursion or deep supervision helps a *text classifier* (the router) is untested in the sources found.

---

## 2. The JEPA family, especially LLM-JEPA: objective, two-view data (NL↔code/regex/SQL), results, compute overhead, 2026 follow-ups, and pairing problem statements with solution code

### Takeaway
LLM-JEPA (Huang, LeCun, Balestriero; arXiv Sept 2025; ICLR 2026) adds an embedding-prediction term to next-token fine-tuning. A predictor (k appended [PRED] tokens through the same LLM) maps the last-token embedding of view A (e.g. an NL description) to the embedding of view B (e.g. regex or SQL). The term is scored by cosine similarity. Gains over plain fine-tuning are large on NL→regex (+14.2 pp on Llama-3.2-1B) and modest elsewhere (+1.2 pp GSM8K on DeepSeek-R1-Distill-Qwen-1.5B, the team's base model). Training costs about 2× (reducible with loss dropout), and inference cost is zero. There are no competitive-programming results yet. A 2026 follow-up (DLLM-JEPA) brings the idea to diffusion LMs at lower cost, and a JEPA-style text→code retriever has been reported.

### Cited Findings

**Mechanism**
- **Loss.** The loss is Σ next-token loss over the text + λ·d(Pred(Enc(Text)), Enc(Code)).
- **Embedding.** Enc is the last-layer hidden state of the last token. Text and Code are packed into one context with a custom attention mask so the two views cannot attend to each other.
- **Predictor.** The predictor is tied to the LLM's own weights: k [PRED] tokens are appended and the last one's embedding is the prediction (k=0 means identity).
- **Metric and search grid.** Distance is cosine similarity. The hyperparameter grid was (k, λ) ∈ {0..4} × {0.5, 1, 2, 4}, and learning rate was searched over {1e-5…8e-5}.
- Setting the next-token weight to 0 produced only empty outputs, so next-token prediction stays essential.
- Source for all of the above: [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)
- **Two-view datasets:** NL-RX-SYNTH and NL-RX-TURK (NL description ↔ regex), Spider (DB id + description ↔ SQL), NQ-Open (question ↔ answer), HellaSwag (context ↔ continuation), and GSM8K. A paraphrase corpus is used for pretraining. — [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)

**Results (Llama-3.2-1B-Instruct, baseline NTP → LLM-JEPA)**

| Dataset | Baseline (NTP) | LLM-JEPA |
|---|---|---|
| NL-RX-SYNTH | 57.29 ± 5.32 | **71.46** ± 1.34 |
| NL-RX-TURK | 22.49 | **30.94** |
| GSM8K | 32.36 | **36.36** |
| Spider | 47.52 | **50.55** |
| NQ-Open | 20.12 | **21.59** |
| HellaSwag | 69.40 | **70.51** |

All differences have p ≤ 0.014. — [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)

- **Other models:** Qwen3-1.7B GSM8K 44.32→45.00. **DeepSeek-R1-Distill-Qwen-1.5B GSM8K 13.87→15.04.** Llama-3.2-3B NL-RX-SYNTH 74.55→77.16. — [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)
- **Overfitting resistance.** In LoRA fine-tuning, the baseline plateaus and degrades while LLM-JEPA keeps improving with more epochs. The method also speeds up PEFT convergence. — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac818ec2da1c976b267de32f59709c69-Abstract-Conference.html); [BDTechTalks](https://bdtechtalks.com/2025/09/29/llm-jepa/)
- **Pretraining (preliminary):**
  - NL-RX-SYNTH from scratch: 54.38→60.59.
  - Paraphrase pretraining then fine-tuning: Rotten Tomatoes 56.57→57.76; Yelp 26.46→27.15.
  - Source: [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)

**Compute overhead (version conflict)**
- Search-indexed summaries of the paper cite a "3-fold compute cost during training". I could not pin this to a specific version.
- The v2 text instead describes "two forward passes instead of three" using the masked packing, and a "2-fold increase in compute cost during training". There is no extra cost at inference.
- **Mitigation:** random per-batch JEPA-loss dropout with rate α brings cost to (2−α)× standard fine-tuning (keep λ·(1−α) roughly constant). At α=0.75, accuracy still beat the baseline at equal PFLOPs.
- Source: [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)
- **Stated limitations:** two extra hyperparameters with costly tuning; the need for datasets with non-trivial two-view structure; preliminary pretraining results. — [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2)
- **Code:** [galilai-group/llm-jepa](https://github.com/galilai-group/llm-jepa) (per search listing). Published at ICLR 2026. — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac818ec2da1c976b267de32f59709c69-Abstract-Conference.html)

**2025–2026 follow-ups**
- **DLLM-JEPA (S. Nam, May 2026; ICML 2026 SPIGM workshop).** This is JEPA for masked diffusion LMs.
  - The two views are the same input at different masking rates, so no explicit pairs are needed.
  - It uses one gradient-carrying forward pass per step, so training FLOPs are **33% lower than LLM-JEPA**.
  - Backbones: LLaDA-8B (GSM8K up to +18.7 pp over diffusion-only fine-tuning) and Dream-7B (+11.4 pp).
  - The paper reports "consistent positive gains" on Spider, NL-RX-SYNTH and Django, without numbers in the abstract.
  - Wide-t config on LLaDA-8B: GSM8K 67.1 vs 65.2, with MMLU kept at base level.
  - Source: [arXiv 2606.00091](https://arxiv.org/abs/2606.00091)
- **Rabtriever (Apr 2026; secondary).** A JEPA-style predictor maps text embeddings toward code embeddings for retrieval. It reportedly loses at most 1–2 points of Recall@1/MRR vs generative rerankers while running faster, and matches or beats dual encoders on MS MARCO/BEIR. — [Emergent Mind topic page](https://www.emergentmind.com/topics/llm-jepa) (secondary; primary not read)
- **"JEPA for Code Generation and Optimization" (CentraleSupélec Lusis chair, 2026–27).** This is a research *proposal* in which a JEPA model guides candidate selection while a pretrained generator writes code. It has no results. — [Lusis](https://chaire-lusis.centralesupelec.fr/graduate/2026/2026-jepa-symbolic-world-models/)
- **VL-JEPA (Meta FAIR et al., Dec 2025; ICLR 2026).** It predicts continuous embeddings of the target text, with a lightweight decoder invoked only when needed. "Selective decoding" cuts decoding operations by about 2.85×. With 1.6B parameters it is comparable to InstructBLIP/QwenVL on GQA, TallyQA, POPE and POPEv2. This is vision-language, not code. — [VL-JEPA (arXiv 2512.10942)](https://arxiv.org/pdf/2512.10942); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006956)

### Inferences
- **Natural view pairs for the team:**
  - (A) problem statement ↔ accepted Python solution (rStar-Coder / verifiable-coding-problems);
  - (B) statement ↔ blueprint ([GOAL]/[LOGIC]/[PSEUDOCODE]/[CORNER CASES]);
  - (C) blueprint ↔ code.
  - NL-RX (NL → short formal program, +14.2 pp at 1B) is the closest published analog; Spider (+3 pp) is more typical. On the team's exact base model the published gain was only +1.2 pp (GSM8K), so expect small-to-moderate effects and A/B test them.
  - The very low R1-distill GSM8K baseline in that paper (13.87%) suggests a format or eval mismatch with long-reasoning distills. Use the team's own blueprint output format when testing.
- **Recipe sketch:**
  - Add the JEPA term to the Stage 2 LoRA SFT with k∈{1,2}, λ∈{0.5,1,2}, and JEPA-loss dropout α≈0.5–0.75, giving ~1.25–1.5× training cost.
  - Because LLM-JEPA resisted LoRA overfitting where NTP plateaued, it is most promising exactly where the team is data-poor (~2k blueprint traces).
- **Retrieval upgrade.** The predictor output Pred(Enc(statement)) is trained to land near Enc(solution). Indexing library problems by *solution* embedding and querying with the predicted embedding would "retrieve by predicted solution" instead of by surface statement wording. This could replace all-MiniLM-L6-v2 in M3, and Rabtriever suggests feasibility. Unverified for competitive programming, where statements' "story" text is a strong distractor.
- **Router.** A small topic/difficulty head on the JEPA-trained embedding is plausible. The JEPA term pushes the statement embedding to encode solution-relevant structure (the algorithm) rather than the story.
- **Diffusion branch.** If the team experiments with a diffusion coder (e.g. CoDA-1.7B, §4), DLLM-JEPA gives a pair-free, 33%-cheaper variant.
- **Not a substitute for RL or execution feedback.** All JEPA gains are measured against plain next-token fine-tuning.

### Gaps
- No LLM-JEPA or DLLM-JEPA results on code-generation benchmarks (HumanEval, MBPP, LiveCodeBench, CodeContests) were found. DLLM-JEPA's Django/Spider numbers are not in the abstract.
- I could not confirm which paper version states the "3-fold" cost.
- The Rabtriever primary paper was not read.
- LeJEPA, V-JEPA 2 and Meta/LeCun 2026 JEPA work were not checked in this session.

---

## 3. GEPA (reflective Genetic-Pareto prompt evolution) and related optimizers (MIPROv2, TextGrad, OPRO, ACE): mechanism, results vs GRPO/MIPROv2, rollout efficiency, and applicability to prompts/blueprints for a small code model

### Takeaway
GEPA evolves the prompts of a multi-module LLM system. It reflects in natural language on execution traces, proposes edits, and keeps a Pareto frontier of candidates that each win on some subset of examples. Reported results: +6% average (v2; +10% in v1) and up to +20% over GRPO with up to 35× fewer rollouts, and >10% over MIPROv2 (+12% on AIME-2025). In 2026 the "optimize_anything" API generalized it to any text artifact, including code and agent skills. One independent study found gains only when prompts absorbed domain knowledge, which is what blueprints are. ACE (an evolving, curated "playbook" context) is the closest analog to growing the team's blueprint library from execution feedback.

### Cited Findings

**GEPA mechanism and results**
- GEPA samples system-level trajectories (reasoning, tool calls, tool outputs). It reflects on them in natural language to diagnose problems, proposes and tests prompt updates, and combines complementary lessons from the Pareto frontier of its own attempts. — [GEPA (arXiv 2507.19457)](https://arxiv.org/abs/2507.19457)
- **v1:** across 4 tasks, GEPA beats GRPO by 10% on average and up to 20%, with up to 35× fewer rollouts. — [GEPA v1](https://arxiv.org/pdf/2507.19457v1)
- **v2 (supersedes v1):** across 6 tasks, the average gain over GRPO falls to **6%** (still up to 20%), with up to 35× fewer rollouts. GEPA beats MIPROv2 by >10%, e.g. **+12% accuracy on AIME-2025**. — [GEPA v2](https://arxiv.org/pdf/2507.19457v2)
- The GRPO baseline used **24,000 rollouts on Qwen3-8B**. Tasks include HotpotQA, HoVer and IFBench. — [GEPA HTML](https://arxiv.org/html/2507.19457v1)
- **Implementation:** GEPA ships as an open-source library with an `optimize_anything` API whose default backend is GEPA's evolutionary search. — [GEPA blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/); [optimize_anything paper (arXiv 2605.19633)](https://arxiv.org/html/2605.19633v1)

**2026 extensions (self-reported unless noted)**
- **Coding-agent skills (Feb 2026):** GEPA-optimized skills raised resolve rates from 24% to 93% on one repository and from 55% to 82% on another. The skills transferred to Claude Code and cut resolution time by 47%. — [GEPA blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)
- **CUDA kernels:** 87% of generated kernels matched or beat the baseline, and 48% were at least 10% faster. — [GEPA code-optimization docs](https://www.mintlify.com/gepa-ai/gepa/use-cases/code-optimization)
- **AIME 2025 with gpt-4.1-mini:** 46.67%→60.00% per the blog archive, but 46.6%→56.6% per the artifact repo, so the figure varies by run. — [GEPA 2026 archive](https://gepa-ai.github.io/gepa/blog/archive/2026/); [artifact repo](https://github.com/gepa-ai/optimize-anything-artifact)
- **Blackbox optimization:** matched Optuna on a 56-problem benchmark. — [GEPA 2026 archive](https://gepa-ai.github.io/gepa/blog/archive/2026/)
- **Independent caveat (FEM-Bench):**
  - GEPA "did not yield any broadly applicable, domain-agnostic prompt improvements".
  - Meaningful gains arose only when optimized prompts incorporated substantial domain-specific knowledge.
  - The reflection model had no consistent effect at lighter budgets.
  - Source: [FEM-Bench (arXiv 2512.20732)](https://arxiv.org/pdf/2512.20732)

**Related optimizers**
- **ACE (Agentic Context Engineering; Zhang et al., Oct 2025; ICLR 2026).**
  - **Problem it targets:** "context collapse", where an LLM that fully rewrites the accumulated context shrinks it into less informative summaries.
  - **Design:** ACE treats context as an evolving playbook with three roles (Generator, Reflector, Curator) and applies structured, incremental updates.
  - **Result:** AppWorld average gain of **+10.6%** over ICL, MIPROv2, GEPA and Dynamic Cheatsheet, for 59.5% average accuracy. It matches the top production agent, IBM-CUGA (GPT-4.1), on average and surpasses it on the harder test-challenge split.
  - **Supervision:** it works without labels by using execution feedback.
  - **Cost:** secondary sources report −86.9% adaptation latency (unverified).
  - Sources: [ACE (arXiv 2510.04618)](https://arxiv.org/abs/2510.04618); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10008343); [InfoQ](https://infoq.com/news/2025/10/agentic-context-eng)
- **MIPROv2:** jointly optimizes instructions and few-shot demonstrations for multi-stage LM programs (the DSPy optimizer GEPA is compared against). — [MIPRO (arXiv 2406.11695)](https://arxiv.org/abs/2406.11695)
- **TextGrad:** "automatic differentiation via text", where LLM-written feedback is back-propagated through compound systems as textual gradients. — [TextGrad (arXiv 2406.07496)](https://arxiv.org/abs/2406.07496)
- **OPRO:** an LLM proposes new prompts from a meta-prompt that lists past prompts and their scores. Optimized prompts beat human-designed ones by up to 8% on GSM8K and up to 50% on Big-Bench Hard. — [OPRO (arXiv 2309.03409)](https://arxiv.org/abs/2309.03409)

### Inferences
- **GEPA fits the team's design unusually well.**
  - Every lever is text: the system prompt, the blueprint template, the 7 hand-written blueprints, and the router instructions.
  - Competitive programming gives rich textual feedback for reflection: tracebacks, wrong-answer diffs on sample tests, TLE.
- **Concrete loop.**
  - Task LM = the local 1.5B coder (cheap rollouts).
  - Reflection LM = a strong open model (Gemma-3-27B / Gemma 4 31B / gpt-oss-120b).
  - Metric = pass rate on hidden tests, with failing-test I/O and stderr as feedback text.
  - Optimize each module (router → blueprint retrieval → plan → code) in a DSPy-style program.
  - Keep a strict train/val/test split so prompts are not tuned to the 200-problem evaluation set.
- **Use the Pareto frontier as a library generator.** GEPA keeps candidates that win on *different* problems, which yields topic-specialized blueprints the router can select between.
- **Grow the library with ACE-style curation.** Use execution feedback to grow 7 blueprints into an itemized playbook (per-topic strategies, pitfalls, corner cases) with incremental "delta" updates, to avoid context collapse. Retrieve only the relevant bullets to keep phone prompts short.
- **Small-model caveats.**
  - GEPA's published wins are on Qwen3-8B and larger models. A 1.5B model may follow long instructions poorly, so measure gain against prompt length (prefill cost matters on phones).
  - FEM-Bench suggests domain knowledge, not generic phrasing, is where gains come from.
- **Distill the prompt into weights.** Generate SFT data with the GEPA-optimized prompts, then fine-tune so the deployed model needs a shorter prompt or none. Untested here.

### Gaps
- I found no GEPA or ACE results on competitive programming (Codeforces/LiveCodeBench) with models of 2B or fewer parameters.
- I did not verify GEPA's conference venue, DSPy API names, or ACE code availability.
- The quantitative effect of prompt length on small-model performance is unknown.

---

## 4. Diffusion language models for code (Mercury Coder, Gemini Diffusion, DiffuCoder, Dream-Coder, Seed Diffusion, CoDA, DiffusionGemma, LLaDA): speed vs quality, small sizes, phone suitability, infilling/repair

### Takeaway
Diffusion LMs (dLLMs) generate by iteratively denoising many tokens in parallel. Vendors report roughly 700–2,100 tok/s on datacenter GPUs, and code quality is roughly at parity with speed-tier AR models on short-code benchmarks (Gemini Diffusion LiveCodeBench v6 30.9 vs Flash-Lite 28.5). They lag on reasoning-heavy evaluations (GPQA 40.4 vs 56.5), and Google's open DiffusionGemma (June 2026) reportedly loses quality relative to its AR parent. The only phone-sized open diffusion coder found is CoDA-1.7B (HumanEval 54.3). No verified on-device benchmarks exist. Diffusion's natural strengths, editing, infilling and self-correction, make it more promising for code *repair* or plan→code translation than as the primary competitive-programming reasoner.

### Cited Findings

**Closed or commercial models**
- **Gemini Diffusion (Google DeepMind, I/O May 2025; experimental demo).** Pass@1, Gemini Diffusion vs Gemini 2.0 Flash-Lite:

| Benchmark | Gemini Diffusion | Gemini 2.0 Flash-Lite |
|---|---|---|
| LiveCodeBench v6 | 30.9 | 28.5 |
| BigCodeBench | 45.4 | 45.8 |
| LBPP v2 | 56.8 | 56.0 |
| SWE-Bench Verified (single-turn edit) | 22.9 | 28.5 |
| HumanEval | 89.6 | 90.2 |
| MBPP | 76.0 | 75.8 |
| GPQA Diamond | 40.4 | 56.5 |
| AIME 2025 | 23.3 | 20.0 |
| BIG-Bench Extra Hard | 15.0 | 21.0 |
| Global MMLU Lite | 69.1 | 79.0 |

  - Speed: 1,479 tok/s excluding a 0.84 s overhead.
  - The page says the model "excel[s] at tasks like editing" and can "correct errors during generation".
  - Source: [Gemini Diffusion page](https://deepmind.google/models/gemini-diffusion/)
- **Mercury Coder (Inception Labs, 2025; closed).** On H100, Mini runs at 1,109 tok/s and Small at 737 tok/s. Its LiveCodeBench figures could not be reliably extracted (possibly 17.0 for Mini and 25.0 for Small, alignment unconfirmed). — [Mercury paper (arXiv 2506.17298)](https://arxiv.org/pdf/2506.17298v1); [Inception](https://inceptionlabs.ai/introducing-mercury)
  - **Mercury 2 (2026) supersedes it** and is reported at about 1,000 tok/s (secondary). One listing gives LiveCodeBench 67.3, not comparable to the original Mercury Coder figures and unverified. — [dev.to](https://dev.to/jangwook_kim_e31e7291ad98/mercury-2-inceptions-diffusion-llm-at-1000-tokenss-1i0d)
- **Seed Diffusion Preview (ByteDance Seed, Aug 2025; closed).** About 2,146 tok/s on H20 GPUs, "competitive" on code benchmarks, and claimed faster than Mercury and Gemini Diffusion. Hardware differs across these comparisons. — [Seed Diffusion](https://huggingface.co/papers/2508.02193)

**Open-weight models**
- **DiffuCoder (Apple + HKU, June 2025; ICLR 2026).** A 7B masked-diffusion code model trained on 130B code tokens. "Coupled-GRPO" adds +4.4% on EvalPlus using 21K RL samples. Higher sampling temperature diversifies generation order and raises pass@10. Weights: `apple/DiffuCoder-7B-cpGRPO`; code at github.com/apple/ml-diffucoder. — [Apple ML Research](https://machinelearning.apple.com/research/diffucoder); [arXiv 2506.20639](https://arxiv.org/html/2506.20639v2)
- **Dream-Coder 7B (HKU + Huawei Noah's Ark, Sept 2025).** Adapted from Qwen2.5-Coder-7B. The Instruct model scores **21.4% pass@1 on LiveCodeBench (2410–2505)**. It adapts its decoding order to the task, and checkpoints, training recipes and inference code are released. — [Dream-Coder (arXiv 2509.01142)](https://arxiv.org/abs/2509.01142)
- **CoDA-1.7B (Salesforce, Sept 2025).** Qwen3-1.7B adapted to discrete diffusion. HumanEval is **54.3 (Instruct)** and 29.3 (Base). The authors claim it matches or surpasses diffusion models up to 7B on HumanEval, MBPP and EvalPlus (self-reported). The TPU training pipeline and weights are open (`Salesforce/CoDA-v0-Instruct`, GitHub SalesforceAIResearch/CoDA). — [CoDA (arXiv 2510.03270)](https://arxiv.org/html/2510.03270v1); [HF model card](https://huggingface.co/Salesforce/CoDA-v0-Base); [Salesforce blog](https://www.salesforce.com/blog/coda/)
- **DiffusionGemma (Google DeepMind, 10 June 2026; open weights, Apache 2.0, "experimental").**
  - 26B MoE with about 3.8B active parameters, warm-started from Gemma 4 26B-A4B.
  - Refines blocks of up to 256 tokens in parallel.
  - Up to 4× faster generation; vendor figures exceed 1,000 tok/s on H100 and 700 tok/s on an RTX 5090.
  - A secondary source reports AIME 2026 falling from 88.3 (Gemma 4) to 69.1.
  - Sources: [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/); [MarkTechPost](https://www.marktechpost.com/2026/06/10/google-ai-releases-diffusiongemma-a-26b-moe-open-model-using-text-diffusion-for-up-to-4x-faster-generation/); [NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-local-gemma-diffusion/); [Tech report (arXiv 2608.00146)](https://arxiv.org/html/2608.00146v1); [Digital Applied](https://www.digitalapplied.com/blog/google-diffusiongemma-open-weight-text-diffusion-model-guide)
- **LLaDA-8B and Dream-7B** are the standard open dLLM backbones used in 2026 research (e.g. DLLM-JEPA). — [arXiv 2606.00091](https://arxiv.org/abs/2606.00091)

**Verified vs hype**
- All throughput numbers are vendor- or lab-reported, on different hardware (H100 vs H20 vs RTX 5090). I found no independent cross-model speed benchmark.
- Quality claims are mostly self-reported. DiffusionGemma's own positioning ("experimental, not production-ready") and the AIME drop indicate a real quality cost when converting AR models to diffusion. — [Digital Applied](https://www.digitalapplied.com/blog/google-diffusiongemma-open-weight-text-diffusion-model-guide); [MarkTechPost](https://www.marktechpost.com/2026/06/10/google-ai-releases-diffusiongemma-a-26b-moe-open-model-using-text-diffusion-for-up-to-4x-faster-generation/)

### Inferences
- **For competitive programming, keep an AR reasoner.** dLLMs look near-parity on HumanEval/MBPP-style short code but weaker on reasoning-heavy evaluations (GPQA, BBEH, AIME after conversion), and competitive programming is dominated by algorithmic reasoning.
- **Most promising diffusion roles:**
  - **(a) Repair by infilling.** Mask the lines implicated by a failing test or traceback and re-denoise, conditioning on both prefix and suffix. This is what bidirectional dLLMs do natively.
  - **(b) Fast blueprint→code translation** once the plan is fixed, since translation is local and fits parallel decoding.
- **Experiment sketch.**
  - Fine-tune CoDA-1.7B-Instruct, whose Qwen3-1.7B backbone is close in size and family to the current base, on the team's (blueprint → code) pairs.
  - Compare Pass@1 and wall-clock against the AR 1.5B trained on the same data.
  - Then try "AR plan + diffusion code/repair".
- **Phone suitability is unproven.**
  - Vendor speedups come from parallel token updates on high-FLOP GPUs. On phones, AR decoding at batch 1 is memory-bandwidth-bound, so parallel denoising *could* help by using more compute per weight read. But multiple full-sequence denoising steps without a standard KV cache may cost more total compute.
  - Benchmark on the target device before committing.
  - DiffusionGemma's ~4B active / 26B total is too large for phone RAM.

### Gaps
- I found no verified on-device (phone) dLLM benchmarks.
- I found no dLLM results on Codeforces-style stdin/stdout sets beyond LiveCodeBench.
- Mercury 2's benchmark numbers and Mercury Coder's LiveCodeBench values are unverified.
- LLaDA 1.5/2.x and other small open dLLMs (e.g. sub-1B code dLLMs) were not checked in this session.
- DiffusionGemma's code-benchmark numbers were not found.

---

## 5. Latent / looped / recurrent-depth reasoning (Coconut, Huginn, Mixture-of-Recursions, looped LMs such as Ouro): can they cut visible reasoning tokens at small scale, and what results exist on code?

### Takeaway
Latent-token reasoning (Coconut) cuts visible tokens sharply on logical-search tasks but *trails* explicit CoT on math (GSM8K 34.1% vs 42.9%), and I found no code results. Recurrent-depth and looped models scale compute by reusing weights. The strongest small-scale evidence is ByteDance's Ouro, where 1.4B and 2.6B looped LMs trained on 7.7T tokens match 4B and 8B transformers, including a reported MBPP 80.40 for 2.6B vs Qwen3-8B 79.00 (secondary). But they mainly raise per-parameter capability rather than replacing CoT. Huginn's latent "reasoning" was found to be weak under independent probing. For a phone, looping saves RAM, not latency.

### Cited Findings

**Coconut (Meta, Dec 2024)**
- Coconut feeds the last hidden state back as the next input embedding ("continuous thought"). — [Emergent Mind summary](https://www.emergentmind.com/papers/2412.06769); [arXiv 2412.06769](https://arxiv.org/abs/2412.06769)
- **GSM8K (GPT-2):** Coconut 34.1% vs no-CoT 16.5%, while CoT reached 42.9%. — [The Decoder](https://the-decoder.com/coconut-from-meta-lets-language-models-think-without-words/)
- **ProsQA:** about 97% vs CoT 77.5%; another summary gives 96.6 vs 76.7. — [The Decoder](https://the-decoder.com/coconut-from-meta-lets-language-models-think-without-words/); [Gonzo ML](https://gonzoml.substack.com/p/chain-of-continuous-thought-coconut)
- **Token use:** 9 vs 92.5 on ProntoQA; 14.2 vs 49.4 on ProsQA. Without its multi-stage curriculum, GSM8K drops to 14.4%. — [Gonzo ML](https://gonzoml.substack.com/p/chain-of-continuous-thought-coconut)
- No code-generation experiments were found for Coconut. — (search found none)

**Huginn-0125 (recurrent depth; Feb 2025)**
- A 3.5B latent recurrent-depth model trained on 800B tokens. It iterates a recurrent block so it can unroll to arbitrary depth at test time. The model card calls it "surprisingly capable in reasoning and code" given its budget. — [Huginn paper (arXiv 2502.05171)](https://arxiv.org/html/2502.05171v2); [model card](https://huggingface.co/tomg-group-umd/huginn-0125/raw/refs%2Fpr%2F6/README.md)
- **Independent probing (Brown University, 2025):** more recurrent steps only marginally improve GSM8K, which falls "far short" of explicit CoT. Token-rank trajectory analysis gives "little evidence" of latent CoT. — [arXiv 2507.02199](https://arxiv.org/html/2507.02199v1)

**Mixture-of-Recursions (MoR; KAIST/Google, July 2025; NeurIPS 2025)**
- Shared-parameter recursion with lightweight routers that assign a recursion depth to each token. Attention is restricted to tokens still active at a given depth, so fewer KV pairs are cached. — [MoR (arXiv 2507.10524)](https://arxiv.org/abs/2507.10524); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/118085)
- At 135M–1.7B and equal training compute, MoR gives lower validation perplexity and better few-shot accuracy than vanilla and recursive baselines, plus higher throughput. One summary reports up to **2.06×** throughput (secondary). Scales of 3B+ are not validated. — [MoR (arXiv 2507.10524)](https://arxiv.org/abs/2507.10524); [HF Papers](https://huggingface.co/papers/2507.10524)

**Ouro looped LMs (ByteDance Seed et al., Oct 2025)**
- The 1.4B and 2.6B LoopLMs were pretrained on 7.7T tokens. They "match the performance of 4B and 8B standard transformers respectively across nearly all benchmarks", about 2–3× parameter efficiency. — [Ouro (arXiv 2510.25741)](https://arxiv.org/pdf/2510.25741)
- Ouro-1.4B architecture: 24 layers, hidden size 2048, 4 recurrent steps with shared weights. — [HF model card](https://huggingface.co/ByteDance/Ouro-1.4B)
- **Training tokens:** 6T pretraining, 1.4T annealing, 20B long-context and 300B mid-training. — [HF model card](https://huggingface.co/ByteDance/Ouro-1.4B)
- **Early exit:** the `early_exit_threshold` defaults to 1.0, so all steps always run. vLLM bypasses adaptive exit.
- **License:** Apache-2.0, though the card says "intended for research purposes only". The model card also says to use transformers<4.56.0. — [HF model card](https://huggingface.co/ByteDance/Ouro-1.4B)
- A reasoning-specialized SFT variant, Ouro-1.4B-Thinking, exists. — [HF](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking)
- **Code (secondary):** Ouro-2.6B is reported to beat Qwen3-8B on MBPP (80.40 vs 79.00), BBH (80.46 vs 77.65) and MMLU-Pro (55.73 vs 53.72). — [InsiderLLM](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/) (not checked against the paper)
- Ouro 1.4B-R4 is competitive with 4B models and 2.6B-R4 with 8B models. Looping does not add knowledge storage (~2 bits/parameter for both looped and standard models), so gains are in manipulating knowledge rather than memorizing it. — [Moonlight review](https://www.themoonlight.io/review/scaling-latent-reasoning-via-looped-language-models)
- **2026 follow-ups** target looped-LM decoding speed (titles only): "LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers" ([arXiv 2609.17184](https://arxiv.org/pdf/2609.17184)) and "LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models" ([arXiv 2610.21594](https://arxiv.org/pdf/2610.21594)).

### Inferences
- **Cutting visible reasoning tokens at 1.5B is not yet demonstrated for code.** Coconut's token savings came with lower math accuracy than CoT, and Huginn's latent reasoning looks weak under probing. Competitive programming needs math-like multi-step reasoning, so pure latent reasoning is a high-risk research bet.
- **Structured short traces are still the practical path.** The team's blueprint approach already gives most of the token savings with explicit, checkable plans.
- **Looped models are a credible "sharper per parameter" base.** Ouro-1.4B (or -Thinking) could be A/B tested as an alternative to R1-Distill-Qwen-1.5B on the team's data, with the license and research-use caveat in mind.
- **Phone trade-off.**
  - A 1.4B looped model needs only about 1.4B parameters of RAM (~0.7–0.8 GB at 4-bit, by arithmetic).
  - But each loop re-reads the shared weights during bandwidth-bound decoding, so per-token latency should be closer to a ~4× deeper model unless early exit triggers.
  - Looping therefore buys RAM, not speed. Adaptive exit is needed, and vLLM support is already missing.
- **The router idea has architectural analogs.** MoR learns per-token depth, and Ouro has an early-exit threshold. A problem-level router predicting a "thinking budget" is the coarse-grained, easy-to-build version. It could set the number of repair or refinement steps, or the reasoning-length cap.

### Gaps
- Ouro's primary code tables (HumanEval, MBPP) were not extracted; only a secondary MBPP figure was found.
- Huginn's code numbers were not found.
- I found no latent-reasoning (Coconut-style) results on code benchmarks.
- On-device runtime support (llama.cpp/ExecuTorch/MLC) for looped or MoR models is unverified.
- "Retrofitting" recurrence into an existing pretrained 1.5B model was not checked in this session.

---

## 6. Other viral 2025–2026 ideas that make small models sharp (SEAL, test-time training/TTT-Discover, AlphaEvolve/OpenEvolve/ThetaEvolve, Darwin Gödel Machine, Recursive Language Models, etc.)

### Takeaway
The most consistent 2025–2026 pattern is search plus verification plus learning at test time. ARC Prize 2025 called "refinement loops" the year's central theme. Evolutionary or test-time-RL systems now let *small open models* reach frontier-level discoveries: ThetaEvolve with an 8B model set new best-known bounds, and TTT-Discover with gpt-oss-120b narrowly beat the best humans on two AtCoder Heuristic Contests. These systems are far too expensive per problem for a phone. Their main use for a 1.5B competitive-programming model is as *offline data engines*: discover verified solutions and blueprints, then distill them. SEAL and DGM are powerful but costly self-improvement loops.

### Cited Findings

**SEAL (Self-Adapting Language Models; MIT, June 2025; NeurIPS 2025)**
- The LLM generates "self-edits" (its own fine-tuning data and update directives). They are applied by fine-tuning, and RL uses the updated model's downstream performance as the reward. — [SEAL (arXiv 2506.10943)](https://arxiv.org/pdf/2506.10943)
- **Few-shot abstract reasoning (ARC subset), Llama-3.2-1B-Instruct:**
  - SEAL 72.5% success;
  - self-edits without RL 20%;
  - in-context learning 0%;
  - Oracle TTT upper bound 100%.
- SEAL-generated data beat GPT-4.1-generated synthetic data for knowledge incorporation. Catastrophic forgetting is a known limitation. — [SEAL (arXiv 2506.10943)](https://arxiv.org/pdf/2506.10943); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/118690)

**TTT-Discover ("Learning to Discover at Test Time", Jan 2026; ICML 2026 spotlight)**
- Reinforcement learning at test time on a single problem, using the open gpt-oss-120b, for a few hundred dollars per problem (one summary says ~$500). — [arXiv 2601.16175](https://arxiv.org/html/2601.16175); [ICML 2026](https://icml.cc/virtual/2026/spotlight/84891)
- **AtCoder Heuristic Contest results:**

| Contest | TTT-Discover | Best human | Previous best AI |
|---|---|---|---|
| AHC039 | 567,062 | 566,997 | 558,026 |
| AHC058 | 848,414,228 | 847,674,723 | 848,373,282 |

  - The margins are tiny, and these are optimization-score contests, not exact-answer competitive programming.
  - Source: [arXiv 2601.16175](https://arxiv.org/html/2601.16175)
- A pip package exists. — [PyPI ttt-discover](https://pypi.org/project/ttt-discover/)

**AlphaEvolve → OpenEvolve → ThetaEvolve**
- **AlphaEvolve (Google DeepMind, May 2025)** found a way to multiply 4×4 complex-valued matrices with 48 scalar multiplications, improving on Strassen (1969) in that setting. — [DeepMind blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
  - **Hype check:** for real matrices the scheme needs more than 48 real multiplications. Commenters also point to Winograd's 1967 48-multiplication method over commutative rings, which is not recursive. — [E. Davis notes (NYU)](https://cs.nyu.edu/~davise/papers/AlphaEvolveNotes.pdf); [36Kr](https://www.36kr.com/p/3299453439477768)
- **OpenEvolve** is the open-source implementation of AlphaEvolve. — [GitHub codelion/openevolve](https://github.com/codelion/openevolve)
  - A critical blog post is titled as if OpenEvolve "cheated" on benchmark algorithms. It is member-only and was not read, so treat it as an unverified warning about evaluator gaming. — [Towards AI](https://pub.towardsai.net/i-pointed-openevolve-at-18-algorithms-the-6-2k-star-alphaevolve-clone-cheated-by-321-8912d8f11e72)
- **ThetaEvolve (Microsoft + UW, Nov 2025; ICML 2026).**
  - Uses one small open model (**DeepSeek-R1-0528-Qwen3-8B**) to set new best-known bounds on circle packing and the first autocorrelation inequality, two open problems from AlphaEvolve.
  - It is a single LLM rather than an ensemble, with a population of up to 10,000 (vs 70 in OpenEvolve), lazy penalties, and optional RL during evolution.
  - Test-time RL consistently beat inference-only runs, and the RL-trained checkpoints transferred to other problems.
  - Code: github.com/ypwang61/ThetaEvolve.
  - Sources: [ThetaEvolve (arXiv 2511.23473)](https://arxiv.org/html/2511.23473v1); [ICML 2026](https://icml.cc/virtual/2026/poster/63428)
- **ARC Prize 2025's 2nd-place paper**, "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI", is another evolutionary-plus-self-improvement result with LMs. — [ARC Prize 2025](https://arcprize.org/blog/arc-prize-2025-results-analysis)

**Darwin Gödel Machine (DGM; Sakana AI + UBC, May 2025)**
- A self-improving coding agent that rewrites its own scaffold through open-ended evolution. Over 80 iterations, SWE-bench went from **20.0%→50.0%** and Polyglot from **14.2%→30.7%**. — [DGM (arXiv 2505.22954)](https://arxiv.org/html/2505.22954v3)
- Without self-improvement it reached 39.0% on SWE-bench (secondary). One 80-iteration SWE-bench run costs about **$22,000** and about 2 weeks. — [HF Papers summary](https://huggingface.co/papers/2505.22954)

**Recursive Language Models (RLM; Zhang, Kraska, Khattab; blog Oct 2025, paper Dec 2025)**
- This is not related to TRM. The long prompt is kept as an external variable in a REPL, and the model writes code to inspect it and recursively calls itself on pieces. — [RLM paper (arXiv 2512.24601)](https://arxiv.org/html/2512.24601v2); [blog](https://alexzhang13.github.io/blog/2025/rlm/)
- **OOLONG results conflict across sources.** One reports RLM(GPT-5-mini) at 64.9% vs GPT-5 at 30.3% (132k tokens). Another reports GPT-5 at 44.0 vs RLM(GPT-5) at 56.5. — [Buttondown summary](https://buttondown.com/dodatathings/archive/recursive-language-models-how-to-process-infinite)
- RLM-Qwen3-8B beats Qwen3-8B by 28.3% on average. — [RLM paper](https://arxiv.org/html/2512.24601v2)

### Inferences
- **Offline data engine (highest value).**
  - Run OpenEvolve/ThetaEvolve-style search with the 1.5B model (plus a larger mutator if needed) on *training* problems that it currently fails, using hidden-test pass rate as fitness.
  - Keep the verified solutions, ask the teacher to write short blueprints for them, and distill (SFT/RFT). This is a cheap form of expert iteration.
  - ThetaEvolve's result, an 8B model plus search and test-time RL reaching state-of-the-art bounds, is direct evidence that small models plus verifiers plus search punch far above their size.
- **Phone inference.** Per-problem TTT/RL (TTT-Discover) is infeasible. A light version is feasible: generate code, run the provided sample tests locally, then repair (1–3 iterations). This is the same refinement-loop principle at tiny cost.
- **SEAL-style self-edits.** Let the model write its own (blueprint, code, tests) training items for topics where it fails, and reward self-edits by post-update pass rate. The cost (an inner LoRA update per RL sample) is high but affordable with ample compute. Watch for forgetting.
- **DGM-style scaffold evolution.** Evolve the *pipeline*: router thresholds, number of retrieved blueprints, repair-loop policy. This works far cheaper with a 1.5B task model than DGM's $22k frontier-model runs.
- **Hype flags.**
  - TTT-Discover's wins are sub-0.1% margins on heuristic-score contests.
  - AlphaEvolve's matrix result has important caveats.
  - DGM's numbers rely on expensive frontier-model loops.
  - None of these show a *standalone* tiny model becoming strong at exact-answer competitive programming without heavy search.

### Gaps
- SEAL's knowledge-incorporation numbers, AlphaEvolve's "0.7% of Google compute" figure, and the OpenEvolve "cheating" critique were not verified.
- Not checked this session, though possibly relevant to small code models: Absolute Zero (self-play with a code executor), DeepConf (confidence-filtered parallel thinking), ShinkaEvolve (sample-efficient evolution), L1/LCPO length-controlled reasoning, and RL recipes on R1-Distill-1.5B for code (e.g. DeepCoder-1.5B). These likely overlap with adjacent researchers' scope.

---

## 7. Name disambiguation: which "G model" and "JEV model" did the team hear about, and which are relevant?

### Takeaway
"G model" most likely refers to **GEPA**, a prompt optimizer rather than a model. "JEV model" most likely refers to **JEPA**, specifically **LLM-JEPA** for NL↔code. Both are relevant, GEPA more immediately. Other plausible "G" matches: the **Gemma** family (2026's Gemma 4 E2B for phones, FunctionGemma-270M as a router base, EmbeddingGemma for on-device retrieval, DiffusionGemma for research only), **gpt-oss-20b** (a teacher, and a precedent for reasoning-effort control), **GLM-5** (teacher only) and **Gemini Diffusion** (closed; evidence only). The plausible "JEV" alternative is **Jet-Nemotron-2B**, an efficient small LM whose biggest speedups are at long contexts.

### Cited Findings

**"G" candidates**
- **GEPA:** see §3. Reflective prompt evolution that beat GRPO by 6% on average in v2, with up to 35× fewer rollouts. — [GEPA v2](https://arxiv.org/pdf/2507.19457v2)
- **Gemma 4 (2026)** comes in E2B, E4B, 26B MoE and 31B open-weight sizes. — [Codersera guide](https://codersera.com/blog/gemma-4-complete-guide-2026/); [Gemma 4 tech report (arXiv 2607.02770)](https://arxiv.org/pdf/2607.02770)
  - E2B is about 2.3B effective parameters, or 5.1B including embeddings. — [Medium](https://medium.com/@aristojeff/what-you-miss-if-you-read-gemma-4-as-just-another-open-model-5188e8c735b3)
  - Google's memory table gives about 1.1 GB for the mobile build (0.84 GB text-only). — [Google AI docs](https://ai.google.dev/gemma/docs/core) (figures vary by source and quantization)
- **FunctionGemma (Dec 2025):** a 270M model intended as a base for fine-tuned, specialized function-calling models. The quantized text-only version reportedly uses about 191 MB of RAM on a Pixel. — [innFactory](https://innfactory.ai/en/ai-models/google-gemma/) (secondary)
  - A 2026 paper extends it for on-device mobile function calling. — [arXiv 2609.25373](https://arxiv.org/pdf/2609.25373)
- **EmbeddingGemma (on-device embeddings):** a secondary listing describes an "EmbeddingGemma 2" with 740M parameters (270M text-only), 8,192-token context, Apache 2.0, and under 600 MB RAM targeted. — [innFactory](https://innfactory.ai/en/ai-models/google-gemma/) (secondary; unverified)
- **DiffusionGemma (June 2026):** 26B-A4B, open, experimental text diffusion. Too large for phones. — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)
- **gpt-oss-20b (OpenAI, Aug 2025):**
  - 21B total and 3.6B active parameters (32 experts, top-4), running within 16 GB thanks to MXFP4.
  - Reasoning effort has three levels (low, medium, high), set in the system prompt, e.g. "Reasoning: high".
  - Sources: [LM Studio](https://lmstudio.ai/models/openai/gpt-oss-20b); [vLLM recipes](https://recipes.vllm.ai/openai/gpt-oss-20b)
- **GLM-5 (Zhipu/Z.ai, Feb 2026):** 744B total and ~40B active parameters, MIT-licensed weights. Point releases (5.x) are reported at ~753B (lower confidence). — [Artificial Analysis](https://artificialanalysis.ai/models/releases/glm-5); [Noqta](https://www.noqta.tn/en/news/zhipu-glm-5-open-source-model-rivals-western-ai-2026)
- **Gemini Diffusion:** closed experimental demo (§4). — [DeepMind](https://deepmind.google/models/gemini-diffusion/)

**"JEV" candidates**
- **LLM-JEPA, DLLM-JEPA and VL-JEPA:** see §2. — [LLM-JEPA v2](https://arxiv.org/html/2509.14252v2); [VL-JEPA](https://arxiv.org/pdf/2512.10942)
- **Jet-Nemotron (NVIDIA + MIT Han Lab, Aug 2025), built with Post Neural Architecture Search (PostNAS):**
  - Jet-Nemotron-2B has higher MMLU-Pro accuracy than Qwen3-1.7B-Base with **47× generation throughput** (H100, 64K context).
  - At 256K context it shows up to 6.14× prefill and 53.6× decoding speedups.
  - The paper argues KV-cache size matters more than parameter count for throughput.
  - Sources: [Jet-Nemotron (arXiv 2508.15884)](https://arxiv.org/html/2508.15884v3); [MIT Han Lab](https://hanlab.mit.edu/projects/jet-nemotron)
  - Secondary coverage says it is built on Qwen2.5-1.5B and averages 49.6 on math (+6.3 over Qwen3-1.7B-Base). — [36Kr](https://eu.36kr.com/en/p/3440505371022981)

### Inferences
- **Relevance ranking for this project:**
  1. **GEPA (high).** Cheap, model-agnostic, and directly optimizes the blueprints and prompts.
  2. **FunctionGemma-270M / EmbeddingGemma (high, for the router and retrieval on phone).** A fine-tuned 270M classifier or router and an on-device embedder fit easily alongside a ~1 GB 4-bit 1.5B coder.
  3. **LLM-JEPA (medium).** A training-time add-on with zero inference cost; gains are uncertain for code.
  4. **gpt-oss-20b/120b (medium, as teacher or reflection LM).** Its three-level reasoning-effort control is a direct precedent for the planned "thinking budget" router.
  5. **Gemma 4 E2B (medium).** A phone-ready alternative base, but not code-specialized; its competitive-programming quality is unverified.
  6. **Jet-Nemotron (low–medium).** An efficiency architecture whose gains concentrate at long contexts, while competitive-programming prompts plus blueprints are short.
  7. **GLM-5 (teacher only)**, **DiffusionGemma and Gemini Diffusion (research evidence only)**, **VL-JEPA (not relevant; vision)**.
- **Possible confusions.** "G" could also be GRPO, the RL algorithm GEPA is benchmarked against. "Recursive" could be confused between TRM (tiny recursive nets) and RLM (recursive LM calls over context); they are unrelated.

### Gaps
- I found no published code or competitive-programming benchmarks for Gemma 4 E2B or FunctionGemma-based routers.
- The EmbeddingGemma-2 specifications come only from secondary sources.
- Jet-Nemotron's performance on phone hardware and at short contexts was not found.
- The team's actual referents for "G" and "JEV" cannot be confirmed from sources; the ranking above is by plausibility and relevance.
