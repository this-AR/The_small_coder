# Prior-art and novelty check: research ideas for a ~1.5B competitive-programming code model (router, blueprints and related directions)

Scope and conventions (as of 2026-11-19). All web searches used generic technical terms only. About 48 searches and page fetches were run. "(abstract checked)" means I opened the arXiv abstract page and confirmed the details given. "[bg]" marks a well-known paper whose arXiv ID or venue comes from background knowledge and was not reopened this session; check it before citing in a paper. Novelty ratings are my judgement from the searches here. They are not exhaustive literature reviews: if I found nothing, the work may still exist.

---

## Idea 1: a tiny router (≤100M encoder or classifier) that predicts algorithm tags, difficulty and thinking budget, and uses them to pick topic-specific LoRA experts and algorithm-template retrieval

### Takeaway
Every component already has prior art:
- predicting tags and difficulty for competitive-programming (CP) problems (2023);
- generating tags or knowledge before coding (KareCoder, 2024);
- retrieving thought templates (Buffer of Thoughts 2024, ReasonFlux 2025);
- predicting a per-query token budget (TALE, SelfBudgeter, Learning How Hard to Think, DART);
- routing among LoRA adapters (MapCoder-Lite's role LoRAs, hard-routed reasoning LoRAs).

I found no work where a ≤100M router picks **algorithm-topic** LoRA experts, templates and a budget for a **≤2B CP coder**, and none that does it on-device. **Novelty: medium.** The novelty lies in the combination and the small-model regime, not in any single component.

### Cited Findings
**What the router predicts (tags and difficulty)**
- Kim, Cho, Na, "Problem-Solving Guide: Predicting the Algorithm Tags and Difficulty for Competitive Programming Problems." arXiv 2310.05791 (v1 Oct 2023, v2 Oct 2024), preprint.
  - Predicts algorithm tags and difficulty jointly on the new AMT dataset, mostly Codeforces problems.
  - Claims to be the first work to predict algorithm-problem difficulty.
  - The abstract page gives no accuracy figures (abstract checked). — [arXiv](https://arxiv.org/abs/2310.05791)
- An ENIAC (Brazil) paper on multi-label tagging of programming problems with pre-trained LMs reports two things: using only the problem statement can predict better, and a Llama-based classifier did well on graph tags (search snippet only). — [SBC/ENIAC PDF](https://sol.sbc.org.br/index.php/eniac/article/download/33783/33574)
- CodeElo (2025) gives each Codeforces problem its contest division, difficulty rating and algorithm tags. This is ready-made supervision for a router. — [arXiv 2501.01257](https://arxiv.org/html/2501.01257v1)
- AlgoSimBench (2025) argues that platform tags such as "dynamic programming" are usually too general to capture a solution strategy. It calls for finer tags such as "Longest Increasing Subsequence". — [arXiv 2507.15378](https://arxiv.org/pdf/2507.15378)

**Predict tags or knowledge first, then generate code**
- Huang, Sun, Jin, Li, Lyu, "Knowledge-Aware Code Generation with Large Language Models" (KareCoder). arXiv 2401.15940, accepted at ICPC 2024.
  - Builds a Knowledge Library for Python contest problems and injects library knowledge into ChatGPT's code-generation reasoning.
  - Reports +23.3% relative Pass@1 over direct ChatGPT on CodeF post2021-9 (abstract checked). — [arXiv](https://arxiv.org/abs/2401.15940)
  - A search snippet of the PDF says APPS lacks algorithm tags, so the authors use a ChatGPT-prompted tag generator, checked with Cohen's kappa. — [PDF](https://arxiv.org/pdf/2401.15940)
- AlphaCode (Li et al., Science 2022):
  - Trained with metadata in the prompt: ratings, tags, language and whether the solution was correct.
  - At sampling time it randomised the tags and ratings (ratings 800–3,500) to increase diversity, because tags and ratings are hidden during a contest. — [arXiv 2203.07814](https://arxiv.org/abs/2203.07814)
  - AlphaCode 2 also randomises the difficulty rating and categorical tags in the prompt. — [AlphaCode 2 Tech Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)
  - A secondary summary says the metadata contributed only about +0.6% solve rate (unverified). — [GreaterWrong summary](https://www.greaterwrong.com/posts/ZmxkmCjXJBpwJkgrw/competitive-programming-with-alphacode)

**Template retrieval (what M3 does)**
- Buffer of Thoughts (Yang et al., NeurIPS 2024; arXiv 2406.04271) keeps a "meta-buffer" of high-level thought templates in six categories, one of which is Code Programming. It retrieves templates by embedding similarity. — [arXiv](https://arxiv.org/abs/2406.04271); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cde328b7bf6358f5ebb91fe9c539745e-Abstract.html)
- ReasonFlux (Yang et al., 2025; arXiv 2502.06772) has a library of about 500 structured thought templates, each with a name, retrieval tags and a description. At inference it retrieves templates and chains them into trajectories. ReasonFlux-32B solves about 56.7% of AIME problems. — [arXiv](https://arxiv.org/abs/2502.06772)

**Predicting the budget and routing by difficulty**
- TALE, "Token-Budget-Aware LLM Reasoning" (arXiv 2412.18547), estimates a token budget per question before reasoning. It does this in one of three ways:
  - a zero-shot self-estimate;
  - a separately trained regression estimator, which is an LLM;
  - post-training (TALE-PT).

  It reports about 67% fewer tokens with an accuracy drop under 3%; the figure varies across versions. — [arXiv](https://arxiv.org/abs/2412.18547)
- Damani, Shenfeld et al., "Learning How Hard to Think: Input-Adaptive Allocation of LM Computation," ICLR 2025:
  - Small probes on the LM's hidden states predict the marginal reward of extra compute for each query.
  - On Math and Code it matches non-adaptive methods with up to 50% less compute.
  - It also routes queries between a large and a small model. — [arXiv 2410.04707](https://arxiv.org/abs/2410.04707); [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ff414825df833edb8b1839e3d5d495e9-Abstract-Conference.html)
- SelfBudgeter (arXiv 2505.11274) trains the model to estimate its own reasoning budget. — [arXiv](https://arxiv.org/html/2505.11274v6)
- AdaCtrl (arXiv 2505.18822) budgets by self-assessed difficulty, with a difficulty-aware length reward. It is evaluated on math only. — [arXiv](https://arxiv.org/abs/2505.18822)
- Route-to-Reason (RTR, 2025) routes over both LLMs and reasoning strategies, using predicted performance and predicted output length. — [ResearchGate](https://www.researchgate.net/publication/392104768_Route_to_Reason_Adaptive_Routing_for_LLM_and_Reasoning_Strategy_Selection)
- Lee et al., "DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets in Hybrid Reasoning Models" (arXiv 2606.23181; EMNLP 2026 Findings):
  - Samples two cheap no-think drafts. If they agree, it answers directly; if not, it sets a thinking budget from the drafts' entropy.
  - Up to +22.5 points on code (execution-based equivalence) with 32–73% fewer thinking tokens.
  - The first-stage signal works from 0.6B to 32B (abstract checked). — [arXiv](https://arxiv.org/abs/2606.23181)

**Routing among LoRA experts**
- Lee, Cho, Choi, "MapCoder-Lite" (arXiv 2509.17489). v1 (Sep 2025) was titled "Squeezing Multi-Agent Coding into a Single Small LLM"; v2 (Feb 2026) is "Distilling Multi-Agent Coding into a Single Small LLM".
  - A single 7B model with agent-wise LoRA fine-tuning for the retriever, planner, coder and debugger roles.
  - xCodeEval accuracy rises from 13.2% to 28.3%. Format failures are eliminated, and GPU memory and token-generation time are 4× lower than a 32B model (abstract checked). — [arXiv](https://arxiv.org/abs/2509.17489)
  - Summaries of v1 describe a frozen Qwen2.5-7B backbone with rank-32 adapters per role, adding under 3% parameters. — [arXiv v1](https://arxiv.org/html/2509.17489v1)
- Molavi et al., "Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs" (arXiv 2606.31413, Jun 2026):
  - Domain LoRA experts are trained with RL from verifiable rewards (RLVR) and then frozen.
  - A lightweight router then picks exactly one expert per token (hard top-1).
  - It finds that normalised soft mixtures often concentrate most of the routing weight on a single expert. The abstract page names no domains (abstract checked). — [arXiv](https://arxiv.org/abs/2606.31413)
- Other LoRA-routing work:
  - LoRA-Mixer (arXiv 2507.00029; ICLR 2026 poster) routes over modular or frozen LoRA experts. — [arXiv](https://arxiv.org/html/2507.00029v2); [ICLR](https://iclr.cc/virtual/2026/poster/10010501)
  - MoLoRA (arXiv 2603.15965, Microsoft Research) routes per token among domain adapters, motivated by requests that mix code and math. — [arXiv](https://arxiv.org/abs/2603.15965)
  - LD-MoLE (ICLR 2026) uses learnable dynamic routing. — [arXiv](https://arxiv.org/html/2509.25684)

### Inferences
- **Novelty: medium.**
  - The prediction targets have been predicted before: tags, difficulty and budget.
  - "Predict tags, then generate" has been done with ChatGPT (KareCoder).
  - Template retrieval exists (BoT, ReasonFlux), and so does adapter routing (MapCoder-Lite, MoR-LoRA).
  - What is new is the regime: a ≤100M router driving algorithm-topic experts, templates and a budget for a ≤2B CP model on a phone.
- **Open questions** (from my searches):
  1. Nobody has tested whether algorithm-topic LoRA experts beat a single generalist LoRA at ≤2B for CP.
  2. Nobody has measured how router accuracy turns into Pass@1, against an oracle-routing upper bound.
  3. It is untested whether an external tiny encoder predicts the budget better than the coder's own estimate (TALE-style regression estimator versus SelfBudgeter or AdaCtrl).
  4. The on-device memory and latency of router plus adapter switching has not been measured.
- **Risks reviewers will raise:**
  - AlgoSimBench suggests platform tags are too coarse to choose a strategy.
  - MoR-LoRA's finding that soft mixtures collapse argues for hard routing.
  - MapCoder-Lite specialises by *role*, so reviewers will ask why specialise by *topic*.
  - The decisive baseline is one LoRA with the same total trainable parameters.
- **Framing for a novel paper:** treat the router as a cheap "control plane" and report:
  - (a) the oracle-routing upper bound and the curve from router accuracy to Pass@1;
  - (b) hard top-1 topic routing versus a single LoRA versus no routing, at matched parameters;
  - (c) selective routing, where the system abstains and falls back to longer thinking or generic blueprints when the router is unsure;
  - (d) on-device latency and memory.

  Results (a) and (c) are the parts least covered by prior work.

### Gaps
- I could not retrieve the PSG/AMT accuracy numbers or dataset size from the abstract page. A snippet mentioned about 8k problems, unverified.
- KareCoder's tag-generator details come from a snippet.
- RTR's details come from a ResearchGate listing.
- MoR-LoRA's abstract gives no code-specific results.
- A targeted search for DP/graph/greedy-specific LoRA experts for CP found nothing. That is weak evidence the gap is real.

---

## Idea 2: short structured "blueprint" reasoning (goal / key insight / pseudocode / corner cases) as the distillation target for small students, instead of long free-form chain of thought

### Takeaway
**Novelty: low to medium on its own.**
- For math, it is established that ≤3B students learn better from short than from long chain of thought (CoT) (Small-Model Learnability Gap, ICML 2025).
- Pseudocode and plan formats already exist as compact supervision: CodePlan, PGPO, a 2025 pseudocode-distillation paper, and prompting-era work such as SCoT and self-planning.
- **"Blueprint2Code"** (Frontiers in AI, 2025) already uses the word "blueprint" for planning in code generation, so the paper must address the naming clash.

What remains open is a controlled, execution-verified CP study at 1.5B that **separates the effect of structure from the effect of length**.

### Cited Findings
- Li et al., "Small Models Struggle to Learn from Strong Reasoners" (arXiv 2502.12143, Feb 2025; listed on the ICML 2025 site):
  - Students of ≤3B do not reliably improve from long CoT or from larger teachers. They do better with short CoT or smaller teachers.
  - Qwen2.5-1.5B-Instruct scores more than 10 points lower on MATH and AMC when trained on long CoT.
  - Mix Distillation, which mixes long and short CoT, beats either alone.
  - The evaluation is on math. — [arXiv](https://arxiv.org/abs/2502.12143); [ICML listing](https://icml.cc/virtual/2025/52467); [project page](https://small-model-gap.github.io)
- MI-Distillation (secondary summary): short CoT gives better supervision for 1.5B and 3B students, and long CoT only marginally overtakes it at 7B. — [Lacuna summary](https://lacuna.tiptreesystems.com/work/mi-distillation-selecting-from-model-interpolated-instruct-reasoning-data/wrk_cfc1488c8a6867f4b47a67b34de2b5c3)
- "Efficient Long CoT Reasoning in Small Language Models" (arXiv 2505.18440) cuts redundant "overthinking" steps from long CoT before distilling into small LMs. The experiments are on math. — [arXiv](https://arxiv.org/abs/2505.18440)
- Zyphra's ZR1-1.5B is an RL-trained math and code model that uses 60% fewer tokens than R1-Distill-1.5B. — [Zyphra](https://zyphra.com/post/introducing-zr1-1-5b-a-small-but-powerful-math-code-reasoning-model)
- ShorterBetter (arXiv 2504.21370), at 1.5B, reduces repetition, excessive self-verification and over-exploration in reasoning traces. — [arXiv](https://arxiv.org/abs/2504.21370)
- CodePlan, "Unlocking Reasoning Potential in Large Language Models by Scaling Code-form Planning" (arXiv 2409.12452; thu-coai), trains models to produce "code-form plans", i.e. pseudocode. The authors credit code's structure for plan labels that are more concise and easier to learn. — [arXiv](https://arxiv.org/abs/2409.12452); [GitHub](https://github.com/thu-coai/CodePlan)
- PGPO (arXiv 2506.01475) uses GPT-4o to rewrite ReAct-style plans into pseudocode-style plans. The authors argue the gains come from the structure itself, not only from knowledge distillation. The setting is agents, not CP. — [arXiv](https://arxiv.org/pdf/2506.01475)
- "Symmetry-Aware Code Generation: Distilling Pseudocode Reasoning for Lightweight Deployment of Large Language Models" (MDPI *Symmetry* 17(8):1325, 2025):
  - A teacher writes pseudocode and then code; a smaller student learns to produce both, with a combined loss.
  - It explicitly targets lightweight deployment.
  - These details come from a search snippet; the page returned HTTP 403 when fetched. — [MDPI](https://www.mdpi.com/2073-8994/17/8/1325)
- o1-Coder (arXiv 2412.00154) uses pseudocode as an intermediate step inside an MCTS-guided reasoning loop. — [arXiv](https://arxiv.org/pdf/2412.00154)
- Mao et al., "Blueprint2Code: a multi-agent pipeline for reliable code generation via blueprint planning and repair" (Frontiers in Artificial Intelligence, Oct 2025; doi 10.3389/frai.2025.1660912):
  - Four agents (Previewing, Blueprint, Coding, Debugging). Plans are ranked by confidence, there are up to 5 debug rounds, and the system replans if debugging fails.
  - Reports pass@1 of 96.3% on HumanEval, 88.4% on MBPP and 24.6% on APPS.
  - It is a prompting pipeline with API LLMs. A summary cites 89.1% HumanEval with GPT-4o-mini, which it calls an "8B" model; that size is not an official figure. — [Frontiers](https://www.frontiersin.org/articles/10.3389/frai.2025.1660912); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12575318/)
- Hadhoud et al., "Idea First, Code Later: Disentangling Problem Solving from Code Generation in Evaluating LLMs for Competitive Programming" (arXiv 2601.11332, Jan 2026):
  - Writing a natural-language editorial before the code improves solve rates for some LLMs; expert "gold" editorials help much more.
  - Even with gold editorials, models still struggle with implementation.
  - Uses 83 ICPC-style problems with gold editorials and evaluates 19 LLMs (abstract checked). — [arXiv](https://arxiv.org/abs/2601.11332)
- Prompting-era structured-planning baselines that reviewers will expect [bg]:
  - SCoT, structured CoT for code — [arXiv 2305.06599](https://arxiv.org/abs/2305.06599)
  - Self-planning code generation — [arXiv 2303.06689](https://arxiv.org/abs/2303.06689)
  - CodeChain (ICLR 2024) — [arXiv 2310.08992](https://arxiv.org/abs/2310.08992)
  - "Think Outside the Code: Brainstorming Boosts LLMs in Code Generation" — [arXiv 2305.10679](https://arxiv.org/abs/2305.10679)
  - PlanSearch — [arXiv 2409.03733](https://arxiv.org/abs/2409.03733)
- Related (title only, not opened): "Distilling Algorithmic Reasoning from LLMs via Explaining Solution Programs". — [Papers with Code](https://astro.paperswithcode.com/paper/distilling-algorithmic-reasoning-from-llms)

### Inferences
- **Novelty: low to medium.**
  - The Stage-1 collapse (0–1% Pass@1 after long-CoT SFT) looks like the code version of the learnability gap. This is consistent with Li et al. 2025 but not yet shown for CP code.
  - The four-field blueprint format is close to SCoT, self-planning, CodePlan pseudocode and Blueprint2Code's blueprint stage.
  - The distinguishing points must be (a) it is a *training target for a ≤2B student*, not a prompting pipeline for API models, and (b) the controlled evidence.
- **Open questions:**
  1. There is no execution-verified evidence at 1.5B on CP that compares long CoT, short CoT, blueprint, pseudocode only, and code only, from the same teacher on the same problems.
  2. Structure has not been separated from length, for example blueprint versus long CoT truncated or compressed to the same number of tokens.
  3. It is unknown which fields matter. Ablate [CORNER CASES], [PSEUDOCODE] and the others.
  4. The teacher effect is untested: blueprints written by an instruct model versus R1 long traces compressed into blueprints.
  5. It is unknown whether blueprint SFT is a better starting point for RL than long-CoT SFT.
- **Framing:** "Format learnability, not length, limits small coders". Use a controlled experiment matrix with Mix Distillation and long-CoT pruning as baselines. Either rename the format (e.g., "algorithm sketch") or explicitly set it apart from Blueprint2Code.

### Gaps
- I found no direct head-to-head CP distillation result at 1.5B (absence in search, not proof).
- I could not open the MDPI paper; its details come from a snippet.
- The SCoT, self-planning, CodeChain, Brainstorm and PlanSearch IDs are from background knowledge.

---

## Idea 3: conditioning generation on predicted algorithm tags or hints, and measuring the gain against oracle tags

### Takeaway
**Novelty: medium. This is the most tractable novel angle.** Hints are known to help LLMs on CP:
- AlphaCode's metadata conditioning;
- KareCoder's knowledge library;
- the TEACH hint generator (IEEE TSE 2025);
- small human hints on USACO problems;
- explanations derived from oracle solutions;
- gold editorials.

I found **no systematic study for ≤2B models** of a "hint-granularity ladder": none → predicted tag → oracle tag → oracle technique → oracle blueprint → gold editorial. Nor did I find one that measures how much of the oracle gain a tiny predictor recovers.

### Cited Findings
- AlphaCode trains with tag and rating metadata, randomises them at sampling, and they are unavailable at contest time (see Idea 1). — [arXiv 2203.07814](https://arxiv.org/abs/2203.07814)
- KareCoder uses a knowledge library plus a ChatGPT tag generator for +23.3% relative Pass@1 (see Idea 1). The abstract page mentions no comparison against oracle tags. — [arXiv 2401.15940](https://arxiv.org/abs/2401.15940)
- Zhang, Dong et al., "A Little Help Goes a Long Way: Tutoring LLMs in Solving Competitive Programming through Hints" (TEACH), IEEE Transactions on Software Engineering 2025 (doi 10.1109/TSE.2025.3614204):
  - A hint generator fine-tuned on curated hints from competition platforms, combined with structured CoT.
  - Up to +6.56 points (17.4% relative) pass@1 on LeetCode, with smaller gains on APPS and ASAC.
  - These numbers come from a ResearchGate abstract snippet. — [oa.mg record](https://oa.mg/work/10.1109/tse.2025.3614204); [ResearchGate](https://www.researchgate.net/publication/395900769_A_Little_Help_Goes_a_Long_Way_Tutoring_LLMs_in_Solving_Competitive_Programming_through_Hints)
- Shi, Tang, Narasimhan, Yao, "Can Language Models Solve Olympiad Programming?" (USACO benchmark, arXiv 2404.10952, 2024; COLM 2024 [bg]):
  - 307 USACO problems. GPT-4 reaches 8.7% zero-shot CoT pass@1, and 20.2% with self-reflection plus retrieval over episodic knowledge.
  - In a human-in-the-loop study, a few targeted hints let GPT-4 solve 13 of 15 problems no model or method had solved.
  - Only some models can make use of feedback. — [arXiv](https://arxiv.org/abs/2404.10952); [project](https://princeton-nlp.github.io/USACOBench)
- "Explaining Competitive-Level Programming Solutions using LLMs" (Li et al., 2023; arXiv 2307.05337; appears as an ACL 2023 NLRSE workshop paper):
  - Explanations are generated from <problem, oracle solution> pairs and fed to the solver as hints.
  - With step-by-step explanations, CodeContests solve@10 rose from 6.1% to 42.4% (secondary summary).
  - One-sentence hints helped GPT-3.5 on easier problems. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2307.05337); [Lacuna summary](https://lacuna.tiptreesystems.com/work/explaining-competitive-level-programming-solutions-using-llms/wrk_a888bad8ad7e4b27bcca1848e4004494); [UT Austin PDF](https://www.cs.utexas.edu/users/ml/papers/li.acl23nlrse.pdf)
- "Idea First, Code Later" (arXiv 2601.11332): gold editorials help much more than editorials the model writes itself, and implementation remains a bottleneck even with gold editorials. — [arXiv](https://arxiv.org/abs/2601.11332)
- AlgoSimBench: platform tags are too coarse to capture the strategy. — [arXiv 2507.15378](https://arxiv.org/pdf/2507.15378)
- Jha & Jha, "Where Do Large Language Models Fail on Competitive Programming? A Taxonomy of Failures by Algorithm Type and Difficulty Rating" (arXiv 2606.05228, Jun 2026):
  - 315 Codeforces problems across 7 algorithm categories and 3 difficulty tiers, with GPT-4o and Claude Sonnet 4.6.
  - Forced CoT dropped GPT-4o's pass rate from 46.0% to 36.8%.
  - Wrong Answer accounts for over 90% of GPT-4o's failed solutions and about 70% of Claude's.
  - The authors argue that prompt engineering does not close the gap. The abstract page shows no hint or tag ablation (abstract checked). — [arXiv](https://arxiv.org/abs/2606.05228)
- A critique of a CP repair framework (arXiv 2506.22954) notes that part of its gain comes from author-written algorithm hints in the prompt. This shows reviewers do scrutinise hint leakage. — [Pith review](https://pith.science/paper/2506.22954)
- Solvita (arXiv 2605.15301, 2026) uses tagged problems, 107 algorithmic tags after filtering, to build training data for CP. — [arXiv](https://arxiv.org/html/2605.15301v1)
- HintMR (arXiv 2604.12229, 2026) studies hints to elicit math reasoning in small LMs (title and listing only). — [arXiv](https://arxiv.org/html/2604.12229v1)

### Inferences
- **Novelty: medium.** It could be higher if run carefully at ≤2B with the full ladder and with a router that recovers a measured share of the oracle gap. Prior hint work targets large models (GPT-4, ChatGPT), uses human or LLM hints, or (AlphaCode) uses tags for diversity rather than for test-time guidance.
- **Open questions:**
  1. The oracle gap for small models. A 1.5B coder may be implementation-bound even with oracle tags; "Idea First, Code Later" shows that large LLMs still fail at implementation with gold editorials.
  2. Robustness to *wrong* tags: what happens with top-1 versus top-k tags, and with confidence gating.
  3. Training-time tag conditioning (AlphaCode-style, during SFT) versus conditioning only at inference.
  4. Which hint granularity is learnable and cost-effective at 1.5B: a coarse tag, a fine technique such as "LIS" or "two pointers", a blueprint or an editorial.
- **Framing:** "How much of a small coder's failure is algorithm selection versus implementation?" Report:
  - a table of Pass@1 against hint level and difficulty tier;
  - the share of the oracle-tag gain recovered by the tiny router;
  - the cost of a wrong hint.

  This also gives Idea 1 its scientific core and directly explains the M3 result (11% with retrieved blueprints).

### Gaps
- I could not open the TEACH full text; its numbers come from a snippet.
- The "Idea First, Code Later" abstract gives no figures.
- I found no paper measuring predicted against oracle tags for small models. This is absence in my search, not proof of absence.

---

## Idea 4: a JEPA-style auxiliary embedding-prediction loss between problem statement and solution code during code-LLM fine-tuning

### Takeaway
**Novelty: low to medium.** LLM-JEPA (ICLR 2026) already uses natural language and code as two views of the same thing: NL↔regex (NL-RX) and NL↔SQL (Spider). It already works with LoRA fine-tuning. Pairing a CP problem statement with its solution is a direct instance of that set-up. The method costs about 3× the forward passes, and I found no 2026 follow-up specific to code generation.

### Cited Findings
- Huang, LeCun, Balestriero, "LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures" (arXiv 2509.14252; ICLR 2026):
  - Adds a JEPA term to the next-token loss. The views are paired text and code (NL-RX, Spider), plus GSM8K and RottenTomatoes.
  - The embedding is the last-layer hidden state of the final token. The two views are encoded in separate forward passes.
  - It beats standard objectives across the Llama3, OpenELM, Gemma2 and OLMo families.
  - Under LoRA fine-tuning, the baseline plateaus and then degrades, while LLM-JEPA keeps improving with more epochs. — [arXiv](https://arxiv.org/abs/2509.14252); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ac818ec2da1c976b267de32f59709c69-Abstract-Conference.html)
- Cost and follow-ups (aggregator):
  - About 3× training compute because of the extra forward passes.
  - 2026 follow-ups include DLLM-JEPA (views from a masked-diffusion model) and a retrieval-focused variant.
  - No follow-up specific to code generation was found. — [emergentmind topic page](https://www.emergentmind.com/topics/llm-jepa)

### Inferences
- **Novelty: low to medium.** "Statement ↔ solution code" is the natural CP instance of LLM-JEPA's own text↔code views.
- **Open questions:**
  - CP problems have many correct solutions per statement, which calls for a multi-positive JEPA loss.
  - CP outputs are long.
  - It is unknown whether JEPA alignment makes the coder's hidden states *algorithm-aware*, i.e. whether a linear probe can recover the tag.
  - Cheaper single-pass variants are untested.
- **Possible framing** (low priority): JEPA as a way to make the coder's own embedding serve as the router, removing the separate encoder. Report tag-probe accuracy, Pass@1 and behaviour over epochs. LLM-JEPA's reported resistance to overfitting with LoRA is relevant to small (about 2k-sample) blueprint SFT sets.

### Gaps
- I found no independent replication.
- The emergentmind pages are aggregators.

---

## Idea 5: TRM/HRM-style recursive latent refinement applied to code generation or program repair

### Takeaway
**Novelty: medium to high, but low feasibility and expected impact for CP today.**
- The only direct code study (CASCON 2026) is a 28M autoregressive TRM on natural-language-to-Python.
- Its advantage is attributed to resisting overfitting, at about 175× the per-step compute.
- I found **no application to program repair**.

### Cited Findings
- Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (TRM; arXiv 2510.04871, 2025):
  - A single tiny network recursively updates an answer and a latent scratchpad.
  - With 7M parameters it reaches 45% on ARC-AGI-1 and 8% on ARC-AGI-2 (aggregator summary). — [arXiv](https://arxiv.org/abs/2510.04871); [emergentmind](https://www.emergentmind.com/papers/2510.04871)
- HRM (Wang et al., "Hierarchical Reasoning Model," arXiv 2506.21734 [bg]) couples two recurrent modules that run at different time scales; TRM simplifies it. — [arXiv](https://arxiv.org/abs/2506.21734)
- Sirivella, Newaz, Melo, "Evaluating Tiny Recursive Models Across Training for Code Generation" (arXiv 2608.29376, Aug 2026; accepted at CASCON 2026):
  - Compares a ~28M TRM-AR with parameter-matched and depth-matched transformers over 40 epochs and 3 seeds, on natural-language-to-Python generation.
  - At equal parameters, TRM-AR fits, generates and generalises better. It recovers about 45% of the validation-loss gap and 57% of the generation-quality gap between the two controls.
  - This costs about 175× the per-step compute.
  - At equal effective depth, the larger transformer wins. The authors attribute TRM-AR's edge to resistance to overfitting.
  - The abstract page names no dataset (abstract checked). — [arXiv](https://arxiv.org/abs/2608.29376)
- Jim, Kolawole, Busoye, Bagai, Smith, "What Survives When You Compress a Recursive Reasoner for the Edge?" (arXiv 2606.26488, 2026, in review):
  - Aggressive compression keeps local predictions but destroys global reasoning; pruning, distillation and linear attention all do this.
  - Per-channel calibrated INT4 brings the backbone to 3.3 MB.
  - One recursive cycle matches full-depth accuracy at 6× fewer FLOPs.
  - The tasks are puzzle-style (abstract checked). — [arXiv](https://arxiv.org/abs/2606.26488)
- Related titles, not opened: "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator" — [arXiv 2511.16886](https://arxiv.org/pdf/2511.16886); "Simpler Tiny Recursive Models" — [arXiv 2611.23677](https://arxiv.org/pdf/2611.23677); Tab-TRM (TRM on tabular insurance pricing) — [arXiv 2601.07675](https://arxiv.org/pdf/2601.07675)
- An aggregator lists "TRM-driven iterative code fixers trained on bug/patch corpora" only as a speculative application, not a reported result. — [emergentmind](https://www.emergentmind.com/papers/2510.04871)

### Inferences
- **Novelty: medium to high for repair**, since I found no prior work. But TRM targets fixed-size grid outputs, while CP code is long text of variable length. The one code study shows regularisation benefits, not added capability, and at very high compute.
- **More feasible variants:** apply recursion to a *small, fixed-size* part of the system, for example iteratively refining the router's algorithm or plan latent, or predicting the verdict. Do not apply it to the coder itself.
- **Recommendation:** not a main contribution. At most an exploratory appendix.

### Gaps
- I found no TRM or HRM program-repair studies.
- The CASCON paper's dataset is not named on its abstract page.

---

## Idea 6: planner + coder decomposition or multi-agent pipelines built entirely from ≤2B models for CP

### Takeaway
**Novelty: medium. The novelty is mainly the size regime.**
- MapCoder, AgentCoder, CodeSim and Blueprint2Code assume large models.
- MapCoder-Lite (2025/26) reports that such systems *fail when scaled down*, and fixes this at **7B** with trajectory distillation plus per-role LoRA.
- ≤2B is untested. "MapCoder-Lite at 1.5B" is an obvious incremental step, so it needs a sharper scientific question.

### Cited Findings
- Multi-agent CP pipelines that reviewers will expect as baselines [bg]:
  - MapCoder (Islam, Ali, Parvez; ACL 2024) — [arXiv 2405.11403](https://arxiv.org/abs/2405.11403)
  - AgentCoder (Huang et al.) — [arXiv 2312.13010](https://arxiv.org/abs/2312.13010)
  - CodeSim (Islam et al.; NAACL 2025 Findings) — [arXiv 2502.05664](https://arxiv.org/abs/2502.05664)
- MapCoder-Lite's abstract says existing multi-agent coding systems "either need very large models (>30B) or fail when scaled down to small open-source models". Its fix has three parts:
  - pass-based trajectory distillation from strong LLMs;
  - supervisor-guided correction;
  - agent-wise LoRA.

  On xCodeEval accuracy goes from 13.2% to 28.3% at 7B (abstract checked). — [arXiv 2509.17489](https://arxiv.org/abs/2509.17489)
- Blueprint2Code: a four-agent blueprint pipeline using API LLMs (see Idea 2). — [Frontiers](https://www.frontiersin.org/articles/10.3389/frai.2025.1660912)
- Sharma & Mehta's survey of small language models for agentic systems (arXiv 2510.03847) argues that 1–12B models suit agent tasks constrained by schemas or APIs. It is not code-focused. — [arXiv](https://arxiv.org/abs/2510.03847)
- NVIDIA's position paper "Small Language Models are the Future of Agentic AI" [bg]. — [arXiv 2506.02153](https://arxiv.org/abs/2506.02153)

### Inferences
- **Open questions:**
  1. Does decomposition help or hurt at ≤2B, where errors compound across agents?
  2. Does a two-role (planner/coder) setup beat a single model trained on blueprint SFT at the same token and compute budget?
  3. A shared backbone with role LoRAs that switch on a phone.
  4. Combining role adapters with topic adapters.
- **Framing:** the blueprint is already a "planner" output, so the experiment is: one model emits blueprint plus code, versus a planner adapter and a coder adapter on a shared 1.5B backbone. Report by difficulty tier when the split pays off. Keep the multi-agent pipeline minimal; full MapCoder-style pipelines multiply on-device latency.

### Gaps
- I did not reopen the MapCoder, AgentCoder or CodeSim papers.
- I found no multi-agent CP numbers at ≤2B.

---

## Idea 7: a fully on-device generate → self-test → repair loop (local Python sandbox on a phone)

### Takeaway
**Algorithmic novelty: low**, because self-debugging and self-repair are well studied. But I found **no work measuring a code model's generate-test-repair loop on a phone with local execution**, with accuracy weighed against latency and energy. **Novelty: medium, as a systems or empirical contribution.**

Two cautions: small models struggle to self-correct unless trained to (CoCoS), and self-repair gains are modest once its cost is counted (Olausson et al.).

### Cited Findings
- Self-Debugging (Chen et al., ICLR 2024) teaches models to explain their code and use execution feedback through few-shot prompting, without training. — [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/2460396f2d0d421885997dd1612ac56b-Paper-Conference.pdf)
- Olausson et al., "Is Self-Repair a Silver Bullet for Code Generation?" (arXiv 2306.09896; ICLR 2024 [bg venue]):
  - Once repair cost is counted, gains are often modest, vary across subsets, and are sometimes absent.
  - The bottleneck is the quality of the model's feedback on its own code. — [arXiv](https://arxiv.org/abs/2306.09896)
- Cho, Kang, Kim, Lee, "Self-Correcting Code Generation Using Small Language Models" (CoCoS; EMNLP 2025 Findings; arXiv 2505.23060):
  - 1B-scale models struggle to revise their own code, whether self-correction is prompted or trained.
  - Online RL with an accumulated multi-turn reward plus a fine-grained reward fixes this.
  - Gains over baselines are +35.8% on MBPP and +27.7% on HumanEval (abstract checked). — [arXiv](https://arxiv.org/abs/2505.23060)
- On-device measurement infrastructure:
  - llm.npu (ASPLOS 2025) uses 35.63–59.52× less energy than llama.cpp on CPU for 1024-token prompts. — [PDF](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf)
  - "Understanding Large Language Models in Your Pockets: Performance Study on COTS Mobile Devices" (arXiv 2410.03613). — [arXiv](https://arxiv.org/html/2410.03613v2)
  - "LLM Inference at the Edge" (arXiv 2603.23640, 2026) reports an average 146±11 mJ per token on a Galaxy S24 Ultra under sustained load. — [arXiv](https://arxiv.org/html/2603.23640v2)
- Adjacent: a preprint on self-healing JSON in small LMs finds that 45.6% of valid self-healed responses still contain at least one wrong field, and that the best repair strategy depends on the model. — [Research Square](https://www.researchsquare.com/article/rs-11641294/v1)
- Searches combining "on-device / smartphone + code generation + local execution + self-repair" returned no matching research paper. This is noted as absence in search.

### Inferences
- **Framing:** "Cost-aware self-repair on a phone".
  - Metrics: problems solved per joule and per second, under thermal steady state.
  - Compare repair against simply resampling at equal energy, as in Olausson et al.
  - Use the problem's sample I/O plus self-generated tests.
  - Make repair verdict-aware (see Idea 12).
  - CoCoS-style RL is likely needed for a 1.5B model to benefit at all.
- **Feasibility: medium.** The engineering work is an embedded Python sandbox with time and memory limits on Android or iOS.
- **Impact: medium.** It is a strong demo and fits a systems or demo track, but it is incremental as ML.

### Gaps
- I found no direct prior work.
- I did not verify which mobile Python runtimes are suitable.

---

## Idea 8: hypernetwork-generated LoRA adapters per algorithm topic, produced by a large model for a small coder

### Takeaway
**Novelty: low.** Three 2025 papers already generate LoRA adapters for small or edge models from a text description, and one of them covers coding:
- Text-to-LoRA (ICML 2025);
- Drag-and-Drop LLMs (NeurIPS 2025; includes coding with Qwen2.5-1.5B);
- LoRA-Gen (ICML 2025; a cloud model generates LoRA for an edge model).

MoEGen (2026) makes the generated adapters instance-adaptive. A per-topic CP version would be incremental and needs a lot of data.

### Cited Findings
- Charakorn, Cetin, Tang, Lange, "Text-to-LoRA: Instant Transformer Adaption" (ICML 2025; arXiv 2506.06105):
  - A hypernetwork turns a task description into a LoRA adapter in one forward pass.
  - Trained on 9 pre-trained LoRAs (GSM8K, Arc and others), it matches the task-specific adapters and generalises to unseen tasks.
  - Training with the SFT objective generalises better than reconstruction. — [arXiv](https://arxiv.org/abs/2506.06105); [Sakana AI](https://sakana.ai/text-to-lora/)
- Liang et al., "Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights" (NeurIPS 2025; arXiv 2506.16406):
  - Maps prompts to LoRA weights with up to 12,000× lower overhead than full fine-tuning.
  - Up to 30% average gains over the strongest training LoRAs on unseen tasks, including coding. It uses Qwen2.5-1.5B for math and coding and reports LiveCodeBench at 7B.
  - It fails to generalise to new datasets when training samples are few. — [arXiv](https://arxiv.org/abs/2506.16406); [NeurIPS](https://papers.neurips.cc/paper_files/paper/2025/hash/e15b6539e6d61aab5dee4593222ba40e-Abstract-Conference.html)
- Xiao et al., "LoRA-Gen: Specializing Large Language Model via Online LoRA Generation" (ICML 2025; arXiv 2506.11638):
  - A cloud-side LLM produces meta-tokens from the system prompt, which yield LoRA parameters merged into the edge-side model by reparameterisation.
  - With TinyLLaMA-1.1B it beats vanilla LoRA fine-tuning with a 2.1× speedup.
  - With Gemma-2B on agent tasks it reaches a 10.1× compression ratio.
  - One secondary account says it mixes a pool of pre-trained LoRA experts rather than producing raw matrices. — [arXiv](https://arxiv.org/abs/2506.11638); [PMLR](https://proceedings.mlr.press/v267/xiao25e.html)
- Zeng et al., "MoEGen: Mixture-of-Experts for Instance-Adaptive LoRA Generation" (arXiv 2608.03275, Aug 2026):
  - Routes each input over small learned "expert codes" that condition a hypernetwork, which generates input-specific low-rank updates.
  - Tested on commonsense, medical and legal tasks, not code (abstract checked). — [arXiv](https://arxiv.org/abs/2608.03275)
- "Semantic-guided LoRA Parameters Generation" (arXiv 2509.10535), title and listing only. — [listing](https://arxiv.deeppaper.ai/papers/2509.10535v1)

### Inferences
- **Novelty: low.** "A large model generates adapters for an edge model" is LoRA-Gen, and "a prompt generates coding adapters for a 1.5B model" is close to DnD.
- **Feasibility: low to medium.** A hypernetwork needs many trained topic adapters as training data, and DnD degrades with few samples.
- **Dependency:** first show that *static* topic LoRAs beat a single LoRA (Idea 1). If they do not, generating them is moot. Not recommended.

### Gaps
- I did not check DnD's exact coding numbers.

---

## Idea 9: RL with budget-conditioned length rewards, where the budget comes from the router's predicted difficulty

### Takeaway
**Novelty: low to medium. The area is crowded.**
- L1/LCPO trains **exactly DeepSeek-R1-Distill-Qwen-1.5B** to obey a length budget given in the prompt.
- Elastic Reasoning covers code (E1-Code-14B, LiveCodeBench).
- TALE trains a separate budget-regression estimator.
- Learning How Hard to Think uses external difficulty probes and includes code.
- DART sets budgets without training, and AdaCtrl and SelfBudgeter let the model estimate its own budget.

The only remaining difference is "a budget from an external tiny router, for CP code at 1.5B".

### Cited Findings
- Aggarwal & Welleck, "L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning" (arXiv 2503.04697):
  - LCPO trains DeepSeek-R1-Distill-Qwen-1.5B into L1-Max and L1-Exact, which follow a length constraint in the prompt.
  - The authors claim it matches GPT-4o at equal reasoning lengths. — [arXiv](https://arxiv.org/abs/2503.04697)
- LEASH (arXiv 2512.21540; listed in the ACL 2026 Anthology) treats length control as constrained optimisation with a Lagrangian penalty. On R1-Distill-Qwen-1.5B it cuts average length by 62.7% and raises accuracy by 0.8 points. — [arXiv](https://arxiv.org/html/2512.21540v1); [ACL Anthology](https://aclanthology.org/2026.acl-long.129.pdf)
- HAPO (AAAI) trains the same base model for correctness and efficiency: 49% shorter outputs with about a 2% accuracy drop. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/40373/44334)
- "Scalable Chain of Thoughts via Elastic Reasoning" (arXiv 2505.05315):
  - Splits the budget between thinking and solution, and adds a budget-constrained rollout to GRPO.
  - Releases E1-Math-1.5B and E1-Code-14B.
  - Uses 37.4% fewer tokens on LiveCodeBench even without budget constraints at inference. — [arXiv](https://arxiv.org/abs/2505.05315)
- Related methods: ShorterBetter (1.5B) — [arXiv 2504.21370](https://arxiv.org/abs/2504.21370); AdaCtrl — [arXiv 2505.18822](https://arxiv.org/abs/2505.18822); SelfBudgeter — [arXiv 2505.11274](https://arxiv.org/html/2505.11274v6); TALE — [arXiv 2412.18547](https://arxiv.org/abs/2412.18547); Learning How Hard to Think — [arXiv 2410.04707](https://arxiv.org/abs/2410.04707); DART — [arXiv 2606.23181](https://arxiv.org/abs/2606.23181)

### Inferences
- **Novelty: low to medium.** The pieces are an L1-style policy plus a TALE-style external estimator, applied to CP code.
- **Best use:** an ablation inside the router paper. Compare a router-predicted budget, a budget the model predicts for itself, a fixed budget, and DART-style draft agreement, using L1 checkpoints or LCPO as the controllable policy.
- **Possible small novelty:** per-topic budget priors (e.g., greedy versus DP) learned from the router's tags.

### Gaps
- I did not open the L1, HAPO or LEASH papers in full; the numbers come from search snippets.

---

## Idea 10: speculative decoding with a tiny drafter specialised for Python CP code, or n-gram lookup from retrieved templates

### Takeaway
**Novelty: low to medium.**
- Retrieval-based drafting from a code datastore exists (REST), and its compacted successor works especially well on code (CREST).
- Speculative decoding on phones is an active 2025–2026 systems topic (Lever, AHASD, sd.npu).
- "Retrieved templates double as drafts" is a small twist. It is a good engineering add-on for the phone story, not a headline.

### Cited Findings
- He, Zhong, Cai, Lee, He, "REST: Retrieval-Based Speculative Decoding" (arXiv 2311.08252; NAACL 2024 [bg venue]):
  - Training-free. Drafts come from a datastore of 2.7M Python samples (27 GB) from The Stack, used to accelerate CodeLlama 7B and 13B.
  - Speedups of 1.62–2.36×.
  - Retrieval uses a suffix array on CPU, matching the longest n-gram up to 16 tokens. — [arXiv](https://arxiv.org/abs/2311.08252); [project](https://sites.google.com/view/rest-llm/)
- CREST (arXiv 2408.04678) compacts the REST datastore to the smallest and most common n-grams, held in a hash map. The advantage over REST is larger on code generation. — [arXiv](https://arxiv.org/abs/2408.04678)
- DOUBLE (arXiv 2601.05524, 2026) uses retrieval on both the draft side and the target side. — [arXiv](https://www.arxiv.org/pdf/2601.05524)
- On-device speculative decoding:
  - Lever (arXiv 2605.16786) keeps the drafter in DRAM and the target model in flash. Latency is 2.93× lower than flash-offloaded inference and 1.50× lower than conventional speculative decoding. — [arXiv](https://arxiv.org/html/2605.16786v1)
  - AHASD (arXiv 2604.25326) targets mobile NPU-PIM hardware. — [arXiv](https://arxiv.org/html/2604.25326v3)
  - sd.npu (arXiv 2510.15312) finds that decoding takes 73.5% of on-device context-augmented inference time. — [arXiv](https://arxiv.org/abs/2510.15312v2)

### Inferences
- With a 1.5B target, a separate drafter has to be tiny to pay off on a phone, whereas n-gram lookup from the retrieved template or problem text is almost free and lossless.
- **Framing:** "retrieval does double duty". The retrieved algorithm template both conditions generation and supplies draft tokens. Report acceptance rate and on-phone wall-clock time.
- Low risk, but it only improves latency, not Pass@1.

### Gaps
- I did not find acceptance-rate data for CP Python specifically.

---

## Idea 11: a curriculum ordered by algorithm topic and difficulty rating rather than by token length

### Takeaway
**Novelty: low.** Existing work already covers:
- curriculum learning for small code LMs (ACL 2024 SRW);
- mixed results for "pedagogical" curricula on CodeT5;
- a bandit curriculum over problem categories (Self-Evolving Curriculum, COLM 2026);
- difficulty-aware curation for RL on competitive code (DRIVE, 2025).

Use it as an ablation, not a headline.

### Cited Findings
- Naïr et al., "Curriculum Learning for Small Code Language Models" (ACL 2024 SRW; arXiv 2407.10194):
  - A code-difficulty metric that combines cyclomatic complexity and Halstead difficulty.
  - Trained on a code-execution task, a hybrid curriculum reached 79.23% accuracy versus 74.58% for the baseline. — [ACL Anthology](https://aclanthology.org/2024.acl-srw.44.pdf); [alphaXiv](https://www.alphaxiv.org/abs/2407.10194)
- "Should Code Models Learn Pedagogically? A Preliminary Evaluation of Curriculum Learning for Real-World Software Engineering Tasks" (arXiv 2502.03806) found that a schedule based on conventional code difficulty had little effect on CodeT5. — [arXiv](https://arxiv.org/html/2502.03806v1)
- Chen et al., "Self-Evolving Curriculum for LLM Reasoning" (arXiv 2505.14970; COLM 2026 per Microsoft Research):
  - Treats each problem category, such as a difficulty level or problem type, as an arm of a non-stationary bandit.
  - Uses the absolute policy-gradient advantage as a proxy for learning gain, updated with TD(0).
  - Reports gains on planning, inductive reasoning and math. — [arXiv](https://arxiv.org/abs/2505.14970); [MSR](https://www.microsoft.com/en-us/research/publication/self-evolving-curriculum-for-llm-reasoning/)
- DRIVE (arXiv 2511.06307) argues that most RL-for-code methods treat all difficulties the same. It shows that curriculum design and data curation give substantial gains for a 32B competitive-code model. — [arXiv](https://arxiv.org/pdf/2511.06307)
- Related titles, not opened: SpecCoder (curriculum dual-task RL) — [arXiv 2609.06041](https://arxiv.org/pdf/2609.06041); "LLMs as Judges of difficulty in programming" — [arXiv 2511.18597](https://arxiv.org/pdf/2511.18597)

### Inferences
- **Novelty: low.** If used, run a curriculum over the router's own topic labels and Codeforces ratings (an SEC-style bandit) during blueprint SFT or RL, and report it as an ablation.
- Sorting by length (the Stage 1 practice) is itself a weak baseline, given evidence that long traces hurt small students.

### Gaps
- I found no SFT curriculum ordered by Codeforces rating at ≤2B.

---

## Idea 12: failure-type-targeted data synthesis (from wrong-answer, runtime-error and time-limit analysis) and learning-from-mistakes / self-correction training for code

### Takeaway
**Novelty: medium.** The following already exist:
- training models to self-debug (LeDex, NeurIPS 2024);
- self-correction RL for small models (CoCoS, EMNLP 2025 Findings);
- RL from execution feedback (RLEF [bg]);
- synthesising training data per error type, but only in math (Self-Error-Instruct).

**Synthesis targeted by judge verdict (WA/TLE/RE) for ≤2B CP models looks open.** Evidence that Wrong Answer dominates failures and that time-limit errors are the hardest to repair supports this kind of targeting.

### Cited Findings
- Jiang et al., "Training LLMs to Better Self-Debug and Explain Code" (LeDex; NeurIPS 2024 [bg venue]) fine-tunes on data that pairs an explanation of incorrect code with a refinement, both generated from execution feedback. — [Amazon Science PDF](https://assets.amazon.science/46/bf/3743cf75474290526f1147d9373f/training-llms-to-better-self-debug-and-explain-code.pdf)
- CoCoS (EMNLP 2025 Findings): online RL with accumulated and fine-grained rewards lets 1B-scale models self-correct code (see Idea 7). — [arXiv 2505.23060](https://arxiv.org/abs/2505.23060)
- "Unlocking LLM Code Correction with Iterative Feedback Loops" (arXiv 2606.17514, 2026; seen via alphaXiv):
  - Splits errors into compile, runtime, wrong answer and time limit exceeded.
  - Wrong answers are fixed about 35% of the time and TLE only about 20%, the hardest type, because it usually reflects a time-complexity flaw.
  - A WA signal gives no localisation, so the model must re-think its whole approach. — [alphaXiv](https://www.alphaxiv.org/abs/2606.17514)
- "A Systematic Study of Time Limit Exceeded Errors in Online Programming Assignments" (arXiv 2510.14339) notes that TLE gives only a one-line message. It proposes sandbox-verified patches that fix the timing while keeping the output correct. — [arXiv](https://arxiv.org/html/2510.14339)
- "Generalizing from Errors for LLMs Mathematical Reasoning" (Self-Error-Instruct; arXiv 2505.22591) groups errors by type and synthesises new training data for each type. It is math-only. — [arXiv](https://arxiv.org/pdf/2505.22591)
- A 2025 Nemotron-based paper (arXiv 2505.17040) builds a targeted code-repair dataset from an "error report" plus correct open-source snippets: 1,406 samples after filtering (snippet-level detail). — [arXiv](https://arxiv.org/pdf/2505.17040)
- Failure taxonomy (arXiv 2606.05228): Wrong Answer accounts for over 90% of GPT-4o's failed CP solutions. — [arXiv](https://arxiv.org/abs/2606.05228)
- RLEF, "Grounding Code LLMs in Execution Feedback with RL" (Gehring et al.) [bg]. — [arXiv 2410.02089](https://arxiv.org/abs/2410.02089)

### Inferences
- **Novelty: medium.** Learning from mistakes is not new; the open parts are verdict-specific synthesis for small CP coders and linking it to the topic router. Suggested responses per verdict:
  - TLE → rewrite to lower complexity, guided by a blueprint;
  - WA → find a counterexample and update [CORNER CASES];
  - RE → fix recursion limits, bounds and input parsing, which are common Python CP pitfalls.

  Statistics of failure type × topic, from the router's tags, decide what data to synthesise (weakness-driven).
- **Feasibility: high.** The team already has an execution harness.
- **Impact: medium to high.** It targets the dominant failure mode, and the same machinery powers the on-device repair loop (Idea 7).
- **Framing:** "verdict-aware repair curriculum for small coders". Do SFT on targeted pairs, then CoCoS-style multi-turn RL. Evaluate repair versus resampling at equal compute.

### Gaps
- I could not open the full text of arXiv 2606.17514 (alphaXiv page only).
- The RLEF details are from background knowledge.

---

## Are there survey papers mapping this space (efficient reasoning, small LMs for code, small-LM agents) that would help position the paper?

### Takeaway
There are several 2025 surveys on efficient reasoning, surveys on code agents and code reasoning, and one survey of small-LM agents. **I found no survey specifically on small LMs for code or competitive programming.** That gap is itself a positioning opportunity: the paper sits where efficient reasoning (shorter), small models (smaller) and CP meet.

### Cited Findings
- **Efficient reasoning:**
  - "Don't Overthink It: A Survey of Efficient R1-style Large Reasoning Models" (arXiv 2508.02120, Aug 2025) splits the field into single-model efficiency and multi-model collaboration, and keeps a public GitHub tracker. — [arXiv](https://arxiv.org/abs/2508.02120)
  - "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models" (Sui et al., 2025) separates efficient reasoning (shorter outputs) from model compression. — [arXiv 2503.16419](https://arxiv.org/abs/2503.16419) [ID bg]
  - "Efficient Reasoning Models: A Survey" (Feng et al., TMLR 2025) groups methods as shorter, smaller and faster. — [arXiv 2504.10903](https://arxiv.org/abs/2504.10903) [ID bg]
  - "A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond" (arXiv 2503.21614). — [arXiv](https://arxiv.org/abs/2503.21614); [review](https://www.themoonlight.io/review/a-survey-of-efficient-reasoning-for-large-reasoning-models-language-multimodality-and-beyond)
- **Small LMs and agents:**
  - Sharma & Mehta, "Small Language Models for Agentic Systems: A Survey of Architectures, Capabilities, and Deployment Trade-offs" (arXiv 2510.03847), covering 1–12B models, function calling, structured outputs and LoRA/QLoRA. — [arXiv](https://arxiv.org/abs/2510.03847)
  - NVIDIA's position paper "Small Language Models are the Future of Agentic AI" [bg]. — [arXiv 2506.02153](https://arxiv.org/abs/2506.02153)
- **Code agents and code reasoning:**
  - "A Survey on Code Generation with LLM-based Agents" (Dong et al., arXiv 2508.00083). — [arXiv](https://arxiv.org/abs/2508.00083)
  - "AI Agentic Programming: A Survey of Techniques, Challenges, and Opportunities" (arXiv 2508.11126). — [arXiv](https://arxiv.org/html/2508.11126v1)
  - "From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence" (arXiv 2511.18538). — [arXiv](https://arxiv.org/html/2511.18538v1)
  - Ramji & Ray, a 2025 survey on code reasoning that calls itself the first dedicated one (arXiv 2506.13932 per search listing). — [arXiv](https://arxiv.org/html/2506.13932v1)
- **CP context (benchmarks and reports, not surveys):**
  - OpenAI, "Competitive Programming with Large Reasoning Models". — [arXiv 2502.06807](https://arxiv.org/abs/2502.06807)
  - LiveCodeBench Pro. — [arXiv 2506.11928](https://arxiv.org/pdf/2506.11928)
  - OJBench (232 NOI/ICPC problems; reasoning models outperform). — [arXiv 2506.16395](https://arxiv.org/html/2506.16395v2)
  - LiveOIBench. — [arXiv 2510.09595](https://arxiv.org/pdf/2510.09595v1)
  - The 2026 failure taxonomy by algorithm and difficulty. — [arXiv 2606.05228](https://arxiv.org/abs/2606.05228)

### Inferences
- **Positioning:** use the "Efficient Reasoning Models" shorter/smaller/faster split.
  - Blueprints = shorter.
  - A 1.5B model on a phone = smaller.
  - Speculative or n-gram decoding = faster.
  - A router plus coder is "multi-model collaboration" in the "Don't Overthink It" taxonomy.
- State clearly that no survey covers small CP coders and that the paper fills an empirical gap at ≤2B.

### Gaps
- I did not open the full texts of the surveys.
- The arXiv IDs for "Stop Overthinking" and "Efficient Reasoning Models" are from background knowledge; check them.

---

## Which 2–3 ideas offer the best combination of novelty, feasibility and expected impact for a publishable paper, and why?

### Takeaway
1. **Ideas 3 + 1, merged (headline):** an *oracle-gap study* of algorithm-hint conditioning for ≤2B coders, closed by a *tiny router*. It asks how much of the failure is algorithm selection and how much is implementation.
2. **Idea 12 (+7):** verdict-aware repair data and training, used in an on-device, cost-aware test-repair loop.
3. **Idea 2 as supporting evidence:** a controlled ablation of format learnability (structure versus length), largely already in hand.

Do not make Ideas 4, 5, 8 or 9 the headline: each is crowded or high-risk. Use Ideas 10 and 11 as engineering add-ons or ablations.

### Cited Findings
**Why Ideas 3 + 1**
- Hints and editorials help large models a lot:
  - USACO: 13 of 15 previously unsolved problems were solved with a few hints. — [arXiv 2404.10952](https://arxiv.org/abs/2404.10952)
  - Explanations derived from oracle solutions raised solve@10 from 6.1% to 42.4%. — [ar5iv 2307.05337](https://ar5iv.labs.arxiv.org/html/2307.05337)
  - Gold editorials help far more than self-generated ones, but implementation still fails. — [arXiv 2601.11332](https://arxiv.org/abs/2601.11332)
- Tags have only been used for diversity (AlphaCode) or with ChatGPT-scale models (KareCoder), and are argued to be too coarse (AlgoSimBench). — [AlphaCode](https://arxiv.org/abs/2203.07814); [KareCoder](https://arxiv.org/abs/2401.15940); [AlgoSimBench](https://arxiv.org/pdf/2507.15378)

**Why Idea 12**
- Wrong Answer dominates CP failures (over 90% for GPT-4o). — [arXiv 2606.05228](https://arxiv.org/abs/2606.05228)
- TLE and WA are the hardest verdicts to repair (about 20% and 35% fix rates). — [alphaXiv 2606.17514](https://www.alphaxiv.org/abs/2606.17514)
- Small models need training to self-correct. — [CoCoS](https://arxiv.org/abs/2505.23060)
- Self-repair must be judged with its cost counted. — [Olausson et al.](https://arxiv.org/abs/2306.09896)

**Why Idea 2 (supporting only)**
- The learnability gap is shown for math, not CP code, at ≤3B. — [arXiv 2502.12143](https://arxiv.org/abs/2502.12143)
- Structured plans are already proposed (CodePlan, PGPO), and "blueprint" is already used as a name. — [CodePlan](https://arxiv.org/abs/2409.12452); [PGPO](https://arxiv.org/pdf/2506.01475); [Blueprint2Code](https://www.frontiersin.org/articles/10.3389/frai.2025.1660912)

**Why not the others as headlines**
- Idea 9: L1 uses the same 1.5B base model. — [arXiv 2503.04697](https://arxiv.org/abs/2503.04697)
- Idea 8: DnD already generates coding adapters for Qwen2.5-1.5B, and LoRA-Gen already generates adapters cloud-to-edge. — [DnD](https://arxiv.org/abs/2506.16406); [LoRA-Gen](https://arxiv.org/abs/2506.11638)
- Idea 4: LLM-JEPA already uses text↔code views. — [arXiv 2509.14252](https://arxiv.org/abs/2509.14252)
- Idea 5: TRM for code gains through regularisation at about 175× per-step compute. — [arXiv 2608.29376](https://arxiv.org/abs/2608.29376)

### Inferences
**Summary scorecard** (my judgement):

| # | Idea | Closest prior art | Novelty | Feasibility | Expected impact |
|---|---|---|---|---|---|
| 1 | Tiny router → topic LoRA + templates + budget | PSG/AMT; KareCoder; BoT/ReasonFlux; TALE, Learning How Hard to Think, DART; MapCoder-Lite; MoR-LoRA | Medium | High | Medium–High (with oracle analysis) |
| 2 | Blueprint distillation target | Learnability gap; CodePlan; PGPO; MDPI pseudocode distillation; Blueprint2Code; SCoT | Low–Med | High (largely done) | Medium |
| 3 | Predicted vs oracle tags/hints | AlphaCode; KareCoder; TEACH; USACO; Idea First, Code Later; Explaining-CP | Medium (≤2B ladder) | High | High |
| 4 | JEPA auxiliary loss | LLM-JEPA (ICLR 2026) | Low–Med | Medium (3× compute) | Low–Med |
| 5 | TRM/HRM recursion for code or repair | TRM-AR for code (CASCON 2026); edge-compression study | Med–High | Low | Low (near term) |
| 6 | ≤2B planner + coder / multi-agent | MapCoder-Lite (7B); MapCoder/CodeSim; Blueprint2Code | Medium | Medium | Medium |
| 7 | On-device test-repair loop | Self-Debugging; Olausson; CoCoS; on-device inference studies | Medium (systems) | Medium | Medium |
| 8 | Hypernetwork topic LoRA | Text-to-LoRA; DnD; LoRA-Gen; MoEGen | Low | Low–Med | Low–Med |
| 9 | Router-budgeted length-reward RL | L1/LCPO (same base); Elastic Reasoning; TALE; Learning How Hard to Think; DART; AdaCtrl | Low–Med | Medium | Medium |
| 10 | Spec decoding / template n-gram drafts | REST; CREST; Lever; sd.npu | Low–Med | Medium | Low–Med (latency only) |
| 11 | Topic/difficulty curriculum | Curriculum for small code LMs; SEC; DRIVE | Low | High | Low–Med |
| 12 | Verdict-targeted synthesis / self-correction | LeDex; CoCoS; Self-Error-Instruct; TLE study | Medium | High | Medium–High |

**A suggested paper built from the top three ideas**

Working title: "What does a 1.5B coder actually need? Algorithm selection, format and repair on a phone".
- **Contribution 1, diagnosis:** a hint-granularity ladder (none / predicted tag / oracle tag / oracle technique / oracle blueprint / gold editorial) crossed with difficulty tier. It shows how much headroom routing could unlock.
- **Contribution 2, the router:** a ≤100M router that predicts tags, difficulty and a budget. Report:
  - router accuracy against the share of the oracle gap it recovers;
  - hard topic routing versus one generalist LoRA at matched parameters;
  - abstention and fallback when the router is unsure;
  - on-device memory and latency.
- **Contribution 3, format:** a controlled comparison of structure versus length (blueprint versus token-matched long or short CoT, with field ablations) explains why Stage 1 collapses. Contrast it with Blueprint2Code, SCoT and CodePlan.
- **Contribution 4 (optional, if time):** verdict-aware repair training (WA, TLE and RE-specific synthesis plus CoCoS-style RL), evaluated on-device as solved problems per joule, against resampling at equal energy.

**Evaluation hygiene that reviewers will check**
- The current numbers come from small samples:
  - 11% on 100 problems has a Wilson 95% interval of about [6.3%, 18.6%].
  - 8% on 200 problems has an interval of about [5.0%, 12.6%].
  - Because the problem sets also differ, "8% → 11%" is not yet evidence of improvement. This is my calculation, not a cited figure.
- Use one fixed, larger test set, preferably time-split (e.g., LiveCodeBench or Codeforces problems after the model's training cutoff), and average pass@1 over several samples.
- Check overlap between the training sources and the test set.

**Baselines to include**
- The base DeepSeek-R1-Distill-Qwen-1.5B, with and without budget forcing.
- L1 checkpoints (same base). — [arXiv 2503.04697](https://arxiv.org/abs/2503.04697)
- ZR1-1.5B. — [Zyphra](https://zyphra.com/post/introducing-zr1-1-5b-a-small-but-powerful-math-code-reasoning-model)
- A MapCoder-Lite-style role-LoRA set-up scaled down to 1.5B. — [arXiv 2509.17489](https://arxiv.org/abs/2509.17489)
- A ReasonFlux/BoT-style template-retrieval baseline. — [arXiv 2502.06772](https://arxiv.org/abs/2502.06772)

### Gaps
- My novelty judgements rest on about 48 targeted searches and fetches, not a systematic review. Several 2026 preprints (DART, MoR-LoRA, the TRM code paper, the failure taxonomy) may be revised.
- Absence of evidence (e.g., no ≤2B oracle-tag study, no on-device CP repair study, no topic-LoRA CP study) is not proof of absence. Run a final Semantic Scholar / Google Scholar check with the exact framing before submission.
- I did not check strong 1.5B CP RL baselines beyond ZR1-1.5B, or CP-specific retrieval benchmarks that would be stronger M3 baselines than MiniLM top-1 retrieval.
