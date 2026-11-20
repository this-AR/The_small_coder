# Topic/difficulty routers, adaptive thinking budgets, concise-reasoning training, adapter routing and retrieval for small competitive-programming code models

*Compiled 2026-11-19/20. Numbers are as each source reports them. Results for models of 7B or smaller are prioritised. "(secondary)" marks figures taken from a search summary or a third-party review rather than read in the primary text. The session's web-search budget ran out partway through; the items that could not be checked are listed under Gaps.*

## 1. Predicting algorithm tags and difficulty from problem statements, and how much correct tags or hints raise solve rates

### Takeaway
Predicting tags from the statement alone works, but the output is noisy. Macro-F1 is about 0.41–0.43 on the 20 most frequent Codeforces tags, and TF-IDF, gradient boosting and BigBird all land in that range. On 8 coarse tags it reaches about 0.69–0.75. Predicting the exact difficulty is much harder: about 11% exact accuracy over 28 rating classes, with a mean error of about 4.5 classes (roughly 450 rating points). However, a linear probe on a large solver's own hidden states recovers human difficulty well (Spearman ρ ≈ 0.81 on Codeforces). The most important result for router design: telling an LLM only the algorithm name gives almost no gain (GPT-3.5 on CodeContests, solve@10 stays at 6.1%). An oracle step-by-step plan for the specific problem raises solve@10 from 6.1% to 42.4%. So a router helps through what it fetches or plans, not through the tag itself.

### Cited Findings
#### Datasets and tag-prediction accuracy
- AMT dataset: 7,976 Codeforces problems that have tags. Labels are the 20 most frequent tags, led by implementation, math and greedy. Difficulty is the Codeforces rating, 28 levels from 800 to 3500. The PSG model is a BigBird encoder with a multi-label tag head and an ordinal difficulty head — [AMT/PSG paper](https://arxiv.org/html/2310.05791)
- AMT test results (AUROC / macro-F1): single-task BigBird 80.70 / 42.78; SVM on bag-of-words + TF-IDF 79.26 / 40.33; CNN ensemble 58.12 / 20.14; XGBoost 73.47 / 43.09; LightGBM 74.52 / 42.94; multi-task PSG (λ=10) 79.12 / 41.08. The multi-task model uses about half the parameters of two single-task models. The paper has no BERT, CodeBERT or LLM baselines — [AMT/PSG paper](https://arxiv.org/html/2310.05791)
- AMT difficulty results: single-task BigBird gets 11.24% exact accuracy, 23.71% within ±3 classes, 34.54% within ±5 classes, and MAE 4.55 classes. PSG (λ=10) gets 10.10% exact and MAE 4.79. The column mapping in the extracted table was approximate — [AMT/PSG paper](https://arxiv.org/html/2310.05791)
- Hobbyist, unreviewed benchmark: 4,982 Codeforces problems, 8 coarse tags (math, graphs, strings, number theory, trees, geometry, games, probabilities), 994 test problems. Test macro-F1:
  - LLM yes/no-question pipeline "Jev": 0.7515 with statement and code, 0.7367 with the statement only
  - zero-shot API LLM: 0.7407 (the repo names it "gpt-6-luna"; that name cannot be verified)
  - frozen F2LLM-0.6B embeddings + ridge probe: 0.7415
  - ModernBERT-base, full fine-tune: 0.7039
  - ModernBERT-large, LoRA: 0.6954
  - TF-IDF + linear SVM: 0.6888
  - label prior: 0.1487
  - The 95% confidence intervals are about ±0.03, and the top five methods are statistically tied — [codeforces-tag-prediction repo](https://github.com/msobroza/codeforces-tag-prediction)
- Same repo: fine-tuning ModernBERT gave no dev-set gain over TF-IDF "despite hours of GPU tuning". A fixed 0.5 threshold costs 0.039 macro-F1 compared with per-tag thresholds. Adding solution code changed the best method's dev score by 0.000. Per-tag test F1 ranges across methods: math 0.62–0.67, graphs 0.56–0.72, number theory 0.52–0.70, trees 0.73–0.79, games 0.88–0.93. The author flags possible contamination — [repo](https://github.com/msobroza/codeforces-tag-prediction)
- One study of 2,400 problems with single-category tagging reached at most 78.0% accuracy, using an MLP (secondary) — [arXiv 2308.01863](https://arxiv.org/pdf/2308.01863)
- Another study reports that an ensemble using the statement plus submitted solutions beats text-only models (secondary) — [arXiv 2301.04597](https://arxiv.org/html/2301.04597v1). This conflicts with the hobby repo's finding that code added nothing.
- TACO: 26,443 problems (25,443 train / 1,000 test). Each problem has `raw_tags`, `tags` ("manually annotated algorithms needed to solve the problem") and `skill_types`, plus 5 difficulty levels from EASY to VERY_HARD — [TACO dataset card](https://huggingface.co/datasets/BAAI/TACO). The original raw labels number 968, most of them rare. A figure of "36 algorithm topics" appears only in a third-party summary. GPT-4 reaches only 31.5 pass@1 on EASY problems, and other code models are mostly below 10 at every level — [alphaXiv overview](https://www.alphaxiv.org/overview/2312.14852)
- CodeElo does not show tags to the models (or to humans during contests), so tags are never used as hints there. Top models exceed 20% pass@1 on math, implementation and sorting problems, but are often below 5% on DP and trees — [CodeElo](https://arxiv.org/pdf/2501.01257); [EmergentMind summary](https://www.emergentmind.com/topics/codeelo)

#### Difficulty prediction
- A Codeforces problem's rating is defined as the rating of a hypothetical contestant whose expected rank equals the number of people who solved it — [Codeforces blog](https://codeforces.com/blog/entry/62865)
- A hobby predictor that uses accepted-solution metadata (language, runtime, memory) reports an average error of about 450 rating points (secondary; matched to this blog entry from the search summary) — [Codeforces blog](https://codeforces.com/blog/entry/102457)
- "LLMs Encode How Difficult Problems Are" trained linear probes on about 60 models (0.5B–72B, 18 families) using Easy2Hard-Bench:
  - E2H-Codeforces, human-labelled difficulty: Spearman ρ = 0.8115 (Llama-3.1-70B-Instruct, layer 42, final token)
  - AMC: ρ = 0.8842
  - Labels derived from LLM performance are much weaker (GSM8K ρ = 0.5799)
  - The best probes sit in middle layers; the scaling exponent for Codeforces is α = 0.066
  - The paper does not test routing or compute allocation — [arXiv 2510.18147](https://arxiv.org/html/2510.18147v1)
- During GRPO on Qwen2.5-Math-1.5B, the human-difficulty probe gets stronger and tracks test accuracy. Steering activations toward "easy" improved MATH500 accuracy and shortened outputs — [arXiv 2510.18147](https://arxiv.org/abs/2510.18147)

#### Effect of tags and hints on solve rates
- "Explaining Competitive-Level Programming Solutions" setup: GPT-3.5 solves the 165 CodeContests test problems, given hints written by an LLM that saw the gold solution. Results (solve@1 / @5 / @10):
  - no hint: 1.8 / 3.6 / 6.1
  - **"used algorithm" hint: 1.8 / 4.2 / 6.1**
  - time-complexity hint: 1.8 / 3.6 / 6.7
  - one-sentence summary: 4.2 / 9.1 / 13.9
  - explanation of the solution: 6.1 / 17.6 / 23.6
  - **step-by-step description: 13.3 / 32.2 / 42.4**
  - self-generated "general-to-specific" prompt (no oracle): 2.4 / 5.4 / 9.1 — [ar5iv 2307.05337](https://ar5iv.labs.arxiv.org/html/2307.05337)
- With hints written by GPT-4, solve@1 was: step-by-step 15.8%, explanation 4.8%, one-sentence 4.2%, time complexity 2.4%, **used algorithm 1.2% (below the 1.8% baseline)** — [same](https://ar5iv.labs.arxiv.org/html/2307.05337)
- Gains shrink as the rating rises. On easy problems a one-sentence hint lets GPT-3.5 solve about 70% versus about 30% without it, but "no explanation can help solve complex problems". Hint diversity matters: one-sentence hints drawn from 10 different human solutions gave solve@10 of 13.9%, against 12.1% for 10 explanations of one solution and 6.7% for 10 sampled programs — [same](https://ar5iv.labs.arxiv.org/html/2307.05337)
- Platform tags are "usually too general (e.g., dynamic programming)" — [UT Austin thesis proposal](https://www.cs.utexas.edu/~ml/papers/li.proposal25.pdf)
- USACO human tutoring: on 15 problems that no automatic method had solved, GPT-4 given minimal human hints solved 13 (86.7%). GPT-3.5 solved 0 — [USACO-bench](https://arxiv.org/html/2404.10952v1)
- PlanSearch/IdeaSearch: accuracy when conditioned on a given idea sketch tends toward either 0% or 100%. Whether the sketch is correct explains most of the variance in passing — [PlanSearch v2](https://arxiv.org/html/2409.03733v2)
- Llama-3 multi-turn study. Its reasoning prompts included "predict problem tag", but the tag prompt's effect was not isolated in the text I read. "Solution"-type instructions (naive or weak solution first) worked best. For 8B models, the best prompt combinations barely moved pass@1 (Llama 3.1 8B 3.8 → 4.0; Llama 3.0 8B 1.6 → 1.5) but raised pass@100 (22.8 → 26.1 and 12.3 → 17.3) — [arXiv 2410.08105](https://arxiv.org/html/2410.08105)
- "Where do LLMs fail" (June 2026): 315 Codeforces problems, with tags used only to categorise. Zero-shot chain-of-thought lowered GPT-4o from 46.0% to 36.8% and moved Claude Sonnet 4.6 from 62.2% to 63.5%. On the hard tier (rating 1901–3500) they scored 9.5% and 28.6%. 92% of GPT-4o's failures were wrong answers — [arXiv 2606.05228](https://arxiv.org/html/2606.05228v1)

### Inferences
- A router that only predicts a tag and injects a generic blueprint for that tag matches the "used algorithm" condition, which gave zero gain. Value appears only with problem-specific content (a summary, explanation or steps), and it grows with specificity.
- Expect a statement-only tag router to be wrong often: macro-F1 is about 0.4 on 20 tags and about 0.7 on 8 coarse tags. Use calibrated top-k or soft routing with per-tag thresholds, plus a generalist fallback, rather than hard top-1 routing over 7 topics.
- The cheapest good tag model in the evidence is a frozen small embedder (about 0.6B) with a linear head. Fine-tuned ModernBERT gave no gain over TF-IDF there, and that result is from a single unreviewed repo.
- For difficulty or budget, a probe on the coder's own prefill hidden state needs no extra model. Strong correlation is shown only at 70B; whether it holds at 1.5B is untested (see Gaps). Training labels are available from TACO (5 levels), CodeContests and Codeforces ratings, and E2H-Codeforces.

### Gaps
- I found no study that isolates tag-only hints for models of 7B or smaller on competitive programming. AlphaCode's tag and rating metadata conditioning could not be checked because search ran out.
- I found no peer-reviewed numbers for CodeBERT or ModernBERT tag classifiers, and no LLM zero-shot tag-prediction accuracy from a reviewed source.
- Difficulty-probe accuracy for 1.5B models on Codeforces is not reported separately.
- The "36 TACO algorithm topics" figure is unverified.

## 2. Adaptive reasoning: whether and how long to think (accuracy vs. token savings, especially on code)

### Takeaway
At 0.6B–4B, thinking is worth roughly 2.5–3.4× on LiveCodeBench: Qwen3-1.7B scores 33.2 with thinking against 11.6 without. So "skip reasoning" can only be a route for easy problems; the main design knob is the length budget. The RL methods that learn when to think (Thinkless, AdaptThink, AutoThink) cut tokens by 50–90% at the 1.5B scale, but were shown only on math. The evidence on code comes from larger models: DEER early exit (about −65% length on programming tasks), Elastic Reasoning on a 14B model (−37.4% tokens at equal LiveCodeBench score, with graceful degradation under 1K-token budgets), and the Sonata prefill-time budget predictor (−20% to −60% thinking tokens, LiveCodeBench included). Qwen itself moved from one hybrid think/no-think model to separate models in its 2507 updates, which outperform the hybrid in both modes.

### Cited Findings
#### Does thinking matter for small models on code?
- Qwen3 report, LiveCodeBench v5, thinking vs. non-thinking: Qwen3-0.6B 12.3 vs. 3.6; Qwen3-1.7B 33.2 vs. 11.6; Qwen3-4B 54.2 vs. 21.3. Qwen3-4B's Codeforces Elo is 1671 (92.8th percentile) vs. 842 (33.7th). DeepSeek-R1-Distill-Qwen-1.5B, which thinks, scores 13.2 — [Qwen3 report](https://arxiv.org/html/2505.09388v1)
- Thinking budget: the report describes "scalable and smooth performance improvements correlated to the allocated thinking budget". Budget control works by halting thinking and inserting a stop-thinking instruction, an ability that emerged from Thinking Mode Fusion without explicit training — [Qwen3 report](https://arxiv.org/html/2505.09388v1)
- Fusing the two modes has a cost. Qwen3-32B's LiveCodeBench v5 score in thinking mode went 68.4 (after reasoning RL) → 67.2 (after mode fusion) → 65.7 (after general RL). AIME'24 went 83.8 → 81.9 → 81.4 — [Qwen3 report](https://arxiv.org/html/2505.09388v1)
- **Superseded: the hybrid design was replaced by separate models.** Qwen3-4B-Instruct-2507 "supports only non-thinking mode" and scores 35.1 on LiveCodeBench v6, against 26.4 for the hybrid Qwen3-4B in non-thinking mode — [Qwen3-4B-Instruct-2507 card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Qwen3-4B-Thinking-2507 scores 55.2 on LiveCodeBench v6 vs. 48.4 for the hybrid's thinking mode, with CFEval 1852 vs. 1671 and OJBench 17.9 vs. 16.1. It has "an increased thinking length", and the card recommends up to 81,920 output tokens for competition programming — [Qwen3-4B-Thinking-2507 card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- NoThinking (prompting DeepSeek-R1-Distill-Qwen with an empty thinking block): when token budgets are matched, it beats Thinking on 7 datasets, coding among them. On AMC 23 with a 700-token budget it scores 51.3 vs. 28.9. With parallel sampling plus best-of-N selection it matches Thinking at up to 9× lower latency — [arXiv 2504.09858](https://arxiv.org/abs/2504.09858)

#### Learning when to think (all evaluated on math with DeepSeek-R1-Distill-Qwen-1.5B)
- Thinkless: the model chooses a short or long control token first, trained with Decoupled GRPO. It cuts long-chain thinking by 50–90% on Minerva Algebra, MATH-500 and GSM8K. The abstract lists no code benchmark — [arXiv 2505.13379](https://arxiv.org/abs/2505.13379). Checkpoints are public — [Thinkless-1.5B-Warmup](https://huggingface.co/Vinnnf/Thinkless-1.5B-Warmup)
- AdaptThink: −53% average length and +2.4% accuracy across GSM8K, MATH500 and AIME24. Per dataset, length falls 50.9%, 63.5% and 44.7% and accuracy rises 4.1, 1.4 and 1.6 points. The 1.5B model chose NoThinking on 86.9% of GSM8K problems and 40.4% of AIME24 problems. EMNLP 2025 — [arXiv 2505.13417](https://arxiv.org/abs/2505.13417)
- AutoThink (NeurIPS 2025): an ellipsis prompt plus 3-stage RL. Version 1 reports +6.4% relative accuracy and −52% tokens on 5 math benchmarks. Later versions give no single headline figure — [v1](https://arxiv.org/html/2505.10832v1); [v3](https://arxiv.org/abs/2505.10832v3)
- DuP-PO: about 15% fewer tokens than plain GRPO on DeepSeek-R1-Distill-Qwen-1.5B (secondary) — [arXiv 2506.23840](https://arxiv.org/pdf/2506.23840)

#### Predicting and controlling the budget
- TALE: the zero-shot budget-estimation variant cuts tokens by 67% with less than 3% accuracy loss; the post-trained variant cuts about 50% — [arXiv 2412.18547](https://arxiv.org/pdf/2412.18547)
- SelfBudgeter: the model first writes a predicted budget in a `<budget>` block, then is trained with GRPO. The ACL 2026 Findings version reports 61% average compression on math at maintained accuracy. **Superseded figure:** v1 reported up to 74.47% on MATH — [v5](https://arxiv.org/html/2505.11274v5); [ACL 2026](https://aclanthology.org/2026.findings-acl.1063.pdf)
- s1 budget forcing: SFT on 1,000 samples (Qwen2.5-32B-Instruct). Forcing longer thinking extrapolates AIME24 from 50% to 57% — [arXiv 2501.19393](https://arxiv.org/abs/2501.19393). L1/LCPO beats s1 budget forcing by up to 100% relative and 20% absolute — [L1 project page](https://cmu-l3.github.io/l1)
- Sonata, "Adaptive Thinking" (ICLR 2026): a lightweight adapter reads last-layer hidden states during prefill, predicts self-consistency, and sets the thinking budget before reasoning starts. Overhead is under 1‰. It cuts thinking tokens by 20–60% at the same accuracy, or gains up to 2% accuracy at the same cost. Models: Qwen3-8B/32B, GPT-OSS-120B, Qwen3-235B-A22B. LiveCodeBench is among the benchmarks, but the abstract gives no per-benchmark numbers — [ICLR poster](https://iclr.cc/virtual/2026/poster/10011708)
- Elastic Reasoning (ICLR 2026): separate thinking and solution budgets, trained with 1K tokens for each.
  - E1-Code-14B, unconstrained: LiveCodeBench 58.4 vs. 58.1 for DeepCoder-14B, using 11,145 vs. 17,815 tokens (−37.4%). Codeforces rating 1987 (96.0th percentile) vs. 1945. HumanEval+ 91.4 vs. 90.8.
  - E1-Code-14B with a 1K solution budget and thinking budget t: t=1K gives LiveCodeBench 37.3%, t=2K 41.6%, t=3K 44.1%, t=4K 47.0%. DeepCoder-14B stays below 10% under 4K.
  - E1-Math-1.5B: AIME24 35.0 vs. 41.0 for DeepScaleR-1.5B, with −32.1% tokens. Not evaluated on code — [arXiv 2505.05315v2](https://arxiv.org/html/2505.05315v2)

#### Early exit and code-specific triggers
- DEER is training-free: it exits when a trial answer is confident. **Version note:** v1 reports −64.9% length and +2.1 pass@1 on programming benchmarks; v2 reports −19.1% to −80.1% length and +0.3 to +5.0 accuracy. Smaller models get larger reductions. On DeepSeek-R1-Distill-Qwen-14B, HumanEval and BigCodeBench lengths fall more than 60% (secondary) — [DEER](https://arxiv.org/pdf/2504.15895); [v2](https://arxiv.org/html/2504.15895v2)
- UnCert-CoT: at each new line of code, high uncertainty (entropy or probability differential) triggers chain-of-thought; otherwise the model writes code directly. Up to +6.1% PassRate on MHPP — [arXiv 2503.15341](https://arxiv.org/html/2503.15341v1)
- Think-Anywhere (March 2026): the model invokes reasoning at high-entropy token positions inside the code, tested on LeetCode, LiveCodeBench, HumanEval and MBPP. The abstract claims state of the art without numbers; a search summary reports a net token reduction — [arXiv 2603.29957](https://arxiv.org/abs/2603.29957)
- Reasoning Shift: extra context silently shortens reasoning. Qwen3.5-27B produced about 3.6× fewer reasoning tokens on LiveCodeBench in a multi-turn setting (secondary) — [arXiv 2604.01161](https://arxiv.org/pdf/2604.01161)

#### Model routers
- RouteLLM: on MT Bench it reaches 95% of GPT-4 quality with 26% of calls going to GPT-4 (about 48% cheaper than a random router), or 14% of calls with LLM-judge-augmented training data (75% cheaper) — [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/). Compared with always using GPT-4, cost falls over 85% on MT Bench, 45% on MMLU and 35% on GSM8K — [Berkeley project page](https://sky.cs.berkeley.edu/project/routellm/)
- Arch-Router: a 1.5B generative router that maps queries to user-defined domain–action policies. It beats top proprietary LLMs by 7.71% on average at matching human preferences, and new models can be added without retraining — [arXiv 2506.16655](https://arxiv.org/html/2506.16655v1)
- R2-Reasoner: a task decomposer plus a subtask allocator over 9 models ranging from under 1B to hundreds of billions of parameters, trained with SFT then GRPO. **Version note:** v1 reports −86.85% API cost on 4 benchmarks; v2 reports −84.46% on 6 benchmarks with +3.73% average accuracy — [arXiv 2506.05901v2](https://arxiv.org/abs/2506.05901v2)

### Inferences
- At phone scale, a no-think route throws away about two-thirds of competitive-programming solves (Qwen3-1.7B: 33.2 vs. 11.6). The router should mainly pick a budget (for example short, 2K, 4K or 8K tokens). It should choose no-think only when it is confident the problem is easy.
- Elastic-style training (separate thinking and solution budgets, trained at about 1K each) is the best-evidenced way to make a coder degrade gracefully under tight budgets. A Sonata-style prefill probe is the best-evidenced budget predictor, and costs under 1‰ overhead.
- For models of 2B or smaller, when-to-think RL has no published code evaluation. Applying Thinkless or AdaptThink to code would be a novel contribution, not a replication.
- DEER-style early exit needs no training and helps small models most, so it is a cheap baseline to try on device.
- Injecting retrieved context may silently shorten reasoning (Reasoning Shift), so measure response length with and without retrieval.

### Gaps
- No code results for Thinkless, AdaptThink, AutoThink, TALE or SelfBudgeter.
- No per-benchmark Sonata numbers on LiveCodeBench.
- No RouteLLM or Arch-Router routing accuracy on code.
- Qwen's stated reason for dropping hybrid thinking was not found; the model cards give none. The 2507 cards report LiveCodeBench v6, so their scores are not directly comparable with the report's v5 scores.

## 3. Training small models to reason concisely without losing accuracy (results for ~1.5B models on code)

### Takeaway
The team's Stage-1 failure matches a documented effect called Long CoT Degradation. Small models fine-tuned on only about 8k long chain-of-thought (CoT) examples lose up to 75% of their prior performance. Models of 3B or smaller learn better from short CoT, or from a long:short mix of about 1:4. Concise-RL methods at 1.5B (L1, ShorterBetter, Concise-RL, AdaptThink, Self-Braking) cut length by 40–80% at equal accuracy, but almost entirely on math. Direct code evidence: DeepCoder-1.5B, trained with RL from the same base model, raised LiveCodeBench from 16.9 to 25.1. ASAP's step-level pruning on a 7B model gave −23.5% tokens and +2.6 pass@1. For code, token-level pruning such as TokenSkip degrades results.

### Cited Findings
- "Small Models Struggle to Learn from Strong Reasoners" (Findings of ACL 2025): models of 3B or smaller do not consistently benefit from long CoT or from distilling a large teacher; small students do better with short CoT and smaller teachers. Mixing long and short CoT at 1:4 gives Qwen2.5-3B-Instruct more than 8 points on MATH and AMC over long CoT alone. Mixing large- and small-teacher data gives more than 7 points on MATH, AIME and AMC — [arXiv 2502.12143](https://arxiv.org/abs/2502.12143)
- "Through the Valley" (EMNLP 2025), on Long CoT Degradation:
  - With 8k long-CoT examples, models lose up to 75% of their original performance. This holds across Qwen2.5, LLaMA3 and Gemma3 from 0.5B to 14B.
  - The cause is length inflation leading to error accumulation.
  - More data recovers performance, faster for larger models, but some very small models fail to recover even with 220k examples.
  - The degradation can hurt downstream RL. Large-scale SFT (128k examples) leads to faster RL gains (secondary) — [arXiv 2506.07712v2](https://arxiv.org/abs/2506.07712v2)
- L1/LCPO: RL that follows a target length given in the prompt, with an exact mode and a maximum mode. The 1.5B model beats GPT-4o at equal reasoning length and improves up to 10% over its original. Length control generalises out of domain to GPQA, LSAT and MMLU, but code was not evaluated — [L1 project page](https://cmu-l3.github.io/l1)
- ShorterBetter (NeurIPS 2025): the reward targets the shortest correct sample in each group. On DeepSeek-R1-Distill-Qwen 1.5B and 7B it cuts length 50–80% in and out of domain at maintained accuracy — [arXiv 2504.21370](https://arxiv.org/html/2504.21370v4). Training used 40k DeepScaleR math problems, and the out-of-domain evaluation included LiveCodeBench (secondary) — [review](https://www.themoonlight.io/review/shorterbetter-guiding-reasoning-models-to-find-optimal-inference-length-for-efficient-reasoning)
- "Concise Reasoning via RL": two RL phases, the second using only 8 problems. On R1-Distill 1.5B and 7B, outputs are 40–54% shorter on MATH, AIME, AMC and MMLU-STEM, with accuracy preserved or improved (secondary for the percentages) — [arXiv 2504.05185](https://arxiv.org/html/2504.05185v1); [EmergentMind](https://www.emergentmind.com/papers/2504.05185)
- Self-Braking Tuning (NeurIPS 2025): up to 62.8% fewer tokens. The two variants average 48.3% and 43.9% reductions. On Qwen2.5-Math-7B-Instruct they cut 30.7% and 23.0% for accuracy drops of 2.65 and 1.95 points — [arXiv 2505.14604](https://arxiv.org/pdf/2505.14604)
- O1-Pruner (length-harmonizing fine-tuning): the abstract claims a large overhead reduction and higher accuracy on math, with no numbers — [arXiv 2501.12570](https://arxiv.org/abs/2501.12570). C3oT: CoT more than 50% shorter on 4 arithmetic and commonsense datasets, "without compromising effectiveness" — [arXiv 2412.11664](https://arxiv.org/abs/2412.11664)
- TokenSkip (EMNLP 2025): on Qwen2.5-14B-Instruct, GSM8K reasoning tokens fall from 313 to 181 (−40%) with under 0.4% loss, using LoRA — [arXiv 2502.12067](https://arxiv.org/abs/2502.12067)
- ASAP, a code-specific method: step-level pruning guided by first-token surprisal. On DeepSeek-R1-Distill-Qwen-7B it scores 36.19% pass@1 on LiveCodeBench v4–v5, against 33.58% for the best baseline (SPIRIT). Tokens fall from 7,892 to 6,035 (−23.5%) and latency from 4.62 s to 2.61 s (−43.5%). Token-level pruners (Selective Context, LLMLingua-2, TokenSkip) degrade code reasoning — [HF papers page](https://huggingface.co/papers/2508.05988); [arXiv v1](https://arxiv.org/html/2508.05988v1)
- "Think Less, Locally" (November 2026): local preference learning. Length falls 41.7–45.4% on 5 math benchmarks and 23.4–36.3% on 3 code benchmarks, with accuracy maintained or improved and the best AUC on every code benchmark — [arXiv 2611.04814](https://arxiv.org/abs/2611.04814)
- DeepCoder-1.5B-Preview, trained with RL from DeepSeek-R1-Distill-Qwen-1.5B:
  - LiveCodeBench v5: 25.1 vs. 16.9
  - Codeforces rating: 963 (28.5th percentile) vs. 615 (1.9th)
  - HumanEval+: 73.0 vs. 58.3
  - Recipe: GRPO+ with offline difficulty filtering, no entropy or KL loss, overlong filtering (loss masked on truncated sequences) and a raised upper clip bound. Trained at 32K context, evaluated up to 64K. About 24K problems — [model card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- DAPO reaches 50 on AIME 2024 from Qwen2.5-32B base using "four key techniques". The abstract does not name them — [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
- Survey: "Stop Overthinking" (TMLR 2025) groups methods as model-based, output-based or prompt-based, and covers reasoning in small LMs — [arXiv 2503.16419](https://arxiv.org/abs/2503.16419)

### Inferences
- Stage 1 (12k long traces, 0–1% pass@1) sits inside the "valley". The evidence-backed fixes:
  - scale long-CoT SFT to 100k+ examples, or
  - mix long and short or blueprint traces at about 1:4 rather than training on them in sequence, or
  - skip long-CoT SFT and run verifiable-reward RL from R1-Distill-1.5B, which gained +8.2 on LiveCodeBench for DeepCoder. Add a length-aware reward (ShorterBetter-style shortest-correct target, L1 target length, or Elastic budgets) and overlong filtering.
- The blueprint traces are a form of step-level compression. For code, step-level pruning beats token-level pruning, so keep the steps whole.
- Concise-RL numbers at 1.5B come from math, so a code version needs its own validation on the team's benchmark.

### Gaps
- I verified no concise-RL result for a ~1.5B model trained on code. ShorterBetter's LiveCodeBench out-of-domain numbers were not seen.
- O1-Pruner and C3oT have no numbers in their abstracts.
- DAPO's overlong reward shaping details were not verified. GFPO and similar methods were not covered because search ran out.

## 4. Mixture of LoRA experts and adapter routing: do topic-specialised adapters beat one generalist for small code models?

### Takeaway
There is no solid evidence that routing among topic-specialised LoRAs beats a single generalist for small code models. The classic methods (LoraHub, MoLE, X-LoRA) were evaluated on BBH, NLP and vision-language, or biology and materials, not on code. Reported gains are small: Self-MoE gains 3.7 points on HumanEval over its base; Arrow gains about 2% over uniform routing. Dynamic per-token adapter routing slows decoding by 2.5× or more unless kernels are fused (LoRA-Switch). One paper shows routed LoRAs are functionally equivalent to partitioning the rank of a single LoRA.

### Cited Findings
- LoraHub composes LoRAs from a few examples, evaluated on Big-Bench Hard. On average it does not beat in-context learning, but it sets a better upper bound and uses fewer tokens per example — [arXiv 2307.13269](https://arxiv.org/abs/2307.13269)
- MoLE uses hierarchical gating over LoRAs, evaluated on NLP and vision-language. It claims better fusion than arithmetic merging, with no numbers in the abstract — [arXiv 2404.13628](https://arxiv.org/abs/2404.13628)
- X-LoRA gates pretrained LoRAs per layer and per token from hidden states, applied to protein mechanics and molecular design. The abstract has no benchmark numbers — [arXiv 2402.07148](https://arxiv.org/abs/2402.07148)
- Arrow (ICML 2024) routes per token and per layer, zero-shot, by matching hidden states to each LoRA's direction of maximum variance. Tested on Phi-2 and Mistral. For private libraries it beats other routing methods; for multi-task-clustered libraries routing matters less. With a 256-expert private library it gains +1.8% (Phi-2) and +2.5% (Mistral) over uniform routing (secondary) — [arXiv 2405.11157](https://arxiv.org/html/2405.11157v1)
- A 2026 comparison on LLaMA2-7B (normalised averages): LoRAHub 68.6, Arrow 82.5, SpectR 74.2, LoraRetriever 92.9, LoRAuter 101.2 (secondary; the setup differs from the original papers) — [EmergentMind topic page](https://www.emergentmind.com/topics/lorauter)
- LoRA-Switch: existing dynamic adapters slow decoding by "2.5+ times". Token-wise routing with a fused CUDA kernel cuts decoding latency by more than 2.4× with similar accuracy gains — [arXiv 2405.17741](https://arxiv.org/abs/2405.17741)
- Self-MoE/MiXSE (ICLR 2025) trains self-specialised LoRA experts for knowledge, reasoning, math and coding, plus a router. HumanEval goes 34.1 (base) → 36.0 (instance merging) → 37.8 (MiXSE); MBPP goes 37.8 → 39.6. The average gain across domains is 6.5 points, and the router sends coding tasks to the coding expert — [arXiv 2406.12034](https://arxiv.org/html/2406.12034v2)
- SMoRA ("Each Rank Could be an Expert"): dynamic routing of multiple LoRAs is functionally equivalent to rank partitioning and block activation inside one LoRA. At equal parameters, finer partitioning helps. Plain LoRA struggles on multi-domain tasks such as code (DS-1000) — [paper](https://www.researchgate.net/publication/388422159_Each_Rank_Could_be_an_Expert_Single-Ranked_Mixture_of_Experts_LoRA_for_Multi-Task_Learning)
- Mixture-of-LoRAs: domain LoRA experts plus routing reduce forgetting and task interference — [arXiv 2403.03432](https://arxiv.org/pdf/2403.03432). In contrast, "Align, Don't Divide" challenges the premise that multi-task adaptation needs that kind of structural complexity — [arXiv 2508.05078](https://arxiv.org/pdf/2508.05078)

### Inferences
- With a noisy topic router (macro-F1 about 0.4–0.7, see Q1) and small expected expert gains (about 2–4 points), misrouting can cancel the benefit. If adapters are used on a phone, select one adapter per problem at prefill rather than routing per token, which costs about 2.5× in decode speed. Keep a generalist adapter as the fallback.
- The cleaner experiment: one generalist LoRA trained on all topics versus per-topic LoRAs with oracle routing versus per-topic LoRAs with predicted routing, on the same benchmark. The oracle condition bounds the maximum value of routing.

### Gaps
- I found no head-to-head comparison of topic-specialised LoRAs against a generalist on competitive programming for models of 2B or smaller.
- MoLE, X-LoRA and LoraHub have no code results.
- Apple-style on-device adapter swapping and LoRA-Switch's mobile performance could not be verified because search ran out.

## 5. Retrieval that actually helps small models on competitive programming

### Takeaway
Retrieving whole solved episodes (a similar problem with its solution, explanation and code) helps most. On USACO, GPT-3.5 went from 0.59% to 5.49%, or 8.79% with reflection, while textbook retrieval helped far less. The query works best when it includes a first solution attempt; retrieving problem descriptions alone kept only 3.18% of the gain. On LiveCodeBench, retrieved solutions gave DeepSeekCoder-7B only +1.3 to +3.3 points and did nothing or harm for a weak 7B model. General small embedders are poor at telling which problems are algorithmically similar: Qwen3-Embedding-0.6B averages 56.8 NDCG@10 on CPRet, against 84.6 for a retriever trained on competitive programming. Text similarity misleads, and matching attempted solutions plus BM25 adds 11.8% over embedding models.

### Cited Findings
- USACO-bench, zero-shot pass@1: GPT-4 8.70, Claude-3-Sonnet 2.61, GPT-3.5 0.59, DeepSeek Coder 7B 1.04, CodeLlama 7B 0.19; the human average is 35.83. The 7B models were not tested with retrieval — [arXiv 2404.10952](https://arxiv.org/html/2404.10952v1)
- USACO-bench methods (pass@1, GPT-3.5 / GPT-4):

  | Method | GPT-3.5 | GPT-4 |
  |---|---|---|
  | Reflexion | 0.97 | 12.38 |
  | Semantic (textbook) retrieval | 2.04 | 10.26 |
  | Episodic retrieval | 5.49 | 14.33 |
  | Semantic + episodic | 6.76 | 12.54 |
  | Episodic + Reflexion | 8.79 | 20.20 |
  | All three | 7.52 | 18.05 |

  — [same](https://arxiv.org/html/2404.10952v1)
- USACO-bench retrieval details:
  - Episodic documents are other USACO problems, each with its statement, English solution and Python code. The semantic store is the cp-algorithms textbook (30 chapters).
  - Retrieval is BM25, with a query made of the problem plus an initial generated solution.
  - Number of documents: GPT-4 used 2 and GPT-3.5 used 1. A sweep over 1–4 documents gave 13.03 / 14.33 / 13.11 / 12.38.
  - Retrieving problem descriptions only retained 3.18% of the performance — [same](https://arxiv.org/html/2404.10952v1)
- CodeRAG-Bench (Findings of NAACL 2025), LiveCodeBench pass@1 without retrieval: StarCoder2-7B 1.5, CodeGemma-7B 21.5, CodeLlama-7B 13.5, DeepSeekCoder-7B 30.5, Llama3-8B 2.8, GPT-4o 43.8 — [arXiv 2406.14497](https://arxiv.org/html/2406.14497v2)
- CodeRAG-Bench, LiveCodeBench with the top 5 retrieved programming solutions:

  | Retriever | StarCoder2-7B | DeepSeekCoder-7B |
  |---|---|---|
  | None | 1.5 | 30.5 |
  | BM25 | 1.0 | 31.8 |
  | GIST-large | 0.5 | 33.8 |
  | Voyage-code | 0.3 | 31.8 |
  | OpenAI | 1.5 | 32.0 |

  — [same](https://arxiv.org/html/2406.14497v2)
- Cases where retrieval hurt in CodeRAG-Bench:
  - DeepSeekCoder-7B on HumanEval fell from 70.1 to 66.3–68.9, and to 53.0 with OpenAI reranking. The authors cite "over-complicated and ungrammatically repetitive generations".
  - Models copy functions from the context, sometimes overwriting the function being asked for.
  - Reranking chunks "greatly degrades" results.
  - Top 5 documents worked best in most cases, as did keeping the first ~800 tokens of each document; adding more low-ranked documents adds noise.
  - Average retriever NDCG@10 over the non-LiveCodeBench tasks: BM25 57.7, GIST-large 61.7, BGE-base 58.8, SFR-Mistral 67.0, OpenAI 61.7, Voyage-code 63.7 — [same](https://arxiv.org/html/2406.14497v2)
- CPRet (NeurIPS 2025 Datasets & Benchmarks): four tasks (text-to-code, code-to-code, problem-to-duplicate, simplified-to-full). Training data: 38.8K problems and 2.93M solutions. CPRetriever-Code is 2B, built on SFR-Embedding-Code-2B_R (Gemma-2-2B). Average NDCG@10:

  | Model | Avg NDCG@10 |
  |---|---|
  | CPRetriever-Prob-Qwen3-4B | 84.60 (text-to-code 80.84, code-to-code 87.10, duplicate 74.33, simplified-to-full 96.15) |
  | CPRetriever-Code-Qwen3-4B | 75.54 |
  | Qwen3-Embedding-4B | 71.15 |
  | CPRetriever-Prob (2B) | 69.50 |
  | Qwen3-Embedding-8B | 68.67 |
  | CPRetriever-Code (2B) | 65.28 |
  | SFR-Embedding-Code-2B | 59.84 |
  | Qwen3-Embedding-0.6B | 56.83 |

  — [arXiv 2505.12925v2](https://arxiv.org/html/2505.12925v2)
- CPRet also analysed 388 LiveCodeBench problems released after 2024-09-01: pass rates rise with a problem's similarity to earlier problems, most strongly at medium difficulty. The models are open source and there is a demo — [same](https://arxiv.org/html/2505.12925v2); [abstract](https://arxiv.org/abs/2505.12925v2)
- AlgoSimBench (IJCAI 2026): 402 multiple-choice questions, each pairing one algorithmically similar problem with three textually similar distractors; LLMs struggle. Matching attempted solutions improves accuracy by 9% on average. Combining it with BM25 gains 11.8% over state-of-the-art embedding models — [arXiv 2507.15378](https://arxiv.org/abs/2507.15378). The UT page reports gains of 6.7–11.7 points and up to 52.2% retrieval accuracy — [UT Austin page](https://www.cs.utexas.edu/~ai-lab/pub-view.php?PubID=128167). An earlier version reported o3-mini as best at 65.9% on the multiple-choice task (secondary).
- Small code embedders (vendor-reported, overall average over 25 benchmarks / MTEB-Code average):

  | Model | Size, backbone | Overall | MTEB-Code |
  |---|---|---|---|
  | jina-code-embeddings-0.5b | 494M, Qwen2.5-Coder-0.5B | 78.41 | 78.72 |
  | jina-code-embeddings-1.5b | 1.54B | 79.04 | 78.94 |
  | Qwen3-Embedding-0.6B | 600M | 73.49 | 74.69 |
  | voyage-code-3 | closed | 79.23 | 79.84 |
  | gemini-embedding-001 | closed | 77.38 | 76.48 |

  No per-task competitive-programming scores were given — [Jina blog](https://jina.ai/news/jina-code-embeddings-sota-code-retrieval-at-0-5b-and-1-5b/). The vendor also lists CoIR nDCG@10 of 73.94 for the 0.5B model — [Jina model page](https://jina.ai/models/jina-code-embeddings-0.5b/)

### Inferences
- The current retrieval setup (top-1 over 7 hand-written topic blueprints, embedded with general-purpose MiniLM, queried by the statement) is the weakest configuration in this evidence on four counts: the corpus is topic-level rather than episodic, the query is the statement only, the embedder is general-text, and the corpus is tiny. Higher-value changes:
  - Build an episodic corpus of problems with solutions and explanations. TACO, CodeContests and CPRet training data are candidates.
  - Query with a draft attempt or plan plus the statement, using BM25 and/or a retriever trained on competitive programming.
  - Keep k at 1–2 and truncate each document to about 800 tokens.
  - Do not rerank chunks.
- Fine-tuning a ≤0.6B embedder on CPRet-style data is likely necessary on device, since stock Qwen3-Embedding-0.6B is weak at problem-level similarity.
- Retrieval gains depend on how capable the coder is: StarCoder2-7B saw none on LiveCodeBench. Measure copying and distraction on the 1.5B model, and gate retrieval with the router's confidence.

### Gaps
- I found no published retrieval result on competitive programming for coders of 2B or smaller.
- No verified competitive-programming retrieval numbers for nomic-embed-code, CodeRankEmbed, CodeSage or CodeT5+ embeddings.
- No reranker results on competitive programming (Qwen3-Reranker, bge-reranker).
- KareCoder and other algorithm-knowledge retrieval systems were not verified because search ran out.

## 6. Systems pairing a tiny planner or router with a small coder (plan-then-code, self-planning, PlanSearch, CodeChain, structured CoT), with results at 7B or smaller

### Takeaway
An external plan generator reliably lifts small coders: plans from COTTON, a 7B planner, add 6–17 points of HumanEval pass@1 to 0.2B–7B coders. Small models cannot write good plans themselves. In competitive programming, plan correctness dominates: accuracy is bimodal on whether the sketch is right (PlanSearch), and oracle step-by-step plans give about 7× (Q1). But a fine-tuned 7B planner gives only small absolute gains on competitive programming: DeepSeek-Coder-7B solve@10 went from 1.8% to 4.2%. A tiny planner will probably be better used for routing and budgeting than for algorithm design.

### Cited Findings
- Self-planning: up to 25.4% relative Pass@1 gain over direct generation, and up to 11.9% over CoT — [arXiv 2303.06689](https://arxiv.org/abs/2303.06689). Structured CoT (SCoT): up to 13.79% Pass@1 over CoT prompting with ChatGPT and Codex on HumanEval, MBPP and MBCPP — [arXiv 2305.06599](https://arxiv.org/abs/2305.06599)
- MapCoder uses four agents (recall examples, plan, code, debug). With a strong backbone it reports pass@1 of HumanEval 93.9, MBPP 83.1, APPS 22.0, CodeContests 28.5 and xCodeEval 45.3 — [arXiv 2405.11403](https://arxiv.org/abs/2405.11403)
- PlanSearch (ICLR 2025): Claude 3.5 Sonnet on LiveCodeBench reaches pass@200 of 77.0, against 60.6 for repeated sampling and 41.4 pass@1 without search — [arXiv 2409.03733](https://arxiv.org/abs/2409.03733)
- CodeChain (ICLR 2024): relative pass@1 gains of +35% on APPS and +76% on CodeContests. WizardCoder-15B goes from 7.90 to 10.50 on APPS and from 1.11 to 2.35 on CodeContests. The trend is clearer at 7B–34B than at the smallest sizes — [arXiv 2310.08992](https://arxiv.org/pdf/2310.08992); [Salesforce blog](https://blog.salesforceairesearch.com/codechain/)
- COTTON (IEEE TSE 2024) is a CodeLlama-7B LoRA (rank 8) trained on 9,264 CoT pairs in about 6 hours on one RTX 3090. HumanEval pass@1 without CoT → with COTTON plans → with gpt-3.5 plans:

  | Coder | No CoT | COTTON | gpt-3.5 |
  |---|---|---|---|
  | CodeGen-350M | 14.63 | 20.73 | 24.39 |
  | StarCoder-1B | 12.80 | 25.00 | 28.66 |
  | StarCoder-3B | 17.07 | 30.49 | 39.63 |
  | StarCoder-7B | 21.95 | 37.20 | 41.46 |
  | CodeT5+ 220M | 12.20 | 18.90 | 23.17 |
  | CodeT5+ 6B | 26.22 | 42.68 | 47.56 |

  The authors conclude that "ℓLMs cannot independently generate high-quality CoTs"; self-generated CoT gave limited gains or made things worse — [arXiv 2312.05562](https://arxiv.org/html/2312.05562v2)
- Reasoner–Coder distillation: GPT-4 writes editorial-style explanations of gold solutions, and a Reasoner is fine-tuned on them (8,248 examples, 8M tokens). With DeepSeek-Coder-7B, CodeContests solve@1/5/10 goes from 0.8/1.2/1.8 to 1.4/2.9/4.2, and a set of 246 newer Codeforces problems goes from 0.4/0.9/1.6 to 1.0/2.9/4.1. Fine-tuning the Coder directly on problem–solution pairs hurts: GPT-3.5's Codeforces solve@10 fell from 3.3 to 1.6 — [arXiv 2404.08148](https://arxiv.org/html/2404.08148v1)
- Llama-3 study, further findings:
  - For 8B models, reasoning prompts that simplify the problem (self-reflection, explaining the input–output examples) help.
  - Stacking a second reasoning step hurts: −2.0 at 1@3 and −3.1 at 100@300 for 70B.
  - Simple binary execution feedback works best.
  - Rejection-sampling fine-tuning internalises the CoT: the 70B RFT model without CoT prompts scores 29.7 at 1@3 multi-turn, against 27.7 for the prompted model — [arXiv 2410.08105](https://arxiv.org/html/2410.08105)
- Self-reflection only helps strong models: on USACO, GPT-4 goes from 8.70 to 12.38 and GPT-3.5 from 0.59 to 0.97 — [arXiv 2404.10952](https://arxiv.org/html/2404.10952v1)

### Inferences
- The current Gemma-27B blueprint traces are the same idea as COTTON or the Reasoner: plans from a large model distilled into training data. The evidence says the plan's specificity and correctness, not its format, drive gains. Prefer problem-specific step-by-step plans or editorials over topic templates, and sample diverse plans (diversity raised solve@10 from 12.1% to 13.9% in Q1).
- For a "decision model" in the range of 100M–0.5B parameters, the evidence supports three jobs: predicting budget or difficulty, deciding whether to retrieve, and choosing a soft topic. Generating competitive-programming plans is not supported; even 7B planners add only about 2 points of solve@10.
- A two-stage design of a tiny router plus a 1.5B coder that plans and codes, using retrieved episodes and a predicted budget, fits the evidence better than tiny-planner → coder.

### Gaps
- Model sizes for Self-planning and SCoT were not given in the abstracts, and their small-model behaviour was not checked.
- MapCoder results with small open models were not verified.
- I found no competitive-programming results for planners of 1B or smaller.
- Web search ran out mid-session, so AlphaCode metadata conditioning, KareCoder and LiveCodeBench Pro's per-tag analysis were not checked.
