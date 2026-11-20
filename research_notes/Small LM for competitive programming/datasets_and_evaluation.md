# Datasets and Evaluation Methodology for Small (~1.5B) Competitive-Programming Code Models (Python, stdin/stdout)

Research date: 2026-11-19. Sources are mainly Hugging Face dataset and model cards, plus HF API metadata (licence tags, creation dates, and row counts from the HF datasets-server). Other sources: arXiv abstracts and HTML full texts, and the LiveCodeBench and open-r1 GitHub code. Row counts marked "datasets-server" come from https://datasets-server.huggingface.co/size. Where the API flags `partial=True`, the counts are lower bounds.

---

## 1. Training datasets: size, source, teacher, test quality, labels, licence, HF id

### Takeaway
Two families of competitive-programming SFT corpora are the largest and best licensed (CC-BY-4.0):
- **NVIDIA's OpenCodeReasoning line**: OCR, then OCR-2, then Nemotron-Competitive-Programming-v1 (Dec 2025) and v2 (Mar 2026). These are DeepSeek-R1 / R1-0528 traces over about 28–35K questions from TACO, APPS, CodeContests and Codeforces.
- **Microsoft rStar-Coder**: QwQ-32B traces, filtered by tests, over 418K problems with mutually verified synthetic tests.

For problems with trustworthy tests (RL, or rejection-sampled SFT), the best-documented sources are CodeContests+, open-r1/codeforces (generated checkers and tests), HardTests, TACO-verified and DeepCoder's 24K set. Nearly every corpus draws on the same upstream pool (TACO, APPS, CodeContests, Codeforces, AtCoder, AIZU, CodeChef…). This has big consequences for decontamination (see section 5).

### Cited Findings

#### rStar-Coder (`microsoft/rStar-Coder`, CC-BY-4.0, arXiv 2505.21297)
- Contains 418K competition-level problems, 580K long-reasoning solutions and "rich test cases of varying difficulty". Configs: `synthetic_sft`, `synthetic_rl`, `synthetic_rl_testcase`, `seed_sft`, `seed_testcase`. Licence tag: cc-by-4.0. — [HF card](https://huggingface.co/datasets/microsoft/rStar-Coder)
- Row counts from datasets-server (flagged partial): seed_sft 591,660; synthetic_sft 398,183; synthetic_rl 398,107. Only 317 rows of seed_testcase and 4,000 rows of synthetic_rl_testcase were converted, because the test-case shards are huge (the card mentions 797 shards for synthetic_rl_testcase). — [datasets-server](https://datasets-server.huggingface.co/size?dataset=microsoft%2FrStar-Coder); [HF card](https://huggingface.co/datasets/microsoft/rStar-Coder)
- Fields: `seed_sft` has `verified` and `is_passed`, so it includes unverified solutions. `synthetic_sft` has `seed_question` and `seed_source`, so each synthetic problem is a derivative of a seed problem. `seed_testcase` mixes official and "solution-validated generated" cases. `synthetic_rl_testcase` cases "passed through mutual-verification". — [HF card](https://huggingface.co/datasets/microsoft/rStar-Coder)
- Seed sources: "TACO, APPS, CodeContests, CodeContests-Python-Submission, CodeForces from the OpenR1 project, and USA Computing Olympiad 2011-2023". In total 57,215 problems were collected. After cleaning, 37.7K expert-written problems with oracle solutions remained. From these, 1,565K new problems were synthesized and 380K survived verification (418K verified in total). Per-platform table (e.g., AIZU 2,151 verified seeds; AtCoder 2,080; CodeChef 3,765; CodeWars 2,515; GeeksForGeeks 2,680; HackerEarth 1,411; HackerRank 854…). — [rStar-Coder paper HTML](https://arxiv.org/html/2505.21297v1)
- Test generation is a "three-step input generation":
  - GPT-4o writes a test-input generator and a validator using the CYaRon library.
  - Outputs are labelled by **mutual verification**: "We sample multiple long-reasoning solutions from a strong reasoning model (QWQ-32B), and accept both the test outputs and solutions if a majority produce consistent results across all test inputs."
  - The SFT traces come from QwQ-32B, "keeping only those that pass all generated tests."
  
  — [rStar-Coder paper HTML](https://arxiv.org/html/2505.21297v1)
- Decontamination: "we decontaminate our data by removing any problems that overlap (16-gram) with evaluation benchmarks: HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench, and USACO 2025." This does **not** cover the TACO, APPS or CodeContests test splits or open-r1/verifiable-coding-problems. — [rStar-Coder paper HTML](https://arxiv.org/html/2505.21297v1)
- Results:
  - Qwen2.5-7B goes from 17.4% to 57.3% on LiveCodeBench; Qwen2.5-14B from 23.3% to 62.5%.
  - **rStar-Coder-1.5B: LCB 40.1** (versus DeepSeek-R1-Distill-Qwen-1.5B at 16.9), HumanEval 88.4, HumanEval+ 81.7, MBPP 74.1, MBPP+ 60.8.
  - The 7B model averages 16.15% pass@1 on USACO 2025.
  
  — [HF card](https://huggingface.co/datasets/microsoft/rStar-Coder); [arXiv abstract](https://arxiv.org/abs/2505.21297)
- Evaluation setup: LiveCodeBench v5 (problems from 2024.8–2025.2) and USACO 2025 (12 problems, bronze to platinum), "temperature of 0.6, maximum output length to 32k tokens". — [rStar-Coder paper HTML](https://arxiv.org/html/2505.21297v1)

#### NVIDIA OpenCodeReasoning (`nvidia/OpenCodeReasoning`, CC-BY-4.0, arXiv 2504.01943)
- Contains 735,255 Python samples over 28,319 unique questions; the teacher is DeepSeek-R1. Configs: `split_0`, which includes the question text, and `split_1`, where the question must be fetched by dataset/split/index.
- Sources: Codeforces 10,069 questions / 386,948 samples; CodeChef 3,796 / 72,925; AIZU 2,123 / 62,476; AtCoder 2,043 / 47,222; Codewars, GeeksForGeeks, HackerEarth, HackerRank, Kattis, and LeetCode (777).
- Questions were gathered from TACO, APPS, CodeContests and open-r1/codeforces. "We do not include the test split of CodeContests and open-r1/codeforces."
- Each row carries a `difficulty` field and a per-row source `license`. The card states it is "ready for commercial/non-commercial use".

— [HF card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
- Using SFT only, the paper reaches 61.8% on LiveCodeBench and 24.6% on CodeContests. Key finding: "execution filtering negatively affected benchmark accuracy, leading us to prioritize instruction diversity over solution correctness." — [arXiv abstract](https://arxiv.org/abs/2504.01943)
- Generation used temperature 0.6, top-p 0.95, with an injected `<think>` tag. At inference, a temperature sweep found 0.6 and 0.7 best, and 0.6 was used. The paper has a "Verification for Benchmark Contamination" step ("we meticulously check for any potential data leakage or overlap… with evaluation benchmarks"). — [OCR paper HTML](https://arxiv.org/html/2504.01943)

#### OpenCodeReasoning-2 (`nvidia/OpenCodeReasoning-2`, CC-BY-4.0, arXiv 2507.09075)
- Python: 1,398,166 samples over 34,125 questions (Codeforces 15,641 q / 834,523 samples). C++: 1,174,475 samples over 30,092 questions.
- "Solution responses are generated by R1 and critique responses are generated by QwQ". Fields: `r1_generation`, `qwq_critique`, `judgement` (right or wrong), `difficulty`.
- The question field is left blank and must be re-joined from TACO, APPS, CodeContests or open-r1/codeforces. The CodeContests and open-r1/codeforces test splits are excluded.

— [HF card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2)
- The paper describes the dataset as 2.5M question-solution-critique triples over about 35K questions. It also extends LiveCodeBench to C++. — [arXiv abstract](https://arxiv.org/abs/2507.09075)

#### Nemotron competitive-programming sets (2025-12 and 2026-03)
- `nvidia/Nemotron-Competitive-Programming-v1` (created 2025-12-15, CC-BY-4.0):
  - competitive_coding_python 1,821,278; competitive_coding_cpp 932,012; infinibyte 1,174,694. Total 3,927,984, about 178 GB.
  - 34,799 unique questions. References DeepSeek-R1-0528, OCR, OCR-II and GenSelect.
  - InfiniByte "creates entirely new programming problems by bringing together concepts from different scientific fields". The card states it is "ready for commercial use".
  
  — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Competitive-Programming-v1)
- `nvidia/Nemotron-SFT-Competitive-Programming-v2` (created 2026-03-08):
  - competitive_coding_python 336,568; competitive_coding_cpp 332,559; exercism 79,244 (Aider-benchmark-decontaminated); text_to_sql 96,564. Total 844,935.
  - Same 34,799 questions; references DeepSeek-R1-0528.
  - Licence: CC-BY-4.0 "except for Code Forces data, which is licensed under the Open Data Commons Attribution License (ODC-By)"; MIT is also listed.
  
  — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2); [HF API licence tags](https://huggingface.co/api/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2)
- `nvidia/Nemotron-Cascade-SFT-Stage-2` (2025-12, CC-BY-4.0):
  - The code domain has 1,392,831 samples: OpenCodeReasoning 34,747 q / 1,143,266 samples; TACO 5,852 / 58,273; opc-sft-stage2 37,820 / 188,063; apps 159 / 1,575; synthetic 274 / 1,654.
  - Thinking traces come from DeepSeek-R1-0528 and non-thinking responses from DeepSeek-V3.
  
  — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-SFT-Stage-2)
- `nvidia/Llama-Nemotron-Post-Training-Dataset` (CC-BY-4.0): 10,108,883 code samples (SFT). — [HF card](https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset)
- `nvidia/Nemotron-Post-Training-Dataset-v1` (2025-07, CC-BY-4.0): 1,896,395 code samples out of 25,659,642 total. — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)
- `nvidia/Nemotron-Post-Training-Dataset-v2` (2025-08, CC-BY-4.0, arXiv 2508.14444): the repo is gated (auto-approve), so its README could not be read anonymously. — [HF API](https://huggingface.co/api/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)
- `nvidia/AceReason-1.1-SFT` (CC-BY-4.0, arXiv 2506.13284): 2,668,741 math and 1,301,591 code samples, "all responses… generated by DeepSeek-R1". It is the SFT data for AceReason-Nemotron-1.1-7B. — [HF card](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT)
- `nvidia/OpenCodeInstruct` (CC-BY-4.0, arXiv 2504.04030):
  - 5M samples, each with a question, solution, test cases, execution feedback and LLM quality assessment. Fields: `unit_tests`, `tests_execution_status`, `average_test_score`, `llm_judgement`.
  - It is general instruction-style Python, not competitive programming.
  
  — [HF card](https://huggingface.co/datasets/nvidia/OpenCodeInstruct); [arXiv](https://arxiv.org/abs/2504.04030)

#### OpenThoughts (`open-thoughts/*`, Apache-2.0, arXiv 2506.04178)
- `OpenThoughts3-1.2M`: 850K math, 250K code and 100K science questions; "Annotations are done with QwQ-32B". — [HF card](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- OpenThinker3-7B reaches 51% on LiveCodeBench 06/24–01/25, 17.2 points above DeepSeek-R1-Distill-Qwen-7B. — [arXiv abstract](https://arxiv.org/abs/2506.04178)
- Earlier versions: `OpenThoughts-114k` and `OpenThoughts2-1M`, both Apache-2.0. — [HF API](https://huggingface.co/api/datasets/open-thoughts/OpenThoughts-114k)
- Evaluation uses Evalchemy, and values "within 2 standard errors of the best" are bolded. This is a good example of a reporting standard. — [HF card](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)

#### open-r1 family (Hugging Face)
- `open-r1/codeforces` (CC-BY-4.0):
  - Covers "more than 10k unique problems" from the very first contests through 2025. Splits: default 9,556 train / 468 test; `verifiable` 8,338 / 422; `verifiable-prompts` 9,658 / 844.
  - Fields include `rating`, `tags`, `contest_start_year`, `time_limit`, `memory_limit`, `editorial`, `official_tests`, `official_tests_complete`, `generated_checker`, `generated_tests`, `executable`, `input_mode`.
  
  — [datasets-server](https://datasets-server.huggingface.co/size?dataset=open-r1%2Fcodeforces); [HF card](https://huggingface.co/datasets/open-r1/codeforces)
- The card explains "Why most datasets are actually not 'verifiable'":
  - Codeforces truncates displayed tests to about 400 characters, so scraped "verifiable" sets often contain only trivial tests that brute force passes.
  - "Many problems allow for multiple correct answers, and therefore require… a 'checker' (~30% of problems)."
  - open-r1 found which problems need checkers by running real contestant solutions, then generated and validated custom checkers. Editorials cover about 60% of problems.
  
  — [HF card](https://huggingface.co/datasets/open-r1/codeforces)
- `open-r1/codeforces-cots` (CC-BY-4.0):
  - "10k CodeForces problems with up to five reasoning traces generated by DeepSeek R1. We did not filter the traces for correctness, but found that around 84% of the Python ones pass the public tests."
  - Configs: solutions 47,780; solutions_py 9,556; solutions_py_decontaminated 8,133; solutions_w_editorials 29,180; checker_interactor 35,718; test_input_generator 20,620.
  - The `_decontaminated` variants use "8-gram overlap against the AIME24, AIME25, GPQA Diamond, MATH-500, and LiveCodeBench".
  
  — [HF card](https://huggingface.co/datasets/open-r1/codeforces-cots); [datasets-server](https://datasets-server.huggingface.co/size?dataset=open-r1%2Fcodeforces-cots)
- `open-r1/Mixture-of-Thoughts`:
  - 349,317 traces in total; code 83,070 (Python and C++). The code part is sourced from the codeforces-cots `solutions` and `solutions_w_editorials` subsets; the teacher is DeepSeek-R1. It was used to train OpenR1-Distill-7B.
  - **No licence tag on HF.**
  
  — [HF card](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts); [datasets-server](https://datasets-server.huggingface.co/size?dataset=open-r1%2FMixture-of-Thoughts)
- `open-r1/codeforces-submissions` (CC-BY-4.0, 10M–100M rows): configs `selected_accepted` and `selected_incorrect`. These are useful for measuring the true and false positive rates of test suites and checkers. — [HF API](https://huggingface.co/api/datasets/open-r1/codeforces-submissions)
- `open-r1/verifiable-coding-problems-python`:
  - 35,735 rows. "Contains all Python problems from PrimeIntellect's verifiable-coding-problems… otherwise the data is the same."
  - Fields: `source`, `task_type`, `in_source_id`, `problem_statement`, `gold_standard_solution`, `metadata`, `verification_info`. **No licence tag on HF.**
  - A derived `…_decontaminated-tested` variant has 15,068 rows and an extra `test_reward` column. Its card is empty, so the exact filtering is undocumented.
  
  — [HF card](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python); [datasets-server](https://datasets-server.huggingface.co/size?dataset=open-r1%2Fverifiable-coding-problems-python_decontaminated-tested)

#### PrimeIntellect
- `PrimeIntellect/verifiable-coding-problems`: 144,169 rows, with no licence tag on HF. It is "a subset of the task data used to construct SYNTHETIC-1". — [HF card](https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems); [datasets-server](https://datasets-server.huggingface.co/size?dataset=PrimeIntellect%2Fverifiable-coding-problems)
- `PrimeIntellect/SYNTHETIC-1` (Apache-2.0): "Two Million Crowdsourced Reasoning Traces from Deepseek-R1", raw and unfiltered. Its coding part (144K samples) is "curated from Apps, Codecontests, Codeforces and TACO", with "LLM-based post-processing… to translate Python problems into Javascript, Rust and C++". The verifier is "Containerized execution of unit tests". — [HF card](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1); [blog](https://primeintellect.ai/blog/synthetic-1)
- `PrimeIntellect/SYNTHETIC-2` (Apache-2.0, 2025-07): R1-0528 traces, plus "multiple reasoning traces from smaller models which can be used for difficulty estimation". The `SYNTHETIC-2-SFT-verified` subset has 104,913 rows, keeping only R1-0528 responses verified correct. — [HF card](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2); [datasets-server](https://datasets-server.huggingface.co/size?dataset=PrimeIntellect%2FSYNTHETIC-2-SFT-verified)

#### Classic competitive-programming problem sets
- `deepmind/code_contests` (CC-BY-4.0, AlphaCode):
  - Valid 117 and test 165. The 13,610-problem total is cited by CodeContests+. The datasets-server conversion of train is partial, so the official train count is not shown there.
  - Fields: `difficulty`, `cf_rating`, `cf_tags` (example: `cf_rating: 2100`, `cf_tags: ['greedy','two pointers']`), `public_tests`, `private_tests`, `generated_tests`, `incorrect_solutions`, `time_limit`, `memory_limit_bytes`.
  
  — [HF card](https://huggingface.co/datasets/deepmind/code_contests); [datasets-server](https://datasets-server.huggingface.co/size?dataset=deepmind%2Fcode_contests); [CodeContests+ paper](https://arxiv.org/html/2506.05817)
- `ByteDance-Seed/Code-Contests-Plus` (CC-BY-4.0, arXiv 2506.05817):
  - 11,690 problems. Configs `1x`–`5x` average 25/44/62/80/98 tests.
  - Each problem has a generator, validator and **checker**, plus correct and incorrect submissions ("more than 13 million").
  - Validation pass rate is 100%, versus 67.1% for the original CodeContests. Test quality was measured with 1.72M labelled submissions (true and false positive rates): "CodeContests+ can yield twice the number of effective problems compared to CodeContests" at the same thresholds.
  
  — [HF card](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus); [paper HTML](https://arxiv.org/html/2506.05817)
- `BAAI/TACO` (Apache-2.0, arXiv 2312.14852):
  - 25,443 train / 1,000 test, with up to 1.55M solutions.
  - Labels: `difficulty`, `tags`, `raw_tags`, `skill_types`, `Expected Time Complexity`, `Expected Auxiliary Space`, `time_limit`, `memory_limit`, `date`, `source`.
  - "We removed and modified some test cases in test set. Please update to use the newest version."
  
  — [HF card](https://huggingface.co/datasets/BAAI/TACO); [arXiv](https://arxiv.org/abs/2312.14852)
- `likaixin/TACO-verified` (MIT tag): 12,898 problems / 1,043,251 solutions from TACO **train**. That keeps 50.69% of problems and 71.03% of solutions, after removing solutions that fail any test (10 s timeout, Intel E5-2620 v3). — [HF card](https://huggingface.co/datasets/likaixin/TACO-verified)
- `codeparrot/apps` (MIT): 10,000 problems. The CodeContests+ comparison table lists it as "Crawled" tests. — [HF API](https://huggingface.co/api/datasets/codeparrot/apps); [CodeContests+ paper](https://arxiv.org/html/2506.05817)
- `sigcp/hardtests_problems` (no licence tag on HF, arXiv 2505.24098):
  - 47,136 problems from 13 online judges (Codeforces, Luogu, AtCoder, SPOJ, CodeChef, GeeksforGeeks, Codewars, HackerEarth, Aizu…). Tests are stored in a separate dataset.
  - Fields: `difficulty_ratings`, `tags`, `platform`, `contest_date`, `solutions`.
  - The paper abstract instead says "HARDTESTS with 26.6k problems and high-quality synthetic tests", which conflicts with the card's 47,136; possibly only a subset has synthesized tests. HardTests tests gain +11.22 points in precision over existing tests.
  
  — [HF card](https://huggingface.co/datasets/sigcp/hardtests_problems); [arXiv](https://arxiv.org/abs/2505.24098)

#### RL-oriented verified sets
- `agentica-org/DeepCoder-Preview-Dataset` (MIT):
  - "24K problems paired with their test cases": taco 7,436 (TACO-verified), primeintellect 16,252 (SYNTHETIC-1 verified), lcbv5 train 599 (LCB v5, May 1 2023–Jul 31 2024).
  - Test splits: lcbv5 test 279 (Aug 1 2024–Feb 1 2025) and codeforces test 408 (from Qwen/CodeElo).
  - "All problems are fully verifiable and have >= 5 test cases."
  
  — [HF card](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset); [datasets-server](https://datasets-server.huggingface.co/size?dataset=agentica-org%2FDeepCoder-Preview-Dataset)
- `Skywork/Skywork-OR1-RL-Data` (no licence tag on HF, arXiv 2505.22312): 105,055 math and **14,057 code** problems. Problems are tagged with difficulty 0–16 *relative to DeepSeek-R1-Distill-Qwen-{1.5,7,32}B*, and the 0 and 16 values were excluded per model. — [HF card](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data); [datasets-server](https://datasets-server.huggingface.co/size?dataset=Skywork%2FSkywork-OR1-RL-Data)

#### Function-level and synthetic sets (mostly not stdin/stdout competitive programming)
- `KodCode/KodCode-V1` (**CC-BY-NC-4.0**, arXiv 2503.02951):
  - 484,097 train plus 3,335 `use_with_caution` rows, in 12 subsets from basic exercises to competitive programming.
  - Question-solution-test triplets are "validated via a self-verification procedure".
  - Fields: `gpt_difficulty`, `gpt_pass_percentage`, `benchmark_similarity`.
  
  — [HF card](https://huggingface.co/datasets/KodCode/KodCode-V1); [datasets-server](https://datasets-server.huggingface.co/size?dataset=KodCode%2FKodCode-V1); [arXiv](https://arxiv.org/abs/2503.02951)
- `KodCode/KodCode-V1-SFT-R1` (CC-BY-NC-4.0): 268,211 train, plus 210,787 `incorrect` and 4,439 `use_with_caution` rows. Responses come from DeepSeek-R1 "under a test-based reject sampling procedure". — [datasets-server](https://datasets-server.huggingface.co/size?dataset=KodCode%2FKodCode-V1-SFT-R1); [arXiv](https://arxiv.org/abs/2503.02951)
- `TIGER-Lab/AceCode-87K` (MIT, arXiv 2502.01718): 87,149 questions. LLMs "imagine" test cases, and noisy ones are filtered. — [HF card](https://huggingface.co/datasets/TIGER-Lab/AceCode-87K); [arXiv](https://arxiv.org/abs/2502.01718)
- `TIGER-Lab/AceCode-V1.1-69K` (MIT): questions and tests rewritten by o1-mini, filtered with Qwen2.5-Coder-32B-Instruct. Before filtering there were 147,220 samples averaging 20.90 tests. — [HF card](https://huggingface.co/datasets/TIGER-Lab/AceCode-V1.1-69K)
- `newfacade/LeetCodeDataset` (Apache-2.0, arXiv 2504.14655):
  - 2,641 train / 228 test, with a temporal split at July 2024 and 100+ test cases per problem.
  - Fields: `difficulty`, `tags`, `estimated_date`, `entry_point` (function-style).
  - "SFT with only 2.6K model-generated solutions achieves performance comparable to 110K-sample counterparts."
  
  — [datasets-server](https://datasets-server.huggingface.co/size?dataset=newfacade%2FLeetCodeDataset); [arXiv](https://arxiv.org/abs/2504.14655)

#### 2026 additions found
- `nvidia/Nemotron-SFT-Competitive-Programming-v2` (2026-03): see above. — [HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Competitive-Programming-v2)
- `IFM/Code-Reasoning` (2026-08, Apache-2.0, 100M–1B size category, about 35K downloads):
  - Subsets: `code-thinking-v0`, `code-thinking-v1`, `code-thinking-hard`, `code-no-thinking`, `code-no-thinking-hard`, `codeio`, `code-task-synthesis`. Part of the "K2 Horizon" collection.
  - The card I read did not name the teacher model or describe test verification.
  
  — [HF card](https://huggingface.co/datasets/IFM/Code-Reasoning)
- `nvidia/Nemotron-Pretraining-Code-v3` (2026-05, CC-BY-4.0): pretraining code, not competitive-programming SFT. — [HF API](https://huggingface.co/api/datasets/nvidia/Nemotron-Pretraining-Code-v3)
- Many 2026 community re-packagings exist (e.g., prompt-only or embedding versions of Nemotron-CP-v2, OCR "cleaned GRPO" variants). Their provenance and licences vary. — [HF search API](https://huggingface.co/api/datasets?search=competitive%20programming&sort=createdAt&direction=-1)

### Inferences
- For a Python stdin/stdout 1.5B student, the cleanest large SFT pools are the **Python subsets** of the following. All are CC-BY-4.0 (Codeforces part of v2 is ODC-By) and all re-use TACO/APPS/CodeContests/Codeforces questions.
  - Nemotron-Competitive-Programming-v1 (1.82M Python, R1-0528)
  - OCR-2 (1.4M Python, R1, with QwQ correct/incorrect judgements usable as a filter)
  - OCR (735K, R1)
- rStar-Coder is the best-verified (test-filtered QwQ-32B traces) and has a published 1.5B result (LCB v5 40.1) from full SFT on the full corpus. The team's 12K-sample LoRA run is roughly two orders of magnitude smaller than what produced rStar-Coder-1.5B. That suggests under-training rather than a data-format failure.
- NVIDIA found that execution-filtering *hurt* SFT. rStar-Coder filters by tests. For a 1.5B model that "gets lost" in long traces, filtering for *correct and shorter* traces (e.g., the OCR-2 `judgement` field, rStar-Coder `is_passed`, codeforces-cots `solutions_short_and_long`) is a plausible ablation axis. This is a hypothesis, not a reported finding.
- Licence cautions:
  - KodCode is non-commercial.
  - Mixture-of-Thoughts, verifiable-coding-problems(-python), Skywork-OR1-RL-Data, HardTests and SYNTHETIC-2-SFT-verified carry no HF licence tag.
  - NVIDIA cards stress that "underlying datasets are available from the original sources subject to their own license terms."
- For RL with trustworthy rewards, prefer sets that have checkers and strong tests: CodeContests+, open-r1/codeforces `verifiable`, HardTests, TACO-verified, DeepCoder. Avoid relying on raw scraped tests, which are truncated and lack checkers.

### Gaps
- Exact per-config row counts for rStar-Coder (the datasets-server conversion is partial) and the number of test cases per problem were not obtained.
- I did not verify the teacher model or verification procedure for IFM/Code-Reasoning, the 2026 VibeThinker training data, or the Nemotron-Post-Training-Dataset-v2 code subset (gated README).
- The licences of the untagged datasets above were not resolved, nor was the 26.6K vs 47,136 problem-count discrepancy in HardTests.
- How PrimeIntellect's verifiable-coding-problems tests were produced or validated (copied from the source datasets or LLM-generated) is not documented in the cards I read.

---

## 2. Datasets with algorithm tags and difficulty ratings for a topic/difficulty router

### Takeaway
The best supervision for an "algorithm-topic + difficulty" router is:
- **`open-r1/codeforces`**: about 10K problems with Codeforces `rating` and `tags`, dated through 2025, CC-BY-4.0, plus a 468-problem test split.
- **`deepmind/code_contests`**: `cf_rating` and `cf_tags`.
- **`BAAI/TACO`**: `tags`, `skill_types`, `difficulty`, expected complexity.
- **HardTests**: `difficulty_ratings` and `tags` across 13 judges.

For a "thinking-budget" router, Skywork-OR1's per-model difficulty (0–16 measured *on DeepSeek-R1-Distill-Qwen-1.5B*) and SYNTHETIC-2's multi-model traces give model-relative difficulty.

### Cited Findings
- **open-r1/codeforces** fields: `rating`, `tags`, `contest_type`, `contest_start_year`, `time_limit`, `memory_limit`, `editorial`, `generated_checker`, `generated_tests`. Splits are 9,556 train / 468 test. Licence CC-BY-4.0. — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=open-r1%2Fcodeforces); [HF card](https://huggingface.co/datasets/open-r1/codeforces)
- **CodeContests** has `cf_rating`, `cf_tags`, `difficulty`, `cf_contest_id`, `cf_index` (CC-BY-4.0). — [HF card](https://huggingface.co/datasets/deepmind/code_contests)
- **TACO**: "each TACO problem includes several fine-grained labels such as task topics, algorithms, programming skills, and difficulty levels". Fields: `tags`, `raw_tags`, `skill_types`, `difficulty`, `Expected Time Complexity`. Licence Apache-2.0. — [arXiv](https://arxiv.org/abs/2312.14852); [HF card](https://huggingface.co/datasets/BAAI/TACO)
- **TACO-verified** keeps the same label fields on 12,898 problems. — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=likaixin%2FTACO-verified)
- **HardTests**: `difficulty_ratings`, `tags`, `platform`, `contest_date` on 47,136 problems. — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=sigcp%2Fhardtests_problems)
- **CodeElo** (eval only): 408 problems with `rating`, `tags`, `div`. CodeElo compiles "contest divisions, problem difficulty ratings, and problem algorithm tags". — [HF dataset](https://huggingface.co/datasets/Qwen/CodeElo); [arXiv](https://arxiv.org/abs/2501.01257)
- **LiveCodeBench Pro**: "A team of Olympiad medalists annotates every problem for algorithmic categories". The taxonomy includes categories with counts, e.g., Greedy 163 (28%), Dynamic Programming 134 (23%), Number Theory 77, Combinatorics 62, Segment Tree 38, Tree 53, each with a "cognitive focus" (Logic, Observation, Knowledge). — [arXiv HTML](https://arxiv.org/html/2506.11928)
- **LeetCodeDataset**: `difficulty` and `tags` on 2,869 function-style problems; "rich metadata—including difficulty levels, release dates, and topic tags". — [arXiv HTML](https://arxiv.org/html/2504.14655)
- **AetherCode**: `difficulty`, `contest_category` (IOI/ICPC-level, eval only). — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=m-a-p%2FAetherCode)
- **Skywork-OR1-RL-Data** code (14,057): difficulty 0–16 relative to DeepSeek-R1-Distill-Qwen-1.5B/7B/32B. — [HF card](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data)
- **SYNTHETIC-2**: "multiple reasoning traces from smaller models which can be used for difficulty estimation." — [HF card](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2)
- **KodCode**: `gpt_difficulty`, `gpt_pass_percentage` (CC-BY-NC). — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=KodCode%2FKodCode-V1)
- **OCR / OCR-2** carry a per-question `difficulty` label inherited from the source dataset. — [OCR card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
- A 2026 community dataset, `AlexSA45/codeforces-tags-rating-processed`, has 7,585 / 975 / 985 train/val/test splits with `tags_kept` and `rating`. It is aimed at tag and rating prediction, has no licence tag and an empty card. — [datasets-server](https://datasets-server.huggingface.co/info?dataset=AlexSA45%2Fcodeforces-tags-rating-processed)

### Inferences
- Codeforces tags are multi-label and noisy (e.g., "implementation", "brute force"). A router should probably predict a reduced label set, mapped onto the team's ~7 blueprint families, and use the `rating` as a regression or bucket target.
- TACO's labels span many platforms, whereas Codeforces `tags` come from a single platform taxonomy. Merging them requires a label mapping.
- For a thinking-budget router, Skywork-OR1's 0–16 pass-count difficulty is directly measured on the same base model (R1-Distill-Qwen-1.5B), which is exactly the student. The same signal can be produced for any problem set by sampling the student k times.
- Router training data must be deduplicated against every evaluation set, just like SFT data. The router sees problem statements, so leakage would inflate retrieval and routing accuracy.

### Gaps
- No published work found that trains a small topic or difficulty router for competitive programming and reports its accuracy.
- Label-noise statistics for Codeforces tags were not found.

---

## 3. Benchmarks: which are standard for ≤2B reasoning models, and what leading small models score

### Takeaway
For ≤2B reasoning models, the de-facto standard is **LiveCodeBench code generation** with an explicit date window:
- v5 "2408–2502" (279 problems), used by DeepSeek-R1, DeepCoder and rStar-Coder;
- v6 "2408–2505", used by OpenReasoning-Nemotron and VibeThinker.

It is usually paired with a Codeforces rating or percentile (CodeElo-style) and HumanEval+/MBPP+ as secondary signals. LiveCodeBench Pro, OJBench, HLCE and AetherCode are frontier-level (IOI/ICPC) and largely saturate at zero for small models.

Leading ≤2B published LCB scores:
- VibeThinker-1.5B: 51.1 (v6)
- rStar-Coder-1.5B: 40.1 (v5)
- OpenReasoning-Nemotron-1.5B: 28.6 (v6 2408–2505)
- DeepCoder-1.5B: 25.1 (v5)
- R1-Distill-Qwen-1.5B: 16.9

### Cited Findings
**LiveCodeBench versions and windows**
- Release sizes:
  - `release_v1`: 400 problems (May 2023–Mar 2024)
  - `release_v2`: 511 (to May 2024)
  - `release_v3`: 612 (to Jul 2024)
  - `release_v4`: 713 (to Sep 2024)
  - `release_v5`: 880 (to Jan 2025)
  - `release_v6`: 1,055 (to Apr 2025)
  
  `--start_date` and `--end_date` select time windows. `code_generation_lite` prunes tests for speed. — [LCB GitHub README](https://github.com/LiveCodeBench/LiveCodeBench)
- The HF card for `livecodebench/code_generation_lite` still lists only through v5. Its licence tag is "cc". — [HF card](https://huggingface.co/datasets/livecodebench/code_generation_lite)
- Window sizes in use:
  - DeepCoder's LCB v5 test window is 8/1/24–2/1/25, with 279 problems. — [DeepCoder dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)
  - OlympicCoder used the `v4_v5` subset (268 problems) via lighteval. — [OlympicCoder-7B card](https://huggingface.co/open-r1/OlympicCoder-7B)
  - OpenReasoning-Nemotron reports "LiveCodeBench version 6, date range 2408-2505". — [model card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)
- LiveCodeBench collects problems "from contests across three competition platforms, namely LeetCode, AtCoder, and CodeForces". — [arXiv](https://arxiv.org/abs/2403.07974)
- LCB maintains an ERRATA file of "erroneous tests and problems not amenable to autograding". — [LCB README](https://github.com/LiveCodeBench/LiveCodeBench)

**Small-model scores (≤2B, plus a 3B reference)**
- DeepSeek-R1-Distill-Qwen-1.5B: LCB 16.9, HumanEval+ 61.8, MBPP+ 48.5. — [rStar-Coder card table](https://huggingface.co/datasets/microsoft/rStar-Coder)
- The same model in another table: apps 20.95, codecontests 16.79, codeforces 14.13, taco 8.03, HumanEval+ 61.77, LCB 16.80. — [Nemotron-Research-Reasoning-Qwen-1.5B card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- DeepCoder-1.5B-Preview (RL from R1-Distill-1.5B): LCB v5 (8/1/24–2/1/25) **25.1**, Codeforces rating **963** (28.5 percentile), HumanEval+ 73.0. — [model card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- rStar-Coder-1.5B: LCB (v5, 2024.8–2025.2) **40.1**, HumanEval+ 81.7, MBPP+ 60.8. — [HF card](https://huggingface.co/datasets/microsoft/rStar-Coder)
- OpenReasoning-Nemotron-1.5B (Qwen2.5-1.5B derivative, released 2025-07-16): LCB v6 2408–2505 **28.6**. — [model card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)
- Nemotron-Research-Reasoning-Qwen-1.5B (ProRL) claims "+13.9% average pass@1 on coding" over R1-Distill-1.5B. — [model card](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)
- VibeThinker-1.5B (Nov 2025; from Qwen2.5-Math-1.5B; about $7,800 training cost): "On LiveCodeBench V6, it scores **51.1**… [vs] its base model's 0.0". — [arXiv 2511.06221](https://arxiv.org/abs/2511.06221)
- Qwen3-1.7B on LCB v6: 26.9, as quoted in VibeThinker's comparison. This is secondhand via a search summary. — [VibeThinker HTML](https://arxiv.org/html/2511.06221)
- VibeThinker-3B (June 2026, 3B so above the ≤2B band): "80.2 Pass@1 on LiveCodeBench v6", "96.1% acceptance rate on recent unseen LeetCode contests". — [arXiv 2606.16140](https://arxiv.org/abs/2606.16140)

**Harder or alternative benchmarks**
- **CodeElo** (Jan 2025): 408 Codeforces problems (`Qwen/CodeElo`, Apache-2.0). Solutions are submitted to the real platform, and Elo is comparable to humans. o1-mini scored 1578 and QwQ-32B-Preview 1261, while others sit "in the lowest 25 percent". DeepCoder used CodeElo problems as its Codeforces test set. — [arXiv](https://arxiv.org/abs/2501.01257); [DeepCoder dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)
- **Elo caveat (2026)**: "When Elo Lies" studied 37 contests and 13,691 generated tests. Submission order shifted scores by 394 points, contest selection by up to 1,122, and run-to-run variation by up to 349. Reports of "the same model version fluctuated by nearly 500 points". — [arXiv 2602.05891](https://arxiv.org/abs/2602.05891)
- **LiveCodeBench Pro** (June 2025): 584 problems before April 25 2025 from Codeforces, ICPC and IOI, captured "before any accepted solutions, editorials, or discussion threads appear online". The best model reached 53% pass@1 on medium problems and 0% on hard, without tools. HF: `QAQAQAQAQ/LiveCodeBench-Pro` (Apache-2.0, gated) and `QAQAQAQAQ/LiveCodeBench-Pro-Testcase`. — [arXiv](https://arxiv.org/abs/2506.11928); [HF API](https://huggingface.co/api/datasets/QAQAQAQAQ/LiveCodeBench-Pro)
- **OJBench** (June 2025): 232 NOI and ICPC problems with official test data. "For most long-CoT models, using CPP as the programming language performs better than Python." — [arXiv](https://arxiv.org/abs/2506.16395); [HTML](https://arxiv.org/html/2506.16395)
- **HLCE** (June 2025): 235 ICPC World Finals and IOI problems from 2010–2024. o4-mini (high) scored 15.9% pass@1 and Gemini 2.5 Pro 11.4%. — [arXiv](https://arxiv.org/abs/2506.12713)
- **AetherCode** (Aug 2025): IOI and ICPC problems with "expert-validated test suites". `m-a-p/AetherCode` (CC-BY-4.0) has configs v1_2024 (400) and v1_2025 (56), with `checker` and `test_cases` fields. — [arXiv](https://arxiv.org/abs/2508.16402); [datasets-server](https://datasets-server.huggingface.co/size?dataset=m-a-p%2FAetherCode)
- **USACO** (Apr 2024): 307 problems; GPT-4 scores 8.7% zero-shot pass@1. rStar-Coder used a newer USACO 2025 set of 12 problems. — [arXiv](https://arxiv.org/abs/2404.10952); [rStar-Coder HTML](https://arxiv.org/html/2505.21297v1)
- **InteractBench** (Aug 2026): 322 interactive problems from Codeforces, AtCoder, IOI and ICPC, with local interactors for offline evaluation. It finds "even the most advanced reasoning models achieve limited success". — [arXiv 2608.29632](https://arxiv.org/abs/2608.29632)
- **EvoCodeBench** (Feb 2026): evaluates self-evolving coding systems against humans across languages. It is not a static pass@1 benchmark. — [arXiv 2602.10171](https://arxiv.org/abs/2602.10171)
- **EvalPlus and BigCodeBench**: `evalplus/humanevalplus`, `evalplus/mbppplus`, `bigcode/bigcodebench` (all Apache-2.0). Small reasoning models already score high on HumanEval+ (rStar-Coder-1.5B 81.7, DeepCoder-1.5B 73.0). — [HF API](https://huggingface.co/api/datasets/evalplus/humanevalplus); [rStar-Coder card](https://huggingface.co/datasets/microsoft/rStar-Coder); [DeepCoder card](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)
- **Split benchmarks from training sets**: CodeContests test (165), TACO test (1,000), APPS test, open-r1/codeforces test (468). — [datasets-server](https://datasets-server.huggingface.co/size?dataset=deepmind%2Fcode_contests); [TACO card](https://huggingface.co/datasets/BAAI/TACO); [open-r1/codeforces](https://datasets-server.huggingface.co/size?dataset=open-r1%2Fcodeforces)

### Inferences
- A publishable small-model setup should report:
  1. LCB with a stated version and date window. v5 2408–2502 is the most comparable to prior 1.5B work; v6 2408–2505 is the newest common window. Use only windows after the training-data cutoff.
  2. A Codeforces-style held-out set with checkers, e.g., the open-r1/codeforces test split.
  3. Optionally HumanEval+/MBPP+ as a sanity check.
  
  Codeforces Elo numbers should only be compared within one standardized protocol.
- LCB Pro hard, OJBench, HLCE and AetherCode will likely give near-zero scores for 1.5B models, which makes them uninformative as primary metrics. LCB Pro "easy" or USACO bronze/silver could serve as stretch evaluations.
- The team's current evaluation (200 problems from verifiable-coding-problems-python) is not a recognised benchmark. It also comes from the same upstream sources as their training data (section 5), so results are not comparable to the literature.

### Gaps
- No LiveCodeBench release beyond v6 was confirmed in the README I read. The date windows reported for "v6" also vary (README says "to Apr 2025", NVIDIA says "2408–2505").
- LiveCodeBench Pro, OJBench, HLCE and AetherCode results for ≤2B models were not found. The LCB Pro Easy/Medium/Hard rating thresholds were not verified in this session.
- I found no documentation of an "open-r1 Codeforces eval" scoring protocol (e.g., a lighteval task) beyond the dataset test split. A search did not surface it.
- Qwen3-1.7B's 26.9 on LCB v6 needs confirmation from a primary table.

---

## 4. Evaluation methodology: pass@k, sampling, checkers, tolerance, time limits, sandboxing, decontamination, statistics, reporting

### Takeaway
The current norm for R1-style distills: sample **n ≥ 8–64** completions per problem at **temperature 0.6, top-p 0.95**, with a **32K-token** budget (Qwen3 recommends about 38,912 tokens for competition problems), and report **pass@1 as the mean over samples** using the unbiased pass@k estimator. Grade with per-problem checkers or tolerance-aware comparison, not exact string match. Run code in an isolated sandbox with explicit time and memory limits. Report the benchmark version and window, the decoding settings and confidence intervals.

### Cited Findings
**Sampling settings**
- DeepSeek-R1 distills: "the maximum generation length is set to 32,768 tokens… temperature of 0.6, a top-p value of 0.95, and generate 64 responses per query to estimate pass@1." Also: "Set the temperature within the range of 0.5-0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs." — [DeepSeek-R1-Distill-Qwen-1.5B card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- Qwen3 thinking mode: "Temperature=0.6, TopP=0.95, TopK=20, and MinP=0… DO NOT use greedy decoding". Recommended output length is 32,768, or 38,912 "for benchmarking on highly complex problems, such as… programming competitions". — [Qwen3-1.7B card](https://huggingface.co/Qwen/Qwen3-1.7B)
- NVIDIA reports "Pass@1 (Avg@64)", "Majority@64" and GenSelect on LCB v6 2408–2505. — [OpenReasoning-Nemotron-1.5B card](https://huggingface.co/nvidia/OpenReasoning-Nemotron-1.5B)
- rStar-Coder uses temperature 0.6 and 32K max output. — [paper HTML](https://arxiv.org/html/2505.21297v1)
- OCR's sweep found temperatures 0.6 and 0.7 best. — [paper HTML](https://arxiv.org/html/2504.01943)
- The LiveCodeBench harness defaults to `n=10` and reports pass@1 and pass@5. — [LCB README](https://github.com/LiveCodeBench/LiveCodeBench)

**pass@k**
- The unbiased estimator from the Codex/HumanEval paper: generate n ≥ k samples, count c correct, and compute pass@k = E[1 − C(n−c,k)/C(n,k)]. The paper shows repeated sampling raises solve rate (70.2% with 100 samples). — [Chen et al. 2021](https://arxiv.org/abs/2107.03374)

**Output checking and float tolerance**
- How LiveCodeBench grades stdin/stdout:
  - It strips and splits outputs into lines and fails if the line counts differ.
  - It compares each line as a string first.
  - Failing that, it converts whitespace-separated tokens to Python `Decimal` and compares "for good floating point comparision".
  
  This is a Decimal equality check, not an epsilon-tolerance check. — [LCB testing_util.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py)
- About 30% of Codeforces problems need a checker because multiple answers are valid. open-r1 generated and validated custom checkers by running real accepted contestant solutions. — [open-r1/codeforces card](https://huggingface.co/datasets/open-r1/codeforces)
- CodeContests+ ships a checker per problem, plus validators and generators. — [HF card](https://huggingface.co/datasets/ByteDance-Seed/Code-Contests-Plus)
- AetherCode includes a `checker` field. — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=m-a-p%2FAetherCode)
- InteractBench ships local interactors for interactive problems. — [arXiv 2608.29632](https://arxiv.org/abs/2608.29632)
- CodeElo lists "lack of support for special judges" as a flaw of LiveCodeBench and USACO, and solves it by submitting to Codeforces directly. — [arXiv](https://arxiv.org/abs/2501.01257)

**Time limits and Python slowness**
- LCB default `timeout=6` (seconds) per test. A global guard of `(timeout + 1) * num_tests + 5` applies. — [compute_code_generation_metrics.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/compute_code_generation_metrics.py)
- "Time limits can cause slight (< 0.5) points of variation"; lower `--num_process_evaluate` or raise `--timeout` if variation is large. — [LCB README](https://github.com/LiveCodeBench/LiveCodeBench)
- TACO-verified used a 10 s timeout on an Intel E5-2620 v3. — [HF card](https://huggingface.co/datasets/likaixin/TACO-verified)
- CodeElo: "for most models, the best-performing language is C++, instead of Python". — [CodeElo HTML](https://arxiv.org/html/2501.01257v2)
- OJBench: C++ beats Python for most long-CoT models. — [OJBench HTML](https://arxiv.org/html/2506.16395)
- open-r1/codeforces stores per-problem `time_limit` and `memory_limit`. The strongest tests are "very large test inputs". — [HF card](https://huggingface.co/datasets/open-r1/codeforces)

**Sandboxing**
- LiveCodeBench executes generated code in-process with `signal.alarm` timeouts, `faulthandler` and a `reliability_guard()` call. These are guards against destructive calls, not OS-level isolation. — [LCB testing_util.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py)
- open-r1 runs code rewards in remote or containerized sandboxes: E2B (Python), Morph (multi-language), and Piston workers for IOI and Codeforces. — [open-r1 README](https://github.com/huggingface/open-r1)
- SYNTHETIC-1's verifier is "Containerized execution of unit tests". — [HF card](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1)
- HLCE built "a harmonized online-offline sandbox that guarantees fully reproducible evaluation". — [arXiv](https://arxiv.org/abs/2506.12713)

**Decontamination methods in use**
- open-r1 `scripts/decontaminate.py`: word-level n-gram overlap after normalization, default `--ngram_size 8`, run against eval sets including LCB `v4_v5`. — [decontaminate.py](https://github.com/huggingface/open-r1/blob/main/scripts/decontaminate.py)
- rStar-Coder: 16-gram overlap against HumanEval(+), MBPP(+), LCB and USACO 2025. — [paper HTML](https://arxiv.org/html/2505.21297v1)
- codeforces-cots: 8-gram against AIME24/25, GPQA-D, MATH-500 and LCB. — [HF card](https://huggingface.co/datasets/open-r1/codeforces-cots)
- KodCode stores `benchmark_similarity`, `benchmark_instruction` and `benchmark_task_id` per row, i.e., similarity to the nearest benchmark item. — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=KodCode%2FKodCode-V1)
- LiveCodeBench's own approach is temporal: score only problems released after a model's cutoff. The paper reports results only on problems after August 2023 "to counter contamination in the DeepSeek models". — [LCB README](https://github.com/LiveCodeBench/LiveCodeBench)
- LeetCodeDataset uses a pre/post July 2024 temporal split. — [arXiv](https://arxiv.org/abs/2504.14655)

**Statistics and reporting standards**
- "Adding Error Bars to Evals" treats eval questions as drawn from a super-population. It gives formulas for analysing eval data, "measuring differences between two models" (paired), and "planning an evaluation experiment", i.e., power analysis. — [Miller 2024, arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- "A Sober Look at Progress in LM Reasoning" finds benchmark results "highly sensitive to… decoding parameters, random seeds, prompt formatting, and even hardware and software configurations". RL gains on small benchmarks are often "far below prior claims", and the paper proposes standardized reporting. — [arXiv 2504.07086](https://arxiv.org/abs/2504.07086)
- OpenThoughts bolds every value "within 2 standard errors of the best" (Evalchemy). — [HF card](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)
- Elo reporting needs strict standardization: order, contest selection and run-to-run variance each move scores by hundreds of points. — [arXiv 2602.05891](https://arxiv.org/abs/2602.05891)

### Inferences
**Concrete protocol for the team:**
1. **Sampling**: n = 16 (or more) samples per problem, T = 0.6, top-p = 0.95, max_new_tokens ≥ 16K (32K preferred, per the base model card). Report pass@1 as the mean over n, plus pass@k via the unbiased estimator. Avoid a single low-temperature sample: the R1 and Qwen3 cards warn against greedy or low-temperature decoding.
2. **Grading**:
   - Use LCB-style token comparison with Decimal fallback, or better, the problem's checker. Exact string match will wrongly fail roughly 30% of Codeforces problems with multiple valid answers.
   - For float outputs, implement the tolerance the statement specifies (commonly an absolute/relative bound). LCB's Decimal equality alone is stricter than many statements allow.
3. **Time limits**: use each problem's limit times a documented Python multiplier, or a fixed limit such as LCB's 6 s. Run with fixed CPU pinning and low parallelism, and report the choice, because timing noise shifts scores.
4. **Sandbox**: run in containers (Docker, nsjail or firejail) with no network, read-only filesystem and memory caps, or use hosted Piston/E2B. In-process guards like LCB's are not a security boundary.
5. **Statistics**:
   - With 200 problems at 8% accuracy, the 95% CI is about ±3.8 points (normal approximation: √(0.08·0.92/200) ≈ 1.9 points SE). With 100 problems at 11%, it is about ±6.1 points. The current 8% vs 11% results come from different problem sets, so they are not comparable and not significantly different.
   - Compare systems **on the same problems** (paired). Use McNemar's test for single-sample correctness, or paired bootstrap / Wilcoxon on per-problem pass rates when n > 1. Report bootstrap CIs resampling problems.
   - Unpaired detection of a 10% → 15% improvement at 80% power needs about 680 problems per arm. Pairing and multiple samples per problem reduce this substantially, so the test sets should grow to several hundred problems (e.g., full LCB v5/v6 windows plus the open-r1/codeforces test split).
6. **Reporting**: state the benchmark version and window, problem count, n, temperature, top-p, max tokens, time limit, checker policy, sandbox, decontamination method and seeds. Release raw generations.

### Gaps
- I found no authoritative per-benchmark standard for a Python time-limit multiplier.
- I did not verify which float tolerance each benchmark's official judge uses (beyond LCB's Decimal logic).
- I found no 2026 paper giving formal significance-testing recommendations specific to code benchmarks beyond Miller (2024) and the "Sober Look" paper (2025).

---

## 5. Known overlap between rStar-Coder, verifiable-coding-problems, CodeContests, TACO, APPS and LiveCodeBench, and how to decontaminate

### Takeaway
Overlap is structural, not incidental:
- rStar-Coder's seeds come from TACO, APPS, CodeContests (plus Python submissions), open-r1 Codeforces and USACO.
- open-r1/verifiable-coding-problems-python is a Python subset of PrimeIntellect's 144K problems, which come from APPS, CodeContests, Codeforces and TACO.
- TACO itself integrates APPS and CodeContests.

rStar-Coder was decontaminated only against HumanEval/MBPP(+), LiveCodeBench and USACO 2025. Its synthetic problems are seed-derived. So a test set drawn from verifiable-coding-problems almost certainly shares problems, or near-paraphrases, with the rStar-Coder training data. LiveCodeBench windows after the data cutoff are the cleanest option.

### Cited Findings
- **rStar-Coder seed sources**: TACO, APPS, CodeContests, CodeContests-Python-Submission, OpenR1 Codeforces, USACO 2011–2023. — [rStar-Coder HTML](https://arxiv.org/html/2505.21297v1)
- **rStar-Coder decontamination scope**: 16-gram against "HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench, and USACO 2025" only. — [rStar-Coder HTML](https://arxiv.org/html/2505.21297v1)
- **Seed derivation**: synthetic_sft and synthetic_rl rows carry `seed_question` and `seed_source`, i.e., each synthetic problem is generated from a seed problem. — [HF card](https://huggingface.co/datasets/microsoft/rStar-Coder)
- **verifiable-coding-problems-python lineage**: it contains all Python problems of PrimeIntellect/verifiable-coding-problems, whose coding tasks are "curated from Apps, Codecontests, Codeforces and TACO". — [open-r1 card](https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python); [SYNTHETIC-1 card](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1)
- Each verifiable-coding-problems row keeps `source` and `in_source_id`, which allow ID-level joins back to the origin dataset. — [datasets-server info](https://datasets-server.huggingface.co/info?dataset=open-r1%2Fverifiable-coding-problems-python)
- **TACO / APPS / CodeContests overlap**: TACO "integrated existing datasets, including APPS, CodeContest, and Description2code". "There are 3,392 duplicate problems shared between APPS and CodeContest." The overlap coefficient between TACO and CodeContests is 0.4617, and between TACO and APPS 0.3778. — [TACO paper HTML](https://arxiv.org/html/2312.14852)
- **NVIDIA sets**: OCR, OCR-2 and Nemotron-CP questions come from TACO, APPS, CodeContests and open-r1/codeforces. They exclude the CodeContests and open-r1/codeforces *test* splits, but not explicitly the TACO or APPS test splits. — [OCR card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning); [OCR-2 card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2)
- **DeepCoder and LCB**: DeepCoder's training set *includes* 599 LCB v5 problems from May 2023–Jul 2024, and it tests on Aug 2024–Feb 2025. Any evaluation on earlier LCB windows would be contaminated for models trained on it. — [DeepCoder dataset](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset)
- **Same platforms**: rStar-Coder's per-platform table (AIZU, AtCoder, CodeChef, CodeWars, GeeksForGeeks, HackerEarth, HackerRank…) mirrors OCR's source table (AIZU, AtCoder, CodeChef, Codeforces, Codewars, GeeksForGeeks, HackerEarth, HackerRank, Kattis, LeetCode). — [rStar-Coder HTML](https://arxiv.org/html/2505.21297v1); [OCR card](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)
- **LCB contamination evidence**: LiveCodeBench reports "empirical findings on contamination" and restricts DeepSeek results to post-Aug-2023 problems. Its problems come from LeetCode, AtCoder and Codeforces, the same platforms TACO, APPS and CodeContests scraped, but LCB's value lies in its later release dates. — [LCB arXiv](https://arxiv.org/abs/2403.07974); [LCB README](https://github.com/LiveCodeBench/LiveCodeBench)
- **Tools for decontamination**:
  - open-r1's 8-gram script — [decontaminate.py](https://github.com/huggingface/open-r1/blob/main/scripts/decontaminate.py)
  - rStar-Coder's 16-gram filter — [HTML](https://arxiv.org/html/2505.21297v1)
  - KodCode's per-row `benchmark_similarity` — [info](https://datasets-server.huggingface.co/info?dataset=KodCode%2FKodCode-V1)
  - open-r1's already-decontaminated `verifiable-coding-problems-python_decontaminated(-tested)` variants (15,068 rows in the tested one) — [datasets-server](https://datasets-server.huggingface.co/size?dataset=open-r1%2Fverifiable-coding-problems-python_decontaminated-tested)

### Inferences
**Recommended decontamination pipeline**, from cheapest to most thorough:
1. **ID/URL joins**: map every training and test problem to (platform, contest_id, index) or URL. Fields to use: `source`/`in_source_id` in verifiable-coding-problems; `seed_source`/`question_id` in rStar-Coder; `url`/`source` in TACO; `cf_contest_id`/`cf_index` in CodeContests; `contest_id`/`index` in open-r1/codeforces. Then drop exact matches.
2. **n-gram overlap** on normalized statements: 8-gram as in open-r1, or 13- to 16-gram as in rStar-Coder. Apply it to rStar-Coder seed *and* synthetic problems (`seed_question` too), because synthetic variants paraphrase seeds.
3. **Embedding similarity**: e.g., cosine on statement embeddings with a manually calibrated threshold, plus LLM-judge review of borderline pairs. This catches translations (AIZU/AtCoder Japanese → English) and paraphrases that n-grams miss.
4. **Temporal holdout**: evaluate on LCB v5/v6 windows, or on Codeforces contests after the latest training-data date. For rStar-Coder, OCR and Nemotron-CP, the Codeforces pools run "all the way to 2025", so use problems from after mid-2025, e.g., LCB Pro or newer Codeforces rounds.

Report how many test items each stage removed.

**Implication for the current paper**: the 200-problem test set from open-r1/verifiable-coding-problems-python should be treated as potentially contaminated by rStar-Coder unless steps 1–3 show otherwise. The headline evaluation should move to LCB (stated window) plus a decontaminated Codeforces test split with checkers.

### Gaps
- I found no published quantitative overlap numbers between rStar-Coder (seed or synthetic) and PrimeIntellect/open-r1 verifiable-coding-problems, or between either and LiveCodeBench. This needs to be measured directly.
- What open-r1's "decontaminated-tested" variant was decontaminated against is undocumented (the card is empty).
- NVIDIA's OCR contamination check method (n-gram vs embedding) is mentioned but its details were not captured.
