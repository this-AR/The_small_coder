# Runbook: running the plan on the HPC cluster

This is the code for `reports/Chosen plans - datasets, sources and project blueprint.md`. Every step lists the command, what it writes, and the decision gate after it. The old notebooks stay untouched; the new code lives in `smallcoder/`, `scripts/`, `configs/`, `phone/`, `quant/` and `hpc/`.

## Branches: one idea per branch

| Branch | Idea | Contents | Built on |
|---|---|---|---|
| `core/eval-harness` | Track 0: fixed evaluation | loaders, judge, generation + budget forcing, harness, stats, decontamination, SFT script, HPC templates, research notes | `main` |
| `idea/hint-ladder` | Track A1: hint granularity | `ladder.py`, hint generation, Figure 1 report | core |
| `idea/system-one-controller` | Track A2–A4: controller + gated retrieval | labels, policy, controller heads, option logits, draft agreement, RAG | core |
| `idea/verdict-repair` | Track B: verdict-aware repair | `repair.py`, pair mining, repair vs resample, GRPO | core |
| `idea/quant-robustness` | Track C: quantization | GGUF build, llama-server eval | core |
| `integration/on-device` | the phone system | merges the controller, repair and quantization branches, plus `phone/` | all three |

Each idea branch only adds files, so idea branches merge into one another without conflicts. Steps below name the branch they need. A step that reads another idea's *outputs* (for example, repair mining reads the label-generation rows) needs only those files on the cluster, not the other branch.

## 0. Setup (once)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu,train,rag,test]"         # vLLM + transformers + TRL + BM25/sentence-transformers
pytest -q                                      # CPU-only unit tests, no downloads (run on a login node)
mkdir -p logs data results models
```

**Security note.** `smallcoder.judge` runs model-written code in a resource-limited subprocess. That is **not** a sandbox. Run judging jobs inside a container (Apptainer/Singularity), with no network and a scratch-only writable filesystem.

**Check the think tags once per model.** VibeThinker's tags were not verified.
```bash
python scripts/inspect_template.py WeiboAI/VibeThinker-1.5B
python scripts/inspect_template.py deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```
If a template does not use `<think>`/`</think>`, edit `think_start`/`think_end` in its config.

## Track 0: evaluation sets and baselines

```bash
huggingface-cli download livecodebench/code_generation_lite --repo-type dataset --local-dir $SCRATCH/lcb
huggingface-cli download open-r1/codeforces --repo-type dataset --include 'generated_tests/*.parquet' --local-dir $SCRATCH/cf
python scripts/build_eval_sets.py --lcb-dir $SCRATCH/lcb --cf-tests-dir $SCRATCH/cf/generated_tests \
    --team-suite test_suite_200_FINAL.jsonl --out data/eval
sbatch --array=0-7 hpc/eval_array.sbatch configs/eval_vibethinker.yaml
sbatch --array=0-7 hpc/eval_array.sbatch configs/eval_r1distill.yaml      # copy for old Stage1/2/M3 checkpoints
# after EVERY sharded run, merge into the config's `out` name (later steps read the merged file):
cat results/track0/vibethinker_lcb_v5.shard*.jsonl > results/track0/vibethinker_lcb_v5.jsonl
cat results/track0/r1distill_lcb_v5.shard*.jsonl  > results/track0/r1distill_lcb_v5.jsonl
python scripts/summarize.py results/track0/*_lcb_v5.jsonl --compare r1distill-1.5b:base vibethinker-1.5b:base
```

**Gate (week 2).** VibeThinker should land near its card at 8K tokens; it will be lower than the card's 40K-budget number. If it is far off, debug the harness first: extraction rate, checkers, time multiplier.

## Training pool (shared by Tracks A and B)

```bash
python scripts/build_train_pool.py --eval data/eval/*.jsonl --cf-tests-dir $SCRATCH/cf/generated_tests \
    --out data/train/pool.jsonl --embed-check
python scripts/sample_problems.py data/train/pool.jsonl 10000 data/train/labels_pool.jsonl
```

Read `data/train/pool.decontam_report.json`. It holds removal counts per stage for the paper, plus the embedding-flagged pairs to review by hand.

## Track A1: hint ladder

```bash
python scripts/ladder_make_hints.py configs/ladder_hints.yaml          # needs 2 GPUs for Qwen3-32B
sbatch --array=0-3 hpc/eval_array.sbatch configs/eval_ladder_vibethinker.yaml
cat results/ladder/vibethinker.shard*.jsonl > results/ladder/vibethinker.jsonl
python scripts/ladder_report.py results/ladder/vibethinker.jsonl --problems data/ladder/problems.jsonl --out results/ladder/fig1.csv
```

**Gate (week 4).** If even `editorial` barely beats `none`, the coder is implementation-bound. Shift weight to Track B.

## Track A4: retrieval index (needed before labels)

```bash
python scripts/rag_finetune_retriever.py --pool data/train/pool.jsonl --out models/retriever_ft     # optional
python scripts/rag_build_index.py --pool data/train/pool.jsonl --embedder models/retriever_ft --out data/rag/index
python scripts/rag_retrieve.py configs/eval_vibethinker.yaml data/train/labels_pool.jsonl --index data/rag/index --out data/rag/train_contexts.jsonl
python scripts/rag_retrieve.py configs/eval_vibethinker.yaml data/eval/lcb_v5_2408_2502.jsonl --index data/rag/index --out data/rag/lcb_contexts.jsonl
sbatch --array=0-3 hpc/eval_array.sbatch configs/eval_vibethinker_rag.yaml       # rag vs rag_random on the test set
cat results/track0/vibethinker_lcb_v5_rag.shard*.jsonl > results/track0/vibethinker_lcb_v5_rag.jsonl
python scripts/summarize.py results/track0/vibethinker_lcb_v5.jsonl results/track0/vibethinker_lcb_v5_rag.jsonl \
    --compare vibethinker-1.5b:rag_random vibethinker-1.5b:rag
```

## Track A2–A3: labels, controller, evaluation

```bash
sbatch --array=0-15 hpc/eval_array.sbatch configs/labels_vibethinker.yaml        # the big generation job
python scripts/labels_build.py results/labels/vibethinker_train.shard*.jsonl --rag-contexts data/rag/train_contexts.jsonl \
    --out data/labels/vibethinker_train.json
# test labels = Track 0 base rows + the RAG rows (name the merged files; read_rows also drops duplicate rows)
python scripts/labels_build.py results/track0/vibethinker_lcb_v5.jsonl results/track0/vibethinker_lcb_v5_rag.jsonl \
    --rag-contexts data/rag/lcb_contexts.jsonl --out data/labels/vibethinker_lcb.json
python scripts/controller_features.py configs/eval_vibethinker.yaml data/train/labels_pool.jsonl --out data/features/vibe_train.npz
python scripts/controller_features.py configs/eval_vibethinker.yaml data/eval/lcb_v5_2408_2502.jsonl --out data/features/vibe_lcb.npz --option-probs
python scripts/controller_train.py --features data/features/vibe_train.npz --labels data/labels/vibethinker_train.json \
    --problems data/train/labels_pool.jsonl --out models/controller_vibethinker        # --hidden 0 = linear probe ablation
python scripts/controller_eval.py --model models/controller_vibethinker --features data/features/vibe_lcb.npz \
    --labels data/labels/vibethinker_lcb.json --rows results/track0/vibethinker_lcb_v5.jsonl --out results/controller/vibe_lcb
```

**How the comparison works.** `controller_eval.py` scores every arm exactly in expectation from the measured test-set pass rates: the controller, option logits, the draft-agreement heuristic, fixed budgets and the oracle. For the paper, also run the **official DART code** (github.com/js-lee-AI/DART) as the training-free baseline. `selfroute.draft_agreement` is only a DART-inspired heuristic.

**Gate (week 9).** If the controller does not beat DART at matched tokens, report it. Calibration (ECE, risk–coverage) and the oracle gap then become the contribution.

## Track B: repair

```bash
python scripts/repair_mine_pairs.py results/labels/vibethinker_train.shard*.jsonl --problems data/train/labels_pool.jsonl \
    --config configs/eval_vibethinker.yaml --out data/repair/sft.jsonl
sbatch hpc/train.sbatch scripts/sft_train.py --model WeiboAI/VibeThinker-1.5B --data data/repair/sft.jsonl --out models/vibe_repair_sft
sbatch hpc/train.sbatch scripts/repair_rl.py --model models/vibe_repair_sft --data data/repair/sft.jsonl \
    --problems data/train/labels_pool.jsonl --out models/vibe_repair_rl --use-vllm
python scripts/repair_eval.py configs/repair_eval.yaml        # also run once with the untouched base (zero-shot repair)
```

## Track C: quantization

```bash
LLAMA_CPP=~/llama.cpp bash quant/build_gguf.sh WeiboAI/VibeThinker-1.5B models/gguf/vibe
for T in f16 q8_0 q4_k_m q3_k_m; do LLAMA_CPP=~/llama.cpp bash quant/serve_and_eval.sh models/gguf/vibe-$T.gguf configs/quant_vibe.yaml $T; done
python scripts/summarize.py results/quant/vibe_*.jsonl
```

The `f16` GGUF run is the reference, so precision is the only thing that changes.

## Phone

```bash
python phone/export_bundle.py configs/eval_vibethinker.yaml models/controller_vibethinker phone_bundle/
python phone/parity_check.py WeiboAI/VibeThinker-1.5B models/gguf/vibe-f16.gguf phone_bundle/ data/eval/cf_test.jsonl   # expect cos > 0.99
# on the phone (Termux), from a copy of this repo (the phone path needs no torch, scipy or transformers):
#   pkg install python python-numpy clang cmake; pip install llama-cpp-python rank_bm25; pip install -e . --no-deps
python phone/bench.py --gguf vibe-q4_k_m.gguf --bundle phone_bundle/ --problems cf_test_small.jsonl --out bench.jsonl
```

## Known limits (read before reporting numbers)

- **Unverified on real data or GPUs.** This code was written and statically checked off-cluster. Run `pytest -q`, then a 20-problem smoke run, before any large job.
- **Checker scale.** open-r1 checkers print a score on a 0–1 or 0–100 scale, and only full marks count. A 0–100 checker printing exactly `1` would be misread as a pass.
- **Functional (LeetCode) problems.** They follow two conventions: LCB uses one JSON value per input line, TACO uses a JSON argument list. The convention is set per loader in `meta.fn_args`. `team200` is assumed to use TACO-style; spot-check it.
- **Repair RL** is single-turn with partial credit, not CoCoS's multi-turn accumulated reward. Its prompts are plain strings in the exact repair_eval format (template + empty thinking block); check that the TRL version does not add a second BOS for templates that already contain one (R1-distill).
- **Time multiplier.** The train labels use 2.0× (the pool is mostly Codeforces/TACO with C++ limits); the LCB test labels use 1.0× because LCB's 6 s limit is already Python-sized. Keep these fixed across every arm and model, and state them in the paper.
- **Ladder leak filter** counts C/C++-looking lines (`;` at line end, `#include`, `for (;;)`) as code. If `data/ladder/rejects.json` shows many `pseudo:python_lines` rejects, the teacher is writing semicolon pseudocode: raise `max_py_lines` or tighten the prompt. `hint_tokens` there are counted with the teacher's tokenizer, not the coder's.
- **Phone tokenization.** `phone/pipeline.py` forces special-token parsing for `embed()`, and assumes `create_completion` already does this (llama-cpp-python ≥ 0.2.x). `parity_check.py` catches it if not.
- **Phone retrieval** is BM25-only (`shortcut` comment in `phone/pipeline.py`). The phone energy meter reads sysfs and is approximate; validate it once against a USB power meter.
- **Not written yet:**
  - the NanoJev separate-model arm (its data format is unverified);
  - the human failure/fix pairs from `open-r1/codeforces-submissions` (on-policy pairs are used instead);
  - the Track 0b SFT plus on-policy distillation run (`sft_train.py` covers the SFT half; distillation would use TRL's GKD trainer);
  - a native Android (Kotlin) app; the phone path runs in Termux.
