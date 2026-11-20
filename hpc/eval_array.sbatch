#!/bin/bash
# Generation + judging, sharded over a SLURM array. Edit partition/account/modules for your cluster.
#   sbatch --array=0-7 hpc/eval_array.sbatch configs/eval_vibethinker.yaml
#SBATCH --job-name=sc-eval
#SBATCH --partition=gpu            # <- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32         # the judge runs programs on these cores
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
# module load cuda/12.4            # <- if your cluster uses modules
source .venv/bin/activate
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
python scripts/run_eval.py "$1" --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NSHARDS"
# merge afterwards: cat results/<dir>/<name>.shard*.jsonl > results/<dir>/<name>.jsonl
