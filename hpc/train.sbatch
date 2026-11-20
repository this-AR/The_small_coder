#!/bin/bash
# Multi-GPU training (SFT or GRPO) on one node with accelerate + FSDP.
#   sbatch hpc/train.sbatch scripts/sft_train.py --model WeiboAI/VibeThinker-1.5B --data data/repair/sft.jsonl --out models/vibe_repair_sft
#SBATCH --job-name=sc-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source .venv/bin/activate
accelerate launch --config_file hpc/accelerate_fsdp.yaml --num_processes 8 "$@"
