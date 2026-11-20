"""Repair RL with TRL's GRPOTrainer and an execution reward with partial credit.

Approximates CoCoS for one repair turn: reward = fraction of tests passed + 1.0 bonus for passing all
(CoCoS also accumulates reward over a multi-turn trajectory, which this single-turn version does not).
Written against the TRL >= 0.14 GRPO API (reward functions receive dataset columns as kwargs);
check argument names against the TRL version installed on the cluster.

  accelerate launch scripts/repair_rl.py --model models/vibe_repair_sft --data data/repair/rl.jsonl \
      --problems data/train/labels_pool.jsonl --out models/vibe_repair_rl
"""
import argparse
import json

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from smallcoder.extract import extract_code, split_thinking
from smallcoder.judge import judge
from smallcoder.problems import load_problems

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True, help="repair_mine_pairs.py output (prompts are reused, targets ignored)")
ap.add_argument("--problems", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--think-start", default="<think>", help="'' for non-thinking models")
ap.add_argument("--think-end", default="</think>")
ap.add_argument("--max-completion", type=int, default=2048)
ap.add_argument("--num-generations", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--max-steps", type=int, default=500)
ap.add_argument("--use-vllm", action="store_true")
a = ap.parse_args()

probs = {p.pid: p for p in load_problems(a.problems)}
rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
tok = AutoTokenizer.from_pretrained(a.model)


def _prompt(messages):
    """Plain-text prompt in exactly the inference format of repair_eval.py (generate_budgeted at budget 0):
    chat template, open thinking block, empty thinking. The policy then writes only the program."""
    s = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if not a.think_start:
        return s
    if not s.rstrip().endswith(a.think_start):
        s += a.think_start + "\n"
    return s + "\n" + a.think_end + "\n\n"


ds = Dataset.from_list([{"prompt": _prompt(r["messages"][:-1]), "pid": r["pid"]} for r in rows if r["pid"] in probs])


def _text(c):
    return c if isinstance(c, str) else c[-1]["content"]


def exec_reward(prompts, completions, pid, **kwargs):
    out = []
    for c, p in zip(completions, pid):
        _, ans = split_thinking(_text(c), a.think_end)
        code = extract_code(ans or _text(c))
        res = judge(code, probs[p], stop_on_fail=False, time_multiplier=2.0)
        frac = res.passed / res.total if res.total else 0.0
        out.append(frac + (1.0 if res.verdict == "AC" else 0.0))
    return out


cfg = GRPOConfig(output_dir=a.out, learning_rate=a.lr, num_generations=a.num_generations,
                 max_completion_length=a.max_completion, max_steps=a.max_steps, bf16=True,
                 per_device_train_batch_size=a.num_generations, gradient_accumulation_steps=4,
                 temperature=1.0, logging_steps=5, save_steps=100, use_vllm=a.use_vllm, report_to="none")
GRPOTrainer(model=a.model, reward_funcs=[exec_reward], args=cfg, train_dataset=ds).train()
