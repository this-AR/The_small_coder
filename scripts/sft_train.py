"""Full-parameter chat SFT with loss on the assistant turn only (repair pairs, or Track 0b traces).
Plain transformers Trainer, so it runs under `accelerate launch` with FSDP/DeepSpeed configs.

  accelerate launch --config_file hpc/accelerate_fsdp.yaml scripts/sft_train.py \
      --model WeiboAI/VibeThinker-1.5B --data data/repair/sft.jsonl --out models/vibe_repair_sft
"""
import argparse
import json

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--batch-size", type=int, default=1)
ap.add_argument("--grad-accum", type=int, default=16)
ap.add_argument("--warmup", type=float, default=0.03)
ap.add_argument("--think-start", default="<think>", help="'' for non-thinking models")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


class ChatSFT(Dataset):
    def __init__(self, path):
        self.items, skipped = [], 0
        for line in open(path, encoding="utf-8"):
            m = json.loads(line)["messages"]
            prompt = tok.apply_chat_template(m[:-1], add_generation_prompt=True, tokenize=False)
            if a.think_start and not prompt.rstrip().endswith(a.think_start):
                prompt += a.think_start + "\n"  # same rule as smallcoder.gen.open_thinking at inference
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            # targets continue an open thinking block: "\n</think>\n\n```python ...```"
            a_ids = tok(m[-1]["content"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if len(p_ids) + len(a_ids) > a.max_len:
                skipped += 1  # never truncate the target: a cut program teaches broken code
                continue
            self.items.append((p_ids, a_ids))
        print(f"{len(self.items)} examples, {skipped} skipped as too long")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, t = self.items[i]
        return {"input_ids": p + t, "labels": [-100] * len(p) + t}


def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), tok.pad_token_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n], lab[i, :n], att[i, :n] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
args = TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                         per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
                         warmup_ratio=a.warmup, lr_scheduler_type="cosine", bf16=True, logging_steps=10,
                         save_strategy="epoch", gradient_checkpointing=True, report_to="none",
                         remove_unused_columns=False)
Trainer(model=model, args=args, train_dataset=ChatSFT(a.data), data_collator=collate).train()
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
