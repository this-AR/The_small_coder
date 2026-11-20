"""Fine-tune the small dense retriever on (statement -> solution document of the same problem) pairs,
with in-batch negatives. Statement text is stripped from the positive so the model must learn
problem -> idea/code similarity, not string matching.

  python scripts/rag_finetune_retriever.py --pool data/train/pool.jsonl --base Qwen/Qwen3-Embedding-0.6B \
      --out models/retriever_ft
"""
import argparse

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

from smallcoder.problems import load_problems
from smallcoder.rag import make_doc

ap = argparse.ArgumentParser()
ap.add_argument("--pool", nargs="+", required=True)
ap.add_argument("--base", default="Qwen/Qwen3-Embedding-0.6B")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--batch-size", type=int, default=64)
a = ap.parse_args()

pairs = {"anchor": [], "positive": []}
for f in a.pool:
    for p in load_problems(f):
        d = make_doc(p)
        if d:
            pos = d["text"].split("### Solution idea", 1)[-1] if "### Solution idea" in d["text"] else \
                d["text"].split("### Accepted Python solution", 1)[-1]
            pairs["anchor"].append(p.statement[:2000])
            pairs["positive"].append(pos)
print(f"{len(pairs['anchor'])} pairs")
model = SentenceTransformer(a.base)
args = SentenceTransformerTrainingArguments(output_dir=a.out, num_train_epochs=a.epochs,
                                            per_device_train_batch_size=a.batch_size, learning_rate=2e-5,
                                            warmup_ratio=0.05, bf16=True, save_strategy="epoch",
                                            batch_sampler="no_duplicates")
SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_dict(pairs),
                           loss=MultipleNegativesRankingLoss(model)).train()
model.save(a.out)
