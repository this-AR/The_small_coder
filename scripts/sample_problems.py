"""Random, seeded subset of a problems file (e.g. 10K problems of the pool for label generation).

  python scripts/sample_problems.py data/train/pool.jsonl 10000 data/train/labels_pool.jsonl
"""
import random
import sys

from smallcoder.problems import load_problems, save_problems

src, k, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ps = load_problems(src)
random.Random(seed).shuffle(ps)
print(save_problems(ps[:k], dst), "->", dst)
