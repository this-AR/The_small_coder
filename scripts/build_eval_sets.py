"""Build the Track 0 evaluation sets as problems jsonl files.

  python scripts/build_eval_sets.py --lcb-dir $DATA/lcb --cf-tests-dir $DATA/cf_generated_tests \
      --team-suite test_suite_200_FINAL.jsonl --out data/eval
"""
import argparse
from pathlib import Path

from smallcoder.problems import (load_lcb, load_open_r1_codeforces, load_team_suite, save_problems)

LCB_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]  # release_v6

ap = argparse.ArgumentParser()
ap.add_argument("--lcb-dir", required=True, help="local copy of livecodebench/code_generation_lite")
ap.add_argument("--cf-tests-dir", default=None, help="open-r1/codeforces generated_tests/ directory")
ap.add_argument("--team-suite", default=None)
ap.add_argument("--out", default="data/eval")
a = ap.parse_args()
out = Path(a.out)
files = [Path(a.lcb_dir) / f for f in LCB_FILES]

# v5 window used by DeepSeek-R1 / DeepCoder / rStar-Coder (279 problems) and the v6 extension.
print("lcb_v5_2408_2502", save_problems(load_lcb(files, "2024-08-01", "2025-02-01"), out / "lcb_v5_2408_2502.jsonl"))
print("lcb_v6_2502_2505", save_problems(load_lcb(files, "2025-02-01", "2025-05-01"), out / "lcb_v6_2502_2505.jsonl"))
print("cf_test", save_problems(load_open_r1_codeforces("test", "verifiable", a.cf_tests_dir), out / "cf_test.jsonl"))
if a.team_suite:
    print("team200", save_problems(load_team_suite(a.team_suite), out / "team200.jsonl"))
