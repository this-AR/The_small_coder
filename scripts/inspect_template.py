"""Print the rendered chat prompt for a model so think tags in the config can be checked.

  python scripts/inspect_template.py WeiboAI/VibeThinker-1.5B
"""
import sys

from transformers import AutoTokenizer

from smallcoder.problems import Problem, Test
from smallcoder.prompts import chat_prompt

tok = AutoTokenizer.from_pretrained(sys.argv[1])
p = Problem(pid="x", source="x", statement="Read two integers and print their sum.", tests=[Test("1 2", "3")])
s = chat_prompt(tok, p)
print(repr(s[-300:]))
print("ends inside <think>:", s.rstrip().endswith("<think>"))
print("special tokens containing 'think':", [t for t in tok.get_vocab() if "think" in t.lower()][:20])
