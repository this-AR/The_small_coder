"""Track B: verdict-aware repair.

On a phone only the PUBLIC (sample) tests are available, so repair feedback is built from the
public-test failure. Training pairs may also use hidden-test failures (flag `allow_hidden`).
"""
from __future__ import annotations

from .problems import Problem
from .prompts import user_message

FEEDBACK = {
    "WA": ("Your program printed a wrong answer on this test.\nInput:\n{input}\nExpected output:\n{expected}\n"
           "Your output:\n{got}\nFind the bug: think of the corner case this test exposes, then fix the program."),
    "TLE": ("Your program exceeded the time limit ({tl:g} s) on this test.\nInput:\n{input}\n"
            "Your approach is too slow for the input limits stated in the problem. Use an asymptotically faster "
            "algorithm and fast input (sys.stdin), then rewrite the program."),
    "MLE": ("Your program exceeded the memory limit on this test.\nInput:\n{input}\n"
            "Reduce memory usage (avoid storing large intermediate structures), then rewrite the program."),
    "RE": ("Your program crashed on this test.\nInput:\n{input}\nError:\n{stderr}\n"
           "Fix the cause (index bounds, recursion depth, input parsing, division by zero), then rewrite the program."),
    "CE": ("Your program has a syntax error:\n{stderr}\nFix it and rewrite the program."),
    "NOCODE": ("You did not output a program. Write the complete program in a ```python code block."),
}


def feedback(verdict: str, fail: dict | None, p: Problem, max_chars: int = 1500) -> str:
    f = {k: (v or "")[:max_chars] for k, v in (fail or {}).items()}
    tmpl = FEEDBACK.get(verdict, FEEDBACK["WA"])
    return tmpl.format(input=f.get("input", ""), expected=f.get("expected", ""), got=f.get("got", ""),
                       stderr=f.get("stderr", "")[-max_chars:], tl=p.time_limit)


def repair_messages(p: Problem, code: str | None, verdict: str, fail: dict | None) -> list[dict]:
    prev = f"### Your previous program\n```python\n{code}\n```\n\n" if code else ""
    return [{"role": "user", "content": user_message(p) + "\n\n" + prev + "### Judge feedback\n" +
             feedback(verdict, fail, p)}]
