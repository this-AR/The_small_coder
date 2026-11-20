"""Prompt construction. One builder for every experiment so conditions differ only in `context`."""
from __future__ import annotations

from .problems import Problem

STDIN_INSTR = ("Write a complete Python 3 program that reads from standard input and writes to standard "
               "output. Put the final program in a single ```python code block at the end of your answer.")
FILE_INSTR = ("Write a complete Python 3 program that reads from input.txt and writes to output.txt. "
              "Put the final program in a single ```python code block at the end of your answer.")
FUNC_INSTR = ("Complete the function below in Python 3. Put the final code, including the class/function "
              "signature, in a single ```python code block at the end of your answer.")

# Track C "algorithm sketch" output format (formerly "blueprint").
SKETCH_INSTR = ("Before the code, write a short algorithm sketch with these fields: [IDEA] one or two sentences; "
                "[COMPLEXITY] time and memory; [CORNER CASES] a short list; [PSEUDOCODE] at most 15 lines. "
                "Then give the final program.")


def user_message(p: Problem, context: str | None = None, style: str = "default",
                 context_header: str = "Guidance") -> str:
    parts = [p.statement.strip()]
    if p.mode == "functional":
        parts.append(FUNC_INSTR + ("\n\n```python\n" + p.starter_code.strip() + "\n```" if p.starter_code else
                                   f"\n\nThe function name is `{p.fn_name}`."))
    else:
        parts.append(FILE_INSTR if p.mode == "file" else STDIN_INSTR)
    if p.time_limit and p.mode != "functional":
        parts.append(f"Time limit: {p.time_limit:g} s per test. Memory limit: {p.memory_limit_mb} MB.")
    if style == "sketch":
        parts.append(SKETCH_INSTR)
    if context:
        parts.insert(1, f"### {context_header}\n{context.strip()}")
    return "\n\n".join(parts)


def chat_prompt(tokenizer, p: Problem, context: str | None = None, style: str = "default",
                system: str | None = None, context_header: str = "Guidance") -> str:
    """Render with the model's own chat template (identical in training and evaluation)."""
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": user_message(p, context, style, context_header)})
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
