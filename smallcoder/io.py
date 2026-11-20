from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line after a preempted job


ROW_KEY = ("pid", "model", "condition", "budget", "sample")


def read_rows(paths: Iterable[str | Path]) -> list[dict]:
    """Harness rows from several files, keeping the first copy of each (pid, model, condition, budget, sample).
    Guards against a glob that matches both shard files and their merged file."""
    seen, out, dup = set(), [], 0
    for path in paths:
        for r in read_jsonl(path):
            k = tuple(r.get(x) for x in ROW_KEY)
            if k in seen:
                dup += 1
                continue
            seen.add(k)
            out.append(r)
    if dup:
        print(f"read_rows: dropped {dup} duplicate rows (overlapping input files?)")
    return out


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()


def load_contexts(path: str | Path | None, key: str | None = None) -> dict[str, str]:
    """Context files: {"pid": ..., "context": "..."} or {"pid": ..., "contexts": {key: "..."}}."""
    if not path:
        return {}
    out = {}
    for r in read_jsonl(path):
        v = r["contexts"].get(key) if key else r.get("context")
        if v:
            out[r["pid"]] = v
    return out
