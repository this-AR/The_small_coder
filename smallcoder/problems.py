"""One Problem record for every source, plus loaders.

Field formats were checked against the HF datasets-server rows on 2026-11-20:
- open-r1/codeforces: examples/official_tests = list[{input, output}], generated tests in
  separate parquet files (problem_id, input, output, test_case_i), generated_checker = python source.
- agentica-org/DeepCoder-Preview-Dataset: `tests` is a JSON string in one of three shapes
  (list of {input, output[, testtype]}, list of {type: stdin_stdout, input, output},
  or TACO-style {inputs: [...], outputs: [...]}).
- likaixin/TACO-verified: input_output JSON {inputs, outputs[, fn_name]}.
- livecodebench/code_generation_lite: jsonl files; private tests are base64(zlib(pickle(json str))).
- the team's test_suite_200_FINAL.jsonl: test_cases list of {input, output, type, fn_name}.
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Test:
    input: str
    output: str


@dataclass
class Problem:
    pid: str                      # globally unique: "<source>/<native id>"
    source: str                   # lcb | codeforces | deepcoder-taco | taco-verified | ...
    statement: str
    tests: list[Test]             # hidden + public tests used for grading
    public_tests: list[Test] = field(default_factory=list)  # shown in the prompt / used on-device
    mode: str = "stdin"           # stdin | functional | file
    fn_name: str | None = None    # functional (LeetCode-style) problems
    starter_code: str = ""
    time_limit: float = 2.0       # seconds, per test, before the Python multiplier
    memory_limit_mb: int = 256
    checker: str | None = None    # python source of a checker (open-r1 convention)
    rating: int | None = None
    tags: list[str] = field(default_factory=list)
    editorial: str | None = None
    solutions: list[str] = field(default_factory=list)
    date: str | None = None       # ISO date, for time-windowed evaluation
    url: str | None = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "Problem":
        d = dict(d)
        d["tests"] = [Test(**t) for t in d.get("tests", [])]
        d["public_tests"] = [Test(**t) for t in d.get("public_tests", [])]
        return Problem(**d)


def save_problems(problems: Iterable[Problem], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for p in problems:
            f.write(p.to_json() + "\n")
            n += 1
    return n


def load_problems(path: str | Path) -> list[Problem]:
    with Path(path).open(encoding="utf-8") as f:
        return [Problem.from_dict(json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------- helpers

def _norm_nl(s: str) -> str:
    return s.replace("\r\n", "\n")


def _tests_from_any(raw) -> tuple[list[Test], str | None]:
    """Parse the test formats seen in DeepCoder / TACO / PrimeIntellect. Returns (tests, fn_name)."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    fn_name = None
    if isinstance(raw, dict):  # TACO style
        fn_name = raw.get("fn_name")
        ins, outs = raw.get("inputs", []), raw.get("outputs", [])
        tests = [Test(_as_text(i), _as_text(o)) for i, o in zip(ins, outs)]
        return tests, fn_name
    tests = []
    for t in raw:
        tests.append(Test(_as_text(t["input"]), _as_text(t["output"])))
        fn_name = fn_name or t.get("fn_name")
    return tests, fn_name


def _as_list(x) -> list:
    # TACO-family datasets store solutions either as a list or as a JSON-encoded string of a list.
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except json.JSONDecodeError:
            return [x]
    return list(x or [])


def _as_text(x) -> str:
    # TACO functional tests store argument lists / return values as JSON values, not strings.
    if isinstance(x, str):
        return _norm_nl(x)
    return json.dumps(x)


class _NoGlobalsUnpickler(pickle.Unpickler):
    """LCB private tests are a pickled *string*; refuse anything that needs a global."""

    def find_class(self, module, name):  # pragma: no cover - only hit by malicious data
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _decode_lcb_private(s: str) -> list[dict]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        blob = zlib.decompress(base64.b64decode(s.encode("utf-8")))
        return json.loads(_NoGlobalsUnpickler(io.BytesIO(blob)).load())


# ---------------------------------------------------------------- loaders

def load_lcb(jsonl_files: list[str | Path], start: str | None = None, end: str | None = None) -> Iterator[Problem]:
    """LiveCodeBench code_generation_lite. Download the test*.jsonl files on the cluster with
    `huggingface-cli download livecodebench/code_generation_lite --repo-type dataset`.
    start/end are ISO dates (inclusive start, exclusive end) on contest_date."""
    for fp in jsonl_files:
        with Path(fp).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                date = r["contest_date"][:10]
                if (start and date < start) or (end and date >= end):
                    continue
                public = json.loads(r["public_test_cases"])
                private = _decode_lcb_private(r["private_test_cases"])
                meta = json.loads(r.get("metadata") or "{}")
                functional = public and public[0].get("testtype") == "functional"
                pub = [Test(_norm_nl(t["input"]), _norm_nl(t["output"])) for t in public]
                priv = [Test(_norm_nl(t["input"]), _norm_nl(t["output"])) for t in private]
                yield Problem(
                    pid=f"lcb/{r['question_id']}",
                    source=f"lcb-{r['platform']}",
                    statement=r["question_content"],
                    tests=pub + priv,
                    public_tests=pub,
                    mode="functional" if functional else "stdin",
                    fn_name=meta.get("func_name"),
                    starter_code=r.get("starter_code") or "",
                    time_limit=6.0,  # LCB's fixed per-test timeout
                    date=date,
                    meta={"difficulty": r.get("difficulty"), "title": r.get("question_title"),
                          "contest_id": r.get("contest_id"), "fn_args": "lines"},
                )


def _cf_statement(r: dict) -> str:
    parts = [r.get("description") or ""]
    if r.get("input_format"):
        parts.append("Input\n" + r["input_format"])
    if r.get("output_format"):
        parts.append("Output\n" + r["output_format"])
    for i, ex in enumerate(r.get("examples") or [], 1):
        parts.append(f"Example {i}\nInput:\n{_norm_nl(ex['input'])}\nOutput:\n{_norm_nl(ex['output'])}")
    if r.get("note"):
        parts.append("Note\n" + r["note"])
    return "\n\n".join(p.strip() for p in parts if p)


def load_open_r1_codeforces(split: str = "test", config: str = "verifiable",
                            generated_tests_dir: str | Path | None = None,
                            max_generated: int = 50) -> Iterator[Problem]:
    """open-r1/codeforces. If generated_tests_dir is given (the downloaded generated_tests/*.parquet),
    up to max_generated generated tests are added. Interactive problems are skipped."""
    from datasets import load_dataset
    import pandas as pd

    ds = load_dataset("open-r1/codeforces", config, split=split)
    gen_cache: dict[str, "pd.DataFrame"] = {}
    for r in ds:
        if r.get("interaction_format"):
            continue
        tests = [Test(_norm_nl(t["input"]), _norm_nl(t["output"])) for t in (r.get("official_tests") or [])]
        if generated_tests_dir and r.get("generated_tests"):
            cid = r["contest_id"]
            if cid not in gen_cache:
                fp = Path(generated_tests_dir) / f"test_cases_{cid}.parquet"
                gen_cache.clear()  # contests come in order; keep memory bounded
                gen_cache[cid] = pd.read_parquet(fp) if fp.exists() else pd.DataFrame(columns=["problem_id"])
            df = gen_cache[cid]
            sub = df[df["problem_id"] == r["id"]].sort_values("test_case_i").head(max_generated)
            tests += [Test(_norm_nl(i), _norm_nl(o)) for i, o in zip(sub["input"], sub["output"])]
        public = [Test(_norm_nl(t["input"]), _norm_nl(t["output"])) for t in (r.get("examples") or [])]
        yield Problem(
            pid=f"codeforces/{r['id']}",
            source="codeforces",
            statement=_cf_statement(r),
            tests=tests or public,
            public_tests=public,
            mode="file" if r.get("input_mode") == "file" else "stdin",
            time_limit=float(r.get("time_limit") or 2.0),
            memory_limit_mb=int(r.get("memory_limit") or 256),
            checker=r.get("generated_checker"),
            rating=r.get("rating"),
            tags=list(r.get("tags") or []),
            editorial=r.get("editorial"),
            date=str(r["contest_start_year"]) if r.get("contest_start_year") else None,
            url=f"https://codeforces.com/contest/{r['contest_id']}/problem/{r['index']}",
            meta={"aliases": r.get("aliases") or [], "official_tests_complete": r.get("official_tests_complete")},
        )


def load_deepcoder(config: str, split: str = "train") -> Iterator[Problem]:
    """agentica-org/DeepCoder-Preview-Dataset; configs: primeintellect, taco, lcbv5, codeforces."""
    from datasets import load_dataset

    ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", config, split=split)
    for i, r in enumerate(ds):
        tests, fn_name = _tests_from_any(r["tests"])
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            meta = json.loads(meta or "{}")
        fn_name = fn_name or meta.get("func_name")
        yield Problem(
            pid=f"deepcoder-{config}/{split}/{i}",
            source=f"deepcoder-{config}",
            statement=r["problem"],
            tests=tests,
            public_tests=tests[:1],
            mode="functional" if fn_name else "stdin",
            fn_name=fn_name,
            starter_code=r.get("starter_code") or "",
            solutions=_as_list(r.get("solutions")),
            # LCB-derived rows use one JSON value per input line; TACO-derived rows use a JSON arg list.
            meta={"fn_args": "lines" if config == "lcbv5" else "json_list"},
        )


def load_taco_verified(split: str = "train") -> Iterator[Problem]:
    from datasets import load_dataset
    import ast

    ds = load_dataset("likaixin/TACO-verified", split=split)
    for i, r in enumerate(ds):
        tests, fn_name = _tests_from_any(r["input_output"])
        try:
            tags = ast.literal_eval(r.get("tags") or "[]")
        except (ValueError, SyntaxError):
            tags = []
        yield Problem(
            pid=f"taco/{r.get('id', i)}",
            source=f"taco-{r.get('source')}",
            statement=r["question"],
            tests=tests,
            public_tests=tests[:1],
            mode="functional" if fn_name else "stdin",
            fn_name=fn_name,
            starter_code=r.get("starter_code") or "",
            tags=tags,
            solutions=_as_list(r.get("solutions"))[:20],
            date=(r.get("date") or "")[:10] or None,
            url=r.get("url"),
            meta={"difficulty": r.get("difficulty"), "fn_args": "json_list"},
        )


def load_team_suite(path: str | Path) -> Iterator[Problem]:
    """The team's original test_suite_200_FINAL.jsonl (kept for continuity only)."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tcs = r["test_cases"]
            fn_name = next((t.get("fn_name") for t in tcs if t.get("fn_name")), None)
            tests = [Test(_as_text(t["input"]), _as_text(t["output"])) for t in tcs]
            yield Problem(
                pid=f"team200/{r['id']}",
                source="team200",
                statement=r["problem"],
                tests=tests,
                public_tests=tests[:1],
                mode="functional" if fn_name else "stdin",
                fn_name=fn_name,
                meta={"difficulty": r.get("difficulty"), "fn_args": "json_list"},
            )
