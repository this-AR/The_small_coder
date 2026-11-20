"""Execute candidate programs against tests.

Security: this is a resource-limited subprocess, NOT a security boundary. On the cluster run the
judge inside a container or nsjail/firejail with no network (see RUNBOOK.md, Setup).
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .problems import Problem, Test

AC, WA, TLE, RE, MLE, CE, NOCODE = "AC", "WA", "TLE", "RE", "MLE", "CE", "NOCODE"
_INT = re.compile(r"^[+-]?\d+$")
_RESULT_MARK = "\n@@SMALLCODER_RESULT@@"
_IS_POSIX = os.name == "posix"


@dataclass
class JudgeResult:
    verdict: str
    passed: int
    total: int
    first_fail: dict | None = None   # {"input","expected","got","stderr"} (truncated)
    max_time: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ comparison

def tokens_match(got: str, expected: str, float_tol: float = 1e-6) -> bool:
    """Token-wise comparison (whitespace-insensitive, like Codeforces' default checker).
    Integers compare exactly; other numbers within absolute-or-relative float_tol."""
    g, e = got.split(), expected.split()
    if len(g) != len(e):
        return False
    for a, b in zip(g, e):
        if a == b:
            continue
        if _INT.match(a) and _INT.match(b):
            if int(a) != int(b):
                return False
            continue
        try:
            x, y = Decimal(a), Decimal(b)
        except InvalidOperation:
            return False
        if not (x.is_finite() and y.is_finite()):
            return False
        if abs(x - y) > Decimal(str(float_tol)) * max(Decimal(1), abs(y)):
            return False
    return True


def values_match(got, expected, float_tol: float = 1e-6) -> bool:
    """Structural equality for functional problems with float tolerance."""
    if isinstance(got, bool) or isinstance(expected, bool):
        return got == expected
    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(got, int) and isinstance(expected, int):
            return got == expected
        if not (math.isfinite(got) and math.isfinite(expected)):
            return got == expected
        return abs(got - expected) <= float_tol * max(1.0, abs(expected))
    if isinstance(got, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(got) == len(expected) and all(values_match(a, b, float_tol) for a, b in zip(got, expected))
    if isinstance(got, dict) and isinstance(expected, dict):
        return got.keys() == expected.keys() and all(values_match(got[k], expected[k], float_tol) for k in got)
    return got == expected


# ------------------------------------------------------------------ process running

# preexec_fn is unsafe with threads (judge_many), so limits are set by a launcher that then exec()s.
_LAUNCHER = "\n".join([
    "import os, resource, sys",
    "m, c = int(sys.argv[1]), int(sys.argv[2])",
    "resource.setrlimit(resource.RLIMIT_AS, (m, m))",
    "resource.setrlimit(resource.RLIMIT_CPU, (c, c + 1))",
    "resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))",
    "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))",
    "os.execv(sys.argv[3], sys.argv[3:])",
])


def _child_env() -> dict:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"}
    if "SYSTEMROOT" in os.environ:  # Windows Python cannot start without it
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def _run(cmd: list[str], stdin: str, cwd: str, timeout: float, mem_mb: int) -> tuple[str, str, int | None, float]:
    """Returns (stdout, stderr, returncode or None on timeout, wall seconds)."""
    if _IS_POSIX:
        cmd = [sys.executable, "-I", "-c", _LAUNCHER, str(mem_mb << 20), str(int(math.ceil(timeout)) + 1), *cmd]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=_child_env(), start_new_session=_IS_POSIX)
    try:
        out, err = proc.communicate(stdin.encode("utf-8"), timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, err = proc.communicate()
        rc = None
    dt = time.perf_counter() - t0
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), rc, dt


def _kill(proc: subprocess.Popen) -> None:
    try:
        if _IS_POSIX:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError):
        pass


def _err_verdict(stderr: str, rc: int | None) -> str:
    if rc is None:
        return TLE
    if "MemoryError" in stderr:
        return MLE
    if _IS_POSIX and rc in (-signal.SIGXCPU, -signal.SIGKILL):
        return TLE
    return RE


_FUNC_DRIVER = r'''
import json, sys
ns = {"__name__": "__solution__"}
exec("from typing import *\nimport collections, heapq, math, bisect, itertools, functools, string, re\n"
     "from collections import *\nfrom functools import *\nfrom heapq import *\nfrom bisect import *\n"
     "from itertools import *\nfrom math import *", ns)
exec(compile(open("sol.py", encoding="utf-8").read(), "sol.py", "exec"), ns)
fn, mode = sys.argv[1], sys.argv[2]
raw = open("args.txt", encoding="utf-8").read()
if mode == "lines":
    args = [json.loads(l) for l in raw.split("\n") if l.strip()]
else:
    args = json.loads(raw)
    if not isinstance(args, list):
        args = [args]
f = getattr(ns["Solution"](), fn) if "Solution" in ns else ns[fn]
res = f(*args)
if isinstance(res, tuple):
    res = list(res)
sys.stdout.write(MARK + json.dumps(res))
'''.replace("MARK", repr(_RESULT_MARK))


def _trunc(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + f"...[{len(s) - n} more chars]"


def _check_with_checker(checker_src: str, cwd: str, test: Test, got: str) -> bool:
    d = Path(cwd)
    (d / "checker.py").write_text(checker_src, encoding="utf-8")
    (d / "input.txt").write_text(test.input, encoding="utf-8")
    (d / "correct_output.txt").write_text(test.output, encoding="utf-8")
    (d / "solution_output.txt").write_text(got, encoding="utf-8")
    out, _, rc, _ = _run([sys.executable, "-I", "checker.py", "input.txt", "correct_output.txt",
                          "solution_output.txt"], "", cwd, timeout=20.0, mem_mb=2048)
    if rc != 0 or not out.split():
        return False
    try:
        score = float(out.split()[0])
    except ValueError:
        return False
    # open-r1 checkers print a score on a 0-1 or 0-100 scale; only full marks count as a pass.
    # shortcut: a 0-100 checker printing exactly 1 would be misread as a pass; audit checkers if it matters.
    return math.isclose(score, 1.0) or math.isclose(score, 100.0)


def judge(code: str | None, problem: Problem, tests: list[Test] | None = None, *,
          stop_on_fail: bool = True, time_multiplier: float = 1.0, mem_mb: int | None = None,
          float_tol: float = 1e-6) -> JudgeResult:
    tests = problem.tests if tests is None else tests
    if not code:
        return JudgeResult(NOCODE, 0, len(tests))
    try:
        compile(code, "sol.py", "exec")
    except (SyntaxError, ValueError) as e:
        return JudgeResult(CE, 0, len(tests), {"stderr": _trunc(str(e))})
    if not tests:
        return JudgeResult(AC, 0, 0)

    # Python needs headroom over the judge's limit; 512 MB floor avoids false MLE from the interpreter.
    mem = mem_mb or max(512, problem.memory_limit_mb * 2)
    timeout = problem.time_limit * time_multiplier
    passed, worst, first_fail, verdict = 0, 0.0, None, AC
    with tempfile.TemporaryDirectory(prefix="sc_judge_") as cwd:
        Path(cwd, "sol.py").write_text(code, encoding="utf-8")
        if problem.mode == "functional":
            Path(cwd, "driver.py").write_text(_FUNC_DRIVER, encoding="utf-8")
        for t in tests:
            v, got, err, dt = _run_one(problem, t, cwd, timeout, mem, float_tol)
            worst = max(worst, dt)
            if v == AC:
                passed += 1
                continue
            if first_fail is None:
                verdict = v
                first_fail = {"input": _trunc(t.input), "expected": _trunc(t.output),
                              "got": _trunc(got), "stderr": _trunc(err[-4000:])}
            if stop_on_fail:
                break
    return JudgeResult(verdict, passed, len(tests), first_fail, worst)


def _run_one(problem: Problem, t: Test, cwd: str, timeout: float, mem: int, tol: float):
    if problem.mode == "functional":
        Path(cwd, "args.txt").write_text(t.input, encoding="utf-8")
        mode = problem.meta.get("fn_args", "lines")
        out, err, rc, dt = _run([sys.executable, "-I", "driver.py", problem.fn_name or "", mode], "", cwd, timeout, mem)
        if rc != 0:
            return _err_verdict(err, rc), out, err, dt
        if _RESULT_MARK not in out:
            return RE, out, err, dt
        try:
            got = json.loads(out.rsplit(_RESULT_MARK, 1)[1])
            exp = json.loads(t.output)
        except json.JSONDecodeError:
            return WA, out, err, dt
        ok = values_match(got, exp, tol)
        # APPS/TACO convention: expected outputs are often wrapped in a one-element list.
        if not ok and mode == "json_list" and isinstance(exp, list) and len(exp) == 1:
            ok = values_match(got, exp[0], tol)
        return (AC if ok else WA), json.dumps(got), err, dt

    if problem.mode == "file":
        Path(cwd, "input.txt").write_text(t.input, encoding="utf-8")
        outp = Path(cwd, "output.txt")
        outp.unlink(missing_ok=True)
        out, err, rc, dt = _run([sys.executable, "-I", "sol.py"], "", cwd, timeout, mem)
        if rc == 0:
            out = outp.read_text(encoding="utf-8", errors="replace") if outp.exists() else out
    else:
        out, err, rc, dt = _run([sys.executable, "-I", "sol.py"], t.input, cwd, timeout, mem)
    if rc != 0:
        return _err_verdict(err, rc), out, err, dt
    if problem.checker:
        ok = _check_with_checker(problem.checker, cwd, t, out)
    else:
        ok = tokens_match(out, t.output, tol)
    return (AC if ok else WA), out, err, dt


def judge_many(jobs: list[tuple], workers: int = 16, **kw) -> list[JudgeResult]:
    """jobs = [(code, problem) or (code, problem, tests), ...]; results in the same order.
    Threads suffice: the work happens in subprocesses."""
    def one(j):
        return judge(j[0], j[1], j[2] if len(j) > 2 else None, **kw)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, jobs))
