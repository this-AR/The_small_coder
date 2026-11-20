"""CPU-only tests for Part 1 (no models, no downloads). Run on the cluster: pytest -q"""
import json

import numpy as np
import pytest

from smallcoder.decontam import Decontaminator, cf_ids
from smallcoder.extract import extract_code, split_thinking
from smallcoder.gen import Completion, ModelSpec, generate_budgeted
from smallcoder.judge import AC, CE, NOCODE, RE, TLE, WA, judge, tokens_match, values_match
from smallcoder.metrics import ece_binary, mcnemar_exact, pass_at_k
from smallcoder.problems import Problem, Test, _as_list, _decode_lcb_private, _tests_from_any


# ---------------------------------------------------------------- extract
def test_split_and_extract():
    t, a = split_thinking("reason</think>\nans ```python\nprint(1)\n```", "</think>")
    assert t == "reason" and extract_code(a) == "print(1)"
    assert extract_code("```cpp\nint x;\n```\n```python\nprint(2)\n```") == "print(2)"
    assert extract_code("```\nprint(3)\n```") == "print(3)"
    assert extract_code("<code>print(4)</code>") == "print(4)"
    assert extract_code("no code here") is None
    assert split_thinking("unfinished", "</think>") == ("unfinished", "")


# ---------------------------------------------------------------- compare
def test_tokens_match():
    assert tokens_match("1 2\n3\n", "1 2 3")
    assert not tokens_match("1 2", "1 2 3")
    assert tokens_match("0.3333333", "0.33333333")
    assert not tokens_match("1000000000000000000", "1000000000000000001")  # ints never use float tolerance
    assert not tokens_match("YES", "yes")
    assert not tokens_match("inf", "1e308")  # non-finite never passes via tolerance


def test_values_match():
    assert values_match([1, 2.0000001], [1, 2.0])
    assert not values_match([1, 2], [1, 2, 3])
    assert not values_match(True, 1)


# ---------------------------------------------------------------- judge
P = Problem(pid="t/1", source="t", statement="add", tests=[Test("1 2\n", "3\n"), Test("5 5\n", "10\n")],
            time_limit=2.0)


def test_judge_verdicts():
    assert judge("a,b=map(int,input().split());print(a+b)", P).verdict == AC
    r = judge("a,b=map(int,input().split());print(a-b)", P)
    assert r.verdict == WA and r.passed == 0 and r.first_fail["expected"] == "3\n"
    assert judge("raise ValueError()", P).verdict == RE
    assert judge("while True: pass", Problem(**{**P.__dict__, "time_limit": 0.5})).verdict == TLE
    assert judge("def f(:", P).verdict == CE
    assert judge(None, P).verdict == NOCODE
    r = judge("a,b=map(int,input().split());print(3)", P, stop_on_fail=False)
    assert (r.verdict, r.passed, r.total) == (WA, 1, 2)


def test_judge_checker_and_file_mode():
    checker = "import sys\nprint(1 if open(sys.argv[3]).read().split()==['ok'] else 0)\n"
    p = Problem(pid="t/2", source="t", statement="", tests=[Test("x\n", "anything\n")], checker=checker)
    assert judge("print('ok')", p).verdict == AC
    assert judge("print('no')", p).verdict == WA
    pf = Problem(pid="t/3", source="t", statement="", tests=[Test("4\n", "16\n")], mode="file")
    code = "n=int(open('input.txt').read());open('output.txt','w').write(str(n*n))"
    assert judge(code, pf).verdict == AC


def test_judge_functional():
    lines = Problem(pid="t/4", source="t", statement="", tests=[Test("[1,2]\n3", "[1,2,3]")], mode="functional",
                    fn_name="app", meta={"fn_args": "lines"})
    code = "class Solution:\n    def app(self, xs: List[int], y: int) -> List[int]:\n        return xs+[y]\n"
    assert judge(code, lines).verdict == AC
    taco = Problem(pid="t/5", source="t", statement="", tests=[Test(json.dumps([2, 3]), json.dumps([6]))],
                   mode="functional", fn_name="mul", meta={"fn_args": "json_list"})
    assert judge("def mul(a, b):\n    return a*b\n", taco).verdict == AC  # wrapped-output convention


# ---------------------------------------------------------------- loaders
def test_test_formats():
    t, fn = _tests_from_any('{"inputs": ["1\\n"], "outputs": ["2\\n"], "fn_name": "f"}')
    assert t[0].output == "2\n" and fn == "f"
    t, _ = _tests_from_any('[{"type": "stdin_stdout", "input": "a\\r\\n", "output": "b"}]')
    assert t[0].input == "a\n"
    import base64, pickle, zlib
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps([{"input": "1", "output": "2"}])))).decode()
    assert _decode_lcb_private(blob)[0]["output"] == "2"


# ---------------------------------------------------------------- metrics
def test_metrics():
    assert pass_at_k(10, 0, 1) == 0.0 and pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    assert mcnemar_exact(np.array([1, 1, 0]), np.array([1, 1, 0]))["p"] == 1.0
    assert ece_binary(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.0
    with pytest.raises(ValueError):
        pass_at_k(2, 0, 4)  # fewer samples than k used to return 1.0


def test_solutions_any_shape():
    assert _as_list('["a", "b"]') == ["a", "b"] and _as_list(["a"]) == ["a"] and _as_list(None) == []


def test_read_rows_dedupes_overlapping_files(tmp_path):
    from smallcoder.io import append_jsonl, read_rows
    row = {"pid": "p", "model": "m", "condition": "base", "budget": 0, "sample": 0, "verdict": "AC"}
    append_jsonl(tmp_path / "x.shard0.jsonl", [row])
    append_jsonl(tmp_path / "x.jsonl", [row, dict(row, sample=1)])  # merged file repeats the shard
    assert len(read_rows([tmp_path / "x.shard0.jsonl", tmp_path / "x.jsonl"])) == 2


# ---------------------------------------------------------------- decontam
def test_decontam():
    ev = [Problem(pid="codeforces/2063/A", source="codeforces", statement="alpha beta gamma delta " * 5, tests=[])]
    d = Decontaminator(ev, n=4)
    tr = Problem(pid="taco/1", source="taco", statement="x", tests=[],
                 url="https://codeforces.com/problemset/problem/2063/A")
    assert cf_ids(tr) == {"2063/A"} and d.check(tr)["reason"] == "id"
    tr2 = Problem(pid="taco/2", source="taco", statement="zz alpha beta gamma delta zz", tests=[])
    assert d.check(tr2)["reason"] == "4gram"
    assert d.check(Problem(pid="taco/3", source="taco", statement="unrelated words entirely here", tests=[])) is None


# ---------------------------------------------------------------- budget engine (fake model)
class FakeTok:
    """One token per character."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class FakeBackend:
    """Thinking: sample 0 thinks 5 chars and stops, sample 1 thinks forever. Answers: 'ANS'."""
    def __init__(self):
        self.calls = []

    def generate(self, prompts, n, max_tokens, temperature, top_p, stop=None, seed=None):
        self.calls.append((len(prompts), n, max_tokens))
        out = []
        for p in prompts:
            if stop:  # thinking phase
                out.append([Completion("think", None, "stop"), Completion("x" * max_tokens, None, "length")][:n])
            else:
                out.append([Completion("ANS", None, "stop")] * n)
        return out


def test_generate_budgeted_prefix_sharing():
    spec = ModelSpec(name="m", path="m", force_text="|F|")
    be = FakeBackend()
    res = generate_budgeted(be, FakeTok(), spec, ["P"], [0, 3, 10], n=2, answer_tokens=50,
                            temperature=0.6, top_p=0.95)
    by = {(r.sample, r.budget): r for r in res}
    assert set(by) == {(0, 0), (1, 0), (0, 3), (1, 3), (0, 10), (1, 10)}
    assert by[(0, 10)].think_tokens == 5 and not by[(0, 10)].forced      # natural stop fits budget 10
    assert by[(0, 3)].think_tokens == 3 and by[(0, 3)].forced            # cut at 3
    assert by[(1, 10)].forced and by[(1, 10)].think_tokens == 10
    assert by[(0, 0)].think_tokens == 0
    # one thinking call at the max budget, one no-think call, one answer call
    assert [c[2] for c in be.calls] == [50, 10, 50]
