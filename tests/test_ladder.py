"""CPU-only tests for the hint ladder (Track A1)."""
from smallcoder.ladder import TagPredictor, gold_codes, leak_reason, longest_common_run
from smallcoder.problems import Problem


# ---------------------------------------------------------------- ladder
def test_leak_filter():
    assert leak_reason("use ```python\nx```", []) == "code_fence"
    assert leak_reason("def f():\nimport os\nprint(1)\n", []) == "python_lines"
    gold = "n = int(input())\nans = sum(a[i] * b[i] for i in range(n)) % MOD\nprint(ans)"
    assert leak_reason("compute ans = sum(a[i] * b[i] for i in range(n)) % MOD then output", [gold]) == "gold_overlap"
    assert leak_reason("[IDEA] sort and sweep with a stack", [gold]) is None
    assert longest_common_run("a b c d", "x b c d y") == 3
    cpp = "#include <bits/stdc++.h>\nint main() {\n  long long s = 0;\n  for (int i = 0; i < n; i++) s += a[i];\n}"
    assert leak_reason(cpp, []) == "python_lines"  # C++ lines count as code too
    p = Problem(pid="c", source="codeforces", statement="", tests=[],
                editorial="Use prefix sums.\n```cpp\n" + cpp + "\n```")
    assert gold_codes(p) and cpp in gold_codes(p)[0]


def test_tag_predictor_runs():
    ps = [Problem(pid=str(i), source="t", statement=s, tests=[], tags=t) for i, (s, t) in enumerate(
        [("shortest path in a weighted graph dijkstra", ["graphs"]), ("count subsets modulo prime dp table", ["dp"])] * 5)]
    pred = TagPredictor(top_k=1).fit(ps).predict(["find the shortest path between two cities in a graph"])
    assert pred == [["graphs"]]
