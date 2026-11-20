"""CPU-only tests for verdict-aware repair (Track B)."""
from smallcoder.problems import Problem, Test
from smallcoder.repair import feedback, repair_messages


# ---------------------------------------------------------------- repair
def test_repair_feedback():
    p = Problem(pid="x", source="t", statement="s", tests=[Test("1", "2")], time_limit=1.0)
    fb = feedback("WA", {"input": "1", "expected": "2", "got": "3"}, p)
    assert "Expected output:\n2" in fb and "Your output:\n3" in fb
    assert "time limit (1 s)" in feedback("TLE", {"input": "1"}, p)
    msg = repair_messages(p, "print(3)", "WA", {"input": "1", "expected": "2", "got": "3"})
    assert msg[0]["role"] == "user" and "print(3)" in msg[0]["content"]
