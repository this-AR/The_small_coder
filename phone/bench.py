"""On-device benchmark: solve a problem file, log seconds, tokens, tok/s, verdicts and (approximate) energy.
Final correctness is judged on ALL tests after the run (the solver itself only sees public tests).

  python phone/bench.py --gguf vibe-q4_k_m.gguf --bundle phone_bundle/ --problems cf_test_small.jsonl \
      --lam 0.05 --out bench_pixel.jsonl [--rag rag_index/]
"""
import argparse
import threading
import time
from pathlib import Path

from smallcoder.io import append_jsonl, read_jsonl
from smallcoder.judge import judge
from smallcoder.problems import load_problems

from pipeline import OnDeviceSolver


class EnergyMeter:
    """Integrates battery power from sysfs (current_now x voltage_now), if readable without root.
    Approximate: units/sign conventions vary by vendor; validate against a USB power meter once."""
    BASE = Path("/sys/class/power_supply/battery")

    def __init__(self, period=0.5):
        self.period, self.joules, self._stop = period, 0.0, threading.Event()
        self.ok = (self.BASE / "current_now").exists() and (self.BASE / "voltage_now").exists()

    def _read(self):
        i = abs(int((self.BASE / "current_now").read_text())) / 1e6   # uA -> A
        v = int((self.BASE / "voltage_now").read_text()) / 1e6        # uV -> V
        return i * v

    def _loop(self):
        while not self._stop.is_set():
            self.joules += self._read() * self.period
            time.sleep(self.period)

    def __enter__(self):
        if self.ok:
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()


ap = argparse.ArgumentParser()
ap.add_argument("--gguf", required=True)
ap.add_argument("--bundle", required=True)
ap.add_argument("--problems", required=True)
ap.add_argument("--rag", default=None)
ap.add_argument("--lam", type=float, default=0.05)
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--repair-rounds", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

solver = OnDeviceSolver(a.gguf, a.bundle, a.rag, n_threads=a.threads)
done = {r["pid"] for r in read_jsonl(a.out)}
for p in load_problems(a.problems):
    if p.pid in done:
        continue
    with EnergyMeter() as em:
        log = solver.solve(p, lam=a.lam, repair_rounds=a.repair_rounds)
    full = judge(log["code"], p)
    gen_s = sum(s.get("seconds", 0) for s in log["steps"])
    log.update(verdict=full.verdict, joules=em.joules if em.ok else None,
               tok_per_s=log["tokens"] / gen_s if gen_s else None)
    append_jsonl(a.out, [log])
    print(p.pid, log["arm"], full.verdict, f"{log['seconds']:.1f}s", log["tokens"], "tok")
