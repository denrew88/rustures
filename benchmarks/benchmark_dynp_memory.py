"""Measure Dynp prediction-state peak memory in isolated Python processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def peak_bytes() -> int:
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):
            return int(info.peak_wset)
    except ImportError:
        pass

    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class StopAfterAllocation:
    min_size = 1

    def fit(self, signal):
        self.n_samples = len(signal)
        return self

    def error(self, start, end):
        raise RuntimeError("intentional stop after Dynp allocation")

    def error_many(self, starts, ends):
        raise RuntimeError("intentional stop after Dynp allocation")


def worker(n_samples: int, changes: int) -> None:
    import psutil
    import rustures

    signal = np.zeros(n_samples, dtype=np.float64)
    detector = rustures.Dynp(
        custom_cost=StopAfterAllocation(),
        min_size=1,
        jump=1,
        max_memory_bytes=536_870_912,
    ).fit(signal)
    estimated = detector.estimated_memory_bytes(changes)
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    peak_before = peak_bytes()
    try:
        detector.predict(changes)
    except RuntimeError as error:
        if "intentional stop" not in str(error):
            raise
    else:
        raise AssertionError("the allocation probe cost did not stop prediction")
    peak_after = peak_bytes()
    baseline = max(rss_before, peak_before)
    delta = max(0, peak_after - baseline)
    print(
        json.dumps(
            {
                "n_samples": n_samples,
                "changes_k": changes,
                "estimated_bytes": estimated,
                "rss_before_bytes": rss_before,
                "peak_before_bytes": peak_before,
                "peak_after_bytes": peak_after,
                "peak_delta_bytes": delta,
                "delta_over_estimate": delta / estimated,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n", type=int)
    parser.add_argument("--changes", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.worker:
        if args.n is None or args.changes is None:
            parser.error("--worker requires --n and --changes")
        worker(args.n, args.changes)
        return

    cases = [(50_000, 1), (50_000, 8), (50_000, 32), (50_000, 64), (200_000, 64)]
    rows = []
    for n_samples, changes in cases:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--n",
                str(n_samples),
                "--changes",
                str(changes),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        row = json.loads(completed.stdout.strip())
        rows.append(row)
        print(json.dumps(row))

    report = {
        "method": "isolated child process; custom batch cost aborts after Dynp state allocation",
        "platform": sys.platform,
        "python": sys.version,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
