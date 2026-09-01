"""Measure L1-Potts scaling against the pinned ruptures wheel."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
import types

import numpy as np


def timed(callable_) -> float:
    gc.collect()
    start = time.perf_counter()
    callable_()
    return time.perf_counter() - start


def median_pair(rust_call, reference_call, repeats: int = 7) -> tuple[float, float]:
    rust_call()
    reference_call()
    rust_samples = []
    reference_samples = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            rust_samples.append(timed(rust_call))
            reference_samples.append(timed(reference_call))
        else:
            reference_samples.append(timed(reference_call))
            rust_samples.append(timed(rust_call))
    return statistics.median(rust_samples), statistics.median(reference_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-site", type=Path, required=True)
    parser.add_argument("--rustures-site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    version_module = types.ModuleType("ruptures.version")
    version_module.version = "pinned-reference"
    sys.modules["ruptures.version"] = version_module
    sys.path.insert(0, str(args.reference_site.resolve()))
    import ruptures as rpt  # type: ignore
    sys.path.insert(0, str(args.rustures_site.resolve()))
    import rustures as rst

    rng = np.random.default_rng(20260831)
    rows = []
    for n in [256, 512, 1024, 2048]:
        for k in [4, 16, 64, 256]:
            signal = np.arange(n, dtype=np.float64) % k
            rng.shuffle(signal)
            penalty = float(k)
            rust_call = lambda: rst.L1Potts().fit_predict(signal, pen=penalty)
            reference_call = lambda: rpt.L1Potts().fit_predict(signal, pen=penalty)
            rust_result = rust_call()
            reference_result = reference_call()
            rust_time, reference_time = median_pair(rust_call, reference_call)
            rows.append({
                "n": n,
                "k": k,
                "rustures_seconds": rust_time,
                "ruptures_seconds": reference_time,
                "speedup": reference_time / rust_time,
                "parent_bytes": n * k,
                "score_row_bytes": 2 * k * 8,
                "breakpoints_match": rust_result == reference_result,
            })
    report = {
        "seed": 20260831,
        "timing_scope": "fresh L1Potts.fit_predict, median of 7, alternating order",
        "k_definition": "number of distinct observed values / DP states",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
