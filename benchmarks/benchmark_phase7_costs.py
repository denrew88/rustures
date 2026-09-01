"""End-to-end phase 7 Dynp benchmarks against pinned ruptures."""

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


MODELS = ("l1", "rank", "normal", "linear", "ar", "clinear", "mahalanobis")


def load_packages(reference_site: Path, rustures_site: Path):
    version_module = types.ModuleType("ruptures.version")
    version_module.version = "pinned-reference"
    sys.modules["ruptures.version"] = version_module
    sys.path.insert(0, str(reference_site.resolve()))
    import ruptures  # type: ignore

    sys.path.insert(0, str(rustures_site.resolve()))
    import rustures

    return ruptures, rustures


def make_signal(model: str, n: int, changes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    boundaries = np.linspace(0, n, changes + 2, dtype=int)
    levels = rng.normal(0.0, 4.0, changes + 1)
    scalar = np.empty(n, dtype=np.float64)
    for segment, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        scalar[start:end] = levels[segment] + rng.normal(0.0, 0.35, end - start)
    if model != "linear":
        return scalar

    x = np.linspace(-1.0, 1.0, n)
    response = np.empty(n, dtype=np.float64)
    slopes = rng.normal(0.0, 3.0, changes + 1)
    intercepts = rng.normal(0.0, 2.0, changes + 1)
    for segment, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        response[start:end] = (
            intercepts[segment]
            + slopes[segment] * x[start:end]
            + rng.normal(0.0, 0.05, end - start)
        )
    return np.column_stack((response, np.ones(n), x))


def timed(callable_) -> float:
    gc.collect()
    start = time.perf_counter()
    callable_()
    return time.perf_counter() - start


def median_pair(rust_call, reference_call, repeats: int) -> tuple[float, float]:
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
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    rpt, rst = load_packages(args.reference_site, args.rustures_site)

    rows = []
    for model_index, model in enumerate(MODELS):
        for n in (100, 200, 400):
            for changes in (1, 4, 8):
                signal = make_signal(model, n, changes, 20260831 + model_index * 1000 + n + changes)
                kwargs = {"model": model, "min_size": 5, "jump": 5}
                rust_call = lambda: rst.Dynp(**kwargs).fit_predict(signal, n_bkps=changes)
                reference_call = lambda: rpt.Dynp(**kwargs).fit_predict(signal, n_bkps=changes)
                rust_result = rust_call()
                reference_result = reference_call()
                rust_seconds, reference_seconds = median_pair(
                    rust_call, reference_call, args.repeats
                )
                row = {
                    "model": model,
                    "array_len": n,
                    "changes_k": changes,
                    "rustures_seconds": rust_seconds,
                    "ruptures_seconds": reference_seconds,
                    "speedup": reference_seconds / rust_seconds,
                    "breakpoints_match": rust_result == reference_result,
                }
                rows.append(row)
                print(json.dumps(row))

    report = {
        "seed_base": 20260831,
        "timing_scope": f"fresh Dynp.fit_predict, median of {args.repeats}, alternating order",
        "min_size": 5,
        "jump": 5,
        "k_definition": "requested number of change points",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
