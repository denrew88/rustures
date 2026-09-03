"""Measure predict-only scaling across independent Python threads.

The benchmark constructs one fitted detector per task outside the timed region.
That isolates the native search and avoids mutable estimator sharing. Results are
validated against the one-worker baseline; timing values are reported, never used
as correctness assertions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import platform
import statistics
import time
from typing import Any, Callable

import numpy as np
import ruptures
import rustures


def make_signals(
    tasks: int,
    n_samples: int,
    n_features: int,
    n_bkps: int,
    seed: int,
) -> list[np.ndarray]:
    boundaries = np.linspace(0, n_samples, n_bkps + 2, dtype=np.int64)
    signals = []
    for task in range(tasks):
        rng = np.random.default_rng(seed + task)
        signal = np.empty((n_samples, n_features), dtype=np.float64)
        for segment, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        ):
            level = ((segment % 5) - 2) * 1.8
            feature_offsets = np.linspace(0.0, 0.4, n_features)
            signal[start:end] = level + feature_offsets
        signal += rng.normal(0.0, 0.55, signal.shape)
        signals.append(signal)
    return signals


def rustures_factory(signal: np.ndarray, min_size: int):
    return rustures.KernelCPD(
        kernel="linear",
        backend="fused",
        min_size=min_size,
        jump=1,
    ).fit(signal)


def ruptures_factory(signal: np.ndarray, min_size: int):
    return ruptures.KernelCPD(kernel="linear", min_size=min_size).fit(signal)


def measure_library(
    factory: Callable[[np.ndarray, int], Any],
    signals: list[np.ndarray],
    n_bkps: int,
    min_size: int,
    workers: list[int],
    repeats: int,
) -> tuple[dict[str, dict[str, float]], list[list[int]]]:
    baseline: list[list[int]] | None = None
    timings: dict[int, list[float]] = {worker: [] for worker in workers}

    for worker in workers:
        for _ in range(repeats):
            detectors = [factory(signal, min_size) for signal in signals]
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=worker) as pool:
                predictions = list(
                    pool.map(
                        lambda detector: detector.predict(n_bkps=n_bkps),
                        detectors,
                    )
                )
            timings[worker].append(time.perf_counter() - started)
            if baseline is None:
                baseline = predictions
            elif predictions != baseline:
                raise RuntimeError(
                    f"threaded predictions changed with {worker} workers"
                )

    assert baseline is not None
    one_worker = statistics.median(timings[1])
    report = {}
    for worker in workers:
        median = statistics.median(timings[worker])
        report[str(worker)] = {
            "median_seconds": median,
            "jobs_per_second": len(signals) / median,
            "speedup_vs_one_worker": one_worker / median,
        }
    return report, baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=3_000)
    parser.add_argument("--n-features", type=int, default=3)
    parser.add_argument("--n-bkps", type=int, default=12)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = sorted(set(args.workers))
    if not workers or workers[0] != 1:
        raise ValueError("--workers must include 1 for the speedup baseline")
    if any(worker < 1 or worker > args.tasks for worker in workers):
        raise ValueError("every worker count must be between 1 and --tasks")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")

    signals = make_signals(
        args.tasks,
        args.n_samples,
        args.n_features,
        args.n_bkps,
        args.seed,
    )
    libraries = {}
    baselines = {}
    for name, factory in (
        ("rustures", rustures_factory),
        ("ruptures", ruptures_factory),
    ):
        measurements, baseline = measure_library(
            factory,
            signals,
            args.n_bkps,
            args.min_size,
            workers,
            args.repeats,
        )
        libraries[name] = measurements
        baselines[name] = baseline

    report = {
        "benchmark": "predict-only KernelCPD linear thread scaling",
        "config": {
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "n_bkps": args.n_bkps,
            "min_size": args.min_size,
            "tasks": args.tasks,
            "workers": workers,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "environment": {
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "rustures": rustures.__version__,
            "ruptures": ruptures.__version__,
        },
        "same_breakpoints_between_libraries": baselines["rustures"]
        == baselines["ruptures"],
        "libraries": libraries,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


if __name__ == "__main__":
    main()
