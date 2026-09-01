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


def load(name: str, reference_site: Path | None):
    if name == "rustures":
        import rustures

        return rustures
    if reference_site is None:
        raise ValueError("--reference-site is required for ruptures")
    sys.path.insert(0, str(reference_site.resolve()))
    version_module = types.ModuleType("ruptures.version")
    version_module.version = "0.0.0+ee1c8ff"
    sys.modules["ruptures.version"] = version_module
    import ruptures

    return ruptures


def signal(n_samples: int, n_features: int) -> np.ndarray:
    index = np.arange(n_samples)
    segment = np.minimum(index * 4 // n_samples, 3)
    columns = []
    for feature in range(n_features):
        levels = np.roll(np.asarray([0.0, 5.0, -3.0, 7.0]), feature)
        multiplier = 48271 + feature * 2143
        noise = (
            (
                (index * multiplier + index * index * (31 + feature * 2))
                % 104729
            )
            / 104729
            - 0.5
        ) * 0.1
        columns.append(levels[segment] + noise)
    if n_features == 1:
        return columns[0]
    return np.column_stack(columns)


def fit_detector(package, backend: str, kernel: str, values: np.ndarray):
    if backend == "rustures":
        parameters = {"gamma": 0.5} if kernel == "rbf" else {}
        return package.KernelCPD(
            kernel=kernel, min_size=1, **parameters
        ).fit(values)
    parameters = {"params": {"gamma": 0.5}} if kernel == "rbf" else {}
    return package.KernelCPD(
        kernel=kernel, min_size=1, **parameters
    ).fit(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["rustures", "ruptures", "both"])
    parser.add_argument("--kernel", choices=["linear", "rbf", "cosine"], default="rbf")
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--n-features", type=int, default=1)
    parser.add_argument("--changes", nargs="+", type=int, default=[1, 3, 8, 16])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--reference-site", type=Path)
    parser.add_argument("--omit-breakpoints", action="store_true")
    args = parser.parse_args()

    values = signal(args.n_samples, args.n_features)
    if args.backend == "both":
        packages = {
            "rustures": load("rustures", args.reference_site),
            "ruptures": load("ruptures", args.reference_site),
        }
        results = []
        for changes in args.changes:
            samples = {"rustures": [], "ruptures": []}
            fit_samples = {"rustures": [], "ruptures": []}
            total_samples = {"rustures": [], "ruptures": []}
            breakpoints = {"rustures": None, "ruptures": None}
            for repeat in range(args.repeats):
                order = (
                    ("rustures", "ruptures")
                    if repeat % 2 == 0
                    else ("ruptures", "rustures")
                )
                for backend in order:
                    gc.collect()
                    fit_started = time.perf_counter()
                    detector = fit_detector(
                        packages[backend], backend, args.kernel, values
                    )
                    fit_elapsed = time.perf_counter() - fit_started
                    started = time.perf_counter()
                    current = detector.predict(n_bkps=changes)
                    predict_elapsed = time.perf_counter() - started
                    fit_samples[backend].append(fit_elapsed)
                    samples[backend].append(predict_elapsed)
                    total_samples[backend].append(fit_elapsed + predict_elapsed)
                    breakpoints[backend] = [int(value) for value in current]
            rustures_median = statistics.median(samples["rustures"])
            ruptures_median = statistics.median(samples["ruptures"])
            result = {
                "changes": changes,
                "rustures_minimum_seconds": min(samples["rustures"]),
                "rustures_median_seconds": rustures_median,
                "rustures_fit_median_seconds": statistics.median(
                    fit_samples["rustures"]
                ),
                "rustures_total_median_seconds": statistics.median(
                    total_samples["rustures"]
                ),
                "ruptures_minimum_seconds": min(samples["ruptures"]),
                "ruptures_median_seconds": ruptures_median,
                "ruptures_fit_median_seconds": statistics.median(
                    fit_samples["ruptures"]
                ),
                "ruptures_total_median_seconds": statistics.median(
                    total_samples["ruptures"]
                ),
                "ruptures_over_rustures": ruptures_median / rustures_median,
                "ruptures_total_over_rustures": statistics.median(
                    total_samples["ruptures"]
                )
                / statistics.median(total_samples["rustures"]),
            }
            if not args.omit_breakpoints:
                result["rustures_breakpoints"] = breakpoints["rustures"]
                result["ruptures_breakpoints"] = breakpoints["ruptures"]
            results.append(result)
        print(
            json.dumps(
                {
                    "backend": "both",
                    "kernel": args.kernel,
                    "n_samples": args.n_samples,
                    "n_features": args.n_features,
                    "results": results,
                },
                indent=2,
            )
        )
        return

    package = load(args.backend, args.reference_site)
    results = []
    for changes in args.changes:
        samples = []
        fit_samples = []
        total_samples = []
        breakpoints = None
        for _ in range(args.repeats):
            gc.collect()
            fit_started = time.perf_counter()
            detector = fit_detector(package, args.backend, args.kernel, values)
            fit_elapsed = time.perf_counter() - fit_started
            started = time.perf_counter()
            current = detector.predict(n_bkps=changes)
            predict_elapsed = time.perf_counter() - started
            fit_samples.append(fit_elapsed)
            samples.append(predict_elapsed)
            total_samples.append(fit_elapsed + predict_elapsed)
            breakpoints = [int(value) for value in current]
        result = {
            "changes": changes,
            "minimum_seconds": min(samples),
            "median_seconds": statistics.median(samples),
            "fit_median_seconds": statistics.median(fit_samples),
            "total_median_seconds": statistics.median(total_samples),
        }
        if not args.omit_breakpoints:
            result["breakpoints"] = breakpoints
        results.append(result)
    print(
        json.dumps(
            {
                "backend": args.backend,
                "kernel": args.kernel,
                "n_samples": args.n_samples,
                "n_features": args.n_features,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
