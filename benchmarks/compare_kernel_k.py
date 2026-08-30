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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["rustures", "ruptures"])
    parser.add_argument("--kernel", choices=["linear", "rbf", "cosine"], default="rbf")
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--n-features", type=int, default=1)
    parser.add_argument("--changes", nargs="+", type=int, default=[1, 3, 8, 16])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--reference-site", type=Path)
    parser.add_argument("--omit-breakpoints", action="store_true")
    args = parser.parse_args()

    package = load(args.backend, args.reference_site)
    values = signal(args.n_samples, args.n_features)
    results = []
    for changes in args.changes:
        samples = []
        breakpoints = None
        for _ in range(args.repeats):
            gc.collect()
            if args.backend == "rustures":
                parameters = {"gamma": 0.5} if args.kernel == "rbf" else {}
                detector = package.KernelCPD(
                    kernel=args.kernel, min_size=1, **parameters
                ).fit(values)
            else:
                parameters = {"params": {"gamma": 0.5}} if args.kernel == "rbf" else {}
                detector = package.KernelCPD(
                    kernel=args.kernel, min_size=1, **parameters
                ).fit(values)
            started = time.perf_counter()
            current = detector.predict(n_bkps=changes)
            samples.append(time.perf_counter() - started)
            breakpoints = [int(value) for value in current]
        result = {
            "changes": changes,
            "minimum_seconds": min(samples),
            "median_seconds": statistics.median(samples),
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
