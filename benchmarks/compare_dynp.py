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


def load_backend(
    name: str, reference_site: Path | None, rustures_site: Path | None
):
    if name == "rustures":
        if rustures_site is not None:
            sys.path.insert(0, str(rustures_site.resolve()))
        import rustures

        return rustures

    if reference_site is None:
        raise ValueError("--reference-site is required for the ruptures backend")
    sys.path.insert(0, str(reference_site.resolve()))
    version_module = types.ModuleType("ruptures.version")
    version_module.version = "0.0.0+ee1c8ff"
    sys.modules["ruptures.version"] = version_module
    import ruptures

    return ruptures


def make_signal(n_samples: int) -> np.ndarray:
    index = np.arange(n_samples, dtype=np.int64)
    segment = np.minimum(index * 6 // n_samples, 5)
    levels = np.asarray([0.0, 7.0, -4.0, 11.0, 3.0, -8.0])
    noise_code = (index * 48_271 + index * index * 31) % 104_729
    noise = (noise_code.astype(np.float64) / 104_729.0 - 0.5) * 0.1
    return levels[segment] + noise


def measure(
    backend_name: str,
    backend,
    signal: np.ndarray,
    changes: int,
    repeats: int,
) -> dict[str, object]:
    samples: list[tuple[float, float, float, list[int]]] = []
    for _ in range(repeats):
        gc.collect()
        total_started = time.perf_counter()
        fit_started = time.perf_counter()
        if backend_name == "ruptures-kernelcpd":
            detector = backend.KernelCPD(kernel="linear", min_size=1).fit(signal)
        else:
            detector = backend.Dynp(model="l2", min_size=1, jump=1).fit(signal)
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        breakpoints = detector.predict(n_bkps=changes)
        predict_seconds = time.perf_counter() - predict_started
        total_seconds = time.perf_counter() - total_started
        if len(breakpoints) != changes + 1 or breakpoints[-1] != signal.size:
            raise RuntimeError(f"invalid breakpoints: {breakpoints!r}")
        samples.append((total_seconds, fit_seconds, predict_seconds, breakpoints))

    total_seconds, fit_seconds, predict_seconds, breakpoints = min(
        samples, key=lambda sample: sample[0]
    )
    return {
        "n": int(signal.size),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
        "median_total_seconds": statistics.median(sample[0] for sample in samples),
        "breakpoints": breakpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend", choices=["rustures", "ruptures", "ruptures-kernelcpd"]
    )
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--changes", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-site", type=Path)
    parser.add_argument("--rustures-site", type=Path)
    args = parser.parse_args()

    import_backend = "ruptures" if args.backend == "ruptures-kernelcpd" else args.backend
    backend = load_backend(import_backend, args.reference_site, args.rustures_site)
    results = [
        measure(args.backend, backend, make_signal(size), args.changes, args.repeats)
        for size in args.sizes
    ]
    print(
        json.dumps(
            {
                "backend": args.backend,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "changes": args.changes,
                "min_size": 1,
                "jump": 1,
                "repeats": args.repeats,
                "selection": "minimum total time; fit and predict from same repetition",
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
