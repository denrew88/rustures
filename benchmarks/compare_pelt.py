from __future__ import annotations

import argparse
import gc
import hashlib
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
        raise ValueError("--reference-site is required for a ruptures backend")
    sys.path.insert(0, str(reference_site.resolve()))
    version_module = types.ModuleType("ruptures.version")
    version_module.version = "0.0.0+ee1c8ff"
    sys.modules["ruptures.version"] = version_module
    import ruptures

    return ruptures


def make_signal(n_samples: int, scenario: str) -> np.ndarray:
    index = np.arange(n_samples, dtype=np.int64)
    if scenario == "no-change":
        level = np.zeros(n_samples, dtype=np.float64)
    elif scenario == "sparse":
        segment = np.minimum(index * 4 // n_samples, 3)
        level = np.asarray([0.0, 7.0, -4.0, 5.0])[segment]
    elif scenario == "frequent":
        segment = (index // 50) % 4
        level = np.asarray([0.0, 7.0, -4.0, 5.0])[segment]
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    noise_code = (index * 48_271 + index * index * 31) % 104_729
    noise = (noise_code.astype(np.float64) / 104_729.0 - 0.5) * 0.1
    return level + noise


def breakpoint_digest(breakpoints: list[int]) -> str:
    values = np.asarray(breakpoints, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def measure(
    backend_name: str,
    backend,
    signal: np.ndarray,
    scenario: str,
    penalty: float,
    repeats: int,
) -> dict[str, object]:
    samples: list[tuple[float, float, float, list[int]]] = []
    for _ in range(repeats):
        gc.collect()
        total_started = time.perf_counter()
        fit_started = time.perf_counter()
        if backend_name == "ruptures-kernelcpd":
            detector = backend.KernelCPD(kernel="linear", min_size=2).fit(signal)
        else:
            detector = backend.Pelt(model="l2", min_size=2, jump=1).fit(signal)
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        breakpoints = [int(value) for value in detector.predict(pen=penalty)]
        predict_seconds = time.perf_counter() - predict_started
        total_seconds = time.perf_counter() - total_started
        if not breakpoints or breakpoints[-1] != signal.size:
            raise RuntimeError(f"invalid breakpoints: {breakpoints!r}")
        samples.append((total_seconds, fit_seconds, predict_seconds, breakpoints))

    total_seconds, fit_seconds, predict_seconds, breakpoints = min(
        samples, key=lambda sample: sample[0]
    )
    return {
        "scenario": scenario,
        "n": int(signal.size),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": total_seconds,
        "median_total_seconds": statistics.median(sample[0] for sample in samples),
        "changes": len(breakpoints) - 1,
        "breakpoint_digest": breakpoint_digest(breakpoints),
        "first_breakpoints": breakpoints[:5],
        "last_breakpoints": breakpoints[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend", choices=["rustures", "ruptures-pelt", "ruptures-kernelcpd"]
    )
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["no-change", "sparse", "frequent"],
        default=["no-change", "sparse", "frequent"],
    )
    parser.add_argument("--penalty", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-site", type=Path)
    parser.add_argument("--rustures-site", type=Path)
    args = parser.parse_args()

    import_backend = "rustures" if args.backend == "rustures" else "ruptures"
    backend = load_backend(import_backend, args.reference_site, args.rustures_site)
    results = [
        measure(
            args.backend,
            backend,
            make_signal(size, scenario),
            scenario,
            args.penalty,
            args.repeats,
        )
        for scenario in args.scenarios
        for size in args.sizes
    ]
    print(
        json.dumps(
            {
                "backend": args.backend,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "penalty": args.penalty,
                "min_size": 2,
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
