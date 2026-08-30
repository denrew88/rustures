from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


def load(name: str, reference_site: Path | None):
    if name.startswith("rustures"):
        import rustures
        return rustures
    assert reference_site is not None
    sys.path.insert(0, str(reference_site.resolve()))
    version_module = types.ModuleType("ruptures.version")
    version_module.version = "0.0.0+ee1c8ff"
    sys.modules["ruptures.version"] = version_module
    import ruptures
    return ruptures


def signal(n: int) -> np.ndarray:
    index = np.arange(n)
    levels = np.asarray([0.0, 5.0, -3.0, 7.0])
    noise = (((index * 48271 + index * index * 31) % 104729) / 104729 - 0.5) * 0.1
    return levels[np.minimum(index * 4 // n, 3)] + noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["rustures-fused", "rustures-full", "rustures-streaming", "ruptures"])
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-site", type=Path)
    args = parser.parse_args()
    package = load(args.backend, args.reference_site)
    results = []
    for n in args.sizes:
        values = signal(n)
        samples = []
        for _ in range(args.repeats):
            gc.collect()
            started = time.perf_counter()
            if args.backend == "ruptures":
                detector = package.KernelCPD(kernel="rbf", min_size=1).fit(values)
            else:
                detector = package.KernelCPD(
                    kernel="rbf",
                    min_size=1,
                    jump=1,
                    gamma_policy="exact",
                    backend=args.backend.removeprefix("rustures-"),
                ).fit(values)
            bkps = detector.predict(n_bkps=3)
            samples.append((time.perf_counter() - started, [int(x) for x in bkps]))
        elapsed, bkps = min(samples)
        results.append({"n": n, "seconds": elapsed, "breakpoints": bkps})
    print(json.dumps({"backend": args.backend, "results": results}, indent=2))


if __name__ == "__main__":
    main()
