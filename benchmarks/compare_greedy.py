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
    if name == "rustures":
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
    levels = np.asarray([0.0, 8.0, -5.0, 4.0])
    return levels[np.minimum(index * 4 // n, 3)] + ((index * 7919) % 101) * 1e-4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["rustures", "ruptures"])
    parser.add_argument("--algorithm", choices=["binseg", "bottomup", "window"], required=True)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-site", type=Path)
    args = parser.parse_args()
    package = load(args.backend, args.reference_site)
    classes = {"binseg": package.Binseg, "bottomup": package.BottomUp, "window": package.Window}
    results = []
    for n in args.sizes:
        values = signal(n)
        samples = []
        for _ in range(args.repeats):
            gc.collect()
            started = time.perf_counter()
            kwargs = {"model": "l2", "min_size": 2, "jump": 1}
            if args.algorithm == "window":
                kwargs["width"] = max(6, n // 10 // 2 * 2)
            bkps = classes[args.algorithm](**kwargs).fit(values).predict(n_bkps=3)
            samples.append((time.perf_counter() - started, [int(x) for x in bkps]))
        elapsed, bkps = min(samples)
        results.append({"n": n, "seconds": elapsed, "breakpoints": bkps})
    print(json.dumps({"backend": args.backend, "algorithm": args.algorithm, "results": results}, indent=2))


if __name__ == "__main__":
    main()
