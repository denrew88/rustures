"""Measure translation stability of L2 and linear-kernel detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from validate_algorithms import l2_cost_matrix, reference_fixed, reference_penalty


def public_shape(signal: np.ndarray) -> np.ndarray:
    return signal[:, 0] if signal.shape[1] == 1 else signal


def objective(costs: np.ndarray, path: list[int], penalty: float = 0.0) -> float:
    start = 0
    total = penalty * (len(path) - 1)
    for end in path:
        total += float(costs[start, end])
        start = end
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    import rustures

    rng = np.random.default_rng(args.seed)
    signal = rng.normal(scale=0.5, size=(31, 3))
    signal[10:21] += np.array([3.0, -2.0, 1.0])
    signal[21:] += np.array([-1.0, 4.0, -3.0])
    offsets = [0.0, *[10.0**power for power in range(1, 16)]]
    fixed_changes = (1, 3, 6)
    records: list[dict[str, object]] = []

    for offset in offsets:
        shifted = signal + offset
        public = public_shape(shifted)
        costs = l2_cost_matrix(shifted)
        penalty = max(float(costs[0, len(shifted)]) * 0.2, 1.0e-6)

        for algorithm in ("dynp", "pelt"):
            try:
                if algorithm == "dynp":
                    actual = rustures.Dynp(min_size=2, jump=1).fit_predict(public, 3)
                    expected = reference_fixed(costs, 3, 2, 1)
                else:
                    actual = rustures.Pelt(min_size=2, jump=1).fit_predict(public, penalty)
                    expected = reference_penalty(costs, penalty, 2, 1)
                assert expected is not None
                records.append(
                    {
                        "offset": offset,
                        "algorithm": algorithm,
                        "status": "ok" if actual == expected[0] else "wrong",
                        "actual": actual,
                        "expected": expected[0],
                        "objective_gap": objective(
                            costs, actual, penalty if algorithm == "pelt" else 0.0
                        )
                        - expected[1],
                    }
                )
            except Exception as error:  # keep the entire sweep observable
                records.append(
                    {
                        "offset": offset,
                        "algorithm": algorithm,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

        for backend in ("fused", "full", "streaming"):
            detector = rustures.KernelCPD(
                kernel="linear", min_size=2, jump=1, backend=backend
            ).fit(public)
            for changes in fixed_changes:
                expected = reference_fixed(costs, changes, 2, 1)
                assert expected is not None
                try:
                    actual = detector.predict(n_bkps=changes)
                    records.append(
                        {
                            "offset": offset,
                            "algorithm": "kernel-linear-fixed",
                            "backend": backend,
                            "K": changes,
                            "status": "ok" if actual == expected[0] else "wrong",
                            "actual": actual,
                            "expected": expected[0],
                            "objective_gap": objective(costs, actual) - expected[1],
                        }
                    )
                except Exception as error:
                    records.append(
                        {
                            "offset": offset,
                            "algorithm": "kernel-linear-fixed",
                            "backend": backend,
                            "K": changes,
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

            expected = reference_penalty(costs, penalty, 2, 1)
            try:
                actual = detector.predict(pen=penalty)
                records.append(
                    {
                        "offset": offset,
                        "algorithm": "kernel-linear-penalty",
                        "backend": backend,
                        "status": "ok" if actual == expected[0] else "wrong",
                        "actual": actual,
                        "expected": expected[0],
                        "objective_gap": objective(costs, actual, penalty) - expected[1],
                    }
                )
            except Exception as error:
                records.append(
                    {
                        "offset": offset,
                        "algorithm": "kernel-linear-penalty",
                        "backend": backend,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    summary: dict[str, dict[str, int]] = {}
    for record in records:
        key = str(record["offset"])
        counts = summary.setdefault(key, {"ok": 0, "wrong": 0, "error": 0})
        counts[str(record["status"])] += 1
    result = {"seed": args.seed, "summary": summary, "records": records}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({"seed": args.seed, "summary": summary}, indent=2))
    return int(any(record["status"] != "ok" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
