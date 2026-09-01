"""Black-box parity checks for stage 7 costs/search and stage 8 L1-Potts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types

import numpy as np


def load_packages(reference_site: Path, rustures_site: Path):
    version_module = types.ModuleType("ruptures.version")
    version_module.version = "pinned-reference"
    sys.modules["ruptures.version"] = version_module
    sys.path.insert(0, str(reference_site.resolve()))
    import ruptures  # type: ignore

    sys.path.insert(0, str(rustures_site.resolve()))
    import rustures

    return ruptures, rustures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-site", type=Path, required=True)
    parser.add_argument("--rustures-site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rpt, rst = load_packages(args.reference_site, args.rustures_site)

    rng = np.random.default_rng(20260831)
    scalar = np.r_[rng.normal(0, 0.4, 12), rng.normal(5, 0.5, 12), rng.normal(-2, 0.3, 12)]
    multi = np.column_stack((scalar, rng.normal(size=36)))
    x = np.linspace(-2, 2, 36)
    linear = np.column_stack((2 + 3 * x + rng.normal(0, 0.05, 36), np.ones(36), x))
    metric = np.array([[2.0, 0.25], [0.25, 0.75]])
    segments = [(0, 12), (6, 24), (12, 36)]

    cases = [
        ("l1", rpt.costs.CostL1(), rst.CostL1(), multi),
        ("rank", rpt.costs.CostRank(), rst.CostRank(), multi),
        ("normal", rpt.costs.CostNormal(add_small_diag=True), rst.CostNormal(), multi),
        ("linear", rpt.costs.CostLinear(), rst.CostLinear(), linear),
        ("ar", rpt.costs.CostAR(order=2), rst.CostAR(order=2), scalar),
        ("clinear", rpt.costs.CostCLinear(), rst.CostCLinear(), scalar),
        ("mahalanobis", rpt.costs.CostMl(metric=metric), rst.CostMahalanobis(metric), multi),
    ]
    costs = {}
    for name, reference, candidate, signal in cases:
        reference.fit(signal)
        candidate.fit(signal)
        rows = []
        for start, end in segments:
            try:
                expected = float(np.asarray(reference.error(start, end)).item())
                actual = float(candidate.error(start, end))
                rows.append({
                    "segment": [start, end],
                    "reference": expected,
                    "rustures": actual,
                    "absolute_error": abs(actual - expected),
                })
            except Exception as error:
                rows.append({"segment": [start, end], "error": str(error)})
        costs[name] = rows

    search = {}
    search_signals = {
        "l1": scalar,
        "rank": scalar,
        "normal": scalar,
        "linear": linear,
        "ar": scalar,
        "clinear": scalar,
        "mahalanobis": scalar,
    }
    reference_names = {"mahalanobis": "mahalanobis"}
    for name, signal in search_signals.items():
        min_size = 6 if name == "ar" else 3
        try:
            reference = rpt.Dynp(model=reference_names.get(name, name), min_size=min_size, jump=1).fit(signal).predict(2)
        except Exception as error:
            reference = {"error": str(error)}
        try:
            candidate = rst.Dynp(model=name, min_size=min_size, jump=1).fit(signal).predict(2)
        except Exception as error:
            candidate = {"error": str(error)}
        search[name] = {"reference": reference, "rustures": candidate}

    potts_signal = np.array([0.0, 0.0, 1.0, 8.0, 9.0, 9.0, 2.0, 2.0])
    potts = {"rustures": rst.L1Potts().fit_predict(potts_signal, pen=2.5)}
    try:
        potts["reference"] = rpt.L1Potts().fit(potts_signal).predict(pen=2.5)
    except Exception as error:
        potts["reference_error"] = str(error)

    report = {"seed": 20260831, "costs": costs, "dynp": search, "l1_potts": potts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
