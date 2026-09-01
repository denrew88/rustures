from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
import types
from typing import Callable

import numpy as np


SCORE_ABS_TOL = 1.0e-12
SCORE_REL_TOL = 1.0e-12
CHECK_ABS_TOL = 1.0e-8
CHECK_REL_TOL = 1.0e-9


def load_ruptures(reference_site: Path):
    sys.path.insert(0, str(reference_site.resolve()))
    version_module = types.ModuleType("ruptures.version")
    version_module.version = "0.0.0+ee1c8ff"
    sys.modules["ruptures.version"] = version_module
    import ruptures

    return ruptures


def as_2d(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    return values.reshape(-1, 1) if values.ndim == 1 else values


def public_shape(signal: np.ndarray) -> np.ndarray:
    return signal[:, 0] if signal.shape[1] == 1 else signal


def candidate_positions(n_samples: int, jump: int) -> list[int]:
    return [0, *range(jump, n_samples, jump), n_samples]


def tied(left: float, right: float) -> bool:
    return abs(left - right) <= SCORE_ABS_TOL + SCORE_REL_TOL * max(
        abs(left), abs(right)
    )


def objective_close(left: float, right: float) -> bool:
    return bool(
        np.isclose(left, right, rtol=CHECK_REL_TOL, atol=CHECK_ABS_TOL)
    )


def l2_cost_matrix(signal: np.ndarray) -> np.ndarray:
    values = as_2d(signal)
    n_samples = values.shape[0]
    costs = np.full((n_samples + 1, n_samples + 1), np.inf)
    for start in range(n_samples):
        for end in range(start + 1, n_samples + 1):
            segment = values[start:end]
            # Translation before the mean keeps the oracle stable when a
            # small signal variation rides on a very large common baseline.
            translated = segment - segment[0]
            centered = translated - translated.mean(axis=0)
            costs[start, end] = float(np.sum(centered * centered))
    return costs


def kernel_gram(
    signal: np.ndarray,
    kernel: str,
    gamma: float,
    *,
    ruptures_rbf_clipping: bool = False,
) -> np.ndarray:
    values = as_2d(signal)
    if kernel == "linear":
        return values @ values.T
    if kernel == "rbf":
        differences = values[:, None, :] - values[None, :, :]
        distances = np.sum(differences * differences, axis=2) * gamma
        if ruptures_rbf_clipping:
            distances = np.clip(distances, 1.0e-2, 1.0e2)
            np.fill_diagonal(distances, 0.0)
        return np.exp(-distances)
    if kernel == "cosine":
        norms = np.linalg.norm(values, axis=1)
        gram = np.zeros((len(values), len(values)), dtype=np.float64)
        nonzero = norms != 0.0
        if np.any(nonzero):
            normalized = values[nonzero] / norms[nonzero, None]
            gram[np.ix_(nonzero, nonzero)] = normalized @ normalized.T
        zero = ~nonzero
        gram[np.ix_(zero, zero)] = 1.0
        np.fill_diagonal(gram, 1.0)
        return gram
    raise ValueError(f"unknown kernel: {kernel}")


def kernel_cost_matrix(gram: np.ndarray) -> np.ndarray:
    n_samples = gram.shape[0]
    costs = np.full((n_samples + 1, n_samples + 1), np.inf)
    for start in range(n_samples):
        for end in range(start + 1, n_samples + 1):
            block = gram[start:end, start:end]
            value = float(np.trace(block) - block.sum() / (end - start))
            if value < 0.0 and value >= -1.0e-10:
                value = 0.0
            costs[start, end] = value
    return costs


def partition_cost(costs: np.ndarray, breakpoints: list[int]) -> float:
    start = 0
    total = 0.0
    for end in breakpoints:
        total += float(costs[start, end])
        start = end
    return total


def better(
    candidate_cost: float,
    candidate_path: tuple[int, ...],
    best_cost: float,
    best_path: tuple[int, ...] | None,
) -> bool:
    if best_path is None:
        return True
    if tied(candidate_cost, best_cost):
        return candidate_path < best_path
    return candidate_cost < best_cost


def exhaustive_fixed(
    costs: np.ndarray, changes: int, min_size: int, jump: int
) -> tuple[list[int], float] | None:
    n_samples = costs.shape[0] - 1
    internal = candidate_positions(n_samples, jump)[1:-1]
    best_path: tuple[int, ...] | None = None
    best_cost = np.inf
    for selected in combinations(internal, changes):
        path = (*selected, n_samples)
        starts = (0, *selected)
        if any(end - start < min_size for start, end in zip(starts, path)):
            continue
        current = partition_cost(costs, list(path))
        if better(current, path, best_cost, best_path):
            best_path = path
            best_cost = current
    if best_path is None:
        return None
    return list(best_path), best_cost


def exhaustive_penalty(
    costs: np.ndarray, penalty: float, min_size: int, jump: int
) -> tuple[list[int], float]:
    n_samples = costs.shape[0] - 1
    internal = candidate_positions(n_samples, jump)[1:-1]
    best_path: tuple[int, ...] | None = None
    best_cost = np.inf
    for changes in range(len(internal) + 1):
        for selected in combinations(internal, changes):
            path = (*selected, n_samples)
            starts = (0, *selected)
            if any(end - start < min_size for start, end in zip(starts, path)):
                continue
            current = partition_cost(costs, list(path)) + penalty * changes
            if better(current, path, best_cost, best_path):
                best_path = path
                best_cost = current
    assert best_path is not None
    return list(best_path), best_cost


def reference_fixed(
    costs: np.ndarray, changes: int, min_size: int, jump: int
) -> tuple[list[int], float] | None:
    positions = candidate_positions(costs.shape[0] - 1, jump)
    previous: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for _ in range(changes + 1):
        current: dict[int, tuple[float, tuple[int, ...]]] = {}
        for end in positions[1:]:
            for start, (left_cost, left_path) in previous.items():
                if start >= end or end - start < min_size:
                    continue
                candidate_path = (*left_path, end)
                candidate_cost = left_cost + float(costs[start, end])
                incumbent = current.get(end)
                if incumbent is None or better(
                    candidate_cost,
                    candidate_path,
                    incumbent[0],
                    incumbent[1],
                ):
                    current[end] = (candidate_cost, candidate_path)
        previous = current
    result = previous.get(costs.shape[0] - 1)
    if result is None:
        return None
    return list(result[1]), result[0]


def reference_penalty(
    costs: np.ndarray, penalty: float, min_size: int, jump: int
) -> tuple[list[int], float]:
    positions = candidate_positions(costs.shape[0] - 1, jump)
    solutions: dict[int, tuple[float, tuple[int, ...]]] = {0: (-penalty, ())}
    for end in positions[1:]:
        best_path: tuple[int, ...] | None = None
        best_cost = np.inf
        for start in positions:
            if start >= end:
                break
            if end - start < min_size or start not in solutions:
                continue
            left_cost, left_path = solutions[start]
            candidate_path = (*left_path, end)
            candidate_cost = left_cost + float(costs[start, end]) + penalty
            if better(candidate_cost, candidate_path, best_cost, best_path):
                best_path = candidate_path
                best_cost = candidate_cost
        if best_path is not None:
            solutions[end] = (best_cost, best_path)
    result = solutions[costs.shape[0] - 1]
    return list(result[1]), result[0]


def make_signal(index: int, n_samples: int, n_features: int, seed: int) -> tuple[str, np.ndarray]:
    rng = np.random.default_rng(seed + index * 104_729)
    kind = [
        "gaussian",
        "piecewise_mean",
        "trend",
        "wavy",
        "variance_shift",
        "outliers",
        "integer_ties",
        "constant",
        "high_offset",
        "correlated",
    ][index % 10]
    time = np.arange(n_samples, dtype=np.float64)
    if kind == "gaussian":
        values = rng.normal(size=(n_samples, n_features))
    elif kind == "piecewise_mean":
        levels = rng.normal(scale=4.0, size=(3, n_features))
        segment = np.minimum(time.astype(int) * 3 // n_samples, 2)
        values = levels[segment] + rng.normal(scale=0.15, size=(n_samples, n_features))
    elif kind == "trend":
        slopes = rng.normal(scale=0.15, size=n_features)
        values = time[:, None] * slopes + rng.normal(scale=0.2, size=(n_samples, n_features))
    elif kind == "wavy":
        frequencies = np.arange(1, n_features + 1)
        values = np.sin(time[:, None] * frequencies * 0.37)
        values += rng.normal(scale=0.05, size=values.shape)
    elif kind == "variance_shift":
        scales = np.where(time < n_samples // 2, 0.1, 2.5)
        values = rng.normal(size=(n_samples, n_features)) * scales[:, None]
    elif kind == "outliers":
        values = rng.normal(scale=0.2, size=(n_samples, n_features))
        values[:: max(2, n_samples // 4)] += rng.normal(scale=15.0, size=n_features)
    elif kind == "integer_ties":
        values = rng.integers(-2, 3, size=(n_samples, n_features)).astype(np.float64)
    elif kind == "constant":
        values = np.full((n_samples, n_features), float(index % 3 - 1))
    elif kind == "high_offset":
        values = 1.0e12 + rng.normal(scale=0.25, size=(n_samples, n_features))
        values[n_samples // 2 :] += 2.0
    else:
        base = rng.normal(size=(n_samples, 1))
        weights = np.linspace(0.5, 1.5, n_features)[None, :]
        values = base * weights + rng.normal(scale=0.03, size=(n_samples, n_features))
    return kind, np.ascontiguousarray(values, dtype=np.float64)


class Validation:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.divergences: Counter[str] = Counter()
        self.failures: list[dict[str, object]] = []

    def exception(
        self, category: str, error: Exception, context: dict[str, object]
    ) -> None:
        self.counts[category] += 1
        self.failures.append(
            {
                "category": category,
                **context,
                "exception_type": type(error).__name__,
                "exception": str(error),
            }
        )

    def check(
        self,
        category: str,
        actual: list[int],
        expected: tuple[list[int], float],
        costs: np.ndarray,
        penalty: float = 0.0,
        *,
        require_path: bool = False,
        context: dict[str, object],
    ) -> None:
        self.counts[category] += 1
        actual_cost = partition_cost(costs, actual) + penalty * (len(actual) - 1)
        expected_path, expected_cost = expected
        if not objective_close(actual_cost, expected_cost):
            self.failures.append(
                {
                    "category": category,
                    **context,
                    "actual": actual,
                    "expected": expected_path,
                    "actual_objective": actual_cost,
                    "expected_objective": expected_cost,
                }
            )
        elif actual != expected_path:
            if require_path:
                self.failures.append(
                    {
                        "category": category,
                        **context,
                        "actual": actual,
                        "expected": expected_path,
                        "objective": actual_cost,
                        "reason": "non-canonical optimal path",
                    }
                )
            else:
                self.divergences[f"{category}:alternate_optimum"] += 1


def validate_small(
    rustures,
    validation: Validation,
    count: int,
    seed: int,
) -> None:
    for index in range(count):
        n_samples = 6 + index % 7
        n_features = 1 + (index // 7) % 3
        kind, signal = make_signal(index, n_samples, n_features, seed)
        public = public_shape(signal)
        l2_costs = l2_cost_matrix(signal)
        base_cost = max(float(l2_costs[0, n_samples]), 1.0e-6)
        penalties = [base_cost * factor for factor in (0.001, 0.03, 0.2, 1.0, 5.0)]

        for min_size in range(1, min(3, n_samples) + 1):
            for jump in (1, 2, 3):
                for changes in range(0, 4):
                    expected = exhaustive_fixed(l2_costs, changes, min_size, jump)
                    if expected is None:
                        continue
                    actual = rustures.Dynp(min_size=min_size, jump=jump).fit_predict(
                        public, n_bkps=changes
                    )
                    validation.check(
                        "small:dynp:exhaustive",
                        actual,
                        expected,
                        l2_costs,
                        require_path=True,
                        context={
                            "case": index,
                            "kind": kind,
                            "n": n_samples,
                            "d": n_features,
                            "min_size": min_size,
                            "jump": jump,
                            "K": changes,
                        },
                    )
                for penalty in penalties:
                    expected = exhaustive_penalty(l2_costs, penalty, min_size, jump)
                    actual = rustures.Pelt(min_size=min_size, jump=jump).fit_predict(
                        public, pen=penalty
                    )
                    validation.check(
                        "small:pelt:exhaustive",
                        actual,
                        expected,
                        l2_costs,
                        penalty,
                        require_path=True,
                        context={
                            "case": index,
                            "kind": kind,
                            "n": n_samples,
                            "d": n_features,
                            "min_size": min_size,
                            "jump": jump,
                            "penalty": penalty,
                        },
                    )

        if index >= max(12, count // 2):
            continue
        for kernel in ("linear", "rbf", "cosine"):
            gamma = (0.1, 0.5, 2.0)[index % 3]
            costs = (
                l2_cost_matrix(signal)
                if kernel == "linear"
                else kernel_cost_matrix(kernel_gram(signal, kernel, gamma))
            )
            base_kernel_cost = max(float(costs[0, n_samples]), 1.0e-6)
            kernel_penalties = [base_kernel_cost * factor for factor in (0.03, 0.3, 2.0)]
            parameters = {"gamma": gamma} if kernel == "rbf" else {}
            for min_size in (1, 2):
                if min_size > n_samples:
                    continue
                for jump in (1, 2, 3):
                    detectors = {
                        backend: rustures.KernelCPD(
                            kernel=kernel,
                            min_size=min_size,
                            jump=jump,
                            backend=backend,
                            **parameters,
                        ).fit(public)
                        for backend in ("fused", "full", "streaming")
                    }
                    for changes in range(0, 4):
                        expected = exhaustive_fixed(costs, changes, min_size, jump)
                        if expected is None:
                            continue
                        for backend, detector in detectors.items():
                            context = {
                                "case": index,
                                "kind": kind,
                                "n": n_samples,
                                "d": n_features,
                                "kernel": kernel,
                                "backend": backend,
                                "min_size": min_size,
                                "jump": jump,
                                "K": changes,
                            }
                            try:
                                actual = detector.predict(n_bkps=changes)
                            except Exception as error:
                                validation.exception(
                                    f"small:kernel-{kernel}:{backend}:fixed",
                                    error,
                                    context,
                                )
                                continue
                            validation.check(
                                f"small:kernel-{kernel}:{backend}:fixed",
                                actual,
                                expected,
                                costs,
                                require_path=backend != "fused",
                                context=context,
                            )
                    for penalty in kernel_penalties:
                        expected = exhaustive_penalty(costs, penalty, min_size, jump)
                        for backend, detector in detectors.items():
                            context = {
                                "case": index,
                                "kind": kind,
                                "n": n_samples,
                                "d": n_features,
                                "kernel": kernel,
                                "backend": backend,
                                "min_size": min_size,
                                "jump": jump,
                                "penalty": penalty,
                            }
                            try:
                                actual = detector.predict(pen=penalty)
                            except Exception as error:
                                validation.exception(
                                    f"small:kernel-{kernel}:{backend}:penalty",
                                    error,
                                    context,
                                )
                                continue
                            validation.check(
                                f"small:kernel-{kernel}:{backend}:penalty",
                                actual,
                                expected,
                                costs,
                                penalty,
                                require_path=backend != "fused",
                                context=context,
                            )


def validate_medium(
    rustures,
    ruptures,
    validation: Validation,
    count: int,
    seed: int,
) -> None:
    for index in range(count):
        n_samples = (20, 31, 48, 64)[index % 4]
        n_features = 1 + (index // 4) % 4
        kind, signal = make_signal(index + 10_000, n_samples, n_features, seed)
        public = public_shape(signal)
        l2_costs = l2_cost_matrix(signal)
        base_cost = max(float(l2_costs[0, n_samples]), 1.0e-6)
        for min_size in (1, 2, 4):
            if min_size > n_samples:
                continue
            for jump in (1, 2, 5):
                rust_dynp = rustures.Dynp(min_size=min_size, jump=jump).fit(public)
                c_dynp = ruptures.Dynp(model="l2", min_size=min_size, jump=jump).fit(public)
                for changes in (1, 2, 4, 6):
                    expected = reference_fixed(l2_costs, changes, min_size, jump)
                    if expected is None:
                        continue
                    context = {
                        "case": index,
                        "kind": kind,
                        "n": n_samples,
                        "d": n_features,
                        "min_size": min_size,
                        "jump": jump,
                        "K": changes,
                    }
                    rust_path = rust_dynp.predict(changes)
                    validation.check(
                        "medium:dynp:reference",
                        rust_path,
                        expected,
                        l2_costs,
                        require_path=True,
                        context=context,
                    )
                    c_path = c_dynp.predict(changes)
                    validation.check(
                        "medium:ruptures-dynp:reference",
                        c_path,
                        expected,
                        l2_costs,
                        context=context,
                    )
                    if rust_path != c_path:
                        validation.divergences["medium:dynp:rust-vs-ruptures-path"] += 1

                rust_pelt = rustures.Pelt(min_size=min_size, jump=jump).fit(public)
                c_pelt = ruptures.Pelt(model="l2", min_size=min_size, jump=jump).fit(public)
                for factor in (0.01, 0.1, 0.5, 2.0, 10.0):
                    penalty = base_cost * factor
                    expected = reference_penalty(l2_costs, penalty, min_size, jump)
                    context = {
                        "case": index,
                        "kind": kind,
                        "n": n_samples,
                        "d": n_features,
                        "min_size": min_size,
                        "jump": jump,
                        "penalty": penalty,
                    }
                    rust_path = rust_pelt.predict(penalty)
                    validation.check(
                        "medium:pelt:reference",
                        rust_path,
                        expected,
                        l2_costs,
                        penalty,
                        require_path=True,
                        context=context,
                    )
                    c_path = c_pelt.predict(penalty)
                    validation.check(
                        "medium:ruptures-pelt:reference",
                        c_path,
                        expected,
                        l2_costs,
                        penalty,
                        context=context,
                    )
                    if rust_path != c_path:
                        validation.divergences["medium:pelt:rust-vs-ruptures-path"] += 1

        if index >= max(8, count // 2):
            continue
        for kernel in ("linear", "rbf", "cosine"):
            # Avoid SciPy cosine NaNs in the external differential layer.
            kernel_signal = signal.copy()
            zero_rows = np.linalg.norm(kernel_signal, axis=1) == 0.0
            kernel_signal[zero_rows, 0] = 0.25
            kernel_public = public_shape(kernel_signal)
            gamma = (0.1, 0.5, 2.0)[index % 3]
            if kernel == "linear":
                costs = l2_cost_matrix(kernel_signal)
                c_costs = costs
            else:
                costs = kernel_cost_matrix(kernel_gram(kernel_signal, kernel, gamma))
                c_costs = kernel_cost_matrix(
                    kernel_gram(
                        kernel_signal,
                        kernel,
                        gamma,
                        ruptures_rbf_clipping=kernel == "rbf",
                    )
                )
            parameters = {"gamma": gamma} if kernel == "rbf" else {}
            for min_size in (1, 2):
                detectors = {
                    backend: rustures.KernelCPD(
                        kernel=kernel,
                        min_size=min_size,
                        jump=1,
                        backend=backend,
                        **parameters,
                    ).fit(kernel_public)
                    for backend in ("fused", "full", "streaming")
                }
                c_parameters = {"params": {"gamma": gamma}} if kernel == "rbf" else {}
                c_detector = ruptures.KernelCPD(
                    kernel=kernel, min_size=min_size, **c_parameters
                ).fit(kernel_public)
                for changes in (1, 2, 4, 8):
                    expected = reference_fixed(costs, changes, min_size, 1)
                    c_expected = reference_fixed(c_costs, changes, min_size, 1)
                    if expected is None or c_expected is None:
                        continue
                    context = {
                        "case": index,
                        "kind": kind,
                        "n": n_samples,
                        "d": n_features,
                        "kernel": kernel,
                        "min_size": min_size,
                        "K": changes,
                    }
                    rust_paths = {}
                    for backend, detector in detectors.items():
                        backend_context = {**context, "backend": backend}
                        try:
                            rust_paths[backend] = detector.predict(n_bkps=changes)
                        except Exception as error:
                            validation.exception(
                                f"medium:kernel-{kernel}:{backend}:reference",
                                error,
                                backend_context,
                            )
                            continue
                        validation.check(
                            f"medium:kernel-{kernel}:{backend}:reference",
                            rust_paths[backend],
                            expected,
                            costs,
                            require_path=backend != "fused",
                            context=backend_context,
                        )
                    c_path = c_detector.predict(n_bkps=changes)
                    validation.check(
                        f"medium:ruptures-kernel-{kernel}:reference",
                        c_path,
                        c_expected,
                        c_costs,
                        context=context,
                    )
                    if "fused" in rust_paths and rust_paths["fused"] != c_path:
                        validation.divergences[
                            f"medium:kernel-{kernel}:rust-vs-ruptures-path"
                        ] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-site", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--small-cases", type=int, default=60)
    parser.add_argument("--medium-cases", type=int, default=32)
    parser.add_argument("--failure-limit", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import rustures

    ruptures = load_ruptures(args.reference_site)
    validation = Validation()
    validate_small(rustures, validation, args.small_cases, args.seed)
    validate_medium(rustures, ruptures, validation, args.medium_cases, args.seed)
    report = {
        "seed": args.seed,
        "small_cases": args.small_cases,
        "medium_cases": args.medium_cases,
        "checks": dict(sorted(validation.counts.items())),
        "total_checks": sum(validation.counts.values()),
        "expected_or_tied_divergences": dict(sorted(validation.divergences.items())),
        "failure_count": len(validation.failures),
        "failures": validation.failures[: args.failure_limit],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if validation.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
