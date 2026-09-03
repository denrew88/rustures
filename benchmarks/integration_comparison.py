"""Cross-library integration, timing, and process-memory comparison.

The parent process launches one fresh worker per library and case. A worker
imports one package, builds deterministic input, reports its post-import RSS,
then waits. The parent starts RSS sampling and lets it execute one memory probe
followed by warmed timing repetitions. This keeps Rustures and ruptures imports,
caches, and native allocators isolated from each other.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np


SEED = 20260902
COMMON_LIBRARIES = ("rustures", "ruptures")
MODELS = ("l2", "l1", "rank", "normal", "linear", "ar", "clinear", "mahalanobis")
ALGORITHMS = ("dynp", "pelt", "binseg", "bottomup", "window")
KERNELS = ("linear", "rbf", "cosine")
KERNEL_BACKENDS = ("fused", "full", "streaming")
METRICS = ("hausdorff", "precision_recall", "rand_index")
GENERATORS = ("pw_constant", "pw_linear", "pw_normal", "pw_wavy")


def signal_scale(profile: str) -> float:
    return 0.45 if profile == "quick" else 1.0


def scaled(value: int, profile: str, minimum: int = 36) -> int:
    return max(minimum, int(round(value * signal_scale(profile))))


def case_registry(profile: str) -> list[dict[str, Any]]:
    """Return the deterministic feature-coverage and performance matrix."""
    cases: list[dict[str, Any]] = []

    def add(case_id: str, category: str, **values: Any) -> None:
        cases.append({"id": case_id, "category": category, **values})

    model_signals = {
        "l2": "piecewise_mean",
        "l1": "outliers",
        "rank": "integer_ties",
        "normal": "variance_shift",
        "linear": "linear_regression",
        "ar": "autoregressive",
        "clinear": "trend",
        "mahalanobis": "correlated",
    }
    model_features = {
        "l2": 3,
        "l1": 2,
        "rank": 2,
        "normal": 2,
        "linear": 3,
        "ar": 2,
        "clinear": 1,
        "mahalanobis": 3,
    }

    # Public cost classes: fit plus a deterministic set of interval queries.
    for index, model in enumerate(MODELS):
        add(
            f"cost-{model}",
            "cost",
            feature=model,
            signal=model_signals[model],
            n=scaled(420, profile),
            d=model_features[model],
            seed=SEED + index,
            queries=320 if profile == "standard" else 100,
            libraries=list(COMMON_LIBRARIES),
        )

    # Every detector sees several qualitatively different time series.
    detector_signals = ("piecewise_mean", "trend", "variance_shift", "outliers")
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        for signal_index, signal in enumerate(detector_signals):
            add(
                f"{algorithm}-l2-{signal}",
                "detector",
                feature=algorithm,
                algorithm=algorithm,
                model="l2",
                signal=signal,
                n=scaled(720 if algorithm != "pelt" else 1_200, profile),
                d=2 if signal != "trend" else 1,
                seed=SEED + 100 + algorithm_index * 10 + signal_index,
                min_size=8,
                jump=4,
                n_bkps=1 if algorithm == "window" else 2,
                penalty=12.0,
                width=80,
                libraries=list(COMMON_LIBRARIES),
            )

    # Cost/detector composition coverage uses fixed-K Dynp to avoid comparing
    # different PELT pruning assumptions for every statistical model.
    for index, model in enumerate(MODELS):
        add(
            f"dynp-model-{model}",
            "model_search",
            feature=f"Dynp({model})",
            algorithm="dynp",
            model=model,
            signal=model_signals[model],
            n=scaled(360, profile),
            d=model_features[model],
            seed=SEED + 300 + index,
            min_size=10 if model == "ar" else 6,
            jump=4,
            n_bkps=2,
            libraries=list(COMMON_LIBRARIES),
        )

    # Compare every exact Rustures kernel backend with the same public
    # ruptures KernelCPD call. Gamma is fixed so fit does not benchmark a
    # different median heuristic.
    for kernel_index, kernel in enumerate(KERNELS):
        for backend_index, backend in enumerate(KERNEL_BACKENDS):
            add(
                f"kernel-{kernel}-{backend}",
                "kernel",
                feature=f"KernelCPD({kernel},{backend})",
                kernel=kernel,
                backend=backend,
                signal="correlated" if kernel != "rbf" else "piecewise_mean",
                # All exact backends use the same size. Streaming keeps only
                # endpoint state while incrementally sharing kernel pairs
                # across the generic Dynp states.
                n=scaled(720, profile),
                d=3,
                seed=SEED + 500 + kernel_index * 10 + backend_index,
                min_size=6,
                jump=1,
                n_bkps=3,
                gamma=0.35,
                libraries=list(COMMON_LIBRARIES),
            )

    for algorithm in ("dynp", "pelt"):
        add(
            f"custom-scalar-{algorithm}",
            "custom_cost",
            feature=f"{algorithm}(custom scalar)",
            algorithm=algorithm,
            signal="piecewise_mean",
            n=scaled(420 if algorithm == "dynp" else 800, profile),
            d=2,
            seed=SEED + 700 + (algorithm == "pelt"),
            min_size=6,
            jump=4,
            n_bkps=2,
            penalty=12.0,
            batch=False,
            libraries=list(COMMON_LIBRARIES),
        )

    add(
        "custom-pruned-pelt",
        "custom_cost",
        feature="pelt(custom scalar, pruning opt-in)",
        algorithm="pelt",
        signal="piecewise_mean",
        n=scaled(800, profile),
        d=2,
        seed=SEED + 702,
        min_size=6,
        jump=4,
        penalty=12.0,
        batch=False,
        pelt_pruning_constant=0.0,
        libraries=list(COMMON_LIBRARIES),
    )

    add(
        "custom-batch-dynp",
        "rustures_only",
        feature="Dynp(custom error_many)",
        algorithm="dynp",
        signal="piecewise_mean",
        n=scaled(520, profile),
        d=2,
        seed=SEED + 720,
        min_size=6,
        jump=4,
        n_bkps=2,
        batch=True,
        libraries=["rustures"],
        unsupported_reason="ruptures custom costs have no endpoint error_many protocol",
    )

    add(
        "l1-potts",
        "rustures_only",
        feature="L1Potts",
        signal="integer_ties",
        n=scaled(1_400, profile),
        d=1,
        seed=SEED + 730,
        penalty=4.0,
        libraries=["rustures"],
        unsupported_reason="ruptures 1.1.10 has no L1Potts estimator",
    )

    for index, metric in enumerate(METRICS):
        add(
            f"metric-{metric}",
            "metric",
            feature=metric,
            metric=metric,
            n=scaled(200_000, profile, minimum=20_000),
            calls=500 if profile == "standard" else 100,
            seed=SEED + 800 + index,
            libraries=list(COMMON_LIBRARIES),
        )

    for index, generator in enumerate(GENERATORS):
        add(
            f"dataset-{generator}",
            "dataset",
            feature=generator,
            generator=generator,
            n=scaled(80_000, profile, minimum=10_000),
            d=2 if generator == "pw_normal" else (1 if generator == "pw_wavy" else 3),
            n_bkps=5,
            seed=SEED + 900 + index,
            libraries=list(COMMON_LIBRARIES),
        )

    return cases


def make_signal(case: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    n = int(case["n"])
    d = int(case.get("d", 1))
    rng = np.random.default_rng(int(case.get("seed", SEED)))
    boundaries = [n // 3, 2 * n // 3, n]
    segment_ids = np.minimum(np.arange(n) * 3 // n, 2)
    kind = case["signal"]
    time_axis = np.arange(n, dtype=np.float64)

    if kind == "piecewise_mean":
        levels = np.array([[-3.0, 1.0, 4.0], [5.0, -2.0, 0.5], [0.0, 6.0, -4.0]])
        values = levels[segment_ids, :d] + rng.normal(0.0, 0.35, size=(n, d))
    elif kind == "trend":
        values = np.empty((n, d), dtype=np.float64)
        slopes = (-0.035, 0.055, -0.02)
        levels = (0.0, 7.0, -4.0)
        starts = (0, boundaries[0], boundaries[1])
        for segment, (start, end) in enumerate(zip(starts, boundaries)):
            local = np.arange(end - start, dtype=np.float64)
            for feature in range(d):
                values[start:end, feature] = (
                    levels[segment]
                    + slopes[segment] * (feature + 1) * local
                    + rng.normal(0.0, 0.2, end - start)
                )
    elif kind == "variance_shift":
        scales = np.array([0.25, 2.2, 0.55])
        values = rng.normal(size=(n, d)) * scales[segment_ids, None]
    elif kind == "outliers":
        levels = np.array([-2.0, 4.0, 0.5])
        values = levels[segment_ids, None] + rng.normal(0.0, 0.3, size=(n, d))
        values[:: max(11, n // 31)] += rng.choice([-18.0, 18.0], size=(len(values[:: max(11, n // 31)]), d))
    elif kind == "integer_ties":
        levels = np.array([0.0, 6.0, 2.0])
        values = np.repeat(levels[segment_ids, None], d, axis=1)
        values += rng.integers(-1, 2, size=(n, d))
    elif kind == "correlated":
        base = np.array([-3.0, 4.0, 0.0])[segment_ids] + rng.normal(0.0, 0.4, n)
        columns = [base]
        for feature in range(1, d):
            columns.append((feature + 1) * 0.7 * base + rng.normal(0.0, 0.25, n))
        values = np.column_stack(columns)
    elif kind == "linear_regression":
        x = np.linspace(-2.0, 2.0, n)
        slopes = np.array([-2.0, 4.0, 0.75])
        intercepts = np.array([1.0, -3.0, 5.0])
        response = intercepts[segment_ids] + slopes[segment_ids] * x + rng.normal(0.0, 0.08, n)
        values = np.column_stack((response, np.ones(n), x))
    elif kind == "autoregressive":
        coefficients = (0.15, 0.88, -0.45)
        values = np.zeros((n, d), dtype=np.float64)
        values[0] = rng.normal(0.0, 0.2, d)
        for row in range(1, n):
            segment = segment_ids[row]
            innovation = rng.normal(0.0, 0.25, d)
            values[row] = coefficients[segment] * values[row - 1] + innovation
    else:
        raise ValueError(f"unknown signal kind: {kind}")

    public = values[:, 0] if d == 1 else values
    return np.ascontiguousarray(public, dtype=np.float64), boundaries


def normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def cost_instance(package: Any, library: str, model: str, signal: np.ndarray) -> Any:
    names = {
        "l2": "CostL2",
        "l1": "CostL1",
        "rank": "CostRank",
        "normal": "CostNormal",
        "linear": "CostLinear",
        "ar": "CostAR",
        "clinear": "CostCLinear",
        "mahalanobis": "CostMahalanobis" if library == "rustures" else "CostMl",
    }
    namespace = package if library == "rustures" else package.costs
    constructor = getattr(namespace, names[model])
    if model == "normal":
        return constructor(ridge=1.0e-6) if library == "rustures" else constructor(add_small_diag=True)
    if model == "ar":
        return constructor(order=4)
    if model == "mahalanobis":
        width = 1 if signal.ndim == 1 else signal.shape[1]
        metric = np.eye(width, dtype=np.float64)
        return constructor(metric)
    return constructor()


def custom_cost(
    library: str, *, batch: bool, pelt_pruning_constant: float | None
) -> Any:
    if library == "ruptures":
        from ruptures.base import BaseCost

        base = BaseCost
    else:
        base = object

    class CustomL2(base):  # type: ignore[misc, valid-type]
        min_size = 1
        model = "custom_l2"

        def __init__(self) -> None:
            self.signal: np.ndarray | None = None

        def fit(self, signal: Any) -> "CustomL2":
            values = np.asarray(signal, dtype=np.float64)
            self.signal = values.reshape(-1, 1) if values.ndim == 1 else values
            return self

        def error(self, start: int, end: int) -> float:
            assert self.signal is not None
            segment = self.signal[start:end]
            centered = segment - segment.mean(axis=0)
            return float(np.square(centered).sum())

    if batch and library == "rustures":
        def error_many(self: Any, starts: Any, ends: Any) -> np.ndarray:
            return np.asarray(
                [self.error(int(start), int(end)) for start, end in zip(starts, ends)],
                dtype=np.float64,
            )

        setattr(CustomL2, "error_many", error_many)
    if pelt_pruning_constant is not None:
        setattr(CustomL2, "pelt_pruning_constant", pelt_pruning_constant)
    return CustomL2()


def build_operation(
    package: Any, library: str, case: dict[str, Any]
) -> tuple[Callable[[], Any], dict[str, Any]]:
    category = case["category"]

    if category == "cost":
        signal, truth = make_signal(case)
        n = len(signal)
        min_size = 6 if case["feature"] == "ar" else 3
        queries = []
        for index in range(int(case["queries"])):
            start = (index * 37) % (n - min_size)
            maximum = n - start
            length = min_size + ((index * 53) % (maximum - min_size + 1))
            queries.append((start, start + length))

        def operation() -> dict[str, Any]:
            cost = cost_instance(package, library, case["feature"], signal).fit(signal)
            checksum = sum(float(cost.error(start, end)) for start, end in queries)
            return {"checksum": checksum, "queries": len(queries)}

        return operation, {"truth": truth, "n": n}

    if category in ("detector", "model_search"):
        signal, truth = make_signal(case)
        class_name = {
            "dynp": "Dynp",
            "pelt": "Pelt",
            "binseg": "Binseg",
            "bottomup": "BottomUp",
            "window": "Window",
        }[case["algorithm"]]

        def operation() -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "model": case["model"],
                "min_size": case["min_size"],
                "jump": case["jump"],
            }
            if case["algorithm"] == "window":
                kwargs["width"] = min(int(case["width"]), len(signal) // 2)
            detector = getattr(package, class_name)(**kwargs).fit(signal)
            if case["algorithm"] == "pelt":
                result = detector.predict(pen=case["penalty"])
            else:
                result = detector.predict(n_bkps=case["n_bkps"])
            return {"breakpoints": [int(value) for value in result]}

        return operation, {"truth": truth, "n": len(signal)}

    if category == "kernel":
        signal, truth = make_signal(case)

        def operation() -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "kernel": case["kernel"],
                "min_size": case["min_size"],
                "jump": case["jump"],
            }
            if library == "rustures":
                kwargs["backend"] = case["backend"]
                if case["kernel"] == "rbf":
                    kwargs["gamma"] = case["gamma"]
            elif case["kernel"] == "rbf":
                kwargs["params"] = {"gamma": case["gamma"]}
            detector = package.KernelCPD(**kwargs).fit(signal)
            result = detector.predict(n_bkps=case["n_bkps"])
            return {"breakpoints": [int(value) for value in result]}

        return operation, {"truth": truth, "n": len(signal)}

    if category in ("custom_cost", "rustures_only") and "algorithm" in case:
        signal, truth = make_signal(case)

        def operation() -> dict[str, Any]:
            estimator = getattr(package, "Dynp" if case["algorithm"] == "dynp" else "Pelt")(
                custom_cost=custom_cost(
                    library,
                    batch=bool(case.get("batch")),
                    pelt_pruning_constant=case.get("pelt_pruning_constant"),
                ),
                min_size=case["min_size"],
                jump=case["jump"],
            ).fit(signal)
            if case["algorithm"] == "dynp":
                result = estimator.predict(n_bkps=case["n_bkps"])
            else:
                result = estimator.predict(pen=case["penalty"])
            return {"breakpoints": [int(value) for value in result]}

        return operation, {"truth": truth, "n": len(signal)}

    if case["id"] == "l1-potts":
        signal, truth = make_signal(case)

        def operation() -> dict[str, Any]:
            result = package.L1Potts().fit_predict(signal, pen=case["penalty"])
            return {"breakpoints": [int(value) for value in result]}

        return operation, {"truth": truth, "n": len(signal)}

    if category == "metric":
        n = int(case["n"])
        truth = [n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n]
        prediction = [truth[0] + 7, truth[1] - 11, truth[2] + 3, truth[3] - 5, n]
        if library == "rustures":
            metric = getattr(package, case["metric"])
        else:
            import ruptures.metrics as reference_metrics

            name = "randindex" if case["metric"] == "rand_index" else case["metric"]
            metric = getattr(reference_metrics, name)

        def operation() -> dict[str, Any]:
            value: Any = None
            for _ in range(int(case["calls"])):
                value = metric(truth, prediction)
            return {"value": normalize(value), "calls": case["calls"]}

        return operation, {"truth": truth, "n": n}

    if category == "dataset":
        generator = getattr(package, case["generator"])

        def operation() -> dict[str, Any]:
            kwargs: dict[str, Any] = {
                "n_samples": case["n"],
                "n_bkps": case["n_bkps"],
                "seed": case["seed"],
            }
            if case["generator"] in ("pw_constant", "pw_linear"):
                # ruptures.pw_linear returns the time coordinate plus the
                # requested feature count; Rustures returns exactly d columns.
                kwargs["n_features"] = (
                    case["d"] - 1
                    if library == "ruptures" and case["generator"] == "pw_linear"
                    else case["d"]
                )
                kwargs["noise_std"] = 1.0
            elif case["generator"] == "pw_wavy":
                kwargs["noise_std"] = 1.0
            elif library == "rustures":
                kwargs["n_features"] = case["d"]
                kwargs["noise_std"] = 1.0
            values, breakpoints = generator(**kwargs)
            return {
                "shape": list(np.asarray(values).shape),
                "breakpoints": [int(value) for value in breakpoints],
            }

        return operation, {"n": int(case["n"])}

    raise ValueError(f"unsupported case: {case['id']}")


def current_rss() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def high_water_rss() -> int | None:
    import psutil

    info = psutil.Process().memory_info()
    value = getattr(info, "peak_wset", None)
    return int(value) if value is not None else None


def worker_main(library: str, case_json: str, repeats: int) -> int:
    case = json.loads(case_json)
    package = __import__(library)
    operation, metadata = build_operation(package, library, case)
    gc.collect()
    ready = {
        "event": "ready",
        "library": library,
        "version": importlib.metadata.version(library),
        "baseline_rss_bytes": current_rss(),
        "baseline_high_water_bytes": high_water_rss(),
        "metadata": metadata,
    }
    print(json.dumps(ready), flush=True)
    if not sys.stdin.readline():
        return 2

    memory_result = normalize(operation())
    durations_ns: list[int] = []
    timing_result: Any = None
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter_ns()
        timing_result = operation()
        durations_ns.append(time.perf_counter_ns() - started)
    done = {
        "event": "done",
        "result": normalize(timing_result),
        "memory_probe_result": memory_result,
        "durations_ns": durations_ns,
        "final_rss_bytes": current_rss(),
        "final_high_water_bytes": high_water_rss(),
    }
    print(json.dumps(done), flush=True)
    return 0


def measure_worker(
    script: Path,
    library: str,
    case: dict[str, Any],
    repeats: int,
    sample_interval: float,
) -> dict[str, Any]:
    import psutil

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--library",
        library,
        "--case-json",
        json.dumps(case, separators=(",", ":")),
        "--repeats",
        str(repeats),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    assert process.stdin is not None
    assert process.stderr is not None
    ready_line = process.stdout.readline()
    if not ready_line:
        stderr = process.stderr.read()
        process.wait()
        return {"status": "error", "error": stderr.strip() or "worker exited before ready"}
    try:
        ready = json.loads(ready_line)
    except json.JSONDecodeError:
        process.kill()
        return {"status": "error", "error": f"invalid ready message: {ready_line!r}"}

    sampled_peak = int(ready["baseline_rss_bytes"])
    monitored = psutil.Process(process.pid)
    process.stdin.write("run\n")
    process.stdin.flush()
    process.stdin.close()
    while process.poll() is None:
        try:
            sampled_peak = max(sampled_peak, int(monitored.memory_info().rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(sample_interval)
    remainder = process.stdout.read().strip().splitlines()
    stderr = process.stderr.read().strip()
    if process.returncode != 0 or not remainder:
        return {
            "status": "error",
            "error": stderr or f"worker exited with {process.returncode}",
            "ready": ready,
        }
    try:
        done = json.loads(remainder[-1])
    except json.JSONDecodeError:
        return {"status": "error", "error": f"invalid done message: {remainder[-1]!r}"}

    durations = [int(value) for value in done["durations_ns"]]
    baseline_rss = int(ready["baseline_rss_bytes"])
    baseline_high = ready.get("baseline_high_water_bytes")
    final_high = done.get("final_high_water_bytes")
    high_water_delta = 0
    if baseline_high is not None and final_high is not None:
        high_water_delta = max(0, int(final_high) - int(baseline_high))
    peak_delta = max(0, sampled_peak - baseline_rss, high_water_delta)
    return {
        "status": "ok",
        "version": ready["version"],
        "metadata": ready["metadata"],
        "result": done["result"],
        "memory_probe_result": done["memory_probe_result"],
        "seconds": {
            "median": statistics.median(durations) / 1.0e9,
            "minimum": min(durations) / 1.0e9,
            "maximum": max(durations) / 1.0e9,
            "samples": [value / 1.0e9 for value in durations],
        },
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "peak_rss_bytes": max(sampled_peak, baseline_rss + high_water_delta),
            "peak_delta_bytes": peak_delta,
            "final_rss_bytes": int(done["final_rss_bytes"]),
            "sampling_interval_seconds": sample_interval,
        },
        "stderr": stderr,
    }


def valid_breakpoints(values: list[int], n: int) -> bool:
    return bool(values) and values[-1] == n and all(
        left < right for left, right in zip([0, *values[:-1]], values)
    )


def hausdorff(left: list[int], right: list[int]) -> float:
    def directed(first: list[int], second: list[int]) -> int:
        return max(min(abs(value - other) for other in second) for value in first)

    return float(max(directed(left, right), directed(right, left)))


def precision_recall(truth: list[int], predicted: list[int], margin: int) -> tuple[float, float]:
    truth_internal = truth[:-1]
    predicted_internal = predicted[:-1]
    matches = 0
    used: set[int] = set()
    for predicted_value in predicted_internal:
        candidates = [
            (abs(predicted_value - truth_value), index)
            for index, truth_value in enumerate(truth_internal)
            if index not in used and abs(predicted_value - truth_value) < margin
        ]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            matches += 1
    precision = matches / len(predicted_internal) if predicted_internal else float(not truth_internal)
    recall = matches / len(truth_internal) if truth_internal else float(not predicted_internal)
    return precision, recall


def segmentation_rand_index(left: list[int], right: list[int]) -> float:
    n = left[-1]
    if n < 2:
        return 1.0
    left_labels = np.empty(n, dtype=np.int32)
    right_labels = np.empty(n, dtype=np.int32)
    start = 0
    for label, end in enumerate(left):
        left_labels[start:end] = label
        start = end
    start = 0
    for label, end in enumerate(right):
        right_labels[start:end] = label
        start = end
    contingency: dict[tuple[int, int], int] = defaultdict(int)
    left_sizes: dict[int, int] = defaultdict(int)
    right_sizes: dict[int, int] = defaultdict(int)
    for left_label, right_label in zip(left_labels, right_labels):
        contingency[(int(left_label), int(right_label))] += 1
        left_sizes[int(left_label)] += 1
        right_sizes[int(right_label)] += 1
    choose_two = lambda value: value * (value - 1) // 2
    same_both = sum(choose_two(value) for value in contingency.values())
    same_left = sum(choose_two(value) for value in left_sizes.values())
    same_right = sum(choose_two(value) for value in right_sizes.values())
    different_both = choose_two(n) - same_left - same_right + same_both
    return (same_both + different_both) / choose_two(n)


def compare_results(case: dict[str, Any], measurements: dict[str, Any]) -> dict[str, Any]:
    rust = measurements.get("rustures")
    reference = measurements.get("ruptures")
    comparison: dict[str, Any] = {}
    if rust is None or rust.get("status") != "ok":
        comparison["rustures_valid"] = False
        return comparison

    result = rust["result"]
    if case["category"] == "dataset":
        rust_breakpoints = [int(value) for value in result["breakpoints"]]
        n = int(case["n"])
        comparison.update(
            {
                "rustures_valid": bool(result["shape"]) and result["shape"][0] == n
                and valid_breakpoints(rust_breakpoints, n),
            }
        )
    elif "breakpoints" in result:
        breakpoints = [int(value) for value in result["breakpoints"]]
        n = int(rust["metadata"]["n"])
        truth = [int(value) for value in rust["metadata"].get("truth", [n])]
        comparison.update(
            {
                "rustures_valid": valid_breakpoints(breakpoints, n),
                "rustures_truth_hausdorff": hausdorff(truth, breakpoints),
                "rustures_truth_precision_recall": precision_recall(truth, breakpoints, max(3, n // 50)),
                "rustures_truth_rand_index": segmentation_rand_index(truth, breakpoints),
            }
        )
    else:
        comparison["rustures_valid"] = True

    if reference is None:
        comparison["comparable"] = False
        comparison["unsupported_reason"] = case.get("unsupported_reason")
        return comparison
    if reference.get("status") != "ok":
        comparison.update({"comparable": True, "ruptures_valid": False})
        return comparison

    comparison["comparable"] = True
    reference_result = reference["result"]
    if case["category"] == "dataset":
        reference_breakpoints = [int(value) for value in reference_result["breakpoints"]]
        n = int(case["n"])
        comparison.update(
            {
                "ruptures_valid": bool(reference_result["shape"])
                and reference_result["shape"][0] == n
                and valid_breakpoints(reference_breakpoints, n),
                "same_sample_count": result["shape"][0] == reference_result["shape"][0],
                "same_element_count": int(np.prod(result["shape"]))
                == int(np.prod(reference_result["shape"])),
            }
        )
    elif "breakpoints" in result and "breakpoints" in reference_result:
        rust_bkps = [int(value) for value in result["breakpoints"]]
        reference_bkps = [int(value) for value in reference_result["breakpoints"]]
        n = int(reference["metadata"]["n"])
        comparison.update(
            {
                "ruptures_valid": valid_breakpoints(reference_bkps, n),
                "breakpoints_match": rust_bkps == reference_bkps,
                "cross_hausdorff": hausdorff(rust_bkps, reference_bkps),
                "cross_rand_index": segmentation_rand_index(rust_bkps, reference_bkps),
            }
        )
    elif case["category"] == "cost":
        left = float(result["checksum"])
        right = float(reference_result["checksum"])
        comparison.update(
            {
                "ruptures_valid": math.isfinite(right),
                "values_close": math.isclose(left, right, rel_tol=1.0e-6, abs_tol=1.0e-7),
                "relative_difference": abs(left - right) / max(abs(left), abs(right), 1.0),
            }
        )
    elif case["category"] == "metric":
        left = np.asarray(result["value"], dtype=np.float64)
        right = np.asarray(reference_result["value"], dtype=np.float64)
        comparison.update(
            {
                "ruptures_valid": bool(np.all(np.isfinite(right))),
                "values_close": bool(np.allclose(left, right, rtol=1.0e-12, atol=1.0e-12)),
            }
        )
    else:
        comparison.update({"ruptures_valid": True, "same_shape": result.get("shape") == reference_result.get("shape")})

    rust_time = rust["seconds"]["median"]
    reference_time = reference["seconds"]["median"]
    comparison["time_speedup"] = reference_time / rust_time if rust_time > 0 else None
    rust_memory = rust["memory"]["peak_delta_bytes"]
    reference_memory = reference["memory"]["peak_delta_bytes"]
    comparison["memory_ratio"] = reference_memory / rust_memory if rust_memory > 0 else None
    return comparison


def geometric_mean(values: list[float]) -> float | None:
    positives = [value for value in values if value > 0 and math.isfinite(value)]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def fmt_number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def fmt_speedup(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.4f}×" if value < 0.1 else f"{value:.2f}×"


def render_report(payload: dict[str, Any]) -> str:
    rows = payload["cases"]
    paired = [row for row in rows if row["comparison"].get("comparable")]
    completed = [
        row for row in paired
        if row["measurements"].get("rustures", {}).get("status") == "ok"
        and row["measurements"].get("ruptures", {}).get("status") == "ok"
    ]
    exact_rows = [row for row in completed if "breakpoints_match" in row["comparison"]]
    exact_matches = sum(bool(row["comparison"]["breakpoints_match"]) for row in exact_rows)
    value_rows = [row for row in completed if "values_close" in row["comparison"]]
    value_matches = sum(bool(row["comparison"]["values_close"]) for row in value_rows)
    rust_invalid = [row["case"]["id"] for row in rows if not row["comparison"].get("rustures_valid", False)]

    def group_speedup(predicate: Callable[[dict[str, Any]], bool]) -> float | None:
        return geometric_mean(
            [
                float(row["comparison"]["time_speedup"])
                for row in completed
                if predicate(row) and row["comparison"].get("time_speedup") is not None
            ]
        )

    dynp_speedup = group_speedup(
        lambda row: row["case"]["category"] == "detector"
        and row["case"].get("algorithm") == "dynp"
    )
    pelt_speedup = group_speedup(
        lambda row: row["case"]["category"] == "detector"
        and row["case"].get("algorithm") == "pelt"
    )
    fused_speedup = group_speedup(lambda row: row["case"].get("backend") == "fused")
    full_speedup = group_speedup(lambda row: row["case"].get("backend") == "full")
    streaming_speedup = group_speedup(lambda row: row["case"].get("backend") == "streaming")
    dataset_speedup = group_speedup(lambda row: row["case"]["category"] == "dataset")
    pruned_custom_speedup = group_speedup(
        lambda row: row["case"]["id"] == "custom-pruned-pelt"
    )
    faster_cases = sum(
        row["comparison"].get("time_speedup", 0.0) > 1.0 for row in completed
    )

    lines = [
        "# Rustures vs ruptures 통합 정확성·성능 비교",
        "",
        f"생성 시각: {payload['generated_at']}",
        "",
        "## 요약",
        "",
        f"- 프로필: `{payload['profile']}`. 메모리·워밍업 probe 1회 뒤 각 시간을 {payload['repeats']}회 측정해 중앙값을 사용했습니다.",
        f"- 전체 {len(rows)}건: 양쪽 비교 {len(completed)}건, Rustures-only {len(rows) - len(paired)}건입니다.",
        f"- Rustures 결과·불변조건 실패: {len(rust_invalid)}건" + (f" ({', '.join(rust_invalid)})" if rust_invalid else "."),
        f"- 비교 가능한 breakpoint: {exact_matches}/{len(exact_rows)}건이 정확히 일치했습니다.",
        f"- 비교 가능한 cost·metric 값: {value_matches}/{len(value_rows)}건이 허용오차 안에서 일치했습니다.",
        f"- 양쪽 측정이 끝난 {len(completed)}건 중 Rustures가 빠른 경우는 {faster_cases}건입니다.",
        "",
        "## 핵심 성능 결론",
        "",
        f"- L2 Dynp는 기하평균 {fmt_speedup(dynp_speedup)}, L2 Pelt는 {fmt_speedup(pelt_speedup)}였습니다.",
        f"- 기본 fused KernelCPD는 세 kernel 평균 {fmt_speedup(fused_speedup)}였습니다.",
        f"- Full-Gram KernelCPD의 평균 speedup은 {fmt_speedup(full_speedup)}, Gram-free streaming은 {fmt_speedup(streaming_speedup)}였습니다.",
        f"- 데이터 생성기의 평균 speedup은 {fmt_speedup(dataset_speedup)}였습니다.",
        "- 기본 scalar custom Pelt는 안전한 unpruned 경로라 더 느렸지만, pruning을 증명하고 opt-in한 custom Pelt의 speedup은 " + fmt_speedup(pruned_custom_speedup) + "였습니다.",
        "- AR은 두 라이브러리의 구간 경계 정책이 다르므로 cost 값과 한 Dynp breakpoint가 의도적으로 달랐습니다.",
        "",
        "Speedup이 1보다 크면 Rustures가 빠르다는 뜻입니다. Peak RSS는 package import 후",
        "입력까지 준비된 별도 프로세스에서 관측한 전체 working set이고, Δ RSS는 그 기준점에서",
        "연산 중 추가된 최대치입니다. psutil의 1ms sampling과 Windows working-set 정책의 영향을",
        "받으므로 작은 차이는 방향성만 봐야 합니다.",
        "",
        "## 측정 환경",
        "",
        "| Item | Value |",
        "|---|---|",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"| {key} | {str(value).replace('|', '/')} |")

    lines.extend([
        "",
        "## 범주별 성능",
        "",
        "| 범주 | 건수 | 기하평균 speedup | Rustures peak/Δ RSS 중앙값 (MiB) | ruptures peak/Δ RSS 중앙값 (MiB) |",
        "|---|---:|---:|---:|---:|",
    ])
    categories = sorted({row["case"]["category"] for row in completed})
    for category in categories:
        category_rows = [row for row in completed if row["case"]["category"] == category]
        speedups = [float(row["comparison"]["time_speedup"]) for row in category_rows]
        rust_peak = [row["measurements"]["rustures"]["memory"]["peak_rss_bytes"] / 2**20 for row in category_rows]
        reference_peak = [row["measurements"]["ruptures"]["memory"]["peak_rss_bytes"] / 2**20 for row in category_rows]
        rust_memory = [row["measurements"]["rustures"]["memory"]["peak_delta_bytes"] / 2**20 for row in category_rows]
        reference_memory = [row["measurements"]["ruptures"]["memory"]["peak_delta_bytes"] / 2**20 for row in category_rows]
        lines.append(
            f"| {category} | {len(category_rows)} | {fmt_speedup(geometric_mean(speedups))} | "
            f"{fmt_number(statistics.median(rust_peak))}/{fmt_number(statistics.median(rust_memory))} | "
            f"{fmt_number(statistics.median(reference_peak))}/{fmt_number(statistics.median(reference_memory))} |"
        )

    lines.extend([
        "",
        "## Case별 결과",
        "",
        "| Case | 신호 / N | Rustures ms | ruptures ms | Speedup | Rust peak/Δ MiB | ruptures peak/Δ MiB | 일치 여부 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        case = row["case"]
        measurements = row["measurements"]
        rust = measurements.get("rustures", {})
        reference = measurements.get("ruptures", {})
        comparison = row["comparison"]
        signal_label = f"{case.get('signal', case.get('feature'))} / {case.get('n', '—')}"
        rust_ms = rust.get("seconds", {}).get("median")
        reference_ms = reference.get("seconds", {}).get("median")
        speedup = comparison.get("time_speedup")
        rust_mem = rust.get("memory", {}).get("peak_delta_bytes")
        reference_mem = reference.get("memory", {}).get("peak_delta_bytes")
        rust_peak = rust.get("memory", {}).get("peak_rss_bytes")
        reference_peak = reference.get("memory", {}).get("peak_rss_bytes")
        if "breakpoints_match" in comparison:
            agreement = "exact" if comparison["breakpoints_match"] else f"different (H={comparison['cross_hausdorff']:.0f})"
        elif "values_close" in comparison:
            agreement = "close" if comparison["values_close"] else "different semantics/value"
        elif not comparison.get("comparable", True):
            agreement = "Rustures only"
        elif case["category"] == "dataset":
            agreement = "valid outputs" if comparison.get("same_sample_count") else "different length"
        else:
            agreement = "completed"
        lines.append(
            f"| `{case['id']}` | {signal_label} | "
            f"{fmt_number(rust_ms * 1_000 if rust_ms is not None else None, 3)} | "
            f"{fmt_number(reference_ms * 1_000 if reference_ms is not None else None, 3)} | "
            f"{fmt_speedup(speedup)} | "
            f"{fmt_number(rust_peak / 2**20 if rust_peak is not None else None)}/{fmt_number(rust_mem / 2**20 if rust_mem is not None else None)} | "
            f"{fmt_number(reference_peak / 2**20 if reference_peak is not None else None)}/{fmt_number(reference_mem / 2**20 if reference_mem is not None else None)} | {agreement} |"
        )

    lines.extend([
        "",
        "## 검증 범위와 해석",
        "",
        "- Cost: L2, L1, Rank, Normal, Linear, AR, CLinear, Mahalanobis를 모두 포함합니다.",
        "- 탐색기: Dynp, Pelt, Binseg, BottomUp, Window, KernelCPD를 포함합니다.",
        "- Kernel: linear, RBF, cosine 및 Rustures fused/full/streaming을 각각 ruptures KernelCPD와 비교합니다.",
        "- 추가 기능: scalar custom cost, Rustures batch custom cost, L1Potts, metric 세 종류, 합성 데이터 generator 네 종류를 포함합니다.",
        "- Breakpoint 완전 일치는 진단 정보이지 모든 경우의 정확성 조건은 아닙니다. Greedy 동점, Normal/AR 경계, RBF clipping, regularization 정책은 의도적으로 다를 수 있습니다.",
        "- Generator는 같은 seed라도 난수·경계 생성 정책이 달라 동일 배열을 요구하지 않고 shape와 breakpoint 유효성을 검사합니다.",
        "- 시간은 interpreter 시작, package import, 입력 신호 생성을 제외하고 객체 생성·fit·predict 또는 해당 연산 전체를 포함합니다.",
        "- ruptures는 SciPy를 함께 import하므로 baseline RSS가 더 큽니다. 반대로 Δ RSS는 이미 확보된 Python allocator page를 재사용하면 실제 자료구조 크기를 과소평가할 수 있습니다. 두 수치를 함께 보아야 합니다.",
        "- 단일 PC 결과이며 1ms RSS sampling과 1ms 미만 시간은 방향성입니다. 배포 대상 장비에서는 다시 측정해야 합니다.",
        "",
        "## 재현 방법",
        "",
        "```powershell",
        "python -m venv .benchmark-venv",
        ".\\.benchmark-venv\\Scripts\\python -m pip install -r benchmarks/integration-requirements.txt",
        ".\\.benchmark-venv\\Scripts\\python -m pip install --no-index --no-deps --find-links dist rustures",
        "",
        ".\\.benchmark-venv\\Scripts\\python benchmarks/integration_comparison.py --profile standard --repeats 5 `",
        "  --output artifacts/validation/integration-comparison.json `",
        "  --report artifacts/phase-9/08-integration-performance-comparison.md",
        "```",
        "",
    ])
    return "\n".join(lines)


def environment_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "OS": platform.platform(),
        "CPU": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "Python": sys.version.replace("\n", " "),
        "NumPy": np.__version__,
        "psutil": importlib.metadata.version("psutil"),
        "logical CPUs": os.cpu_count(),
    }
    for package in COMMON_LIBRARIES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    try:
        result["SciPy"] = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        result["SciPy"] = "not installed"
    return result


def parent_main(args: argparse.Namespace) -> int:
    import psutil  # noqa: F401 - fail early with a clear missing dependency

    cases = case_registry(args.profile)
    if args.only:
        cases = [case for case in cases if args.only.lower() in case["id"].lower()]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise SystemExit("no cases selected")

    script = Path(__file__).resolve()
    measured_environment = environment_metadata()
    methodology = {
        "timing_scope": "warmed fresh estimator fit+predict/operation, process import excluded",
        "memory_scope": "incremental process RSS after import and input construction",
        "sample_interval_ms": args.sample_interval_ms,
        "thread_limits": 1,
    }
    output_rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        measurements: dict[str, Any] = {}
        libraries = list(case["libraries"])
        if len(libraries) == 2 and index % 2 == 0:
            libraries.reverse()
        for library in libraries:
            measured = measure_worker(
                script,
                library,
                case,
                args.repeats,
                args.sample_interval_ms / 1_000.0,
            )
            measurements[library] = measured
            if measured["status"] == "ok":
                print(
                    f"  {library}: {measured['seconds']['median'] * 1_000:.3f} ms, "
                    f"ΔRSS {measured['memory']['peak_delta_bytes'] / 2**20:.2f} MiB",
                    flush=True,
                )
            else:
                print(f"  {library}: ERROR {measured['error']}", flush=True)
        comparison = compare_results(case, measurements)
        output_rows.append({"case": case, "measurements": measurements, "comparison": comparison})

        # Preserve completed work if a long manual benchmark is interrupted.
        checkpoint = {
            "schema_version": 1,
            "complete": False,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "profile": args.profile,
            "repeats": args.repeats,
            "seed": SEED,
            "environment": measured_environment,
            "methodology": methodology,
            "cases": output_rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {
        "schema_version": 1,
        "complete": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": args.profile,
        "repeats": args.repeats,
        "seed": SEED,
        "environment": measured_environment,
        "methodology": methodology,
        "cases": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")

    failures = [
        row for row in output_rows
        if row["measurements"].get("rustures", {}).get("status") != "ok"
        or not row["comparison"].get("rustures_valid", False)
    ]
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "standard"), default="quick")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-interval-ms", type=float, default=1.0)
    parser.add_argument("--only")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/integration-comparison.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/phase-9/08-integration-performance-comparison.md"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--library", choices=COMMON_LIBRARIES, help=argparse.SUPPRESS)
    parser.add_argument("--case-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.sample_interval_ms <= 0:
        parser.error("--sample-interval-ms must be positive")
    if args.worker and (args.library is None or args.case_json is None):
        parser.error("worker mode requires --library and --case-json")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        raise SystemExit(worker_main(parsed.library, parsed.case_json, parsed.repeats))
    if parsed.render_only:
        existing = json.loads(parsed.output.read_text(encoding="utf-8"))
        parsed.report.parent.mkdir(parents=True, exist_ok=True)
        parsed.report.write_text(render_report(existing), encoding="utf-8")
        print(f"wrote {parsed.report}")
        raise SystemExit(0)
    raise SystemExit(parent_main(parsed))
