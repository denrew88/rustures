"""Compare native, scalar-callback, and batch-callback Dynp costs."""

from __future__ import annotations

import json
import statistics
import time

import numpy as np

import rustures


class PrefixL2Scalar:
    min_size = 1

    def __init__(self) -> None:
        self.prefix = None
        self.square_prefix = None
        self.error_calls = 0

    def fit(self, signal):
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        zeros = np.zeros((1, values.shape[1]), dtype=np.float64)
        self.prefix = np.vstack((zeros, np.cumsum(values, axis=0)))
        self.square_prefix = np.vstack(
            (zeros, np.cumsum(np.square(values), axis=0))
        )
        self.error_calls = 0
        return self

    def error(self, start: int, end: int) -> float:
        self.error_calls += 1
        length = end - start
        sums = self.prefix[end] - self.prefix[start]
        squares = self.square_prefix[end] - self.square_prefix[start]
        return float(np.sum(squares - np.square(sums) / length))


class PrefixL2Batch(PrefixL2Scalar):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0
        self.maximum_batch = 0

    def fit(self, signal):
        super().fit(signal)
        self.batch_calls = 0
        self.maximum_batch = 0
        return self

    def error_many(self, starts, ends):
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        self.batch_calls += 1
        self.maximum_batch = max(self.maximum_batch, len(starts))
        lengths = (ends - starts)[:, None]
        sums = self.prefix[ends] - self.prefix[starts]
        squares = self.square_prefix[ends] - self.square_prefix[starts]
        return np.sum(squares - np.square(sums) / lengths, axis=1).astype(
            np.float64, copy=False
        )


def timed(function, repeats: int = 3):
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), result


def signal_for(n_samples: int, n_features: int) -> np.ndarray:
    rng = np.random.default_rng(20260901 + n_samples + n_features)
    values = rng.normal(scale=0.2, size=(n_samples, n_features))
    values[n_samples // 3 : 2 * n_samples // 3] += 4.0
    values[2 * n_samples // 3 :] -= 3.0
    return values


def main() -> None:
    rows = []
    for n_samples in [100, 200, 400]:
        for n_features in [1, 4]:
            for n_bkps in [2, 5]:
                signal = signal_for(n_samples, n_features)

                native_time, native_result = timed(
                    lambda: rustures.Dynp(min_size=2, jump=1).fit_predict(
                        signal, n_bkps=n_bkps
                    )
                )

                scalar_cost = PrefixL2Scalar()
                scalar_detector = rustures.Dynp(
                    custom_cost=scalar_cost, min_size=2, jump=1
                )
                scalar_time, scalar_result = timed(
                    lambda: scalar_detector.fit_predict(signal, n_bkps=n_bkps)
                )

                batch_cost = PrefixL2Batch()
                batch_detector = rustures.Dynp(
                    custom_cost=batch_cost, min_size=2, jump=1
                )
                batch_time, batch_result = timed(
                    lambda: batch_detector.fit_predict(signal, n_bkps=n_bkps)
                )

                if scalar_result != native_result or batch_result != native_result:
                    raise AssertionError(
                        (n_samples, n_features, n_bkps, native_result, scalar_result, batch_result)
                    )

                rows.append(
                    {
                        "n": n_samples,
                        "d": n_features,
                        "k": n_bkps,
                        "native_ms": native_time * 1_000.0,
                        "scalar_ms": scalar_time * 1_000.0,
                        "batch_ms": batch_time * 1_000.0,
                        "scalar_over_native": scalar_time / native_time,
                        "batch_over_native": batch_time / native_time,
                        "scalar_error_calls_last_repeat": scalar_cost.error_calls,
                        "batch_calls_last_repeat": batch_cost.batch_calls,
                        "maximum_batch": batch_cost.maximum_batch,
                    }
                )

    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
