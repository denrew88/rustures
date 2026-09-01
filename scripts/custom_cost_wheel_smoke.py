"""Smoke-test the installed wheel's Python custom-cost boundary.

Run this script with a Python interpreter where the built wheel and NumPy are
installed. It intentionally imports only the installed package, not repository
helpers.
"""

from __future__ import annotations

import json

import numpy as np

import rustures


class BatchL2:
    min_size = 1

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self.fit_calls = 0
        self.batch_calls = 0

    def fit(self, signal: np.ndarray) -> BatchL2:
        values = np.asarray(signal, dtype=np.float64)
        self.signal = values[:, None] if values.ndim == 1 else values
        self.fit_calls += 1
        return self

    def error(self, start: int, end: int) -> float:
        assert self.signal is not None
        segment = self.signal[start:end]
        return float(np.square(segment - segment.mean(axis=0)).sum())

    def error_many(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        assert self.signal is not None
        self.batch_calls += 1
        return np.asarray(
            [
                np.square(
                    self.signal[start:end]
                    - self.signal[start:end].mean(axis=0)
                ).sum()
                for start, end in zip(starts, ends, strict=True)
            ],
            dtype=np.float64,
        )


def main() -> None:
    signal = np.repeat(np.array([0.0, 7.0, -4.0]), 4)

    dynp_cost = BatchL2()
    dynp = rustures.Dynp(
        custom_cost=dynp_cost,
        min_size=1,
        jump=1,
    )
    dynp_breakpoints = dynp.fit_predict(signal, n_bkps=2)
    assert dynp_breakpoints == [4, 8, 12]
    assert dynp.uses_custom_cost and dynp.uses_batch_callback
    assert dynp_cost.fit_calls == 1 and dynp_cost.batch_calls > 0

    pelt_cost = BatchL2()
    pelt = rustures.Pelt(
        custom_cost=pelt_cost,
        min_size=1,
        jump=1,
    )
    pelt_breakpoints = pelt.fit_predict(signal, pen=1.0)
    assert pelt_breakpoints == [4, 8, 12]
    assert pelt.uses_custom_cost and pelt.uses_batch_callback
    assert pelt_cost.fit_calls == 1 and pelt_cost.batch_calls > 0

    print(
        json.dumps(
            {
                "version": rustures.__version__,
                "dynp": dynp_breakpoints,
                "pelt": pelt_breakpoints,
                "dynp_batch_calls": dynp_cost.batch_calls,
                "pelt_batch_calls": pelt_cost.batch_calls,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
