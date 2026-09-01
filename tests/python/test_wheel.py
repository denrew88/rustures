import json
from importlib.metadata import version as distribution_version
from pathlib import Path
import unittest

import numpy as np

import rustures


class ScalarL2CustomCost:
    min_size = 1

    def __init__(self) -> None:
        self.signal = None
        self.fit_calls = 0
        self.error_calls = 0

    def fit(self, signal):
        self.signal = np.asarray(signal, dtype=np.float64)
        if self.signal.ndim == 1:
            self.signal = self.signal[:, None]
        self.fit_calls += 1
        return self

    def error(self, start: int, end: int) -> float:
        self.error_calls += 1
        segment = self.signal[start:end]
        return float(np.square(segment - segment.mean(axis=0)).sum())


class BatchL2CustomCost(ScalarL2CustomCost):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0
        self.maximum_batch = 0

    def error_many(self, starts, ends):
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        self.batch_calls += 1
        self.maximum_batch = max(self.maximum_batch, len(starts))
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


class BatchBernoulliCustomCost:
    min_size = 1

    def fit(self, signal):
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("Bernoulli observations must be 0 or 1")
        zeros = np.zeros((1, values.shape[1]), dtype=np.float64)
        self.ones_prefix = np.vstack((zeros, np.cumsum(values, axis=0)))
        return self

    @staticmethod
    def costs(ones, trials):
        mixed = (ones > 0.0) & (ones < trials)
        costs = np.zeros_like(ones)
        probabilities = ones[mixed] / trials[mixed]
        costs[mixed] = (
            -ones[mixed] * np.log(probabilities)
            - (trials[mixed] - ones[mixed]) * np.log1p(-probabilities)
        )
        return costs

    def error(self, start: int, end: int) -> float:
        ones = self.ones_prefix[end] - self.ones_prefix[start]
        trials = np.full_like(ones, end - start, dtype=np.float64)
        return float(self.costs(ones, trials).sum())

    def error_many(self, starts, ends):
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        lengths = (ends - starts)[:, None].astype(np.float64)
        ones = self.ones_prefix[ends] - self.ones_prefix[starts]
        trials = np.broadcast_to(lengths, ones.shape)
        return self.costs(ones, trials).sum(axis=1).astype(np.float64, copy=False)


class WheelSmokeTests(unittest.TestCase):
    def test_version_is_exported(self) -> None:
        self.assertEqual(rustures.version(), rustures.__version__)
        self.assertEqual(rustures.__version__, distribution_version("rustures"))
        self.assertTrue(issubclass(rustures.RusturesError, RuntimeError))

    def test_validates_scalar_and_multivariate_signals(self) -> None:
        self.assertEqual(rustures.validate_signal(np.array([1.0, 2.0])), (2, 1))
        signal = np.arange(12.0).reshape(4, 3)
        self.assertEqual(rustures.validate_signal(signal), (4, 3))
        self.assertEqual(rustures.validate_signal(signal[:, ::-1]), (4, 3))

    def test_rejects_empty_high_rank_and_non_finite_signals(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            rustures.validate_signal(np.array([], dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "one- or two-dimensional"):
            rustures.validate_signal(np.zeros((1, 1, 1), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "row 1, column 0"):
            rustures.validate_signal(np.array([[1.0], [np.nan]]))

    def test_cost_l2_matches_direct_scalar_and_multivariate_costs(self) -> None:
        scalar = np.array([1.0, 2.0, 5.0, 6.0])
        cost = rustures.CostL2().fit(scalar)
        self.assertTrue(cost.is_fitted)
        self.assertEqual((cost.n_samples, cost.n_features, cost.min_size), (4, 1, 1))
        self.assertAlmostEqual(cost.error(0, 4), 17.0)
        self.assertAlmostEqual(cost.error(1, 3), 4.5)

        signal = np.arange(30.0).reshape(10, 3)[:, ::-1]
        cost.fit(signal)
        direct = np.square(signal[2:9] - signal[2:9].mean(axis=0)).sum()
        self.assertAlmostEqual(cost.error(2, 9), direct)

    def test_cost_l2_is_stable_and_rejects_invalid_state(self) -> None:
        cost = rustures.CostL2()
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            cost.error(0, 1)

        constant = np.full((8, 2), 1.0e12)
        self.assertEqual(cost.fit(constant).error(0, 8), 0.0)
        mutable = np.array([1.0, 2.0, 5.0, 6.0])
        cost.fit(mutable)
        before_mutation = cost.error(0, 4)
        mutable[:] = 0.0
        self.assertEqual(cost.error(0, 4), before_mutation)
        with self.assertRaisesRegex(ValueError, "invalid segment"):
            cost.error(2, 2)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            cost.fit(np.array([1.0, np.inf]))

    def test_cost_l2_matches_pinned_ruptures_fixture(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "ruptures_dynp_l2_v1.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        signal = np.asarray(fixture["signal"]["values"], dtype=np.float64).reshape(
            fixture["signal"]["shape"]
        )
        cost = rustures.CostL2().fit(signal)
        self.assertEqual(cost.sum_of_costs(fixture["breakpoints"]), fixture["objective"])

        cost_fixture = json.loads(
            (fixture_path.parent / "ruptures_cost_l2_v1.json").read_text(encoding="utf-8")
        )
        signal = np.asarray(cost_fixture["signal"]["values"], dtype=np.float64).reshape(
            cost_fixture["signal"]["shape"]
        )
        cost.fit(signal)
        self.assertEqual(
            cost.sum_of_costs(cost_fixture["breakpoints"]),
            cost_fixture["objective"],
        )

    def test_stage7_cost_classes_and_edge_cases(self) -> None:
        signal = np.array([[0.0, 0.0], [1.0, 2.0], [8.0, 3.0], [9.0, 5.0]])
        l1 = rustures.CostL1().fit(signal)
        direct_l1 = np.abs(signal - np.median(signal, axis=0)).sum()
        self.assertAlmostEqual(l1.error(0, 4), direct_l1)

        metric = np.diag([2.0, 0.5])
        ml = rustures.CostMahalanobis(metric).fit(signal)
        centered = signal - signal.mean(axis=0)
        self.assertAlmostEqual(ml.error(0, 4), np.einsum("ni,ij,nj->", centered, metric, centered))
        self.assertIs(rustures.CostMl, rustures.CostMahalanobis)
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            rustures.CostMahalanobis(np.diag([1.0, -1.0])).fit(signal)

        relation = np.column_stack((2.0 + 3.0 * np.arange(8.0), np.ones(8), np.arange(8.0)))
        self.assertAlmostEqual(rustures.CostLinear().fit(relation).error(0, 8), 0.0, places=10)
        straight = np.column_stack((np.arange(8.0), 2.0 * np.arange(8.0)))
        self.assertAlmostEqual(rustures.CostCLinear().fit(straight).error(0, 8), 0.0, places=10)

        tied = np.array([0.0, 0.0, 1.0, 1.0, 9.0, 9.0, 10.0, 10.0])
        self.assertTrue(np.isfinite(rustures.CostRank().fit(tied).error(0, 4)))
        self.assertTrue(np.isfinite(rustures.CostNormal().fit(np.ones(8)).error(0, 8)))
        with self.assertRaisesRegex(ValueError, "ridge"):
            rustures.CostNormal(ridge=0.0)

        ar_signal = np.array([1.0, 2.0, 3.5, 5.75, 9.125, 14.1875, 21.78125])
        ar = rustures.CostAR(order=1).fit(ar_signal)
        self.assertEqual(ar.min_size, 5)
        self.assertTrue(np.isfinite(ar.error(0, len(ar_signal))))

    def test_stage7_models_compose_with_search_algorithms(self) -> None:
        scalar = np.repeat(np.array([0.0, 5.0, -2.0]), 6)
        x = np.arange(18.0)
        linear = np.column_stack((np.where(x < 9, x, -x + 18), np.ones_like(x), x))
        cases = {
            "l1": (scalar, 2),
            "rank": (scalar, 2),
            "normal": (scalar, 2),
            "linear": (linear, 2),
            "ar": (scalar, 5),
            "clinear": (scalar, 3),
            "mahalanobis": (scalar, 2),
        }
        for model, (signal, min_size) in cases.items():
            detectors = [
                ("Dynp", rustures.Dynp(model=model, min_size=min_size, jump=1), {"n_bkps": 1}),
                ("Pelt", rustures.Pelt(model=model, min_size=min_size, jump=1), {"pen": 2.0}),
                ("Binseg", rustures.Binseg(model=model, min_size=min_size, jump=1), {"n_bkps": 1}),
                ("BottomUp", rustures.BottomUp(model=model, min_size=min_size, jump=1), {"n_bkps": 1}),
                (
                    "Window",
                    rustures.Window(width=max(6, 2 * min_size), model=model, min_size=min_size, jump=1),
                    {"n_bkps": 1},
                ),
            ]
            for detector_name, detector, stopping_rule in detectors:
                with self.subTest(model=model, detector=detector_name):
                    result = detector.fit(signal).predict(**stopping_rule)
                    self.assertEqual(result[-1], len(signal))

    def test_l1_potts_weighted_smoke_and_validation(self) -> None:
        signal = np.array([0.0, 0.0, 1.0, 9.0, 9.0, 10.0])
        detector = rustures.L1Potts()
        self.assertEqual(detector.fit_predict(signal, pen=2.0), [3, 6])
        self.assertTrue(detector.is_fitted)
        self.assertEqual((detector.n_samples, detector.distinct_levels), (6, 4))
        self.assertEqual(detector.predict(2.0), [3, 6])
        self.assertEqual(
            rustures.L1Potts().fit_predict(signal, pen=2.0, weights=np.zeros(6)),
            [6],
        )
        with self.assertRaisesRegex(ValueError, "weights have length"):
            rustures.L1Potts().fit(signal, weights=np.ones(5))
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            rustures.L1Potts().fit(signal, weights=np.array([1, 1, -1, 1, 1, 1], dtype=float))
        with self.assertRaisesRegex(ValueError, "scalar signal"):
            rustures.L1Potts().fit(np.zeros((6, 2)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rustures.L1Potts().fit(np.array([0.0, np.nan]))

    def test_dynp_fit_predict_and_pinned_fixture(self) -> None:
        fixture_directory = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
        )
        for fixture_name in [
            "ruptures_dynp_l2_v1.json",
            "ruptures_dynp_l2_nonzero_v1.json",
        ]:
            with self.subTest(fixture=fixture_name):
                fixture = json.loads(
                    (fixture_directory / fixture_name).read_text(encoding="utf-8")
                )
                signal = np.asarray(
                    fixture["signal"]["values"], dtype=np.float64
                ).reshape(fixture["signal"]["shape"])
                parameters = fixture["parameters"]
                detector = rustures.Dynp(
                    model=parameters["model"],
                    min_size=parameters["min_size"],
                    jump=parameters["jump"],
                )
                self.assertEqual(detector.model, "l2")
                self.assertEqual(
                    detector.fit_predict(signal, n_bkps=parameters["n_bkps"]),
                    fixture["breakpoints"],
                )
                self.assertTrue(detector.is_fitted)
                self.assertEqual(
                    detector.predict(parameters["n_bkps"]), fixture["breakpoints"]
                )
                self.assertAlmostEqual(
                    rustures.CostL2().fit(signal).sum_of_costs(fixture["breakpoints"]),
                    fixture["objective"],
                )

    def test_dynp_errors_and_grid_behavior(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported models: l2"):
            rustures.Dynp(model="rbf")
        with self.assertRaisesRegex(ValueError, "min_size must be positive"):
            rustures.Dynp(min_size=0)
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            rustures.Dynp().predict(1)

        signal = np.array([0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0])
        detector = rustures.Dynp(min_size=1, jump=3).fit(signal)
        self.assertEqual(detector.predict(n_bkps=1), [3, 7])
        with self.assertRaisesRegex(ValueError, "cannot segment"):
            rustures.Dynp(min_size=2, jump=1).fit(np.zeros(5)).predict(2)

    def test_dynp_memory_estimate_limit_and_process_survival(self) -> None:
        with self.assertRaisesRegex(ValueError, "memory limit must be positive"):
            rustures.Dynp(max_memory_bytes=0)

        detector = rustures.Dynp(min_size=1, jump=1, max_memory_bytes=1_000)
        self.assertEqual(detector.max_memory_bytes, 1_000)
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            detector.estimated_memory_bytes(n_bkps=1)

        detector.fit(np.arange(20.0))
        self.assertEqual(detector.estimated_memory_bytes(n_bkps=0), 0)
        self.assertEqual(detector.estimated_memory_bytes(n_bkps=1), 1_140)
        with self.assertRaisesRegex(
            MemoryError,
            r"requires 1140 bytes, above configured limit 1000",
        ):
            detector.predict(n_bkps=1)

        # A rejected allocation must not poison or terminate the interpreter.
        self.assertEqual(rustures.CostL2().fit(np.arange(4.0)).error(0, 4), 5.0)

    def test_pelt_fit_predict_and_grid_behavior(self) -> None:
        signal = np.array([0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0])
        detector = rustures.Pelt(model="l2", min_size=1, jump=1)
        self.assertEqual(detector.model, "l2")
        self.assertEqual(detector.fit_predict(signal, pen=1.0), [3, 7])
        self.assertTrue(detector.is_fitted)
        self.assertEqual(detector.predict(pen=1.0), [3, 7])

        approximate = rustures.Pelt(min_size=1, jump=2).fit(signal)
        self.assertEqual(approximate.predict(pen=1.0), [2, 4, 7])

    def test_pelt_matches_pinned_ruptures_fixture(self) -> None:
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "ruptures_pelt_l2_v1.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        signal = np.asarray(fixture["signal"]["values"], dtype=np.float64).reshape(
            fixture["signal"]["shape"]
        )
        parameters = fixture["parameters"]
        breakpoints = rustures.Pelt(
            model=parameters["model"],
            min_size=parameters["min_size"],
            jump=parameters["jump"],
        ).fit_predict(signal, pen=parameters["pen"])
        self.assertEqual(breakpoints, fixture["breakpoints"])
        segment_cost = rustures.CostL2().fit(signal).sum_of_costs(breakpoints)
        self.assertAlmostEqual(segment_cost, fixture["segment_cost"])
        self.assertAlmostEqual(
            segment_cost + parameters["pen"] * (len(breakpoints) - 1),
            fixture["objective"],
        )

    def test_pelt_errors_and_no_change_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported models: l2"):
            rustures.Pelt(model="rbf")
        with self.assertRaisesRegex(ValueError, "min_size must be positive"):
            rustures.Pelt(min_size=0)
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            rustures.Pelt().predict(1.0)

        detector = rustures.Pelt(min_size=1, jump=1).fit(
            np.array([0.0, 0.0, 10.0, 10.0])
        )
        self.assertEqual(detector.predict(pen=1_000.0), [4])
        with self.assertRaisesRegex(ValueError, "penalty must be finite and positive"):
            detector.predict(pen=0.0)

    def test_dynp_custom_cost_scalar_matches_native_l2(self) -> None:
        signal = np.repeat(np.array([0.0, 8.0, -3.0]), 4)
        custom = ScalarL2CustomCost()
        detector = rustures.Dynp(
            custom_cost=custom,
            min_size=1,
            jump=1,
        )
        self.assertEqual(detector.model, "custom")
        self.assertTrue(detector.uses_custom_cost)
        self.assertFalse(detector.uses_batch_callback)
        self.assertEqual(detector.fit_predict(signal, n_bkps=2), [4, 8, 12])
        self.assertEqual(custom.fit_calls, 1)
        self.assertGreater(custom.error_calls, 0)
        self.assertEqual(
            detector.predict(n_bkps=2),
            rustures.Dynp(min_size=1, jump=1).fit_predict(signal, n_bkps=2),
        )
        self.assertEqual(custom.fit_calls, 1)

    def test_custom_cost_batch_matches_scalar_for_dynp_and_pelt(self) -> None:
        signal = np.column_stack(
            (
                np.repeat(np.array([0.0, 5.0, -2.0]), 4),
                np.repeat(np.array([1.0, -3.0, 7.0]), 4),
            )
        )
        for detector_type, predict_argument in [
            (rustures.Dynp, {"n_bkps": 2}),
            (rustures.Pelt, {"pen": 1.0}),
        ]:
            with self.subTest(detector=detector_type.__name__):
                scalar = ScalarL2CustomCost()
                batch = BatchL2CustomCost()
                scalar_detector = detector_type(
                    custom_cost=scalar, min_size=1, jump=1
                )
                batch_detector = detector_type(
                    custom_cost=batch, min_size=1, jump=1
                )
                expected = scalar_detector.fit_predict(signal, **predict_argument)
                actual = batch_detector.fit_predict(signal, **predict_argument)
                self.assertEqual(actual, expected)
                self.assertTrue(batch_detector.uses_batch_callback)
                self.assertGreater(batch.batch_calls, 0)
                self.assertEqual(batch.error_calls, 0)
                self.assertLessEqual(batch.maximum_batch, len(signal))

    def test_custom_cost_matches_native_l2_across_small_parameter_grid(self) -> None:
        rng = np.random.default_rng(20260901)
        for n_samples in [6, 8, 10]:
            for n_features in [1, 2]:
                signal = rng.normal(size=(n_samples, n_features))
                if n_features == 1:
                    signal = signal[:, 0]
                for min_size in [1, 2]:
                    for jump in [1, 2]:
                        for n_bkps in [0, 1]:
                            with self.subTest(
                                detector="Dynp",
                                n=n_samples,
                                d=n_features,
                                min_size=min_size,
                                jump=jump,
                                n_bkps=n_bkps,
                            ):
                                native = rustures.Dynp(
                                    min_size=min_size, jump=jump
                                ).fit(signal)
                                scalar = rustures.Dynp(
                                    custom_cost=ScalarL2CustomCost(),
                                    min_size=min_size,
                                    jump=jump,
                                ).fit(signal)
                                batch = rustures.Dynp(
                                    custom_cost=BatchL2CustomCost(),
                                    min_size=min_size,
                                    jump=jump,
                                ).fit(signal)
                                expected = native.predict(n_bkps=n_bkps)
                                self.assertEqual(
                                    scalar.predict(n_bkps=n_bkps), expected
                                )
                                self.assertEqual(
                                    batch.predict(n_bkps=n_bkps), expected
                                )

                        for penalty in [0.25, 2.0, 20.0]:
                            with self.subTest(
                                detector="Pelt",
                                n=n_samples,
                                d=n_features,
                                min_size=min_size,
                                jump=jump,
                                penalty=penalty,
                            ):
                                native = rustures.Pelt(
                                    min_size=min_size, jump=jump
                                ).fit(signal)
                                scalar = rustures.Pelt(
                                    custom_cost=ScalarL2CustomCost(),
                                    min_size=min_size,
                                    jump=jump,
                                ).fit(signal)
                                batch = rustures.Pelt(
                                    custom_cost=BatchL2CustomCost(),
                                    min_size=min_size,
                                    jump=jump,
                                ).fit(signal)
                                expected = native.predict(pen=penalty)
                                self.assertEqual(scalar.predict(pen=penalty), expected)
                                self.assertEqual(batch.predict(pen=penalty), expected)

    def test_custom_cost_min_size_and_protocol_validation(self) -> None:
        class MinimumThreeCost(ScalarL2CustomCost):
            min_size = 3

        detector = rustures.Dynp(
            custom_cost=MinimumThreeCost(), min_size=1, jump=1
        )
        self.assertEqual(detector.min_size, 3)

        class MissingMinSize:
            def fit(self, signal):
                return self

            def error(self, start, end):
                return 0.0

        with self.assertRaisesRegex(TypeError, "min_size"):
            rustures.Dynp(custom_cost=MissingMinSize())

        invalid = ScalarL2CustomCost()
        invalid.min_size = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            rustures.Pelt(custom_cost=invalid)

        class MissingError:
            min_size = 1

            def fit(self, signal):
                return self

        with self.assertRaisesRegex(TypeError, "error"):
            rustures.Dynp(custom_cost=MissingError())

        invalid_batch = ScalarL2CustomCost()
        invalid_batch.error_many = 3
        with self.assertRaisesRegex(TypeError, "error_many"):
            rustures.Pelt(custom_cost=invalid_batch).fit(np.arange(6.0))

        self.assertEqual(
            rustures.Dynp(
                model="ignored-when-custom",
                custom_cost=ScalarL2CustomCost(),
                min_size=1,
                jump=1,
            ).model,
            "custom",
        )

    def test_custom_cost_preserves_callback_exception_and_traceback(self) -> None:
        class CustomCostFailure(RuntimeError):
            pass

        class FailingCost(ScalarL2CustomCost):
            def error(self, start: int, end: int) -> float:
                raise CustomCostFailure(f"failed on [{start}, {end})")

        detector = rustures.Dynp(
            custom_cost=FailingCost(), min_size=1, jump=1
        ).fit(np.arange(6.0))
        try:
            detector.predict(n_bkps=1)
        except CustomCostFailure as error:
            self.assertRegex(str(error), "failed on")
            traceback = error.__traceback__
        else:
            self.fail("custom callback exception was not propagated")

        frames = []
        while traceback is not None:
            frames.append(traceback.tb_frame.f_code.co_name)
            traceback = traceback.tb_next
        self.assertIn("error", frames)

    def test_custom_cost_rejects_invalid_scalar_and_batch_outputs(self) -> None:
        class NonFiniteCost(ScalarL2CustomCost):
            def error(self, start: int, end: int) -> float:
                return np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            rustures.Dynp(
                custom_cost=NonFiniteCost(), min_size=1, jump=1
            ).fit_predict(np.arange(6.0), n_bkps=1)

        class WrongBatchLength(BatchL2CustomCost):
            def error_many(self, starts, ends):
                return np.zeros(len(starts) + 1, dtype=np.float64)

        with self.assertRaisesRegex(ValueError, "returned .* values"):
            rustures.Pelt(
                custom_cost=WrongBatchLength(), min_size=1, jump=1
            ).fit_predict(np.arange(6.0), pen=1.0)

        class WrongBatchDtype(BatchL2CustomCost):
            def error_many(self, starts, ends):
                return np.zeros(len(starts), dtype=np.float32)

        with self.assertRaisesRegex(TypeError, "float64 NumPy array"):
            rustures.Dynp(
                custom_cost=WrongBatchDtype(), min_size=1, jump=1
            ).fit_predict(np.arange(6.0), n_bkps=1)

    def test_batch_bernoulli_custom_cost_detects_probability_change(self) -> None:
        signal = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        dynp = rustures.Dynp(
            custom_cost=BatchBernoulliCustomCost(), min_size=1, jump=1
        )
        self.assertEqual(dynp.fit_predict(signal, n_bkps=1), [3, 6])
        self.assertTrue(dynp.uses_batch_callback)

        pelt = rustures.Pelt(
            custom_cost=BatchBernoulliCustomCost(), min_size=1, jump=1
        )
        self.assertEqual(pelt.fit_predict(signal, pen=1.0), [3, 6])

    def test_custom_pelt_is_exact_for_an_arbitrary_additive_cost(self) -> None:
        class TableCost:
            min_size = 1

            def fit(self, signal):
                self.n_samples = len(signal)
                return self

            def error(self, start: int, end: int) -> float:
                return float(
                    ((7 * start + 11 * end + 3 * (end - start) ** 2) % 19) - 9
                )

        n_samples = 7
        penalty = 2.5
        cost = TableCost().fit(np.zeros(n_samples))
        candidates = []
        for mask in range(1 << (n_samples - 1)):
            breakpoints = [
                position
                for position in range(1, n_samples)
                if mask & (1 << (position - 1))
            ]
            breakpoints.append(n_samples)
            start = 0
            raw_cost = 0.0
            for end in breakpoints:
                raw_cost += cost.error(start, end)
                start = end
            objective = raw_cost + penalty * (len(breakpoints) - 1)
            candidates.append((objective, breakpoints))
        expected = min(candidates)[1]

        actual = rustures.Pelt(
            custom_cost=TableCost(), min_size=1, jump=1
        ).fit_predict(np.zeros(n_samples), pen=penalty)
        self.assertEqual(actual, expected)

    def test_greedy_detectors_and_stopping_rules(self) -> None:
        signal = np.repeat(np.array([0.0, 8.0, -5.0, 4.0]), 5)
        expected = [5, 10, 15, 20]
        detectors = [
            rustures.Binseg(min_size=2, jump=1),
            rustures.BottomUp(min_size=2, jump=1),
            rustures.Window(width=6, min_size=2, jump=1),
        ]
        for detector in detectors:
            with self.subTest(detector=type(detector).__name__):
                self.assertEqual(detector.fit_predict(signal, n_bkps=3), expected)
                self.assertTrue(detector.is_fitted)
                self.assertEqual(detector.predict(pen=1.0), expected)
                self.assertEqual(detector.predict(epsilon=0.0), expected)

    def test_greedy_detector_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one stopping rule"):
            rustures.Binseg().fit(np.arange(10.0)).predict()
        with self.assertRaisesRegex(ValueError, "exactly one stopping rule"):
            rustures.BottomUp().fit(np.arange(10.0)).predict(n_bkps=1, pen=1.0)
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            rustures.Window(width=4, min_size=2, jump=1).predict(n_bkps=1)

    def test_metrics_match_small_examples(self) -> None:
        self.assertEqual(rustures.hausdorff([3, 7, 10], [2, 8, 10]), 1.0)
        self.assertEqual(
            rustures.precision_recall([3, 7, 10], [2, 5, 10], margin=2),
            (0.5, 0.5),
        )
        self.assertEqual(rustures.rand_index([2, 4], [2, 4]), 1.0)
        with self.assertRaisesRegex(ValueError, "different sample lengths"):
            rustures.rand_index([5], [6])

    def test_seeded_dataset_generators(self) -> None:
        for generator in [
            rustures.pw_constant,
            rustures.pw_linear,
            rustures.pw_normal,
            rustures.pw_wavy,
        ]:
            with self.subTest(generator=generator.__name__):
                first, first_bkps = generator(
                    60, n_features=2, n_bkps=3, noise_std=0.1, seed=42
                )
                second, second_bkps = generator(
                    60, n_features=2, n_bkps=3, noise_std=0.1, seed=42
                )
                self.assertEqual(first.shape, (60, 2))
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first_bkps, second_bkps)
                self.assertEqual(first_bkps[-1], 60)

    def test_kernel_cpd_full_and_streaming_match(self) -> None:
        signal = np.repeat(np.array([0.0, 5.0, -3.0]), 4)
        full = rustures.KernelCPD(
            kernel="rbf", gamma=0.5, min_size=1, jump=1, backend="full"
        ).fit(signal)
        streaming = rustures.KernelCPD(
            kernel="rbf", gamma=0.5, min_size=1, jump=1, backend="streaming"
        ).fit(signal)
        self.assertEqual(full.predict(n_bkps=2), [4, 8, 12])
        self.assertEqual(streaming.predict(n_bkps=2), full.predict(n_bkps=2))
        self.assertEqual(streaming.predict(pen=1.0), full.predict(pen=1.0))
        self.assertEqual(streaming.stored_gram_entries, 0)
        self.assertGreater(full.stored_gram_entries, 0)
        self.assertEqual(full.gamma, 0.5)

    def test_fused_kernel_backend_matches_full_for_every_kernel(self) -> None:
        signal = np.column_stack(
            [
                np.repeat(np.array([0.0, 5.0, -3.0]), 4),
                np.tile(np.array([0.0, 1.0, 2.0, 1.0]), 3),
            ]
        )
        for kernel in ["linear", "rbf", "cosine"]:
            with self.subTest(kernel=kernel):
                parameters = {"gamma": 0.5} if kernel == "rbf" else {}
                fused = rustures.KernelCPD(
                    kernel=kernel, min_size=1, jump=1, backend="fused", **parameters
                ).fit(signal)
                full = rustures.KernelCPD(
                    kernel=kernel, min_size=1, jump=1, backend="full", **parameters
                ).fit(signal)
                self.assertEqual(fused.predict(n_bkps=2), full.predict(n_bkps=2))
                self.assertEqual(fused.predict(pen=1.0), full.predict(pen=1.0))
                self.assertEqual(fused.stored_gram_entries, 0)
                self.assertEqual(fused.backend, "fused")

    def test_linear_kernel_is_stable_under_large_feature_translations(self) -> None:
        signal = np.array(
            [
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 1.0],
                [5.0, -2.0],
                [5.0, -1.0],
                [5.0, -2.0],
                [-3.0, 4.0],
                [-3.0, 5.0],
                [-3.0, 4.0],
            ]
        )
        translated = signal + np.array([1.0e12, -1.0e12])
        for backend in ["fused", "full", "streaming"]:
            with self.subTest(backend=backend):
                base = rustures.KernelCPD(
                    kernel="linear", min_size=1, jump=1, backend=backend
                ).fit(signal)
                shifted = rustures.KernelCPD(
                    kernel="linear", min_size=1, jump=1, backend=backend
                ).fit(translated)
                self.assertEqual(base.predict(n_bkps=2), [3, 6, 9])
                self.assertEqual(shifted.predict(n_bkps=2), [3, 6, 9])
                self.assertEqual(shifted.predict(pen=1.0), base.predict(pen=1.0))

        extreme = np.array([np.finfo(np.float64).max, -np.finfo(np.float64).max])
        with self.assertRaisesRegex(rustures.RusturesError, "centering linear kernel"):
            rustures.KernelCPD(kernel="linear", backend="fused").fit(extreme)
        self.assertEqual(
            rustures.KernelCPD(kernel="linear", min_size=1)
            .fit(signal)
            .predict(n_bkps=2),
            [3, 6, 9],
        )

    def test_kernel_policies_and_memory_limit(self) -> None:
        signal = np.arange(20.0).reshape(10, 2)
        first = rustures.KernelCPD(
            gamma_policy="sampled", gamma_samples=30, seed=7
        ).fit(signal)
        second = rustures.KernelCPD(
            gamma_policy="sampled", gamma_samples=30, seed=7
        ).fit(signal)
        self.assertEqual(first.backend, "fused")
        self.assertEqual(first.stored_gram_entries, 0)
        self.assertEqual(first.gamma, second.gamma)
        with self.assertRaisesRegex(MemoryError, "Gram prefix requires"):
            rustures.KernelCPD(backend="full", max_gram_bytes=8).fit(signal)
        with self.assertRaisesRegex(ValueError, "supported kernels"):
            rustures.KernelCPD(kernel="unknown")

    def test_phase4_to_6_pinned_ruptures_fixture(self) -> None:
        fixture = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures"
                / "ruptures_phase4_6_v1.json"
            ).read_text(encoding="utf-8")
        )
        greedy = fixture["greedy"]
        signal = np.asarray(greedy["signal"])
        parameters = greedy["parameters"]
        detectors = {
            "binseg": rustures.Binseg(min_size=2, jump=1),
            "bottomup": rustures.BottomUp(min_size=2, jump=1),
            "window": rustures.Window(width=6, min_size=2, jump=1),
        }
        for name, detector in detectors.items():
            self.assertEqual(
                detector.fit_predict(signal, n_bkps=parameters["n_bkps"]),
                greedy["breakpoints"][name],
            )
        kernel = fixture["kernel_rbf"]
        self.assertEqual(
            rustures.KernelCPD(min_size=1, jump=1).fit_predict(
                np.asarray(kernel["signal"]), n_bkps=2
            ),
            kernel["breakpoints"],
        )
        metrics = fixture["metrics"]
        self.assertEqual(
            rustures.hausdorff(metrics["truth"], metrics["hausdorff_prediction"]),
            metrics["hausdorff"],
        )
        self.assertEqual(
            rustures.precision_recall(
                metrics["truth"], metrics["prediction"], metrics["margin"]
            ),
            tuple(metrics["precision_recall"]),
        )
        self.assertEqual(
            rustures.rand_index(metrics["rand_truth"], metrics["rand_prediction"]),
            metrics["rand_index"],
        )


if __name__ == "__main__":
    unittest.main()
