import json
from pathlib import Path
import unittest

import numpy as np

import rustures


class WheelSmokeTests(unittest.TestCase):
    def test_version_is_exported(self) -> None:
        self.assertEqual(rustures.version(), rustures.__version__)
        self.assertEqual(rustures.__version__, "0.1.0")
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
