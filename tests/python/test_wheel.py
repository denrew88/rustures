import unittest

import numpy as np

import rustures


class WheelSmokeTests(unittest.TestCase):
    def test_version_is_exported(self) -> None:
        self.assertEqual(rustures.version(), rustures.__version__)
        self.assertEqual(rustures.__version__, "0.1.0")

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


if __name__ == "__main__":
    unittest.main()

