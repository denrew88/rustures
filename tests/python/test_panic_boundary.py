import unittest

import numpy as np

import rustures
from rustures import _rustures as native


class PanicBoundaryWheelTests(unittest.TestCase):
    def test_internal_panic_is_catchable_and_process_survives(self) -> None:
        caught: Exception | None = None
        try:
            native._panic_test_hook()
        except Exception as error:
            caught = error

        self.assertIsNotNone(caught, "the Rust panic was not exposed as Exception")
        self.assertIsInstance(caught, rustures.RusturesError)
        self.assertIsInstance(caught, RuntimeError)
        self.assertIn("intentional panic-test-hook panic", str(caught))

        # The same interpreter must remain usable after the panic is handled.
        value = rustures.CostL2().fit(np.array([0.0, 0.0, 4.0, 4.0])).error(0, 4)
        self.assertEqual(value, 16.0)


if __name__ == "__main__":
    unittest.main()
