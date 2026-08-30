"""Offline change-point detection powered by Rust."""

from ._rustures import (
    Binseg, BottomUp, CostL2, Dynp, KernelCPD, Pelt, RusturesError, Window,
    hausdorff, precision_recall, pw_constant, pw_linear, pw_normal, pw_wavy,
    rand_index, validate_signal, version,
)

__version__ = version()

__all__ = [
    "CostL2",
    "Binseg",
    "BottomUp",
    "Dynp",
    "KernelCPD",
    "Pelt",
    "Window",
    "RusturesError",
    "hausdorff",
    "precision_recall",
    "pw_constant",
    "pw_linear",
    "pw_normal",
    "pw_wavy",
    "rand_index",
    "__version__",
    "validate_signal",
    "version",
]
