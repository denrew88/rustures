"""Offline change-point detection powered by Rust."""

from ._rustures import (
    Binseg, BottomUp, CostAR, CostCLinear, CostL1, CostL2, CostLinear,
    CostMahalanobis, CostMl, CostNormal, CostRank, Dynp, KernelCPD, L1Potts,
    Pelt, RusturesError, Window,
    hausdorff, precision_recall, pw_constant, pw_linear, pw_normal, pw_wavy,
    rand_index, validate_signal, version,
)

__version__ = version()

__all__ = [
    "CostL2",
    "CostL1",
    "CostRank",
    "CostNormal",
    "CostLinear",
    "CostAR",
    "CostCLinear",
    "CostMahalanobis",
    "CostMl",
    "Binseg",
    "BottomUp",
    "Dynp",
    "KernelCPD",
    "L1Potts",
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
