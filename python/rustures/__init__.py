"""Offline change-point detection powered by Rust."""

from ._rustures import RusturesError, validate_signal, version

__version__ = version()

__all__ = ["RusturesError", "__version__", "validate_signal", "version"]

