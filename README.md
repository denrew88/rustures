<div align="center">

# rustures

**Fast, memory-aware offline change-point detection for Python — powered by Rust.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.83%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3](https://img.shields.io/badge/bindings-PyO3-FFD43B)](https://pyo3.rs/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](#project-status)

Exact dynamic programming, kernel methods, robust costs, custom Python costs,
and familiar `fit` / `predict` APIs in one native extension.

[Quick start](#quick-start) · [Algorithms](#choosing-an-algorithm) · [Custom costs](#custom-python-costs) · [Build](#installation) · [Tutorial](examples/rustures_tutorial_ko.ipynb)

</div>

---

`rustures` finds points where the statistical behaviour of a sequence changes:
its mean, distribution, trend, autoregressive dynamics, or kernel representation.
The search algorithms and built-in cost functions run in Rust while the public API
stays in Python.

The project is inspired by the excellent
[`ruptures`](https://github.com/deepcharles/ruptures) ecosystem, but it is an
independent implementation rather than a complete drop-in replacement.

> [!IMPORTANT]
> `rustures` is currently pre-alpha. The API is usable and heavily tested, but
> packaging, CI platform coverage, and release metadata are still being hardened.

## Why rustures?

- **A native core without a Python loop in the hot path.** Built-in costs and
  detectors execute in optimized Rust.
- **Exact and approximate search strategies.** Use fixed-`K` dynamic programming,
  penalized optimal partitioning, kernel CPD, or faster greedy detectors.
- **Memory is part of the API.** Dynp and full-Gram kernel backends reject oversized
  jobs before allocating their main tables.
- **Kernel CPD without mandatory quadratic storage.** The default fused backend is
  exact and does not materialize a full Gram matrix.
- **Custom Python costs when the built-ins are not enough.** Dynp and Pelt accept
  scalar callbacks and optional endpoint-batched callbacks.
- **Multivariate input is first-class.** Most costs accept an `(n_samples,
  n_features)` NumPy array; scalar signals may remain one-dimensional.
- **Python-safe failure boundaries.** Invalid data, allocation limits, numerical
  failures, and unwinding Rust panics become catchable Python exceptions.
- **Reproducible validation.** Deterministic generators, metrics, exhaustive small
  oracles, parity fixtures, and raw benchmark artifacts live in the repository.

## Quick start

```python
import rustures as rpt

# Deterministic piecewise-constant data and its true breakpoints.
signal, truth = rpt.pw_constant(
    n_samples=600,
    n_features=2,
    n_bkps=3,
    noise_std=0.7,
    seed=42,
)

# Penalized exact segmentation.
prediction = rpt.Pelt(
    model="l2",
    min_size=10,
    jump=1,
).fit_predict(signal, pen=12.0)

precision, recall = rpt.precision_recall(truth, prediction, margin=10)

print("truth:     ", truth)
print("prediction:", prediction)
print(f"precision={precision:.3f}, recall={recall:.3f}")
```

Breakpoints use the half-open interval convention and always include the terminal
sample. A result such as `[120, 360, 600]` represents segments `[0, 120)`,
`[120, 360)`, and `[360, 600)`.

## Choosing an algorithm

| You know… | Start with | What it does |
|---|---|---|
| The number of changes `K` | `Dynp` | Exact fixed-`K` dynamic programming |
| A penalty per additional change | `Pelt` | Exact penalized optimal partitioning; uses pruning only when the cost proves it is valid |
| The change may be nonlinear or distributional | `KernelCPD` | Exact linear, RBF, or cosine kernel segmentation |
| You need a fast exploratory result | `Binseg` | Recursive binary segmentation |
| You prefer merge-based segmentation | `BottomUp` | Starts small and merges neighbouring segments |
| Changes should be found from a local score | `Window` | Window discrepancy with deterministic peak selection |
| A scalar signal is piecewise constant with robust L1 loss | `L1Potts` | Weighted scalar L1-Potts optimization |

### Fixed number of changes

```python
algo = rpt.Dynp(model="normal", min_size=8, jump=2).fit(signal)

print("workspace bytes:", algo.estimated_memory_bytes(n_bkps=3))
breakpoints = algo.predict(n_bkps=3)
```

Dynp defaults to a 512 MiB prediction-workspace limit. Override it explicitly when
you know the process budget:

```python
algo = rpt.Dynp(
    model="l2",
    jump=1,
    max_memory_bytes=256 * 1024 * 1024,
).fit(signal)

# Raises MemoryError before allocating DP states if the limit would be exceeded.
breakpoints = algo.predict(n_bkps=32)
```

### Kernel change-point detection

```python
kernel_algo = rpt.KernelCPD(
    kernel="rbf",
    gamma_policy="sampled",
    gamma_samples=10_000,
    seed=42,
    backend="fused",
    min_size=5,
    jump=1,
)

breakpoints = kernel_algo.fit_predict(signal, n_bkps=3)
```

Available kernels are `"linear"`, `"rbf"`, and `"cosine"`.

| Backend | Exact? | Main storage behaviour |
|---|:---:|---|
| `fused` | Yes | Default fixed-`K` implementation; no full Gram matrix |
| `streaming` | Yes | Computes kernel contributions without retaining a full Gram table |
| `full` | Yes | Stores a full Gram prefix for repeated constant-time segment-cost queries |

The full backend has its own 512 MiB default limit through `max_gram_bytes`.

## Cost models

The following model strings work with the general-purpose detectors:

| Model | Detects changes in… | Notes |
|---|---|---|
| `l2` | Mean | Fast prefix sums; scalar or multivariate |
| `l1` | Median / robust location | Component-wise median absolute deviation |
| `rank` | Distribution | Global ranks with tie handling |
| `normal` | Gaussian mean and covariance | Regularized covariance log-determinant |
| `linear` | Regression relationship | First column is the response; remaining columns are predictors |
| `ar` | Autoregressive dynamics | Default order is 4 |
| `clinear` | Continuous piecewise-linear trend | Endpoint interpolation cost |
| `mahalanobis` | Metric-weighted scatter | Exposed as `CostMahalanobis(metric=...)` / `CostMl` |

Standalone cost objects expose `fit`, `error(start, end)`, and `sum_of_costs`:

```python
cost = rpt.CostL2().fit(signal)
segment_cost = cost.error(100, 220)
partition_cost = cost.sum_of_costs([100, 220, len(signal)])
```

## Custom Python costs

Dynp and Pelt accept any object with this protocol:

```python
class CustomCost:
    min_size: int

    def fit(self, signal): ...
    def error(self, start: int, end: int) -> float: ...

    # Optional: calculate several segments ending at the same endpoint.
    def error_many(self, starts, ends): ...
```

For example, a Bernoulli negative log-likelihood cost can be written as:

```python
import numpy as np
import rustures as rpt


class BernoulliCost:
    min_size = 1

    def fit(self, signal):
        values = np.asarray(signal, dtype=np.float64).reshape(-1)
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("BernoulliCost expects only 0 and 1")
        self.values = values
        self.prefix = np.r_[0.0, np.cumsum(values)]
        return self

    def error(self, start, end):
        length = end - start
        ones = self.prefix[end] - self.prefix[start]
        p = ones / length
        if p == 0.0 or p == 1.0:
            return 0.0
        return -(ones * np.log(p) + (length - ones) * np.log1p(-p))


binary_signal = np.r_[np.zeros(80), np.ones(60), np.zeros(90)]

breakpoints = rpt.Dynp(
    custom_cost=BernoulliCost(),
    min_size=10,
    jump=1,
).fit_predict(binary_signal, n_bkps=2)
```

`error_many` avoids one Python call per segment candidate. The adapter retains only
one endpoint batch, so it does not silently build an `O(n²)` Python cost table.
Exceptions raised by a custom cost preserve their Python type, message, and
traceback. Custom Pelt uses the exact unpruned path because arbitrary user costs do
not automatically satisfy the PELT pruning inequality.

## Included utilities

Deterministic signal generators:

- `pw_constant`
- `pw_linear`
- `pw_normal`
- `pw_wavy`

Evaluation metrics:

- `hausdorff`
- `precision_recall`
- `rand_index`

All generators require an explicit seed and return `(signal, breakpoints)`.

## Performance snapshot

The latest local comparison used Windows 11 x86-64, Python 3.11, a release abi3
wheel, `N=400`, `min_size=5`, `jump=5`, and the median of five alternating runs.
Times include a fresh `Dynp.fit_predict` call.

| Model | K | rustures | pinned ruptures | Relative speed |
|---|---:|---:|---:|---:|
| Linear | 1 | 1.868 ms | 4.526 ms | 2.42× |
| Linear | 4 | 1.936 ms | 115.019 ms | 59.42× |
| Linear | 8 | 2.035 ms | 136.428 ms | 67.04× |
| AR | 1 | 3.738 ms | 6.039 ms | 1.62× |
| AR | 4 | 4.428 ms | 155.526 ms | 35.12× |
| AR | 8 | 4.274 ms | 194.458 ms | 45.50× |

These are machine- and workload-specific results, not universal guarantees. Raw
data is available in
[`artifacts/validation/phase9-final-regression-benchmark.json`](artifacts/validation/phase9-final-regression-benchmark.json),
and the benchmark driver is
[`benchmarks/benchmark_phase7_costs.py`](benchmarks/benchmark_phase7_costs.py).

## Correctness and safety

The repository currently checks correctness through several independent layers:

- exhaustive enumeration for small fixed-`K` and penalized problems;
- black-box parity fixtures generated from pinned `ruptures` behaviour;
- full-Gram, streaming, fused, scalar, and AVX2 backend parity tests;
- deterministic tie-breaking tests;
- finite-input, overflow, singular, collinear, constant, and large-offset cases;
- 2,528 Linear/AR fast-path comparisons against scalar SVD Dynp and Pelt;
- Python exception and Rust panic-boundary process-survival tests.

The AVX2 path uses runtime CPU detection. CPUs without AVX2 automatically use the
scalar implementation instead of failing at import time.

## Installation

### Current compatibility

- Python 3.10 or newer is declared through `abi3-py310`.
- NumPy 1.23 or newer is required.
- The currently verified binary environment is 64-bit CPython 3.11 on Windows 11
  x86-64.
- The current wheel tag is `cp310-abi3-win_amd64`; it does not target 32-bit Python,
  PyPy, or native Windows ARM64.

Published PyPI wheels and a cross-platform CI matrix are not available yet. Build
the extension from source for now.

### Build from source

Prerequisites: Python 3.10+, Rust 1.83+, and a working native compiler toolchain.

```bash
git clone https://github.com/denrew88/rustures.git
cd rustures

python -m venv .venv
```

Activate the environment:

```text
Windows PowerShell:  .venv\Scripts\Activate.ps1
Linux/macOS:         source .venv/bin/activate
```

Build and install an editable release extension:

```bash
python -m pip install --upgrade pip
python -m pip install "maturin>=1.14,<2.0" "numpy>=1.23"
python -m maturin develop --release
```

Or create a wheel:

```bash
python -m maturin build --release
```

The wheel is written to `target/wheels/`.

## Development

```bash
# Rust unit, oracle, and parity tests
cargo test

# Formatting and linting
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Python wheel tests after installing a built wheel
python -m pip install pytest
python -m pytest tests/python/test_wheel.py -q

# Longer Linear/AR regression matrix
cargo test --release --test regression_mass -- --ignored --nocapture
```

The Korean tutorial notebook is available at
[`examples/rustures_tutorial_ko.ipynb`](examples/rustures_tutorial_ko.ipynb).

## Project status

Implemented today:

- Dynp, Pelt, Binseg, BottomUp, Window, KernelCPD, and L1Potts
- eight general-purpose cost models
- linear, RBF, and cosine kernels with three exact backends
- custom Python costs for Dynp and Pelt
- multivariate signal handling, metrics, and deterministic datasets
- typed Python errors, panic isolation, memory preflight, and type hints

Major work still planned:

- automated Windows, Linux, and macOS wheel builds;
- clean-environment tests across supported Python versions;
- PyPI release automation and complete package metadata;
- approximate low-rank kernel backends;
- broader profiling across CPU architectures and feature dimensions.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

The Rust dependencies compiled into the wheel, their selected license options,
copyright notices, and full license texts are recorded in
[`THIRD-PARTY-LICENSES`](THIRD-PARTY-LICENSES). The report is generated from
`Cargo.lock` for the verified Windows x86-64 target with:

```bash
cargo install --locked --features cli cargo-about
cargo about generate --locked --fail -c about.toml -o THIRD-PARTY-LICENSES about.hbs
```

---

<div align="center">

Built for people who want Python ergonomics, Rust execution, and explicit
correctness contracts in offline change-point detection.

</div>
