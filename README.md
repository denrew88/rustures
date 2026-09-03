<div align="center">

# rustures

**Fast, memory-aware offline change-point detection for Python — powered by Rust.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/rustures?logo=pypi&logoColor=white)](https://pypi.org/project/rustures/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-4051B5?logo=materialformkdocs&logoColor=white)](https://denrew88.github.io/rustures/)
[![Rust](https://img.shields.io/badge/Rust-1.83%2B-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![PyO3](https://img.shields.io/badge/bindings-PyO3-FFD43B)](https://pyo3.rs/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](#project-status)

Exact dynamic programming, kernel methods, robust costs, custom Python costs,
and familiar `fit` / `predict` APIs in one native extension.

[Documentation](https://denrew88.github.io/rustures/) · [Quick start](#quick-start) · [Algorithms](#choosing-an-algorithm) · [Custom costs](#custom-python-costs) · [Tutorial](examples/rustures_tutorial_en.ipynb)

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
> public APIs and compatibility guarantees may still change between releases.

## Why rustures?

- **A native core without a Python loop in the hot path.** Built-in costs and
  detectors execute in optimized Rust.
- **Real concurrency from ordinary Python threads.** Built-in predictions release
  the GIL while the native search runs, so independent detections can overlap in a
  `ThreadPoolExecutor` without process-level serialization overhead.
- **Exact and approximate search strategies.** Use fixed-`K` dynamic programming,
  penalized optimal partitioning, kernel CPD, or faster greedy detectors.
- **Memory is part of the API.** Dynp and full-Gram kernel backends reject oversized
  jobs before allocating their main tables.
- **Kernel CPD without mandatory quadratic storage.** The default fused backend is
  exact and does not materialize a full Gram matrix.
- **Endpoint-batched custom Python costs.** Dynp and Pelt accept ordinary scalar
  callbacks plus an optional vectorized `error_many(starts, ends)` protocol that
  is not available in ruptures' scalar-only custom-cost interface.
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

    # Optional pairwise batch: result[i] == error(starts[i], ends[i]).
    # This is not a Cartesian product of every start and end.
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

    def error_many(self, starts, ends):
        starts = np.asarray(starts, dtype=np.intp)
        ends = np.asarray(ends, dtype=np.intp)
        lengths = ends - starts
        ones = self.prefix[ends] - self.prefix[starts]
        probabilities = ones / lengths
        mixed = (ones > 0.0) & (ones < lengths)
        costs = np.zeros(len(starts), dtype=np.float64)
        costs[mixed] = (
            -ones[mixed] * np.log(probabilities[mixed])
            -(lengths[mixed] - ones[mixed])
            * np.log1p(-probabilities[mixed])
        )
        return costs


binary_signal = np.r_[np.zeros(80), np.ones(60), np.zeros(90)]

breakpoints = rpt.Dynp(
    custom_cost=BernoulliCost(),
    min_size=10,
    jump=1,
).fit_predict(binary_signal, n_bkps=2)
```

Unlike ruptures' custom-cost protocol, which calls `error(start, end)` once per
candidate, Rustures can send every candidate ending at the current endpoint through
one `error_many` call. This makes NumPy broadcasting and prefix-array indexing
possible without storing a full `O(n²)` cost table. Search algorithms batch only
the candidates they currently need, while a standalone segment request evaluates
only that segment.

The batch contract is pairwise, not Cartesian. For one-dimensional arrays with
shape `(m,)`, the returned `float64` array must also have shape `(m,)` and satisfy
`costs[i] == error(int(starts[i]), int(ends[i]))`. During endpoint batching, all
entries of `ends` normally contain the same endpoint:

```text
starts = [0,   4,   8]
ends   = [20, 20, 20]
costs  = [C(0,20), C(4,20), C(8,20)]
```

`error_many` must be genuinely vectorized to provide the largest benefit. Wrapping
scalar `error` calls in a Python list comprehension reduces Rust/Python crossings
but retains the Python loop. In a local pruned-Pelt workload (`N=800`, two features,
`jump=4`), a vectorized prefix-L2 callback reduced Rustures' callback count from
6,586 to 203 and predict time from 75.40 ms to 2.96 ms. This 25.5x figure is an
internal scalar-versus-vectorized Rustures comparison, not a claim that every
custom cost or every workload is 25.5x faster than ruptures.
Exceptions raised by a custom cost preserve their Python type, message, and
traceback. Custom Pelt uses the exact unpruned path because arbitrary user costs do
not automatically satisfy the PELT pruning inequality. A cost whose author has
proved the PELT inequality may explicitly expose a finite constant:

```python
class PrunableCustomCost(CustomCost):
    pelt_pruning_constant = 0.0
```

This is a mathematical correctness promise, not a tuning flag: an invalid value
can prune the optimal partition. After fitting, `Pelt.uses_pelt_pruning` reports
whether the optimized path is active.

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

All generators require an explicit seed and return `(signal, breakpoints)`. A seed
is reproducible within a Rustures version; generator streams may change between
releases when the documented RNG implementation is optimized.

## Performance snapshot

The latest integration benchmark used Windows x86-64, Python 3.11, Rustures
0.1.1, ruptures 1.1.10, isolated worker processes, and five warmed timing runs.
It exercised 57 cost, detector, kernel, custom-cost, metric, and dataset cases.
Rustures returned valid results in every case; 39 of 40 comparable breakpoint
results matched exactly. The remaining AR case uses a documented different
segment-boundary policy.

| Measured group | Geometric-mean result versus ruptures |
|---|---:|
| L2 Dynp, four signal families (`N=720`) | 1347.00× faster |
| L2 Pelt, four signal families (`N=1200`) | 668.77× faster |
| Fused KernelCPD, linear/RBF/cosine (`N=720`) | 1.37× faster |
| Full-Gram KernelCPD (`N=720`) | 1.89× slower |
| Gram-free streaming KernelCPD (`N=720`) | 1.12× slower |
| Scalar custom Pelt with proven pruning opt-in (`N=800`) | 1.07× slower |
| Synthetic dataset generators (`N=80000`) | 1.33× faster |

The streaming backend incrementally reuses each symmetric kernel pair while
retaining only `O(n)` endpoint state; fused remains the default high-throughput
path. These are machine- and workload-specific measurements, not universal
guarantees. Raw timing, breakpoint, environment, and process-RSS data is available
in [`artifacts/validation/integration-comparison-optimized-windows-py311.json`](artifacts/validation/integration-comparison-optimized-windows-py311.json),
and the reproducible driver is
[`benchmarks/integration_comparison.py`](benchmarks/integration_comparison.py).

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
- Binary wheels are published for Linux x86-64 and ARM64, Windows x86-64, and
  macOS Intel and Apple Silicon.
- The current wheels target GIL-enabled CPython. They do not target 32-bit Python,
  PyPy, free-threaded CPython, or native Windows ARM64.

Install a published wheel from PyPI:

```bash
python -m pip install rustures
```

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

Tutorial notebooks are available in
[English](examples/rustures_tutorial_en.ipynb) and
[Korean](examples/rustures_tutorial_ko.ipynb).

## Project status

Implemented today:

- Dynp, Pelt, Binseg, BottomUp, Window, KernelCPD, and L1Potts
- eight general-purpose cost models
- linear, RBF, and cosine kernels with three exact backends
- custom Python costs for Dynp and Pelt
- multivariate signal handling, metrics, and deterministic datasets
- typed Python errors, panic isolation, memory preflight, and type hints

Major work still planned:

- approximate low-rank kernel backends;
- broader profiling across CPU architectures and feature dimensions;
- additional interpreter and architecture coverage as the API matures.

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
