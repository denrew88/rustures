# Python API reference

This page lists the public Python surface of Rustures 0.2.0. All detector outputs
are `list[int]` breakpoint lists containing the terminal sample count.

```python
import rustures as rpt
```

## Detectors

### `Dynp`

```python
rpt.Dynp(
    model: str = "l2",
    custom_cost: object | None = None,
    min_size: int = 2,
    jump: int = 5,
    max_memory_bytes: int = 536_870_912,
)
```

Methods:

```python
fit(signal) -> Dynp
predict(n_bkps: int) -> list[int]
fit_predict(signal, n_bkps: int) -> list[int]
estimated_memory_bytes(n_bkps: int) -> int
```

Read-only properties:

```text
model, min_size, jump, max_memory_bytes, is_fitted,
uses_custom_cost, uses_batch_callback
```

### `Pelt`

```python
rpt.Pelt(
    model: str = "l2",
    custom_cost: object | None = None,
    min_size: int = 2,
    jump: int = 5,
)
```

Methods:

```python
fit(signal) -> Pelt
predict(pen: float) -> list[int]
fit_predict(signal, pen: float) -> list[int]
```

Read-only properties:

```text
model, min_size, jump, is_fitted,
uses_custom_cost, uses_batch_callback, uses_pelt_pruning
```

`pen` must be finite and positive. See [Pelt pruning](custom-costs.md#pelt-pruning-is-a-correctness-contract)
before enabling pruning for a custom cost.

### `Binseg`

```python
rpt.Binseg(model: str = "l2", min_size: int = 2, jump: int = 5)
```

```python
fit(signal) -> Binseg
predict(n_bkps=None, pen=None, epsilon=None) -> list[int]
fit_predict(signal, n_bkps=None, pen=None, epsilon=None) -> list[int]
```

Exactly one stopping rule must be supplied. Read-only properties are `model`,
`min_size`, `jump`, and `is_fitted`.

### `BottomUp`

```python
rpt.BottomUp(model: str = "l2", min_size: int = 2, jump: int = 5)
```

`BottomUp` provides the same methods, stopping rules, and properties as `Binseg`.

### `Window`

```python
rpt.Window(
    width: int = 100,
    model: str = "l2",
    min_size: int = 2,
    jump: int = 5,
)
```

`Window` provides the same methods and stopping rules as `Binseg`; it additionally
exposes the read-only `width` property.

### `KernelCPD`

```python
rpt.KernelCPD(
    kernel: str = "rbf",
    min_size: int = 2,
    jump: int = 1,
    gamma: float | None = None,
    gamma_policy: str = "exact",
    gamma_samples: int = 10_000,
    seed: int = 0,
    backend: str = "fused",
    max_gram_bytes: int = 536_870_912,
)
```

```python
fit(signal) -> KernelCPD
predict(n_bkps=None, pen=None) -> list[int]
fit_predict(signal, n_bkps=None, pen=None) -> list[int]
```

Exactly one of `n_bkps` and `pen` must be supplied.

Accepted values:

```text
kernel:       "linear", "rbf", "cosine"
backend:      "fused", "full", "streaming"
gamma_policy: "exact", "sampled"
```

A supplied `gamma` overrides the RBF gamma policy. Read-only properties are
`kernel`, `backend`, `min_size`, `jump`, `gamma`, `stored_gram_entries`, and
`is_fitted`.

### `L1Potts`

```python
rpt.L1Potts()
```

```python
fit(signal, weights=None) -> L1Potts
predict(pen: float) -> list[int]
fit_predict(signal, pen: float, weights=None) -> list[int]
```

The signal must be one-dimensional. Weights, when supplied, must be a finite,
non-negative one-dimensional array with the same length. Read-only properties are
`n_samples`, `distinct_levels`, and `is_fitted`.

## Detector model names

The general-purpose detectors accept:

```text
"l2", "l1", "rank", "normal",
"linear", "ar", "clinear", "mahalanobis"
```

`custom_cost` takes precedence over the model string for Dynp and Pelt.

## Cost classes

All cost classes provide:

```python
fit(signal) -> self
error(start: int, end: int) -> float
sum_of_costs(breakpoints: list[int]) -> float
```

and read-only properties:

```text
n_samples, n_features, min_size, is_fitted
```

Available classes:

```python
rpt.CostL2()
rpt.CostL1()
rpt.CostRank()
rpt.CostLinear()
rpt.CostCLinear()
rpt.CostNormal(ridge: float = 1e-6)
rpt.CostAR(order: int = 4)
rpt.CostMahalanobis(metric)
rpt.CostMl(metric)  # alias
```

Additional properties:

- `CostNormal.ridge`
- `CostAR.order`
- `CostMahalanobis.metric_dimension`

See [Cost functions](costs.md) for model interpretation.

## Custom-cost protocol

Required:

```python
class CustomCost:
    min_size: int
    def fit(self, signal) -> object: ...
    def error(self, start: int, end: int) -> float: ...
```

Optional:

```python
def error_many(self, starts, ends) -> np.ndarray: ...
pelt_pruning_constant: float | None
```

The batch result is pairwise:

```python
costs[i] == error(starts[i], ends[i])
```

Read the full [custom-cost contract](custom-costs.md).

## Metrics

### `hausdorff`

```python
rpt.hausdorff(truth: list[int], prediction: list[int]) -> float
```

### `precision_recall`

```python
rpt.precision_recall(
    truth: list[int],
    prediction: list[int],
    margin: int = 10,
) -> tuple[float, float]
```

### `rand_index`

```python
rpt.rand_index(truth: list[int], prediction: list[int]) -> float
```

Breakpoint inputs must use the terminal-inclusive Rustures convention.

## Synthetic datasets

Each generator returns `(signal, breakpoints)`.

```python
rpt.pw_constant(
    n_samples: int,
    n_features: int = 1,
    n_bkps: int = 3,
    noise_std: float = 1.0,
    seed: int = 0,
)

rpt.pw_linear(...)
rpt.pw_normal(...)
rpt.pw_wavy(...)
```

Signals are returned as NumPy-compatible arrays. Use an explicit seed for
reproducible examples.

## Validation and version helpers

```python
rpt.version() -> str
rpt.__version__: str
rpt.validate_signal(signal) -> tuple[int, int]
```

`validate_signal` returns `(n_samples, n_features)` or raises an exception.

## Exceptions

```python
rpt.RusturesError(RuntimeError)
```

Common mappings:

| Condition | Python exception |
|---|---|
| Invalid value, range, shape, or parameter | `ValueError` |
| Incompatible Python object or callback return type | `TypeError` |
| Prediction before fitting | `RuntimeError` |
| Prediction or Gram workspace exceeds its configured limit | `MemoryError` |
| Native numerical failure or caught Rust panic | `RusturesError` |
| Exception raised by a custom callback | Original Python exception |
