# Custom Python costs

Custom costs let you change the definition of a homogeneous segment without
rewriting Dynp or Pelt. Rustures handles candidate generation, dynamic programming,
pruning policy, tie-breaking, backtracking, and Python error propagation; your object
only evaluates `[start, end)` intervals.

## Protocol

Every custom cost must provide:

```python
class CustomCost:
    min_size: int

    def fit(self, signal):
        ...

    def error(self, start: int, end: int) -> float:
        ...
```

It may additionally provide:

```python
def error_many(self, starts, ends) -> np.ndarray:
    ...
```

The methods have distinct roles:

- `fit(signal)` validates the observations and prepares reusable statistics.
- `error(start, end)` returns one finite additive segment cost.
- `error_many(starts, ends)` optionally evaluates several pairs in one Python call.
- `min_size` declares the shortest interval the cost can evaluate safely.

Custom costs currently work with `Dynp` and `Pelt`.

## The pairwise batch contract

`error_many` is pairwise, not a Cartesian product. If all arrays have length `m`,
the required relationship is:

$$
\operatorname{costs}[i]
=\operatorname{error}(\operatorname{starts}[i],\operatorname{ends}[i]).
$$

```text
starts = [0,   4,   8]
ends   = [20, 20, 20]
costs  = [C(0,20), C(4,20), C(8,20)]
```

The concrete contract is:

- `starts`, `ends`, and the result have shape `(m,)`;
- `starts` and `ends` are one-dimensional NumPy integer arrays;
- all `ends` normally contain the same endpoint during search;
- the result is a one-dimensional NumPy array with exact dtype `float64`;
- every value is finite;
- each batch result agrees with scalar `error` for the same pair.

Rustures validates the returned shape, length, dtype, and finite values.

## A vectorized L2 example

L2 already has a faster native implementation in Rustures, so this class is an
instructional example. The same prefix-and-batch pattern applies to statistical
models that Rustures does not provide.

```python
import numpy as np
import rustures as rpt


class VectorizedL2Cost:
    min_size = 1

    # This promise is valid for an SSE/L2 segment cost.
    pelt_pruning_constant = 0.0

    def fit(self, signal):
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("expected a non-empty 1D or 2D signal")
        if not np.isfinite(values).all():
            raise ValueError("signal must contain only finite values")

        zeros = np.zeros((1, values.shape[1]), dtype=np.float64)
        self.prefix = np.vstack((zeros, np.cumsum(values, axis=0)))
        self.prefix_sq = np.vstack((zeros, np.cumsum(values * values, axis=0)))
        return self

    def error(self, start, end):
        length = end - start
        sums = self.prefix[end] - self.prefix[start]
        sums_sq = self.prefix_sq[end] - self.prefix_sq[start]
        return float(np.sum(sums_sq - sums * sums / length))

    def error_many(self, starts, ends):
        starts = np.asarray(starts, dtype=np.intp)
        ends = np.asarray(ends, dtype=np.intp)
        lengths = (ends - starts)[:, None]
        sums = self.prefix[ends] - self.prefix[starts]
        sums_sq = self.prefix_sq[ends] - self.prefix_sq[starts]
        return np.sum(
            sums_sq - sums * sums / lengths,
            axis=1,
            dtype=np.float64,
        )


detector = rpt.Pelt(
    custom_cost=VectorizedL2Cost(),
    min_size=6,
    jump=4,
).fit(signal)

assert detector.uses_custom_cost
assert detector.uses_batch_callback
assert detector.uses_pelt_pruning

breakpoints = detector.predict(pen=12.0)
```

The interval identity is:

$$
C(s,e)=
\sum_{i=s}^{e-1}\lVert x_i\rVert^2
-\frac{\left\lVert\sum_{i=s}^{e-1}x_i\right\rVert^2}{e-s}.
$$

Prefix arrays remove the need to rescan `signal[start:end]`, and NumPy indexing
evaluates all starts in one operation.

## Why `error_many` is a Rustures feature

The ruptures custom-cost interface invokes scalar `error(start, end)` for each
candidate. Rustures keeps that compatible fallback but can additionally collect
same-endpoint candidates and call `error_many` once.

```text
scalar
  candidate 1 → Python error()
  candidate 2 → Python error()
  candidate 3 → Python error()

endpoint batch
  candidates 1, 2, 3 → one Python error_many()
```

This reduces Python/Rust boundary crossings without allocating an `O(M²)` table of
all interval costs. Only the candidates for the current endpoint and their results
are materialized, using `O(M)` temporary memory.

## A batch method must actually vectorize

This implementation reduces boundary crossings but keeps a Python loop:

```python
def error_many(self, starts, ends):
    return np.asarray(
        [self.error(int(s), int(e)) for s, e in zip(starts, ends)],
        dtype=np.float64,
    )
```

It may provide little benefit when `error` itself is expensive. Prefer array
indexing, broadcasting, compiled NumPy operations, or another genuinely batched
implementation.

In one local pruned-Pelt measurement (`N=800`, `d=2`, `min_size=6`, `jump=4`,
`pen=12`, 15-run median), every implementation returned
`[268, 532, 540, 800]`:

| Rustures custom-cost implementation | Python callbacks | Logical intervals | Predict time |
|---|---:|---:|---:|
| Scalar, direct segment scan | 6,586 | 6,586 | 75.40 ms |
| Batch with an internal Python loop | 203 | 6,586 | 72.66 ms |
| Scalar prefix-L2 | 6,586 | 6,586 | 43.80 ms |
| Vectorized batch prefix-L2 | 203 | 6,586 | **2.96 ms** |

The 25.5x difference between the first and last rows combines three improvements:
prefix statistics, fewer Python callbacks, and NumPy vectorization. It is an
internal Rustures comparison, not a claim that every custom cost is 25.5x faster
than ruptures. A ruptures custom cost can still use prefix statistics inside its
scalar `error`; it cannot expose the endpoint batch protocol.

## Pelt pruning is a correctness contract

Dynp needs an additive segment cost. Safe Pelt pruning needs the stronger condition
that, for every pair of adjacent intervals `A` and `B`, there is a constant `K`
such that:

$$
C(A)+C(B)+K \le C(A\cup B).
$$

Rustures cannot prove this property by inspecting arbitrary Python code. Therefore:

```text
no pelt_pruning_constant → exact unpruned optimal partitioning
finite constant provided → pruning enabled using that constant
```

Only declare the property after proving it for your cost.

```python
class ProvenPrunableCost:
    min_size = 1
    pelt_pruning_constant = 0.0
```

!!! danger "Not a tuning flag"

    A false `pelt_pruning_constant` can discard the globally optimal partition.
    Use `None` or omit the attribute when uncertain.

After fitting, `Pelt.uses_pelt_pruning` reports the selected path.

## Callback exceptions and the GIL

Custom predict must call Python, so Rustures cannot release the GIL for the entire
operation as it does for a native cost. NumPy may release it during individual
compiled operations according to NumPy's own policy.

If `fit`, `error`, or `error_many` raises a Python exception, Rustures preserves the
original exception type, message, and traceback after unwinding the Rust search.

```python
class DomainError(RuntimeError):
    pass

class FailingCost:
    min_size = 1

    def fit(self, signal):
        return self

    def error(self, start, end):
        raise DomainError(f"invalid interval [{start}, {end})")
```

The caller can catch `DomainError`; it is not replaced with an opaque native error.

## Validation checklist

Before relying on a new custom cost:

1. Compare scalar `error` against a slow direct formula on random intervals.
2. Compare `error_many(starts, ends)[i]` against scalar `error` for every pair.
3. Test one- and multi-feature inputs where applicable.
4. Test the shortest accepted segment and reject shorter ones.
5. Verify finite output on constant, extreme, and nearly singular data.
6. Compare Dynp or unpruned Pelt against exhaustive enumeration for small signals.
7. Only then add and benchmark a vectorized `error_many` implementation.
8. Prove the Pelt inequality before declaring `pelt_pruning_constant`.

Use a built-in native cost whenever it already represents the desired model. Custom
costs are most valuable for domain-specific likelihoods, weighted losses, and new
statistical definitions.

