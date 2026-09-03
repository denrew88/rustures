# Performance

Performance results are workload- and machine-specific. This page records what was
measured, the main architectural reasons behind the results, and the limits of each
claim.

## Integration snapshot

The latest full comparison used:

- Windows x86-64;
- Python 3.11;
- Rustures 0.1.1 release ABI3 wheel;
- ruptures 1.1.10;
- isolated worker processes;
- deterministic inputs;
- five warmed timing runs per case;
- process RSS sampling in addition to elapsed time.

It covered 57 cost, detector, kernel, custom-cost, metric, and dataset cases.
Rustures returned valid results in every case. Of 40 comparable detector outputs,
39 breakpoint lists matched exactly. The one AR difference follows a documented
segment-boundary policy difference.

| Measured group | Geometric-mean result versus ruptures |
|---|---:|
| L2 Dynp, four signal families (`N=720`) | 1347.00x faster |
| L2 Pelt, four signal families (`N=1200`) | 668.77x faster |
| Fused KernelCPD, linear/RBF/cosine (`N=720`) | 1.37x faster |
| Full-Gram KernelCPD (`N=720`) | 1.89x slower |
| Gram-free streaming KernelCPD (`N=720`) | 1.12x slower |
| Scalar custom Pelt with proven pruning (`N=800`) | 1.07x slower |
| Synthetic dataset generators (`N=80,000`) | 1.33x faster |

These ratios are not universal guarantees. Signal structure, penalty, change count,
feature dimension, `jump`, CPU architecture, BLAS/NumPy build, and allocator state
can all change the result.

## Why built-in costs can be much faster

Built-in detector paths keep their inner loops and fitted statistics in Rust. They
avoid a Python method call for every candidate interval and can release the GIL
during long computations.

The largest L2 ratios above also reflect implementation structure, not merely the
language boundary: Rustures uses endpoint-major cost batches, compact flat DP state,
and cost-specific prefix scans. Compare algorithms and outputs, not only the source
language labels “Rust” and “Python.”

## Custom-cost batching

Rustures supports an optional pairwise endpoint batch that ruptures does not expose:

```python
costs = custom_cost.error_many(starts, ends)
```

On a local pruned-Pelt workload with `N=800`, two features, `min_size=6`, `jump=4`,
and `pen=12`, 15-run predict medians were:

| Rustures custom-cost implementation | Python callbacks | Logical intervals | Median |
|---|---:|---:|---:|
| Scalar, direct segment scan | 6,586 | 6,586 | 75.40 ms |
| Batch with an internal Python loop | 203 | 6,586 | 72.66 ms |
| Scalar prefix-L2 | 6,586 | 6,586 | 43.80 ms |
| Vectorized batch prefix-L2 | 203 | 6,586 | **2.96 ms** |

All four paths returned `[268, 532, 540, 800]`.

The 25.5x first-to-last speedup combines:

1. prefix statistics instead of rescanning every interval;
2. endpoint batching instead of thousands of Python/Rust crossings;
3. NumPy broadcasting instead of a Python loop inside the batch.

It is an internal Rustures scalar-versus-vectorized comparison. It must not be read
as “every Rustures custom cost is 25.5x faster than ruptures.” A fair framework
comparison must give both libraries the same scalar cost formula; a fair best-public-
API comparison must separately disclose that only Rustures has `error_many`.

## Kernel backend trade-offs

All three backends are exact, but they optimize different resources.

- `fused` avoids a full Gram matrix and is the default throughput path for fixed `K`.
- `streaming` retains only endpoint state and incrementally reuses symmetric kernel
  pairs, but additional control flow can make it slower than fused.
- `full` computes each symmetric pair once and provides constant-time interval cost
  queries, but requires quadratic storage and currently remains a performance target.

Linear and cosine kernels precompute reusable transformed data. RBF still performs
more expensive distance and exponential work per pair.

## Memory measurements

The integration harness launches each library and case in a fresh process, records
post-import RSS, samples peak RSS while the operation runs, and reports both peak and
incremental memory. This reduces contamination from imports, warmed allocators, and
the other library's native state.

RSS is still an operating-system observation rather than a precise attribution of
every allocation. Small differences should not be interpreted as exact object sizes.

Rustures also provides algorithm-level preflight controls:

```python
dynp.estimated_memory_bytes(n_bkps=K)
dynp.max_memory_bytes

kernel.max_gram_bytes  # constructor parameter for the full backend
kernel.stored_gram_entries
```

## Reproducing benchmarks

The comparison driver is maintained in the repository at
[`benchmarks/integration_comparison.py`](https://github.com/denrew88/rustures/blob/main/benchmarks/integration_comparison.py).
Run benchmarks from a release build in an otherwise idle environment, alternate
library order where practical, keep deterministic seeds, and preserve raw outputs.

Before quoting a ratio publicly, record:

- Rustures, ruptures, Python, and NumPy versions;
- operating system, architecture, and CPU;
- input shape and signal family;
- detector, cost, stopping rule, `min_size`, and `jump`;
- warm-up and repetition policy;
- whether fit time is included;
- breakpoint or objective agreement.

