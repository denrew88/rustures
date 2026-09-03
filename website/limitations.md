# Limitations and compatibility

Rustures is pre-alpha. This page states the current boundaries so that users can
decide whether the library fits their workload.

## API stability

Public names and broad `fit` / `predict` workflows are usable, but signatures,
defaults, numerical policies, and compatibility guarantees may change before a
stable release. Pin the package version and include representative regression tests
in production pipelines.

```text
rustures==0.1.1
```

## Relationship to ruptures

Rustures follows familiar ruptures-style workflows where practical, but it is an
independent implementation rather than a complete drop-in replacement.

Known intentional differences include:

- custom Pelt does not assume that arbitrary user costs satisfy the pruning proof;
- Rustures custom costs may provide endpoint-batched `error_many` callbacks;
- AR segment-boundary semantics differ in a documented edge policy;
- exact kernel backends can choose different optimal paths when objective values tie;
- not every ruptures detector, cost, plotting helper, or dataset utility is present.

Always compare outputs on your data when migrating an established workflow.

## Exactness and `jump`

Dynp, Pelt, and KernelCPD can be exact over their selected candidate grid. With
`jump > 1`, internal breakpoint positions are restricted, so the result is not
necessarily the optimum over every sample index.

PELT's worst case is still quadratic. Pruning effectiveness depends on the cost,
penalty, minimum segment length, and data.

## Custom-cost boundaries

- Only Dynp and Pelt accept Python custom costs.
- Custom callbacks require the GIL and therefore do not receive the full native
  GIL-release benefit.
- `error_many` must return a one-dimensional, finite, exact `float64` NumPy array.
- A batch method containing a Python loop may provide little speedup.
- Custom Pelt is unpruned unless the cost author supplies a proven finite
  `pelt_pruning_constant`.
- An invalid pruning constant can produce an incorrect partition.
- Rustures does not maintain an unbounded interval cache or a full quadratic custom
  cost table.

## Input and numerical boundaries

- Signals must be finite and one- or two-dimensional.
- Most public boundaries require `float64` NumPy-compatible input rather than
  silently accepting every dtype.
- L1Potts accepts only scalar signals.
- Regression, covariance, and AR costs impose model-specific minimum lengths and
  may reject numerically unusable segments.
- Very large finite values can still overflow intermediate arithmetic; these cases
  return Python exceptions when detected.

## Memory limits

Dynp and the full-Gram kernel backend protect their principal prediction or Gram
allocations with configurable limits. These checks do not measure every allocation
owned by Python, NumPy, the allocator, or a user-defined custom cost.

The fused and streaming kernel backends avoid mandatory full-Gram storage, but they
still require fitted data, DP state, and temporary endpoint buffers.

## Panic and process safety

Public Python entry points catch unwinding Rust panics and return `RusturesError` so
the interpreter can continue. This does not cover failures that do not unwind:

- `process::abort`;
- operating-system process termination;
- stack overflow;
- allocator abort on unrecoverable out-of-memory conditions;
- undefined behaviour in external native code.

Rustures validates public inputs and preflights predictable large allocations to
avoid these paths, but does not claim that every possible process termination can be
converted into an exception.

## Published wheel targets

The current release policy targets:

| Operating system | Architectures |
|---|---|
| Linux | x86-64, ARM64 manylinux |
| Windows | x86-64 |
| macOS | Intel x86-64, Apple Silicon ARM64 |

The package declares Python 3.10 or newer through a CPython stable-ABI wheel. Current
wheels target GIL-enabled CPython and do not target 32-bit Python, PyPy, free-threaded
CPython, or native Windows ARM64.

## Reproducibility

Synthetic generator seeds are reproducible within a Rustures version. The exact
generated stream may change between releases when RNG or generation algorithms are
optimized.

Benchmark values are not universal guarantees. See [Performance](performance.md)
for the measurement environment and interpretation rules.

## Reporting an issue

Please include:

- a minimal reproducible example;
- Rustures, Python, and NumPy versions;
- operating system and architecture;
- signal shape and dtype;
- detector and all parameter values;
- full Python traceback;
- whether the process survives the exception.

[Open a GitHub issue](https://github.com/denrew88/rustures/issues)

