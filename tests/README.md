# Test layout

The test tree is separated by what it is allowed to observe.

- `unit/`: unit-test module bodies for private Rust implementation details. Each
  production module includes its matching file only under `cfg(test)` with a
  `#[path = ...] mod tests;` declaration. This keeps tests physically outside
  `src/` without weakening Rust privacy or making internals public.
- `support/`: crate-internal test support, including the exhaustive partition
  oracle. `src/lib.rs` links this tree only for Rust test builds.
- `regression_mass.rs`: expensive Rust regression coverage that exercises the
  public crate surface and is normally run explicitly in release mode.
- `python/`: black-box tests for an installed Python wheel. These tests must not
  depend on the source tree or a Rust toolchain at runtime.
- `fixtures/`: pinned independent and `ruptures`-generated compatibility data.

Manual performance programs remain under `benches/`. The detailed fixed-K
kernel profiler stays with the private kernel unit tests because it measures
private solver stages; it is marked `#[ignore]` and runs only when requested.

`benchmarks/integration_comparison.py` is the cross-library black-box harness.
It runs each Rustures/ruptures case in a separate process, validates result
invariants and agreement, and records warmed wall time plus process peak/delta
RSS. Use its `quick` profile while changing the harness and `standard` for a
reportable local comparison.

The feature-gated Python panic probe lives in `src/python/test_support/` because
it must be compiled into a special validation wheel. Normal production wheels
do not enable that feature and do not expose `_panic_test_hook`.
