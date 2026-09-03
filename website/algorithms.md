# Choosing an algorithm

Change-point detection separates one sequence into contiguous segments. The search
algorithm decides which partitions are considered and how the best one is selected;
the cost function decides how well each segment fits one statistical model.

## At a glance

| Algorithm | Stopping rule | Search style | Good first use |
|---|---|---|---|
| `Dynp` | Fixed `n_bkps` | Exact dynamic programming on the grid | You know the number of changes |
| `Pelt` | Positive `pen` | Penalized optimal partitioning with safe optional pruning | You prefer a penalty over a fixed count |
| `KernelCPD` | `n_bkps` or `pen` | Exact kernel segmentation | Mean/variance summaries are insufficient |
| `Binseg` | `n_bkps`, `pen`, or `epsilon` | Recursive greedy splitting | Fast exploration |
| `BottomUp` | `n_bkps`, `pen`, or `epsilon` | Greedy adjacent merging | Merge-based segmentation |
| `Window` | `n_bkps`, `pen`, or `epsilon` | Local discrepancy peaks | Localized changes |
| `L1Potts` | Positive `pen` | Specialized scalar L1 optimization | Robust piecewise-constant scalar signals |

Let `M` be the number of candidate positions after applying `jump`. The estimates
below describe the search state and do not include cost-specific fitted data.

## Dynp

`Dynp` solves the fixed-number-of-changes problem:

$$
\min_{0=t_0<t_1<\cdots<t_{K}<t_{K+1}=n}
\sum_{j=0}^{K} C(t_j,t_{j+1}).
$$

The dynamic program stores the best partition ending at each candidate position
for each segment count. Every possible final segment has a start `s` and end `e`;
the optimal prefix ending at `s` is reused instead of enumerating every complete
partition from scratch.

```python
detector = rpt.Dynp(
    model="l2",
    min_size=8,
    jump=2,
    max_memory_bytes=512 * 1024 * 1024,
).fit(signal)

required = detector.estimated_memory_bytes(n_bkps=4)
breakpoints = detector.predict(n_bkps=4)
```

The DP update is `O(K M²)` in the general case and prediction state is `O(K M)`.
Rustures checks the configured workspace limit before allocating the main tables.

Use Dynp when:

- the number of changes is known or controlled externally;
- an exact grid optimum matters;
- the expected DP state fits the process memory budget.

## Pelt

`Pelt` solves a penalized objective:

$$
\min_{0=t_0<\cdots<t_q=n}
\left[\sum_{j=0}^{q-1} C(t_j,t_{j+1}) + \lambda(q-1)\right].
$$

Here `pen=λ` charges for every internal change point. Larger penalties normally
select fewer changes.

```python
breakpoints = rpt.Pelt(
    model="l2",
    min_size=8,
    jump=1,
).fit_predict(signal, pen=12.0)
```

PELT can discard candidate starts only when the cost satisfies a segment-combination
inequality. Built-in costs opt in only when Rustures has that mathematical contract.
Arbitrary custom costs therefore use exact unpruned optimal partitioning unless the
author explicitly declares `pelt_pruning_constant`.

Worst-case search remains quadratic in `M`. Effective pruning can reduce the number
of evaluated candidates substantially on suitable data, but linear-time behaviour
is not guaranteed for every signal or penalty.

Use Pelt when:

- a penalty is more natural than fixing `K`;
- you want an exact optimum over the candidate grid;
- you understand the cost and penalty scale.

## KernelCPD

Kernel change-point detection measures segment scatter in a feature space. It can
detect changes that a simple mean-shift cost cannot express.

```python
detector = rpt.KernelCPD(
    kernel="rbf",
    gamma_policy="sampled",
    gamma_samples=10_000,
    seed=42,
    backend="fused",
    min_size=6,
    jump=1,
)

breakpoints = detector.fit_predict(signal, n_bkps=3)
```

### Kernels

| Kernel | Interpretation | Important parameter |
|---|---|---|
| `linear` | Changes in ordinary dot-product geometry | None |
| `rbf` | Nonlinear similarity based on squared distance | `gamma` or `gamma_policy` |
| `cosine` | Changes in normalized direction | Zero vectors are handled explicitly |

### Backends

| Backend | Exact | Main storage behaviour | Prefer when |
|---|:---:|---|---|
| `fused` | Yes | No full Gram matrix | Default fixed-`K` throughput |
| `streaming` | Yes | Gram-free endpoint state | Memory is more important than repeated queries |
| `full` | Yes | Full Gram prefix with a memory limit | Constant-time repeated segment queries or reference tie behaviour |

The full backend exposes `stored_gram_entries` and enforces `max_gram_bytes`. RBF
gamma can be given directly, computed exactly, or estimated reproducibly from a
sample according to `gamma_policy`.

!!! note "Tie behaviour"

    The fused backend keeps the first predecessor encountered when objectives tie.
    Use `backend="full"` for audits that require the generic solver's global
    lexicographic tie policy.

## Binseg

Binary segmentation starts with the full signal, finds the best single split, then
recursively splits selected segments.

```python
detector = rpt.Binseg(model="l2", min_size=8, jump=2).fit(signal)

by_count = detector.predict(n_bkps=3)
by_penalty = detector.predict(pen=10.0)
by_budget = detector.predict(epsilon=100.0)
```

It is usually cheaper than exhaustive dynamic programming, but greedy early splits
cannot always be corrected later. Use it for exploration or large problems where an
exact global optimum is less important.

## BottomUp

Bottom-up segmentation starts from small neighbouring blocks and repeatedly merges
the adjacent pair with the smallest increase in cost.

```python
breakpoints = rpt.BottomUp(
    model="l1",
    min_size=8,
    jump=2,
).fit_predict(signal, n_bkps=3)
```

This gives a different approximation path from Binseg: Binseg adds splits, while
BottomUp removes boundaries.

## Window

Window segmentation compares the fit on a combined neighbourhood with the fits on
its left and right halves. Local maxima of that discrepancy become breakpoint
candidates.

```python
breakpoints = rpt.Window(
    width=80,
    model="l2",
    min_size=8,
    jump=2,
).fit_predict(signal, n_bkps=3)
```

`width` determines the neighbourhood scale. It should be wide enough to contain
enough observations for the cost model but smaller than the changes you want to
resolve.

## L1Potts

`L1Potts` is a specialized exact penalized solver for a scalar signal. It balances
absolute deviation from a piecewise-constant level against a penalty for jumps.

```python
breakpoints = rpt.L1Potts().fit_predict(
    signal,
    pen=4.0,
    weights=None,
)
```

Optional non-negative weights control the influence of observations. If `D` is the
number of distinct observed levels, the implementation uses `O(DN)` time, two score
rows, and an `N × D` byte parent matrix.

Use L1Potts when:

- the signal is one-dimensional;
- robustness to spikes matters;
- a piecewise-constant median model is appropriate.

It is not a generic Pelt cost. Pelt searches arbitrary additive segment partitions;
L1Potts exploits the special structure of the scalar L1-Potts problem.

## `min_size`, `jump`, and exactness

Every returned segment must contain at least `min_size` samples. Internal breakpoint
candidates normally occur at multiples of `jump`; the final sample is always added.

```text
n=17, jump=5 → candidates include 0, 5, 10, 15, 17
```

An “exact” result means exact over this candidate set. Choose `jump=1` when every
sample position must be considered.

