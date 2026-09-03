# Cost functions

A cost function answers one question:

> If samples `signal[start:end]` were treated as one homogeneous segment, how much
> information would be lost?

Search algorithms minimize the sum of these segment costs. A breakpoint is useful
when fitting the left and right parts separately reduces the total cost enough to
justify the additional segment or penalty.

## Why lower cost is better

Suppose a scalar signal changes from values near 0 to values near 10. One L2 segment
must use a mean near 5 and receives a large squared-error cost. Splitting at the
change allows separate means near 0 and 10, making the sum of segment costs much
smaller.

```text
one segment:  C(0, 8)
two segments: C(0, 4) + C(4, 8)
```

Fixed-`K` algorithms choose the lowest-cost partition with the requested number of
changes. Penalized algorithms additionally charge for each internal breakpoint so
that splitting every possible position is not automatically optimal.

## Built-in models

| Model string | Public cost class | What changes it detects | Minimum model idea |
|---|---|---|---|
| `l2` | `CostL2` | Mean | Squared deviation from the segment mean |
| `l1` | `CostL1` | Robust location | Absolute deviation from component-wise medians |
| `rank` | `CostRank` | Distribution | Scatter of globally ranked observations |
| `normal` | `CostNormal` | Gaussian mean/covariance | Regularized covariance log-determinant |
| `linear` | `CostLinear` | Regression relationship | Residual sum of squares; first column is response |
| `ar` | `CostAR` | Autoregressive dynamics | Segment-local lagged regression |
| `clinear` | `CostCLinear` | Continuous linear trend | Deviation from endpoint interpolation |
| `mahalanobis` | `CostMahalanobis` / `CostMl` | Metric-weighted location | Scatter under a supplied PSD metric |

## Evaluate a cost directly

All public cost objects follow the same basic flow.

```python
import numpy as np
import rustures as rpt

signal = np.array([0.0, 0.2, -0.1, 5.0, 5.1, 4.9])
cost = rpt.CostL2().fit(signal)

whole = cost.error(0, 6)
split = cost.error(0, 3) + cost.error(3, 6)
same_split = cost.sum_of_costs([3, 6])

assert np.isclose(split, same_split)
```

`error(start, end)` uses the half-open range `[start, end)`. `sum_of_costs` accepts
the same terminal-inclusive breakpoint convention as detector outputs.

## L2: mean shifts

For observations $x_i \in \mathbb{R}^d$, L2 uses:

$$
C_{L2}(s,e)=\sum_{i=s}^{e-1}\lVert x_i-\bar{x}_{s:e}\rVert_2^2.
$$

Rustures fits per-feature prefix sums and squared prefix sums, making an interval
query `O(d)` rather than rescanning every sample. Small negative values caused only
by floating-point cancellation are clamped according to a narrow numerical policy;
larger failures become exceptions.

Use L2 for changes in the mean when large deviations should receive quadratic
weight.

## L1: robust location shifts

L1 fits a component-wise median within each segment and sums absolute deviations:

$$
C_{L1}(s,e)=\sum_{j=1}^{d}\sum_{i=s}^{e-1}
\lvert x_{ij}-\operatorname{median}(x_{s:e,j})\rvert.
$$

It is less sensitive to isolated spikes than L2. Multivariate features are handled
component by component and their costs are added.

## Rank: distributional changes

The rank cost transforms observations into global ranks with explicit tie handling,
then scores segment scatter in rank space. It can respond to distributional changes
that are not well represented by a raw mean.

Use it for ordinal data, heavy tails, or cases where relative ordering is more
meaningful than magnitude.

## Normal: Gaussian mean and covariance

`CostNormal` evaluates a regularized covariance log-determinant objective. It can
detect variance and correlation changes as well as mean changes.

```python
cost = rpt.CostNormal(ridge=1e-6).fit(signal_2d)
value = cost.error(20, 100)
```

The ridge keeps constant or nearly singular covariance matrices numerically usable.
Changing it changes the statistical objective, so record non-default values in
experiments.

## Linear: regression changes

`CostLinear` interprets the first column as the response and all remaining columns
as predictors.

```python
# columns: response, intercept, predictor
design_signal = np.column_stack((y, np.ones(len(y)), x))
cost = rpt.CostLinear().fit(design_signal)
```

Each segment fits its own least-squares relationship. Changes indicate that one set
of regression coefficients no longer explains the entire interval. Rustures uses a
fast endpoint batch path and falls back to a stable SVD calculation for rank-deficient
or poorly conditioned segments.

## AR: autoregressive changes

`CostAR(order=4)` fits segment-local lagged regressions. A breakpoint indicates a
change in how current observations depend on recent observations.

```python
cost = rpt.CostAR(order=6).fit(signal)
```

AR costs need enough samples to construct and fit the lagged design. Rustures has a
documented boundary convention that can differ from ruptures, so validate parity if
you migrate an established pipeline.

## CLinear: continuous trend changes

`CostCLinear` scores deviation from the straight line interpolating a segment's
endpoints. It is intended for piecewise-linear trends whose fitted pieces remain
connected at change points.

It is different from `CostLinear`: CLinear models the signal as a continuous trend,
while Linear treats columns as a response and regression design.

## Mahalanobis: metric-weighted scatter

```python
metric = np.array([[2.0, 0.3], [0.3, 1.0]])
cost = rpt.CostMahalanobis(metric).fit(signal_2d)
```

The metric must be symmetric and positive semidefinite. It determines which feature
directions receive more weight. `CostMl` is a compatibility alias for
`CostMahalanobis`.

## Multivariate interpretation

For shape `(n_samples, n_features)`, segmentation boundaries are shared by all
features. Rustures does not detect an independent breakpoint list for each column.
Instead, the selected cost combines feature evidence into one score for each
candidate segment.

Feature scaling therefore matters. A high-variance column can dominate L1 or L2;
standardize features or choose a metric when equal contribution is desired.

## Numerical and shape rules

- Inputs must be finite one- or two-dimensional `float64` NumPy-compatible arrays.
- Empty signals and higher-rank arrays are rejected.
- Invalid or too-short ranges raise Python exceptions.
- Fitted Rust costs own the statistics they need; they do not keep an unsafe borrowed
  view into the caller's NumPy array.
- Cost and detector `min_size` requirements are combined conservatively.

If no built-in model describes the change you care about, continue with
[Custom Python costs](custom-costs.md).

