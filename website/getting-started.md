# Getting started

This guide installs a published wheel, runs a first segmentation, and explains the
three parameters that most strongly affect results: the objective, `min_size`, and
`jump`.

## Install

Rustures requires Python 3.10 or newer and NumPy 1.23 or newer.

=== "pip"

    ```bash
    python -m pip install rustures
    ```

=== "virtual environment on Windows"

    ```powershell
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install rustures
    ```

=== "virtual environment on Linux/macOS"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install rustures
    ```

Verify the installation:

```python
import rustures

print(rustures.__version__)
```

See [Limitations and compatibility](limitations.md#published-wheel-targets) for
the currently published wheel targets.

## Detect changes with a penalty

`Pelt` is a useful starting point when you do not know the number of changes but
can choose a penalty for adding them.

```python
import rustures as rpt

signal, expected = rpt.pw_constant(
    n_samples=900,
    n_features=1,
    n_bkps=3,
    noise_std=0.6,
    seed=7,
)

detector = rpt.Pelt(model="l2", min_size=20, jump=1)
predicted = detector.fit_predict(signal, pen=10.0)

print("expected: ", expected)
print("predicted:", predicted)
```

A larger penalty normally produces fewer changes; a smaller penalty normally
produces more. The numeric scale depends on the cost function, feature count,
signal scale, and sample count, so there is no universal penalty value.

## Detect a fixed number of changes

If you already know the desired number of changes, use `Dynp`.

```python
detector = rpt.Dynp(
    model="l2",
    min_size=20,
    jump=1,
).fit(signal)

print("estimated DP workspace:", detector.estimated_memory_bytes(n_bkps=3))
predicted = detector.predict(n_bkps=3)
```

`n_bkps=3` means three internal change points and therefore four returned segments.
The returned list still contains the terminal sample index.

## Breakpoint convention

For a signal with 12 samples:

```python
breakpoints = [4, 9, 12]
```

means:

```text
segment 1: signal[0:4]
segment 2: signal[4:9]
segment 3: signal[9:12]
```

Rustures always returns the terminal value `len(signal)`. This convention makes it
possible to iterate over segments without a special final case.

## Multivariate signals

Most detectors and costs accept shape `(n_samples, n_features)`.

```python
signal, expected = rpt.pw_normal(
    n_samples=600,
    n_features=3,
    n_bkps=2,
    noise_std=0.8,
    seed=11,
)

predicted = rpt.Dynp(
    model="normal",
    min_size=10,
    jump=2,
).fit_predict(signal, n_bkps=2)
```

For additive multivariate costs, each feature contributes to the segment score.
The exact interpretation depends on the model: L2 adds squared deviations,
L1 uses component-wise medians, and the normal model evaluates a regularized
covariance objective.

## Understand `min_size`

`min_size` is the shortest allowed segment length.

```python
rpt.Pelt(model="l2", min_size=30, jump=1)
```

This forbids two consecutive breakpoints from being fewer than 30 samples apart.
Use it to express domain knowledge and to prevent tiny segments that cannot support
the selected statistical model. The detector raises an exception when the requested
segmentation is infeasible.

Some costs impose their own minimum. Rustures uses the larger of the detector and
cost requirements.

## Understand `jump`

`jump` controls the internal breakpoint grid.

```text
jump=1 → every sample position is eligible
jump=5 → normally positions 5, 10, 15, ... are eligible
```

The terminal sample is always eligible even when it is not divisible by `jump`.
A larger jump can make search much faster and smaller, but it may move or miss a
change that lies between grid positions.

!!! note "What exact means"

    Dynp and unpruned Pelt return an exact optimum over the selected candidate grid.
    With `jump > 1`, this is not necessarily the optimum over every sample position.

## Select a cost model

```python
rpt.Pelt(model="l2")       # mean shifts
rpt.Pelt(model="l1")       # robust location shifts
rpt.Dynp(model="normal")   # Gaussian mean/covariance shifts
rpt.Dynp(model="rank")     # distributional changes
```

The choice defines what “one homogeneous segment” means. See
[Cost functions](costs.md) before interpreting a detected breakpoint.

## Evaluate known breakpoints

For synthetic data or labelled datasets, Rustures includes common metrics.

```python
hausdorff = rpt.hausdorff(expected, predicted)
precision, recall = rpt.precision_recall(expected, predicted, margin=10)
agreement = rpt.rand_index(expected, predicted)
```

- Hausdorff distance measures the worst breakpoint-location discrepancy.
- Precision and recall match change points within the selected margin.
- Rand index measures agreement between the two induced partitions.

## Handle errors in Python

```python
import numpy as np
import rustures as rpt

try:
    rpt.Pelt().fit_predict(np.array([0.0, np.nan, 1.0]), pen=2.0)
except ValueError as error:
    print("invalid signal:", error)
except rpt.RusturesError as error:
    print("native numerical or panic boundary error:", error)
```

Memory preflight failures use `MemoryError`, invalid parameters normally use
`ValueError`, and prediction before fitting uses `RuntimeError`.

## Where to go next

- [Algorithm guide](algorithms.md)
- [Cost-function guide](costs.md)
- [Custom Python costs](custom-costs.md)
- [Complete API signatures](api-reference.md)

