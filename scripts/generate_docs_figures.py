"""Generate deterministic SVG examples for the public Rustures documentation.

Run this script with a release build of the wheel installed. The generated SVGs
and JSON metadata are committed so the documentation site does not need a native
Rust build or Matplotlib during deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rustures as rpt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "website" / "assets" / "examples"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRUE_COLOR = "#263238"
PREDICTED_COLOR = "#e53935"
SIGNAL_COLOR = "#4051b5"
FEATURE_COLORS = ("#4051b5", "#00a6c8", "#7e57c2")

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "svg.hashsalt": "rustures-docs-v1",
    }
)


def internal(breakpoints: list[int]) -> list[int]:
    """Drop the mandatory terminal sample from graph annotations."""

    return breakpoints[:-1]


def add_breakpoints(
    axis: plt.Axes,
    truth: list[int],
    predicted: list[int],
    *,
    add_labels: bool,
) -> None:
    for index, point in enumerate(internal(truth)):
        axis.axvline(
            point,
            color=TRUE_COLOR,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label="True change" if add_labels and index == 0 else None,
        )
    for index, point in enumerate(internal(predicted)):
        axis.axvline(
            point,
            color=PREDICTED_COLOR,
            linewidth=1.4,
            alpha=0.9,
            label="Predicted change" if add_labels and index == 0 else None,
        )


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(
        OUTPUT / name,
        format="svg",
        metadata={"Date": None, "Creator": "rustures documentation generator"},
    )
    plt.close(figure)


def detector_comparison() -> dict[str, object]:
    rng = np.random.default_rng(20260903)
    truth = [120, 250, 360, 480]
    levels = np.array([0.0, 4.0, -2.0, 3.0])
    segment_ids = np.searchsorted(truth, np.arange(480), side="right")
    signal = levels[segment_ids] + rng.normal(0.0, 0.65, 480)

    operations: list[tuple[str, Callable[[], list[int]], str]] = [
        (
            "Dynp",
            lambda: rpt.Dynp(model="l2", min_size=20, jump=2).fit_predict(
                signal, n_bkps=3
            ),
            "n_bkps=3",
        ),
        (
            "Pelt",
            lambda: rpt.Pelt(model="l2", min_size=20, jump=2).fit_predict(
                signal, pen=18.0
            ),
            "pen=18",
        ),
        (
            "Binseg",
            lambda: rpt.Binseg(model="l2", min_size=20, jump=2).fit_predict(
                signal, n_bkps=3
            ),
            "n_bkps=3",
        ),
        (
            "BottomUp",
            lambda: rpt.BottomUp(model="l2", min_size=20, jump=2).fit_predict(
                signal, n_bkps=3
            ),
            "n_bkps=3",
        ),
        (
            "Window",
            lambda: rpt.Window(
                width=70, model="l2", min_size=20, jump=2
            ).fit_predict(signal, n_bkps=3),
            "width=70, n_bkps=3",
        ),
    ]
    results = {name: operation() for name, operation, _ in operations}

    figure, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    for index, (name, _, parameters) in enumerate(operations):
        axis = flat_axes[index]
        axis.plot(signal, color=SIGNAL_COLOR, linewidth=0.9, alpha=0.82)
        add_breakpoints(axis, truth, results[name], add_labels=index == 0)
        axis.set_title(f"{name} | {parameters}\n{results[name]}", loc="left", fontsize=10)
        axis.set_xlim(0, len(signal) - 1)
        axis.grid(alpha=0.18)
    flat_axes[-1].axis("off")
    flat_axes[0].legend(loc="upper right", frameon=False, fontsize=8)
    for axis in flat_axes[:5]:
        axis.set_ylabel("value")
    for axis in flat_axes[4:]:
        axis.set_xlabel("sample")
    figure.suptitle(
        "Five search strategies on the same piecewise-mean signal",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save(figure, "detector-comparison.svg")

    return {
        "truth": truth,
        "parameters": {name: parameters for name, _, parameters in operations},
        "predictions": results,
    }


def kernel_comparison() -> dict[str, object]:
    rng = np.random.default_rng(20260904)
    truth = [140, 280, 420]
    centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.2]])
    segment_ids = np.searchsorted(truth, np.arange(420), side="right")
    signal = centers[segment_ids] + rng.normal(0.0, 0.45, size=(420, 2))

    kernels = ("linear", "rbf", "cosine")
    results = {
        kernel: rpt.KernelCPD(
            kernel=kernel,
            gamma=0.35 if kernel == "rbf" else None,
            backend="fused",
            min_size=20,
            jump=2,
        ).fit_predict(signal, n_bkps=2)
        for kernel in kernels
    }

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    for index, (axis, kernel) in enumerate(zip(axes, kernels, strict=True)):
        for feature, color in enumerate(FEATURE_COLORS[:2]):
            axis.plot(
                signal[:, feature],
                color=color,
                linewidth=0.9,
                alpha=0.76,
                label=f"feature {feature + 1}" if index == 0 else None,
            )
        add_breakpoints(axis, truth, results[kernel], add_labels=index == 0)
        parameter = "gamma=0.35" if kernel == "rbf" else "no kernel parameter"
        axis.set_title(
            f"{kernel.upper()} kernel | {parameter} | prediction {results[kernel]}",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("value")
        axis.set_xlim(0, len(signal) - 1)
        axis.grid(alpha=0.18)
    axes[0].legend(loc="upper right", frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("sample")
    figure.suptitle(
        "KernelCPD on a two-feature directional shift",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save(figure, "kernel-comparison.svg")

    return {
        "truth": truth,
        "parameters": {
            "backend": "fused",
            "n_bkps": 2,
            "min_size": 20,
            "jump": 2,
            "rbf_gamma": 0.35,
        },
        "predictions": results,
    }


def robust_comparison() -> dict[str, object]:
    rng = np.random.default_rng(20260905)
    truth = [120, 240, 360]
    levels = np.array([0.0, 5.0, -1.0])
    segment_ids = np.searchsorted(truth, np.arange(360), side="right")
    signal = levels[segment_ids] + rng.normal(0.0, 0.32, 360)
    outlier_indices = np.array([28, 57, 91, 151, 178, 211, 276, 307, 338])
    signal[outlier_indices] += np.array([15, -14, 17, -16, 14, -15, 18, -17, 15])

    operations: list[tuple[str, Callable[[], list[int]], str]] = [
        (
            "Dynp with L2",
            lambda: rpt.Dynp(model="l2", min_size=20, jump=1).fit_predict(
                signal, n_bkps=2
            ),
            "quadratic loss, n_bkps=2",
        ),
        (
            "Dynp with L1",
            lambda: rpt.Dynp(model="l1", min_size=20, jump=1).fit_predict(
                signal, n_bkps=2
            ),
            "absolute loss, n_bkps=2",
        ),
        (
            "L1Potts",
            lambda: rpt.L1Potts().fit_predict(signal, pen=20.0),
            "specialized scalar solver, pen=20",
        ),
    ]
    results = {name: operation() for name, operation, _ in operations}

    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    for index, (axis, (name, _, description)) in enumerate(
        zip(axes, operations, strict=True)
    ):
        axis.scatter(
            np.arange(len(signal)),
            signal,
            color=SIGNAL_COLOR,
            s=7,
            alpha=0.65,
            linewidths=0,
            label="observation" if index == 0 else None,
        )
        axis.scatter(
            outlier_indices,
            signal[outlier_indices],
            color="#fb8c00",
            s=18,
            alpha=0.9,
            label="injected outlier" if index == 0 else None,
        )
        add_breakpoints(axis, truth, results[name], add_labels=index == 0)
        axis.set_title(
            f"{name} | {description}\n{results[name]}",
            loc="left",
            fontsize=10,
        )
        axis.set_ylabel("value")
        axis.set_xlim(0, len(signal) - 1)
        axis.grid(alpha=0.18)
    axes[0].legend(loc="upper right", frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("sample")
    figure.suptitle(
        "Robust segmentation with injected outliers",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save(figure, "robust-comparison.svg")

    return {
        "truth": truth,
        "outlier_indices": outlier_indices.tolist(),
        "parameters": {name: description for name, _, description in operations},
        "predictions": results,
    }


def main() -> None:
    results = {
        "rustures_version": rpt.__version__,
        "detector_comparison": detector_comparison(),
        "kernel_comparison": kernel_comparison(),
        "robust_comparison": robust_comparison(),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
