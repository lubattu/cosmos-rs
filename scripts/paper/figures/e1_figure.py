"""Invariant-preservation panels.

Reads the run artifacts of the invariant-preservation cell and the ambient
drift curve produced by the benchmark command, and writes a four-panel
figure: orthogonality error, commutator norm and evaluation accuracy against
epoch for every retraction interval, plus the drift contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_runs(runs: Path) -> list[dict]:
    artifacts = []
    for path in sorted(runs.rglob("run_r*_seed_*.json")):
        with path.open(encoding="utf-8") as handle:
            artifacts.append(json.load(handle))
    return artifacts


def by_interval(artifacts: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact["config"]["retraction_interval"], []).append(artifact)
    return dict(sorted(grouped.items()))


def mean_curve(group: list[dict], selector) -> tuple[np.ndarray, np.ndarray]:
    curves = [np.asarray(selector(artifact), dtype=float) for artifact in group]
    length = min(curve.size for curve in curves)
    stacked = np.vstack([curve[:length] for curve in curves])
    epochs = np.asarray(group[0]["history"]["epoch"][:length], dtype=float)
    return epochs, stacked.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drift", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grouped = by_interval(load_runs(args.runs))
    if not grouped:
        raise SystemExit(f"no run artifacts under {args.runs}")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for interval, group in grouped.items():
        label = f"r = {interval}"
        epochs, orth = mean_curve(
            group, lambda a: a["history"]["per_block"]["total_orth_error_mean"]
        )
        axes[0][0].semilogy(epochs, np.maximum(orth, 1e-18), label=label)
        epochs, comm = mean_curve(group, lambda a: a["history"]["mean_commutator"])
        axes[0][1].semilogy(epochs, np.maximum(comm, 1e-18), label=label)
        epochs, accuracy = mean_curve(group, lambda a: a["history"]["val_acc"])
        axes[1][0].plot(epochs, accuracy, label=label)

    axes[0][0].set(xlabel="epoch", ylabel="orthogonality error", title="Invariant")
    axes[0][1].set(xlabel="epoch", ylabel="commutator norm", title="Compatibility")
    axes[1][0].set(xlabel="epoch", ylabel="evaluation accuracy (%)", title="Accuracy")
    for axis in (axes[0][0], axes[0][1], axes[1][0]):
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)

    drift_path = args.drift or (args.runs.parent / "bench" / "ambient_drift.json")
    if drift_path.exists():
        with drift_path.open(encoding="utf-8") as handle:
            drift = json.load(handle)
        axes[1][1].plot(drift["step"], drift["ambient_distance"], label="dense generator")
        axes[1][1].plot(drift["step"], drift["block_distance"], label="block generator")
        axes[1][1].legend(fontsize=8)
    else:
        axes[1][1].text(0.5, 0.5, "drift curve not found", ha="center", va="center")
    axes[1][1].set(xlabel="step", ylabel="distance from start", title="Drift contrast")
    axes[1][1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(args.out / "e1.png", dpi=200)
    figure.savefig(args.out / "e1.pdf")


if __name__ == "__main__":
    main()
