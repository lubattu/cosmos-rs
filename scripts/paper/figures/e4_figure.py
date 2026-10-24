"""Topology and depth panels.

Reads the run artifacts of the topology cell and draws, for every topology,
evaluation accuracy and mean drift against interior depth, together with the
accuracy and commutator trajectories of the deepest arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def block_mean(flat: list[float], epochs: int) -> np.ndarray:
    values = np.asarray(flat, dtype=float)
    if values.size == 0 or epochs == 0:
        return np.zeros(epochs)
    return values.reshape(epochs, -1).mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    artifacts = []
    for path in sorted(args.runs.rglob("run_r*_seed_*.json")):
        with path.open(encoding="utf-8") as handle:
            artifacts.append(json.load(handle))
    if not artifacts:
        raise SystemExit(f"no run artifacts under {args.runs}")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    topologies = sorted({a["config"]["topology"] for a in artifacts})

    for topology in topologies:
        rows = [a for a in artifacts if a["config"]["topology"] == topology]
        depths = sorted({a["config"]["num_interior_blocks"] for a in rows})
        accuracy, drift = [], []
        for depth in depths:
            group = [a for a in rows if a["config"]["num_interior_blocks"] == depth]
            accuracy.append(np.mean([a["result"]["best_val_acc"] for a in group]))
            drift.append(
                np.mean([
                    block_mean(
                        a["history"]["per_block"]["drift_mean"],
                        len(a["history"]["epoch"]),
                    ).mean()
                    for a in group
                ])
            )
        axes[0][0].plot(depths, accuracy, marker="o", label=topology)
        axes[0][1].plot(depths, drift, marker="o", label=topology)

        deepest = max(rows, key=lambda a: a["config"]["num_interior_blocks"])
        epochs = deepest["history"]["epoch"]
        axes[1][0].plot(epochs, deepest["history"]["val_acc"], label=topology)
        axes[1][1].semilogy(
            epochs,
            np.maximum(np.asarray(deepest["history"]["mean_commutator"]), 1e-18),
            label=topology,
        )

    axes[0][0].set(xlabel="interior blocks", ylabel="best evaluation accuracy (%)",
                   title="Depth response")
    axes[0][1].set(xlabel="interior blocks", ylabel="mean drift", title="Drift")
    axes[1][0].set(xlabel="epoch", ylabel="evaluation accuracy (%)",
                   title="Deepest arm, accuracy")
    axes[1][1].set(xlabel="epoch", ylabel="commutator norm",
                   title="Deepest arm, compatibility")
    for row in axes:
        for axis in row:
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(args.out / "e4.png", dpi=200)
    figure.savefig(args.out / "e4.pdf")


if __name__ == "__main__":
    main()
