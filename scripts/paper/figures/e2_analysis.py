"""Gauge regime metrics.

Derives, for every grid point, the mean evaluation accuracy over the run,
the decay rate of the commutator norm, the mean gauge step size, the worst
orthogonality error and the best evaluation accuracy, writes them to
e2_metrics.csv and draws the accuracy heat map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def block_mean(flat: list[float], epochs: int) -> np.ndarray:
    values = np.asarray(flat, dtype=float)
    if values.size == 0 or epochs == 0:
        return np.zeros(epochs)
    return values.reshape(epochs, -1).mean(axis=1)


def val_auc(accuracy: np.ndarray, epoch: np.ndarray) -> float:
    """val_auc is the trapezoidal area under val_acc(epoch), normalized by the epoch span."""
    if epoch.size < 2 or epoch[-1] == epoch[0]:
        return float(accuracy[0])
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(accuracy, epoch) / (epoch[-1] - epoch[0]))


def metrics(artifact: dict) -> dict:
    history = artifact["history"]
    config = artifact["config"]
    epochs = len(history["epoch"])
    epoch = np.asarray(history["epoch"], dtype=float)
    accuracy = np.asarray(history["val_acc"], dtype=float)
    commutator = np.asarray(history["mean_commutator"], dtype=float)
    positive = commutator > 0
    if positive.sum() >= 2:
        slope = np.polyfit(
            np.arange(epochs)[positive], np.log(commutator[positive]), 1
        )[0]
    else:
        slope = float("nan")
    connection = block_mean(history["per_block"]["connection_norms_mean"], epochs)
    return {
        "lr_gauge": config["lr_gauge"],
        "momentum": config["momentum"],
        "momentum_max_norm": config["momentum_max_norm"],
        "seed": config["seed"],
        "val_auc": val_auc(accuracy, epoch),
        "comm_decay_rate": float(-slope),
        "step_mean": float(config["lr_gauge"] * connection.mean()),
        "max_orth": float(max(history["per_block"]["total_orth_error_mean"])),
        "best_val_acc": float(artifact["result"]["best_val_acc"]),
        "drift_mean": float(block_mean(history["per_block"]["drift_mean"], epochs).mean()),
        "deadband_fraction": float(
            np.mean(history["evolution_scalars"]["deadband_fraction"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(args.runs.rglob("run_r*_seed_*.json")):
        with path.open(encoding="utf-8") as handle:
            rows.append(metrics(json.load(handle)))
    if not rows:
        raise SystemExit(f"no run artifacts under {args.runs}")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "e2_metrics.csv", index=False)

    pivot = frame.pivot_table(
        index="momentum", columns="lr_gauge", values="best_val_acc", aggfunc="mean"
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
    axis.set_xticks(range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [f"{value:g}" for value in pivot.index])
    axis.set(xlabel="gauge learning rate", ylabel="momentum",
             title="Best evaluation accuracy (%)")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(args.out / "e2_heatmap.png", dpi=200)


if __name__ == "__main__":
    main()
