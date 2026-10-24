"""Gauge regime figures.

Reads e2_metrics.csv and draws the two published panels: the accuracy
surface against gauge learning rate for every momentum, and the stability
map of commutator decay rate against mean step size, marked by whether the
orthogonality error stayed within the numerical bound.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORTHOGONALITY_BOUND = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.metrics)

    figure, axis = plt.subplots(figsize=(7, 5))
    for momentum, group in frame.groupby("momentum"):
        aggregated = group.groupby("lr_gauge")["best_val_acc"].agg(["mean", "std"])
        axis.errorbar(
            aggregated.index,
            aggregated["mean"],
            yerr=aggregated["std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=f"momentum {momentum:g}",
        )
    axis.set_xscale("log")
    axis.set(xlabel="gauge learning rate", ylabel="best evaluation accuracy (%)",
             title="Accuracy across the gauge regime")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.out / "e2_accuracy.png", dpi=200)

    figure, axis = plt.subplots(figsize=(7, 5))
    within = frame["max_orth"] <= ORTHOGONALITY_BOUND
    axis.scatter(
        frame.loc[within, "step_mean"],
        frame.loc[within, "comm_decay_rate"],
        marker="o",
        label="invariant preserved",
    )
    axis.scatter(
        frame.loc[~within, "step_mean"],
        frame.loc[~within, "comm_decay_rate"],
        marker="x",
        label="invariant exceeded",
    )
    axis.axhline(0.0, linewidth=0.8)
    axis.set_xscale("log")
    axis.set(xlabel="mean gauge step size", ylabel="commutator decay rate",
             title="Stability map")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.out / "e2_stability.png", dpi=200)
    np.seterr(all="ignore")


if __name__ == "__main__":
    main()
