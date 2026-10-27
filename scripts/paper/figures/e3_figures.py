"""Classification-margin figures.

Reads the run artifacts of the margin-ablation cell and writes two figures.
The first plots, for every swept angular margin, the per-epoch mean angle to
the correct class anchor, the mean smallest angle to a wrong anchor and their
difference, averaged over seeds. The second is the heatmap of the final-epoch
inter-class angle matrix at the default margin, averaged over seeds.

Artifacts lacking the geometry fields, such as baseline arms, are skipped.
Matrix rows index the true class and columns the class anchor; angles are in
radians.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_MARGIN = math.pi / 5.0
HISTORY_FIELDS = ("mean_angle_correct", "mean_angle_wrong_min", "cmr_margin")


def load_runs(runs: Path) -> list[dict]:
    artifacts = []
    for path in sorted(runs.rglob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                artifact = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(artifact, dict):
            artifacts.append(artifact)
    return artifacts


def has_geometry(artifact: dict) -> bool:
    history = artifact.get("history")
    config = artifact.get("config")
    if not isinstance(history, dict) or not isinstance(config, dict):
        return False
    if "cmr_margin" not in config:
        return False
    return all(history.get(field) for field in HISTORY_FIELDS)


def by_margin(artifacts: list[dict]) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = {}
    for artifact in artifacts:
        grouped.setdefault(float(artifact["config"]["cmr_margin"]), []).append(artifact)
    return dict(sorted(grouped.items()))


def mean_curve(group: list[dict], field: str) -> tuple[np.ndarray, np.ndarray]:
    curves = [np.asarray(a["history"][field], dtype=float) for a in group]
    length = min(curve.size for curve in curves)
    stacked = np.vstack([curve[:length] for curve in curves])
    epochs = np.arange(1, length + 1, dtype=float)
    return epochs, stacked.mean(axis=0)


def plot_evolution(grouped: dict[float, list[dict]], out: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    panels = (
        ("mean_angle_correct", "angle to correct anchor (rad)", "Correct class"),
        ("mean_angle_wrong_min", "angle to nearest wrong anchor (rad)", "Nearest wrong class"),
        ("cmr_margin", "realized margin (rad)", "Margin"),
    )
    for margin, group in grouped.items():
        label = f"margin = {margin:.4f} rad"
        for axis, (field, _, _) in zip(axes, panels):
            epochs, curve = mean_curve(group, field)
            axis.plot(epochs, curve, label=label)
    for axis, (_, ylabel, title) in zip(axes, panels):
        axis.set(xlabel="epoch", ylabel=ylabel, title=title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out / "e3_margin_evolution.png", dpi=200)
    figure.savefig(out / "e3_margin_evolution.pdf")
    plt.close(figure)


def plot_angle_matrix(grouped: dict[float, list[dict]], out: Path) -> bool:
    group = next(
        (g for m, g in grouped.items() if math.isclose(m, DEFAULT_MARGIN, rel_tol=1e-9)),
        [],
    )
    matrices = [
        np.asarray(a["class_angle_matrix_final"], dtype=float)
        for a in group
        if a.get("class_angle_matrix_final")
    ]
    if not matrices:
        return False
    mean_matrix = np.mean(np.stack(matrices), axis=0)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(mean_matrix, cmap="viridis")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("mean angle (rad)")
    classes = np.arange(mean_matrix.shape[0])
    axis.set_xticks(classes)
    axis.set_yticks(classes)
    axis.set(
        xlabel="class anchor",
        ylabel="true class",
        title=f"Inter-class angles, margin = {DEFAULT_MARGIN:.4f} rad "
        f"({len(matrices)} seeds)",
    )
    figure.tight_layout()
    figure.savefig(out / "e3_angle_matrix.png", dpi=200)
    figure.savefig(out / "e3_angle_matrix.pdf")
    plt.close(figure)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    artifacts = [a for a in load_runs(args.runs) if has_geometry(a)]
    if not artifacts:
        raise SystemExit(f"no margin run artifacts under {args.runs}")
    grouped = by_margin(artifacts)

    plot_evolution(grouped, args.out)
    if not plot_angle_matrix(grouped, args.out):
        raise SystemExit("no class_angle_matrix_final at the default margin")


if __name__ == "__main__":
    main()
