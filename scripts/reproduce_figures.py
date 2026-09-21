"""Regenerate ICASSP Fig. 3--5 at their final single-column sizes.

Plotting only: all values, intervals, stars, method order, and axis ranges come
from the frozen result files. The PDFs are saved at their exact final physical
dimensions without tight-bounding-box resizing.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


from src.utils import load_config

METHODS = ("Raw-GAT", "GlobalMean", "NodeMean", "ASGC")
DISPLAY_NAME = {"GAT+Refinement": "ASGC"}
COLORS = {"Raw-GAT": "#0072B2", "GlobalMean": "#009E73", "NodeMean": "#E69F00", "ASGC": "#D55E00"}
MARKERS = {"Raw-GAT": "o", "GlobalMean": "s", "NodeMean": "^", "ASGC": "D"}
LINES = {"Raw-GAT": "-", "GlobalMean": "--", "NodeMean": "-.", "ASGC": ":"}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8.8,
    "axes.titlesize": 8.8,
    "axes.labelsize": 8.8,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 8.4,
    "figure.titlesize": 8.8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_axes(ax: plt.Axes, horizontal_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(width=0.7, length=2.5, pad=1.5)
    if horizontal_grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.72)
        ax.set_axisbelow(True)


def save_exact(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(
        output / f"{stem}.pdf",
        format="pdf",
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        metadata={"Creator": "Matplotlib", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        output / f"{stem}.png",
        dpi=600,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
    )
    plt.close(fig)


def draw_fig3(source_dir: Path, output: Path) -> None:
    bound_data = source_dir / "fig3_bound_data.csv"
    source = read_csv(bound_data)
    rows = []
    for row in source:
        method = DISPLAY_NAME.get(row["method"], row["method"])
        ec = float(row["E_c"])
        denominator = max(float(row["S_hat"]), float(row["S_star"]))
        bw = 2.0 * ec / denominator
        bx = float(row["D_X"]) * ec / denominator
        if abs(bw - float(row["B_w"])) > 1e-12 or abs(bx - float(row["B_x"])) > 1e-12:
            raise AssertionError("Frozen bound values do not reproduce")
        rows.append({"method": method, "B_w": bw, "E_w": float(row["E_w"]), "B_x": bx, "E_x": float(row["E_x"])})
    if len(rows) != 80 or any(sum(row["method"] == method for row in rows) != 20 for method in METHODS):
        raise AssertionError("Fig. 3 must contain 20 graphs x 4 methods")
    if max(row["E_w"] - row["B_w"] for row in rows) > 1e-10 or max(row["E_x"] - row["B_x"] for row in rows) > 1e-10:
        raise AssertionError("A deterministic bound is violated")

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.45))
    handles = []
    panels = (("B_w", "E_w", r"$B_w$", r"$E_w$", "(a) Weight bound"), ("B_x", "E_x", r"$B_x$", r"$E_x$", "(b) Signal bound"))
    limits = []
    for panel_index, (bound, error, xlabel, ylabel, title) in enumerate(panels):
        ax = axes[panel_index]
        limit = 1.05 * max(max(row[bound], row[error]) for row in rows)
        limits.append([0.0, float(limit)])
        ax.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=0.9, zorder=1)
        for method in METHODS:
            selected = [row for row in rows if row["method"] == method]
            artist = ax.scatter([row[bound] for row in selected], [row[error] for row in selected], s=14.44, marker=MARKERS[method], c=COLORS[method], alpha=0.85, edgecolors="#222222", linewidths=0.35, zorder=2, label=method)
            if panel_index == 0:
                handles.append(artist)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel, labelpad=0.5)
        ax.set_ylabel(ylabel, labelpad=0.5)
        ax.set_title(title, pad=1.5)
        clean_axes(ax, horizontal_grid=False)
        ax.grid(True, color="#E2E2E2", linewidth=0.4, alpha=0.72)
        ax.set_axisbelow(True)
    fig.legend(handles, METHODS, ncol=4, loc="upper center", bbox_to_anchor=(0.49, 0.995), frameon=False, handletextpad=0.10, columnspacing=0.25, borderaxespad=0.0, borderpad=0.0, markerscale=1.05)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.245, top=0.695, wspace=0.38)
    save_exact(fig, output, "fig3_bound_validation")


def draw_fig4(source_dir: Path, output: Path) -> None:
    rows = read_csv(source_dir / "fig4_method_summary.csv")
    lookup = {(int(row["observation_ratio_percent"]), row["Method"], row["Metric"]): row for row in rows}
    metrics = (("missing_edge_MAE", "(a) Edge MAE"), ("E_c", r"(b) $E_c$"), ("E_w", r"(c) $E_w$"), ("E_x", r"(d) $E_x$"))
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 2.00))
    limits = {}
    for index, (metric, title) in enumerate(metrics):
        ax = axes.flat[index]
        for method in METHODS:
            values = [lookup[(ratio, method, metric)] for ratio in (40, 55, 70)]
            means = np.asarray([float(value["Mean"]) for value in values])
            lower = np.asarray([float(value["95% CI lower"]) for value in values])
            upper = np.asarray([float(value["95% CI upper"]) for value in values])
            ax.errorbar((40, 55, 70), means, yerr=np.vstack([means - lower, upper - means]), color=COLORS[method], marker=MARKERS[method], linestyle=LINES[method], linewidth=1.0, markersize=3.8, markeredgewidth=0.55, markeredgecolor="#222222", elinewidth=0.75, capsize=1.7, capthick=0.75, label=method)
        ax.set_title(title, pad=1.0)
        ax.set_xticks((40, 55, 70))
        clean_axes(ax)
        limits[metric] = [float(value) for value in ax.get_ylim()]
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.998), frameon=False, handlelength=1.15, handletextpad=0.22, columnspacing=0.48, borderaxespad=0.0)
    fig.supxlabel("Observation ratio (%)", x=0.54, y=0.012, fontsize=8.4)
    fig.subplots_adjust(left=0.125, right=0.995, bottom=0.19, top=0.77, wspace=0.38, hspace=0.66)
    save_exact(fig, output, "fig4_observation_ratio")


def draw_fig5(source_dir: Path, output: Path) -> None:
    rows = read_csv(source_dir / "fig5_heatmap_values.csv")
    mechanisms = ("Uniform", "Target-biased", "Block")
    ratios = (40, 55, 70)
    matrix = np.asarray([[float(next(row["E_w_reduction_percent"] for row in rows if row["missingness_mechanism"] == mechanism and int(row["observation_ratio_percent"]) == ratio)) for ratio in ratios] for mechanism in mechanisms])
    significant = np.asarray([[next(row["paired_95_interval_below_zero"] for row in rows if row["missingness_mechanism"] == mechanism and int(row["observation_ratio_percent"]) == ratio).lower() == "true" for ratio in ratios] for mechanism in mechanisms])
    limit = max(abs(matrix.min()), abs(matrix.max()))
    palette = LinearSegmentedColormap.from_list("asgc_soft_diverging", ["#DDAA78", "#F7F5F1", "#5F9F98"])
    fig, ax = plt.subplots(figsize=(3.35, 1.55))
    image = ax.pcolormesh(np.arange(4) - 0.5, np.arange(4) - 0.5, matrix, cmap=palette, vmin=-limit, vmax=limit, shading="flat", edgecolors="white", linewidth=1.0)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_xticks(range(3), ("40%", "55%", "70%"))
    ax.set_yticks(range(3), mechanisms)
    ax.tick_params(axis="both", length=0, pad=2)
    for row in range(3):
        for col in range(3):
            label = f"{matrix[row, col]:+.1f}%" + ("*" if significant[row, col] else "")
            ax.text(col, row, label, ha="center", va="center", fontsize=8.8, color="black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.045)
    cbar.solids.set_rasterized(False)
    cbar.set_label(r"Relative $E_w$ reduction (%)", fontsize=8.8, labelpad=2.5)
    cbar.ax.tick_params(labelsize=8.4, length=2.5, width=0.7, pad=1.5)
    cbar.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.255, right=0.875, bottom=0.14, top=0.985)
    save_exact(fig, output, "fig5_missingness")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce paper figures from released result files."
    )
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source = ROOT / config["frozen_results_dir"]
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    draw_fig3(source, output)
    draw_fig4(source, output)
    draw_fig5(source, output)
    print("figure reproduction smoke test: PASS" if args.smoke_test else output)


if __name__ == "__main__":
    main()
