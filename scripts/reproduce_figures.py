from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import load_config


METHOD_ORDER = ['Raw-GAT', 'GlobalMean', 'NodeMean', 'ASGC']
METHOD_ALIAS = {'GAT+Refinement': 'ASGC'}


def method_label(name: str) -> str:
    """Return the paper-facing method label."""
    return METHOD_ALIAS.get(str(name), str(name))


def save_figure(fig, output: Path, stem: str) -> None:
    """Save PNG and deterministic-metadata PDF versions of a figure."""
    fig.tight_layout()
    fig.savefig(output / f'{stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(
        output / f'{stem}.pdf',
        bbox_inches='tight',
        metadata={
            'Creator': 'Matplotlib',
            'CreationDate': None,
            'ModDate': None,
        },
    )
    plt.close(fig)


def reproduce_fig3(source: Path, output: Path) -> None:
    """Reproduce the two-panel aggregation-bound figure."""
    data = pd.read_csv(source / 'fig3_bound_data.csv').copy()
    data['method_public'] = data['method'].map(method_label)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75))
    panels = [
        (
            'B_w',
            'E_w',
            'Upper bound $B_w$',
            'Observed error $E_w$',
            '(a) Normalized-weight error',
        ),
        (
            'B_x',
            'E_x',
            'Upper bound $B_x$',
            'Observed error $E_x$',
            '(b) Aggregated-signal error',
        ),
    ]

    for axis, (bound, error, xlabel, ylabel, title) in zip(axes, panels):
        for method in METHOD_ORDER:
            group = data[data['method_public'] == method]
            if not group.empty:
                axis.scatter(group[bound], group[error], s=17, label=method)
        limit = max(float(data[bound].max()), float(data[error].max()))
        axis.plot([0.0, limit], [0.0, limit], '--', linewidth=1.0)
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.25)
    save_figure(fig, output, 'fig3_bound_validation')


def reproduce_fig4(source: Path, output: Path) -> None:
    """Reproduce observation-ratio results with bootstrap intervals."""
    summary = pd.read_csv(source / 'fig4_method_summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.9))
    panels = [
        ('missing_edge_MAE', 'Missing-edge MAE', '(a) Missing-edge MAE'),
        ('E_c', 'In-strength error $E_c$', '(b) In-strength error $E_c$'),
        ('E_w', 'Weight error $E_w$', '(c) Weight error $E_w$'),
        ('E_x', 'Signal error $E_x$', '(d) Signal error $E_x$'),
    ]

    for axis, (metric, ylabel, title) in zip(axes.flat, panels):
        subset = summary[summary['Metric'] == metric]
        for method in METHOD_ORDER:
            group = subset[subset['Method'] == method].sort_values(
                'observation_ratio_percent'
            )
            if group.empty:
                continue
            x = group['observation_ratio_percent'].to_numpy(float)
            y = group['Mean'].to_numpy(float)
            lower = group['95% CI lower'].to_numpy(float)
            upper = group['95% CI upper'].to_numpy(float)
            yerr = np.vstack([y - lower, upper - y])
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                marker='o',
                capsize=2.5,
                linewidth=1.1,
                markersize=3.5,
                label=method,
            )
        axis.set_xticks([40, 55, 70])
        axis.set_xlabel('Observation ratio (%)')
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=4,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(bottom=0.17)
    save_figure(fig, output, 'fig4_observation_ratio')


def reproduce_fig5(source: Path, output: Path) -> None:
    """Reproduce missingness-pattern weight-error reductions."""
    values = pd.read_csv(source / 'fig5_heatmap_values.csv')
    patterns = ['Uniform', 'Target-biased', 'Block']
    ratios = [40, 55, 70]
    reductions = np.empty((len(patterns), len(ratios)), dtype=float)
    significant = np.empty_like(reductions, dtype=bool)

    for row_index, pattern in enumerate(patterns):
        for column_index, ratio in enumerate(ratios):
            row = values[
                (values['missingness_mechanism'] == pattern)
                & (values['observation_ratio_percent'] == ratio)
            ].iloc[0]
            reductions[row_index, column_index] = float(
                row['E_w_reduction_percent']
            )
            significant[row_index, column_index] = bool(
                row['paired_95_interval_below_zero']
            )

    fig, axis = plt.subplots(figsize=(5.0, 2.6))
    image = axis.imshow(reductions, aspect='auto')
    axis.set_xticks(range(len(ratios)), [f'{ratio}%' for ratio in ratios])
    axis.set_yticks(range(len(patterns)), patterns)
    axis.set_xlabel('Observation ratio')

    for row_index in range(len(patterns)):
        for column_index in range(len(ratios)):
            marker = '*' if significant[row_index, column_index] else ''
            axis.text(
                column_index,
                row_index,
                f'{reductions[row_index, column_index]:+.1f}%{marker}',
                ha='center',
                va='center',
            )
    fig.colorbar(image, ax=axis, label='Relative $E_w$ reduction (%)')
    save_figure(fig, output, 'fig5_missingness')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reproduce paper figures from released result files.'
    )
    parser.add_argument('--config', default='configs/synthetic.yaml')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    source = ROOT / config['frozen_results_dir']
    output = ROOT / 'figures'
    output.mkdir(exist_ok=True)

    reproduce_fig3(source, output)
    reproduce_fig4(source, output)
    reproduce_fig5(source, output)
    print('figure reproduction smoke test: PASS' if args.smoke_test else output)


if __name__ == '__main__':
    main()
