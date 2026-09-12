from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import pandas as pd

from src.utils import load_config


SYNTHETIC_METHODS = ['Raw-GAT', 'GlobalMean', 'NodeMean', 'ASGC']
METR_METHODS = ['Raw-GAT', 'ASGC']
METRICS = ['E_c', 'E_w', 'E_x']


def synthetic_table(config: dict) -> pd.DataFrame:
    """Build paper Table 1 from released synthetic summaries."""
    source = ROOT / config['frozen_results_dir']
    summary = pd.read_csv(source / 'table1_summary.csv').set_index('Method')
    bootstrap = pd.read_csv(source / 'table1_paired_bootstrap.csv')

    rows = []
    for method in SYNTHETIC_METHODS:
        row = summary.loc[method]
        significant = {metric: False for metric in METRICS}
        if method == 'ASGC':
            for metric in METRICS:
                result = bootstrap[
                    (bootstrap['Comparison'] == 'ASGC - Raw-GAT')
                    & (bootstrap['Metric'] == metric)
                ].iloc[0]
                significant[metric] = bool(result['95% CI upper'] < 0)

        rows.append(
            {
                'Method': method,
                'Edge MAE': float(row['Mean missing_edge_MAE']),
                'E_c': float(row['Mean E_c']),
                'E_w': float(row['Mean E_w']),
                'E_x': float(row['Mean E_x']),
                'E_c_significant': significant['E_c'],
                'E_w_significant': significant['E_w'],
                'E_x_significant': significant['E_x'],
            }
        )
    return pd.DataFrame(rows)


def metr_la_table(config: dict) -> pd.DataFrame:
    """Build paper Table 2 from the ten held-out METR-LA masks."""
    source = ROOT / config['frozen_results_dir'] / 'heldout_masks'
    summary = pd.read_csv(source / 'heldout_10_seed_summary.csv')
    bootstrap = pd.read_csv(source / 'heldout_10_seed_bootstrap.csv')

    rows = []
    for ratio in (0.4, 0.7):
        for method in METR_METHODS:
            row = {
                'Observation': int(round(100 * ratio)),
                'Method': method,
            }
            for metric in METRICS:
                result = summary[
                    (summary['observation_ratio'] == ratio)
                    & (summary['metric'] == metric)
                ].iloc[0]
                row[f'{metric}_mean'] = float(result[f'{method}_mean'])
                row[f'{metric}_std'] = float(result[f'{method}_std'])

                significant = False
                if method == 'ASGC':
                    paired = bootstrap[
                        (bootstrap['observation_ratio'] == ratio)
                        & (bootstrap['metric'] == metric)
                    ].iloc[0]
                    significant = bool(paired['95% CI upper'] < 0)
                row[f'{metric}_significant'] = significant
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_markdown(table: pd.DataFrame) -> str:
    """Format Table 1 as Markdown."""
    lines = [
        '| Method | Edge MAE | $E_c$ | $E_w$ | $E_x$ |',
        '|---|---:|---:|---:|---:|',
    ]
    for _, row in table.iterrows():
        cells = [f"{row['Edge MAE']:.4f}"]
        for metric in METRICS:
            value = f'{row[metric]:.4f}'
            if bool(row[f'{metric}_significant']):
                value += '*'
            cells.append(value)
        lines.append(f"| {row['Method']} | " + ' | '.join(cells) + ' |')

    lines.extend(
        [
            '',
            '* Paired 95% bootstrap confidence interval for ASGC minus '
            'Raw-GAT lies below zero.',
        ]
    )
    return '\n'.join(lines) + '\n'


def metr_la_markdown(table: pd.DataFrame) -> str:
    """Format Table 2 as Markdown."""
    lines = [
        '| Obs. | Method | $E_c$ | $E_w$ | $E_{x,2}^{\\mathrm{METR}}$ |',
        '|---:|---|---:|---:|---:|',
    ]
    for _, row in table.iterrows():
        cells = []
        for metric in METRICS:
            value = (
                f"{row[f'{metric}_mean']:.4f} ± "
                f"{row[f'{metric}_std']:.4f}"
            )
            if bool(row[f'{metric}_significant']):
                value += '*'
            cells.append(value)
        lines.append(
            f"| {int(row['Observation'])}% | {row['Method']} | "
            + ' | '.join(cells)
            + ' |'
        )

    lines.extend(
        [
            '',
            '* Paired 95% bootstrap confidence interval for ASGC minus '
            'Raw-GAT lies below zero.',
            '$E_{x,2}^{\\mathrm{METR}}$ is the mean per-window temporal '
            '$\\ell_2$ deviation used by the METR-LA protocol; it is '
            'distinct from the weighted-$\\ell_1$ synthetic $E_x$ in Eq. (3).',
        ]
    )
    return '\n'.join(lines) + '\n'


def latex_value(row: pd.Series, metric: str) -> str:
    """Format one synthetic metric cell for LaTeX."""
    value = f'{row[metric]:.4f}'
    if bool(row[f'{metric}_significant']):
        return '\\textbf{' + value + '}$^*$'
    return value


def write_latex_table1(table: pd.DataFrame, path: Path) -> None:
    """Write the synthetic paper table in LaTeX format."""
    lines = [
        '\\begin{tabular}{lrrrr}',
        '\\toprule',
        'Method & Edge MAE & $E_c$ & $E_w$ & $E_x$\\\\',
        '\\midrule',
    ]
    for _, row in table.iterrows():
        edge = f"{row['Edge MAE']:.4f}"
        if row['Method'] == 'NodeMean':
            edge = '\\textbf{' + edge + '}'
        lines.append(
            f"{row['Method']} & {edge} & {latex_value(row, 'E_c')} & "
            f"{latex_value(row, 'E_w')} & {latex_value(row, 'E_x')}\\\\"
        )
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_latex_table2(table: pd.DataFrame, path: Path) -> None:
    """Write the METR-LA paper table in LaTeX format."""
    lines = [
        '\\begin{tabular}{clccc}',
        '\\toprule',
        'Obs. & Method & $E_c$ & $E_w$ & '
        '$E_{x,2}^{\\mathrm{METR}}$\\\\',
        '\\midrule',
    ]
    for _, row in table.iterrows():
        cells = []
        for metric in METRICS:
            value = (
                f"{row[f'{metric}_mean']:.4f} $\\pm$ "
                f"{row[f'{metric}_std']:.4f}"
            )
            if bool(row[f'{metric}_significant']):
                value = '\\textbf{' + value + '}$^*$'
            cells.append(value)
        lines.append(
            f"{int(row['Observation'])}\\% & {row['Method']} & "
            + ' & '.join(cells)
            + '\\\\'
        )
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Reproduce the paper tables from released result files.'
    )
    parser.add_argument('--config', default='configs/synthetic.yaml')
    parser.add_argument('--metr-config', default='configs/metr_la.yaml')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    synthetic_config = load_config(args.config)
    metr_config = load_config(args.metr_config)
    output = ROOT / 'results/reproduced/tables'
    output.mkdir(parents=True, exist_ok=True)

    table1 = synthetic_table(synthetic_config)
    table2 = metr_la_table(metr_config)
    table1.to_csv(output / 'table1_paper.csv', index=False)
    table2.to_csv(output / 'table2_paper.csv', index=False)
    (output / 'table1_paper.md').write_text(
        synthetic_markdown(table1), encoding='utf-8'
    )
    (output / 'table2_paper.md').write_text(
        metr_la_markdown(table2), encoding='utf-8'
    )
    write_latex_table1(table1, output / 'table1_paper.tex')
    write_latex_table2(table2, output / 'table2_paper.tex')

    if args.smoke_test:
        print('table reproduction smoke test: PASS')
    else:
        print(output)


if __name__ == '__main__':
    main()
