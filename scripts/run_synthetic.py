from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse

import numpy as np
import pandas as pd
import torch

from src.models.gat import DirectedGATPointModel
from src.utils import load_config

METHOD_ORDER = ['Raw-GAT', 'GlobalMean', 'NodeMean', 'ASGC']
METRICS = ['missing_edge_MAE', 'E_c', 'E_w', 'E_x']


def recompute_table1(per_scenario: pd.DataFrame) -> pd.DataFrame:
    """Recompute Table 1 statistics from released scenario-level results."""
    required = {'scenario_id', 'Method', *METRICS}
    missing = required.difference(per_scenario.columns)
    if missing:
        raise ValueError(f'Missing synthetic result columns: {sorted(missing)}')

    counts = per_scenario.groupby('Method')['scenario_id'].nunique()
    rows = []
    for method in METHOD_ORDER:
        group = per_scenario[per_scenario['Method'] == method]
        if group.empty:
            raise ValueError(f'Missing synthetic method: {method}')
        row = {
            'Method': method,
            'Scenario count': int(counts.loc[method]),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            row[f'Mean {metric}'] = float(values.mean())
            row[f'Sample std {metric}'] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def verify_shared_edge_predictions(per_scenario: pd.DataFrame) -> None:
    """Verify Raw-GAT and ASGC share edge estimates in every scenario."""
    pivot = per_scenario.pivot(
        index='scenario_id',
        columns='Method',
        values='missing_edge_MAE',
    )
    if not np.array_equal(
        pivot['Raw-GAT'].to_numpy(),
        pivot['ASGC'].to_numpy(),
    ):
        raise RuntimeError('Raw-GAT and ASGC missing-edge MAE differ.')


def verify_summary(recomputed: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Check recomputed statistics against the released summary."""
    a = recomputed.set_index('Method').loc[METHOD_ORDER]
    b = frozen.set_index('Method').loc[METHOD_ORDER]
    if list(a.columns) != list(b.columns):
        raise RuntimeError('Synthetic summary schema changed.')

    for column in a.columns:
        if column == 'Scenario count':
            if not np.array_equal(
                a[column].to_numpy(),
                b[column].to_numpy(),
            ):
                raise RuntimeError('Synthetic scenario counts changed.')
            continue

        left = a[column].to_numpy(float)
        right = b[column].to_numpy(float)
        if not np.allclose(left, right, rtol=1e-8, atol=1e-12):
            difference = float(np.max(np.abs(left - right)))
            raise RuntimeError(
                f'Synthetic summary mismatch in {column}: '
                f'max abs={difference}'
            )


def smoke_test() -> None:
    """Check the released synthetic GAT forward path without training."""
    torch.manual_seed(1)
    node_features = torch.randn(5, 4)
    edges = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    model = DirectedGATPointModel(4)
    prediction = model(node_features, edges, edges)
    assert prediction.shape == (4,)
    assert torch.isfinite(prediction).all()
    print('synthetic smoke test: PASS')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Recompute synthetic paper summaries from scenario-level artifacts.'
        )
    )
    parser.add_argument('--config', default='configs/synthetic.yaml')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    cfg = load_config(args.config)
    frozen_dir = ROOT / cfg['frozen_results_dir']
    output_dir = ROOT / cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    per_scenario = pd.read_csv(frozen_dir / 'table1_per_scenario.csv')
    frozen_summary = pd.read_csv(frozen_dir / 'table1_summary.csv')
    verify_shared_edge_predictions(per_scenario)
    recomputed = recompute_table1(per_scenario)
    verify_summary(recomputed, frozen_summary)
    recomputed.to_csv(output_dir / 'table1_summary.csv', index=False)

    indexed = recomputed.set_index('Method')
    raw = indexed.loc['Raw-GAT']
    asgc = indexed.loc['ASGC']
    reductions = {
        metric: (
            100.0
            * (raw[f'Mean {metric}'] - asgc[f'Mean {metric}'])
            / raw[f'Mean {metric}']
        )
        for metric in ('E_c', 'E_w', 'E_x')
    }
    print('Synthetic result reproduction: PASS')
    print(
        {
            name: round(float(value), 4)
            for name, value in reductions.items()
        }
    )


if __name__ == '__main__':
    main()
