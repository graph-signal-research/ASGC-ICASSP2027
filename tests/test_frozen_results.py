import numpy as np
import pandas as pd

from src.utils import REPO_ROOT


def test_synthetic_summary_matches_scenario_level_results():
    per_scenario = pd.read_csv(
        REPO_ROOT / 'results/frozen/synthetic/table1_per_scenario.csv'
    )
    summary = pd.read_csv(
        REPO_ROOT / 'results/frozen/synthetic/table1_summary.csv'
    ).set_index('Method')

    for method, group in per_scenario.groupby('Method'):
        row = summary.loc[method]
        assert row['Scenario count'] == group['scenario_id'].nunique()
        for metric in ['missing_edge_MAE', 'E_c', 'E_w', 'E_x']:
            values = group[metric].to_numpy(float)
            assert np.isclose(
                values.mean(), row[f'Mean {metric}'], rtol=1e-8, atol=1e-12
            )
            assert np.isclose(
                values.std(ddof=1),
                row[f'Sample std {metric}'],
                rtol=1e-8,
                atol=1e-12,
            )


def test_synthetic_table1_rounded_and_reductions():
    data = pd.read_csv(
        REPO_ROOT / 'results/frozen/synthetic/table1_summary.csv'
    ).set_index('Method')
    asgc = data.loc['ASGC']
    raw = data.loc['Raw-GAT']

    assert round(asgc['Mean E_c'], 4) == 3.0947
    assert round(asgc['Mean E_w'], 4) == 0.0668
    assert round(asgc['Mean E_x'], 4) == 0.0043

    reductions = [
        100 * (raw[f'Mean {metric}'] - asgc[f'Mean {metric}'])
        / raw[f'Mean {metric}']
        for metric in ('E_c', 'E_w', 'E_x')
    ]
    assert [round(value, 1) for value in reductions] == [17.3, 14.0, 14.4]


def test_synthetic_ablation_summary():
    data = pd.read_csv(
        REPO_ROOT / 'results/frozen/synthetic/ablation_summary.csv'
    ).set_index('Variant')
    assert np.isclose(data.loc['BiasOnly', 'Value'], 0.081217)
    assert np.isclose(data.loc['AggregateBasic', 'Value'], 0.071538)
    assert np.isclose(data.loc['ASGC', 'Value'], 0.068905)
    assert np.isclose(data.loc['Residual reassignment', 'Value'], 0.095243)


def test_block_70_value():
    data = pd.read_csv(
        REPO_ROOT / 'results/frozen/synthetic/fig5_heatmap_values.csv'
    )
    value = float(
        data[
            (data.missingness_mechanism == 'Block')
            & (data.observation_ratio_percent == 70)
        ].E_w_reduction_percent.iloc[0]
    )
    assert round(value, 1) == 21.4


def test_metr_node_disjoint_table2_values():
    data = pd.read_csv(
        REPO_ROOT
        / 'results/frozen/metr_la_topology_disjoint/test_summary.csv'
    )

    def get(ratio, method, metric):
        return float(
            data[
                (data.observation_ratio == ratio)
                & (data.Method == method)
                & (data.metric == metric)
            ].iloc[0]['mean']
        )

    assert round(get(0.4, 'Raw-GAT', 'E_c'), 4) == 20.0814
    assert round(get(0.4, 'ASGC', 'E_c'), 4) == 14.8649
    assert round(get(0.4, 'ASGC', 'E_w'), 4) == 0.3821
    assert round(get(0.4, 'ASGC', 'E_x2_METR'), 4) == 2.9051
    assert round(get(0.7, 'ASGC', 'E_c'), 4) == 9.8704
    assert round(get(0.7, 'ASGC', 'E_w'), 4) == 0.2575
    assert round(get(0.7, 'ASGC', 'E_x2_METR'), 4) == 1.9839


def test_metr_node_disjoint_signflip():
    data = pd.read_csv(
        REPO_ROOT
        / 'results/frozen/metr_la_topology_disjoint/graph_exact_signflip.csv'
    )

    def p(ratio, metric):
        row = data[
            (data.observation_ratio == ratio) & (data.metric == metric)
        ].iloc[0]
        return float(row.exact_two_sided_signflip_p), int(row.graphs_improved)

    assert p(0.4, 'E_c') == (0.0625, 5)
    assert p(0.4, 'E_w') == (0.03125, 6)
    assert p(0.4, 'E_x2_METR') == (0.03125, 6)
    assert p(0.7, 'E_c') == (0.5, 3)
    assert p(0.7, 'E_w') == (0.03125, 6)
    assert p(0.7, 'E_x2_METR') == (0.03125, 6)


def test_raw_and_asgc_share_edge_metrics_per_scenario():
    data = pd.read_csv(
        REPO_ROOT
        / 'results/frozen/metr_la_topology_disjoint/test_scenario_metrics.csv'
    )
    keys = ['graph_id', 'mask_rep', 'observation_ratio']
    pivot_all = data.pivot(index=keys, columns='Method', values='E_a_all')
    pivot_pos = data.pivot(index=keys, columns='Method', values='E_a_positive')
    assert np.array_equal(pivot_all['Raw-GAT'].to_numpy(), pivot_all['ASGC'].to_numpy())
    assert np.array_equal(pivot_pos['Raw-GAT'].to_numpy(), pivot_pos['ASGC'].to_numpy())
