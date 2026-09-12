import numpy as np
import pandas as pd

from src.utils import REPO_ROOT, sensor_id_hash


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


def test_metr_heldout_summary_exact():
    data = pd.read_csv(
        REPO_ROOT
        / 'results/frozen/metr_la/heldout_masks/heldout_10_seed_summary.csv'
    )

    def get(ratio, metric):
        return data[
            (data.observation_ratio == ratio) & (data.metric == metric)
        ].iloc[0]

    assert round(get(0.4, 'E_c')['ASGC_mean'], 4) == 6.0667
    assert round(get(0.7, 'E_w')['ASGC_mean'], 4) == 0.0813
    assert (
        round(
            get(0.4, 'E_x')['ASGC_improvement_percent_vs_Raw-GAT'],
            2,
        )
        == 29.05
    )


def test_metr_significance_source_is_well_formed():
    data = pd.read_csv(
        REPO_ROOT
        / 'results/frozen/metr_la/heldout_masks/heldout_10_seed_bootstrap.csv'
    )
    assert {
        'observation_ratio',
        'metric',
        '95% CI lower',
        '95% CI upper',
    }.issubset(data.columns)
    assert np.isfinite(
        data[['95% CI lower', '95% CI upper']].to_numpy()
    ).all()


def test_sensor_hash():
    ids = (
        REPO_ROOT
        / 'results/frozen/metr_la/development/selected_sensor_ids.txt'
    ).read_text().splitlines()
    assert sensor_id_hash(ids) == (
        '692779f867902d401354b31816aaf6bbf5d42c57f6a1f056dce9da89a9332bd7'
    )
