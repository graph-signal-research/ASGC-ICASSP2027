from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from src.data.metr_la import (
    candidate_relation_support,
    deterministic_subgraph,
    load_metr_la_graph,
    preprocess_signals,
    read_development_prefix,
)
from src.experiments.metr_la import (
    evaluate_scenario,
    load_checkpoint,
    predict_completed,
)
from src.masking import make_nested_masks
from src.models.asgc import ASGCModel
from src.utils import load_config, sha256_file

METRIC_COLUMNS = ['E_c', 'E_w', 'E_x', 'missing_edge_MAE']
KEY_COLUMNS = ['mask_seed', 'observation_ratio', 'Method']
REPLAY_RTOL = 5e-7
REPLAY_ATOL = 1e-8


def verify_checkpoint_manifest() -> None:
    """Verify released held-out checkpoints against the public manifest."""
    manifest_path = ROOT / 'docs/CHECKPOINT_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for item in manifest:
        path = ROOT / item['file']
        if sha256_file(path) != item['sha256']:
            raise RuntimeError(f'Checkpoint hash mismatch: {item["file"]}')


def verify_edge_invariance(metrics: pd.DataFrame) -> None:
    """Verify ASGC and Raw-GAT retain identical missing-edge predictions."""
    pivot = metrics.pivot_table(
        index=['mask_seed', 'observation_ratio'],
        columns='Method',
        values='missing_edge_MAE',
    )
    if not np.array_equal(
        pivot['Raw-GAT'].to_numpy(),
        pivot['ASGC'].to_numpy(),
    ):
        raise RuntimeError('Raw-GAT and ASGC missing-edge MAE differ.')


def verify_replay(replayed: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Compare replayed held-out metrics with the released per-mask values."""
    merged = replayed[KEY_COLUMNS + METRIC_COLUMNS].merge(
        reference[KEY_COLUMNS + METRIC_COLUMNS],
        on=KEY_COLUMNS,
        suffixes=('_replayed', '_reference'),
    )

    for metric in METRIC_COLUMNS:
        replayed_values = merged[f'{metric}_replayed'].to_numpy(float)
        reference_values = merged[f'{metric}_reference'].to_numpy(float)
        if not np.allclose(
            replayed_values,
            reference_values,
            rtol=REPLAY_RTOL,
            atol=REPLAY_ATOL,
        ):
            difference = float(
                np.max(np.abs(replayed_values - reference_values))
            )
            raise RuntimeError(
                f'Released-result mismatch in {metric}: max abs={difference}'
            )



def recompute_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Recompute Table 2 summary statistics from per-mask metrics."""
    rows = []
    for ratio in sorted(metrics['observation_ratio'].unique()):
        subset = metrics[metrics['observation_ratio'] == ratio]
        pivoted = {
            metric: subset.pivot(
                index='mask_seed', columns='Method', values=metric
            ).sort_index()
            for metric in ['E_c', 'E_w', 'E_x']
        }
        for metric, pivot in pivoted.items():
            raw = pivot['Raw-GAT'].to_numpy(float)
            asgc = pivot['ASGC'].to_numpy(float)
            rows.append(
                {
                    'observation_ratio': float(ratio),
                    'metric': metric,
                    'seed_count': int(len(pivot)),
                    'Raw-GAT_mean': float(raw.mean()),
                    'Raw-GAT_std': float(raw.std(ddof=1)),
                    'ASGC_mean': float(asgc.mean()),
                    'ASGC_std': float(asgc.std(ddof=1)),
                    'ASGC_improvement_percent_vs_Raw-GAT': float(
                        100.0 * (raw.mean() - asgc.mean()) / raw.mean()
                    ),
                    'ASGC_improved_seed_count': int(np.sum(asgc < raw)),
                }
            )
    return pd.DataFrame(rows)


def recompute_bootstrap(
    metrics: pd.DataFrame,
    reference_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute paired mask-level bootstrap intervals using released seeds."""
    rows = []
    for _, reference in reference_bootstrap.iterrows():
        ratio = float(reference['observation_ratio'])
        metric = str(reference['metric'])
        subset = metrics[metrics['observation_ratio'] == ratio]
        pivot = subset.pivot(
            index='mask_seed', columns='Method', values=metric
        ).sort_index()
        differences = (
            pivot['ASGC'].to_numpy(float) - pivot['Raw-GAT'].to_numpy(float)
        )
        repetitions = int(reference['bootstrap_repetitions'])
        seed = int(reference['bootstrap_rng_seed'])
        rng = np.random.default_rng(seed)
        sample_index = rng.integers(
            0,
            len(differences),
            size=(repetitions, len(differences)),
        )
        bootstrap_means = differences[sample_index].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append(
            {
                'observation_ratio': ratio,
                'metric': metric,
                'pairing': 'ASGC - Raw-GAT',
                'seed_count': int(len(differences)),
                'bootstrap_repetitions': repetitions,
                'mean_difference': float(differences.mean()),
                'median_difference': float(np.median(differences)),
                '95% CI lower': float(lower),
                '95% CI upper': float(upper),
                'ASGC_improved_seed_count': int(np.sum(differences < 0)),
                'bootstrap_rng_seed': seed,
            }
        )
    return pd.DataFrame(rows)


def verify_numeric_frame(
    recomputed: pd.DataFrame,
    reference: pd.DataFrame,
    key_columns: list[str],
    label: str,
) -> None:
    """Verify a recomputed tabular artifact against its released reference."""
    merged = recomputed.merge(
        reference,
        on=key_columns,
        suffixes=('_recomputed', '_reference'),
        validate='one_to_one',
    )
    if len(merged) != len(reference) or len(merged) != len(recomputed):
        raise RuntimeError(f'{label} row/key mismatch')

    for column in recomputed.columns:
        if column in key_columns:
            continue
        left = merged[f'{column}_recomputed']
        right = merged[f'{column}_reference']
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                left.to_numpy(float),
                right.to_numpy(float),
                rtol=REPLAY_RTOL,
                atol=REPLAY_ATOL,
            ):
                raise RuntimeError(f'{label} mismatch in {column}')
        elif not np.array_equal(left.astype(str), right.astype(str)):
            raise RuntimeError(f'{label} mismatch in {column}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate ASGC on the ten held-out METR-LA masks.'
    )
    parser.add_argument('--config', default='configs/metr_la.yaml')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    metr_root = ROOT / cfg['frozen_results_dir']
    development_dir = metr_root / 'development'
    heldout_dir = metr_root / 'heldout_masks'
    reference_metrics = pd.read_csv(
        heldout_dir / 'heldout_10_seed_metrics.csv'
    )
    reference_summary = pd.read_csv(
        heldout_dir / 'heldout_10_seed_summary.csv'
    )
    reference_bootstrap = pd.read_csv(
        heldout_dir / 'heldout_10_seed_bootstrap.csv'
    )
    verify_checkpoint_manifest()

    if args.smoke_test:
        verify_edge_invariance(reference_metrics)
        verify_numeric_frame(
            recompute_summary(reference_metrics),
            reference_summary,
            ['observation_ratio', 'metric'],
            'METR-LA held-out summary',
        )
        verify_numeric_frame(
            recompute_bootstrap(reference_metrics, reference_bootstrap),
            reference_bootstrap,
            ['observation_ratio', 'metric'],
            'METR-LA held-out bootstrap',
        )
        print('held-out mask artifact smoke test: PASS')
        return

    data_dir = ROOT / cfg['data_dir']
    sensor_ids, _, adjacency = load_metr_la_graph(data_dir)
    selected_index, selected_ids, sensor_hash = deterministic_subgraph(
        sensor_ids,
        adjacency,
        int(cfg['subgraph']['node_count']),
    )
    if sensor_hash != cfg['subgraph']['sensor_id_sha256']:
        raise RuntimeError('METR-LA sensor selection changed.')

    development, temporal = read_development_prefix(
        data_dir,
        sensor_ids,
        float(cfg['temporal_scope']['train_fraction']),
        float(cfg['temporal_scope']['evaluation_end_fraction']),
    )
    node_features, signal_windows, _ = preprocess_signals(
        development,
        int(temporal['train_stop_index_exclusive']),
        selected_ids,
    )

    truth = adjacency[np.ix_(selected_index, selected_index)].copy()
    np.fill_diagonal(truth, 0.0)
    support = candidate_relation_support(
        truth,
        int(cfg['relation_support']['directed_edge_count']),
    )
    asgc = ASGCModel.load(development_dir / 'asgc_final_model.npz')

    rows = []
    for seed in cfg['heldout_mask_seeds']:
        for ratio in cfg['observation_ratios']:
            observed, _, _, _ = make_nested_masks(
                support,
                int(seed),
                float(ratio),
                float(cfg['gat']['internal_validation_fraction']),
            )
            checkpoint_path = (
                heldout_dir
                / 'models'
                / f'gat_seed_{seed}_r{int(float(ratio) * 100)}.pt'
            )
            model = load_checkpoint(checkpoint_path, cfg['gat'])
            completed = predict_completed(
                model,
                node_features,
                truth,
                observed,
                support,
            )
            for metric_row in evaluate_scenario(
                truth,
                support,
                observed,
                completed,
                signal_windows,
                asgc,
                float(cfg['aggregation']['epsilon']),
            ):
                rows.append(
                    {
                        'mask_seed': int(seed),
                        'observation_ratio': float(ratio),
                        **metric_row,
                    }
                )

    replayed = pd.DataFrame(rows)
    verify_edge_invariance(replayed)
    verify_replay(replayed, reference_metrics)

    summary = recompute_summary(replayed)
    bootstrap = recompute_bootstrap(replayed, reference_bootstrap)
    verify_numeric_frame(
        summary,
        reference_summary,
        ['observation_ratio', 'metric'],
        'METR-LA held-out summary',
    )
    verify_numeric_frame(
        bootstrap,
        reference_bootstrap,
        ['observation_ratio', 'metric'],
        'METR-LA held-out bootstrap',
    )

    output_dir = ROOT / cfg['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    replayed.to_csv(
        output_dir / 'heldout_10_seed_metrics.csv',
        index=False,
    )
    summary.to_csv(
        output_dir / 'heldout_10_seed_summary.csv',
        index=False,
    )
    bootstrap.to_csv(
        output_dir / 'heldout_10_seed_bootstrap.csv',
        index=False,
    )
    print('METR-LA held-out-mask evaluation and Table 2 statistics: PASS')


if __name__ == '__main__':
    main()
