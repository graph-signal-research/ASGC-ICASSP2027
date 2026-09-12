from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import shutil

import numpy as np
import torch

from src.data.metr_la import (
    candidate_relation_support,
    deterministic_subgraph,
    load_metr_la_graph,
    preprocess_signals,
    read_development_prefix,
)
from src.experiments.metr_la import (
    crossfit_ridge,
    predict_completed,
    train_one_gat,
)
from src.masking import make_nested_masks
from src.models.asgc import in_strength_features
from src.utils import load_config


def smoke_test(cfg: dict) -> None:
    """Check deterministic mask construction without loading METR-LA."""
    support = np.ones((6, 6), dtype=bool)
    np.fill_diagonal(support, False)
    _, fit, validation, missing = make_nested_masks(
        support,
        int(cfg['development_mask_seeds'][0]),
        0.4,
        float(cfg['gat']['internal_validation_fraction']),
    )
    assert not np.any(fit & validation)
    assert np.array_equal(fit | validation | missing, support)
    print('METR-LA development smoke test: PASS')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the METR-LA development training protocol.'
    )
    parser.add_argument('--config', default='configs/metr_la.yaml')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Replace only the reproduced development output directory.',
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke_test:
        smoke_test(cfg)
        return

    data_dir = ROOT / cfg['data_dir']
    sensor_ids, _, adjacency = load_metr_la_graph(data_dir)
    selected_index, selected_ids, selected_hash = deterministic_subgraph(
        sensor_ids,
        adjacency,
        int(cfg['subgraph']['node_count']),
    )
    if selected_hash != cfg['subgraph']['sensor_id_sha256']:
        raise RuntimeError('METR-LA sensor selection changed.')

    development, temporal = read_development_prefix(
        data_dir,
        sensor_ids,
        float(cfg['temporal_scope']['train_fraction']),
        float(cfg['temporal_scope']['evaluation_end_fraction']),
    )
    node_features, _, _ = preprocess_signals(
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

    output_dir = ROOT / cfg['output_dir'] / 'development_rerun'
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f'Refusing overwrite: {output_dir}; rerun with --overwrite '
                'to replace reproduced outputs only.'
            )
        shutil.rmtree(output_dir)
    model_dir = output_dir / 'models'
    model_dir.mkdir(parents=True)

    scenarios = {}
    for seed in cfg['development_mask_seeds']:
        for ratio in cfg['observation_ratios']:
            observed, fit, validation, _ = make_nested_masks(
                support,
                int(seed),
                float(ratio),
                float(cfg['gat']['internal_validation_fraction']),
            )
            model, training_record = train_one_gat(
                node_features,
                truth,
                fit,
                validation,
                int(seed),
                float(ratio),
                cfg['gat'],
            )
            completed = predict_completed(
                model,
                node_features,
                truth,
                observed,
                support,
            )
            scenarios[int(seed), float(ratio)] = {
                'features': in_strength_features(
                    truth,
                    support,
                    observed,
                    completed,
                )
            }
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'training_protocol': training_record,
                },
                model_dir
                / f'gat_seed_{seed}_r{int(float(ratio) * 100)}.pt',
            )

    _, scaler, ridge = crossfit_ridge(
        scenarios,
        [int(seed) for seed in cfg['development_mask_seeds']],
        [float(ratio) for ratio in cfg['observation_ratios']],
        float(cfg['ridge']['lambda']),
    )
    reproduced_model_path = output_dir / 'asgc_final_model.npz'
    np.savez_compressed(
        reproduced_model_path,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        ridge_intercept=np.asarray([ridge.intercept_]),
        ridge_coefficients=ridge.coef_,
        ridge_alpha=np.asarray([float(cfg['ridge']['lambda'])]),
    )

    frozen_model_path = (
        ROOT / cfg['frozen_results_dir'] / 'development/asgc_final_model.npz'
    )
    with np.load(reproduced_model_path, allow_pickle=False) as reproduced, np.load(
        frozen_model_path, allow_pickle=False
    ) as frozen:
        if set(reproduced.files) != set(frozen.files):
            raise RuntimeError('ASGC model bundle schema changed')
        for key in reproduced.files:
            left = reproduced[key]
            right = frozen[key]
            if not np.allclose(
                left,
                right,
                rtol=5e-7,
                atol=5e-8,
            ):
                max_difference = float(np.max(np.abs(left - right)))
                raise RuntimeError(
                    f'ASGC development model mismatch in {key}: '
                    f'max abs={max_difference}'
                )

    manifest = {
        'development_seeds': cfg['development_mask_seeds'],
        'observation_ratios': cfg['observation_ratios'],
        'ridge_lambda': cfg['ridge']['lambda'],
        'frozen_model_verification_rtol': 5e-7,
        'frozen_model_verification_atol': 5e-8,
    }
    (output_dir / 'run_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n',
        encoding='utf-8',
    )
    print('METR-LA development training and numerical frozen-model verification: PASS')


if __name__ == '__main__':
    main()
