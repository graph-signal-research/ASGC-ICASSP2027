import numpy as np
import pandas as pd

from src.utils import REPO_ROOT


def test_development_heldout_seed_isolation():
    development = {2027101, 2027102, 2027103, 2027104, 2027105}
    heldout = set(range(2027106, 2027116))
    assert development.isdisjoint(heldout)


def test_heldout_edge_mae_equality():
    path = (
        REPO_ROOT
        / 'results/frozen/metr_la/heldout_masks/heldout_10_seed_metrics.csv'
    )
    data = pd.read_csv(path)
    pivot = data.pivot(
        index=['mask_seed', 'observation_ratio'],
        columns='Method',
        values='missing_edge_MAE',
    )
    assert np.array_equal(
        pivot['Raw-GAT'].to_numpy(), pivot['ASGC'].to_numpy()
    )


def test_gat_split_isolation():
    path = (
        REPO_ROOT
        / 'results/frozen/metr_la/heldout_masks/edge_split_manifest.csv'
    )
    data = pd.read_csv(path)
    overlap = data[
        [
            'fit_validation_overlap',
            'fit_missing_overlap',
            'validation_missing_overlap',
        ]
    ].to_numpy()
    assert (overlap == 0).all()
    assert data.partition_support_exact.all()
    assert (~data.missing_used_for_training).all()
    assert (~data.missing_used_for_early_stopping).all()


def test_mask_hash_determinism():
    from src.masking import mask_manifest_row

    support = np.ones((8, 8), dtype=bool)
    np.fill_diagonal(support, False)
    first = mask_manifest_row(support, 2027106, 0.4, 0.2)
    second = mask_manifest_row(support, 2027106, 0.4, 0.2)
    for key in ['observed_hash', 'fit_hash', 'validation_hash', 'missing_hash']:
        assert first[key] == second[key]


def test_mask_parameter_validation():
    import pytest
    from src.masking import make_nested_masks

    support = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(support, False)
    with pytest.raises(ValueError):
        make_nested_masks(support, 1, 0.0, 0.2)
    with pytest.raises(ValueError):
        make_nested_masks(support, 1, 0.4, 1.0)
