from __future__ import annotations

import numpy as np

from .utils import array_hash, stable_seed


def make_nested_masks(
    support: np.ndarray,
    seed: int,
    ratio: float,
    validation_fraction: float = 0.20,
):
    """Create observed, fit, validation, and missing [source, target] masks."""
    support = np.asarray(support, dtype=bool)
    if support.ndim != 2 or support.shape[0] != support.shape[1]:
        raise ValueError('support must be a square boolean matrix')
    if np.any(np.diag(support)):
        raise ValueError('support must exclude self-loops')
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError('ratio must lie strictly between 0 and 1')
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError('validation_fraction must lie strictly between 0 and 1')

    edges = np.argwhere(support)
    if len(edges) < 2:
        raise ValueError('support must contain at least two candidate relations')
    rng = np.random.default_rng(
        stable_seed(seed, 'metr-la-nested-observed-order')
    )
    ordered_edges = edges[rng.permutation(len(edges))]
    observed_count = int(round(ratio * len(edges)))

    observed = np.zeros_like(support, dtype=bool)
    chosen = ordered_edges[:observed_count]
    observed[chosen[:, 0], chosen[:, 1]] = True

    observed_edges = np.argwhere(observed)
    split_rng = np.random.default_rng(
        stable_seed(seed, 'metr-la-gat-internal-validation', ratio)
    )
    shuffled = observed_edges[split_rng.permutation(len(observed_edges))]
    if len(observed_edges) < 2:
        raise ValueError('observation ratio leaves fewer than two observed edges')
    validation_count = min(
        len(observed_edges) - 1,
        max(1, int(round(validation_fraction * len(observed_edges)))),
    )

    validation = np.zeros_like(support, dtype=bool)
    heldout = shuffled[:validation_count]
    validation[heldout[:, 0], heldout[:, 1]] = True

    fit = observed & ~validation
    missing = support & ~observed

    if (
        np.any(fit & validation)
        or np.any(fit & missing)
        or np.any(validation & missing)
    ):
        raise AssertionError('fit/validation/missing masks overlap')
    if not np.array_equal(fit | validation | missing, support):
        raise AssertionError('masks do not partition support')

    return observed, fit, validation, missing


def mask_manifest_row(
    support: np.ndarray,
    seed: int,
    ratio: float,
    validation_fraction: float = 0.20,
) -> dict:
    """Return counts and hashes for one deterministic mask partition."""
    observed, fit, validation, missing = make_nested_masks(
        support, seed, ratio, validation_fraction
    )
    return {
        'mask_seed': seed,
        'observation_ratio': ratio,
        'support_edge_count': int(support.sum()),
        'observed_edge_count': int(observed.sum()),
        'fit_edge_count': int(fit.sum()),
        'validation_edge_count': int(validation.sum()),
        'missing_edge_count': int(missing.sum()),
        'observed_hash': array_hash(observed),
        'fit_hash': array_hash(fit),
        'validation_hash': array_hash(validation),
        'missing_hash': array_hash(missing),
    }
