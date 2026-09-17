from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_metr_la_graph(data_dir: str | Path):
    """Load the DCRNN adjacency directly as A[source, target]."""
    data_dir = Path(data_dir)
    with (data_dir / 'adj_mx.pkl').open('rb') as handle:
        sensor_ids, id_to_ind, adjacency = pickle.load(
            handle, encoding='latin1'
        )

    sensor_ids = [str(sensor_id) for sensor_id in sensor_ids]
    id_to_ind = {str(key): int(value) for key, value in id_to_ind.items()}
    adjacency = np.asarray(adjacency, float)
    if adjacency.shape != (207, 207):
        raise RuntimeError(f'Unexpected adjacency shape {adjacency.shape}')

    paper_adjacency = adjacency.copy()
    np.fill_diagonal(paper_adjacency, 0.0)
    if not np.isfinite(paper_adjacency).all():
        raise RuntimeError('METR-LA adjacency contains non-finite values')
    if paper_adjacency.min() < 0.0 or paper_adjacency.max() > 1.0:
        raise RuntimeError('METR-LA adjacency weights must lie in [0,1]')
    return sensor_ids, id_to_ind, paper_adjacency


def in_strength(adjacency: np.ndarray) -> np.ndarray:
    """Return c = A.T @ 1 for the convention A[j, i] = j -> i."""
    adjacency = np.asarray(adjacency, float)
    return adjacency.T @ np.ones(adjacency.shape[0])


def read_development_prefix(
    data_dir: str | Path,
    sensor_ids: list[str],
    train_fraction: float = 0.70,
    evaluation_end_fraction: float = 0.80,
):
    """Read the temporal prefix used for training and evaluation."""
    if not 0.0 < train_fraction < evaluation_end_fraction <= 1.0:
        raise ValueError(
            'Require 0 < train_fraction < evaluation_end_fraction <= 1'
        )
    h5_path = Path(data_dir) / 'metr-la.h5'
    with pd.HDFStore(h5_path, 'r') as store:
        keys = store.keys()
        if len(keys) != 1:
            raise RuntimeError(f'Unexpected HDF keys {keys}')

        key = keys[0]
        storer = store.get_storer(key)
        row_count = int(storer.shape[0])
        column_count = int(storer.shape[1])
        train_end = int(math.floor(train_fraction * row_count))
        evaluation_end = int(math.floor(evaluation_end_fraction * row_count))
        prefix = store.select(key, start=0, stop=evaluation_end)

    if column_count != 207 or list(map(str, prefix.columns)) != sensor_ids:
        raise RuntimeError('METR-LA sensor order/dimension mismatch')

    interval = prefix.index[1] - prefix.index[0]
    interval_ns = int(interval.value)
    if interval != pd.Timedelta(minutes=5):
        raise RuntimeError('Timestamp interval mismatch')
    if not np.all(np.diff(prefix.index.view('i8')) == interval_ns):
        raise RuntimeError('Timestamp interval mismatch')

    manifest = {
        'source_hdf_file_size_bytes': h5_path.stat().st_size,
        'train_stop_index_exclusive': train_end,
        'validation_stop_index_exclusive': evaluation_end,
        'rows_loaded': evaluation_end,
    }
    return prefix, manifest
