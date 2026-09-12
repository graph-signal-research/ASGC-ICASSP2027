from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils import sensor_id_hash


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


def candidate_relation_support(
    adjacency: np.ndarray,
    expected_edge_count: int | None = None,
) -> np.ndarray:
    """Return the nonzero, non-self directed candidate-relation support."""
    adjacency = np.asarray(adjacency, float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('adjacency must be a square matrix')
    support = adjacency > 0
    np.fill_diagonal(support, False)
    if expected_edge_count is not None and int(support.sum()) != int(
        expected_edge_count
    ):
        raise RuntimeError(
            'Candidate-relation support changed: '
            f'expected {int(expected_edge_count)}, got {int(support.sum())}'
        )
    return support


def in_strength(adjacency: np.ndarray) -> np.ndarray:
    """Return c = A.T @ 1 for the convention A[j, i] = j -> i."""
    adjacency = np.asarray(adjacency, float)
    return adjacency.T @ np.ones(adjacency.shape[0])


def deterministic_subgraph(
    sensor_ids: list[str],
    adjacency: np.ndarray,
    node_count: int = 20,
):
    """Select the fixed connected METR-LA subgraph used in the paper."""
    undirected = (adjacency > 0) | (adjacency.T > 0)
    np.fill_diagonal(undirected, False)

    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(sensor_ids)):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor_raw in np.where(undirected[node])[0]:
                neighbor = int(neighbor_raw)
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    component = max(components, key=len)
    degree = undirected.sum(axis=1)
    max_degree = max(int(degree[node]) for node in component)
    start = min(
        [node for node in component if int(degree[node]) == max_degree],
        key=lambda node: sensor_ids[node],
    )

    selected = [start]
    selected_set = {start}
    queue = deque([start])
    while queue and len(selected) < node_count:
        node = queue.popleft()
        neighbors = [
            int(neighbor)
            for neighbor in np.where(undirected[node])[0]
            if int(neighbor) in component
            and int(neighbor) not in selected_set
        ]
        neighbors.sort(
            key=lambda neighbor: (
                -int(undirected[neighbor, list(selected_set)].sum()),
                -int(degree[neighbor]),
                sensor_ids[neighbor],
            )
        )
        for neighbor in neighbors:
            if neighbor in selected_set:
                continue
            selected.append(neighbor)
            selected_set.add(neighbor)
            queue.append(neighbor)
            if len(selected) >= node_count:
                break

    if len(selected) != node_count:
        raise RuntimeError(
            f'Could not select {node_count} connected sensors; got {len(selected)}'
        )
    selected_ids = [sensor_ids[index] for index in selected]
    return selected, selected_ids, sensor_id_hash(selected_ids)


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


def preprocess_signals(
    development: pd.DataFrame,
    train_end: int,
    selected_ids: list[str],
):
    """Return GAT features [20,12] and temporal windows [T,20,12]."""
    selected = development[selected_ids].to_numpy(float)
    training = selected[:train_end]
    training_index = development.index[:train_end]
    time_bins = np.asarray([timestamp.hour // 2 for timestamp in training_index])

    node_count = len(selected_ids)
    bin_medians = np.zeros((node_count, 12))
    node_fallback = np.zeros(node_count)

    valid_training = training[
        np.isfinite(training) & (training > 0)
    ]
    global_fallback = (
        float(np.median(valid_training)) if valid_training.size else 0.0
    )

    for node in range(node_count):
        node_values = training[:, node]
        node_values = node_values[
            np.isfinite(node_values) & (node_values > 0)
        ]
        node_fallback[node] = (
            float(np.median(node_values))
            if node_values.size
            else global_fallback
        )

        for time_bin in range(12):
            values = training[time_bins == time_bin, node]
            values = values[np.isfinite(values) & (values > 0)]
            bin_medians[node, time_bin] = (
                float(np.median(values))
                if values.size
                else node_fallback[node]
            )

    imputed = selected.copy()
    for row, timestamp in enumerate(development.index):
        time_bin = timestamp.hour // 2
        invalid = ~np.isfinite(imputed[row]) | (imputed[row] <= 0)
        if np.any(invalid):
            imputed[row, invalid] = bin_medians[invalid, time_bin]

    training_imputed = imputed[:train_end]
    descriptors = np.column_stack(
        [
            training_imputed[time_bins == time_bin].mean(axis=0)
            for time_bin in range(12)
        ]
    )
    scaler = StandardScaler().fit(descriptors)
    node_features = scaler.transform(descriptors)

    signal_windows = np.stack(
        [
            imputed[index - 11 : index + 1].T
            for index in range(train_end, len(development))
        ],
        axis=0,
    )
    timestamps = [
        pd.Timestamp(value) for value in development.index[train_end:]
    ]
    return (
        node_features.astype(float),
        signal_windows.astype(float),
        timestamps,
    )
