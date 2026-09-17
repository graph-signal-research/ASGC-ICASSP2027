from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.metr_la import load_metr_la_graph, read_development_prefix
from src.models.gat import DenseGAT
from src.models.asgc import in_strength_features, project_in_strength
from src.metrics import (
    in_strength_error,
    metr_la_temporal_l2_signal_errors,
    missing_edge_mae,
    weight_error,
)
from src.utils import stable_seed


DEV_GROUPS = [0, 3, 6, 9]
TEST_GROUPS = [1, 2, 4, 5, 7, 8]
OBS_RATIOS = [0.4, 0.7]
MASK_REPS = [0, 1]
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
EPSILON = 0.01
INTERNAL_VALIDATION_FRACTION = 0.20
GAT_CFG = {
    'input_dim': 12,
    'hidden_per_head': 12,
    'heads': 2,
    'learning_rate': 0.005,
    'weight_decay': 0.0001,
    'max_epochs': 300,
    'patience': 30,
    'min_delta': 1.0e-7,
}


def _part1by1(n: int) -> int:
    n &= 0xFFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def location_only_groups(data_dir: Path) -> tuple[list[list[str]], list[str], dict]:
    """Create 10 disjoint 20-node groups using locations only (no adjacency)."""
    loc = pd.read_csv(data_dir / 'graph_sensor_locations.csv', dtype={'sensor_id': str})
    required = {'sensor_id', 'latitude', 'longitude'}
    if not required.issubset(loc.columns):
        raise RuntimeError(f'Location columns missing: {required - set(loc.columns)}')
    loc = loc.copy()
    loc['sensor_id'] = loc['sensor_id'].astype(str)
    if loc['sensor_id'].duplicated().any() or len(loc) != 207:
        raise RuntimeError('Expected 207 unique sensor locations')

    lon = loc['longitude'].to_numpy(float)
    lat = loc['latitude'].to_numpy(float)
    qx = np.floor((lon - lon.min()) / (lon.max() - lon.min()) * 65535.0 + 0.5).astype(np.uint32)
    qy = np.floor((lat - lat.min()) / (lat.max() - lat.min()) * 65535.0 + 0.5).astype(np.uint32)
    morton = np.asarray([
        _part1by1(int(x)) | (_part1by1(int(y)) << 1) for x, y in zip(qx, qy)
    ], dtype=np.uint64)
    order = sorted(
        range(len(loc)),
        key=lambda i: (int(morton[i]), str(loc.iloc[i]['sensor_id'])),
    )
    groups = [
        [str(loc.iloc[i]['sensor_id']) for i in order[g * 20:(g + 1) * 20]]
        for g in range(10)
    ]
    unused = [str(loc.iloc[i]['sensor_id']) for i in order[200:]]

    flat = [s for group in groups for s in group]
    assert len(flat) == 200 and len(set(flat)) == 200
    payload = '\n'.join(f'{g}:{s}' for g, group in enumerate(groups) for s in group) + '\n'
    manifest = {
        'rule': '16-bit Morton/Z-order of (longitude, latitude), tie-break sensor_id; consecutive groups of 20',
        'uses_adjacency_for_split': False,
        'group_count': 10,
        'group_size': 20,
        'development_group_ids': DEV_GROUPS,
        'test_group_ids': TEST_GROUPS,
        'unused_sensor_ids': unused,
        'split_sha256': hashlib.sha256(payload.encode()).hexdigest(),
    }
    return groups, unused, manifest



def preprocess_signals_fast(development: pd.DataFrame, train_end: int, selected_ids: list[str]):
    """Vectorized equivalent of the repository preprocessing for one node set."""
    selected = development[selected_ids].to_numpy(float)
    training = selected[:train_end]
    train_bins = (development.index[:train_end].hour.to_numpy() // 2).astype(int)
    all_bins = (development.index.hour.to_numpy() // 2).astype(int)
    n = len(selected_ids)

    valid = np.isfinite(training) & (training > 0)
    global_vals = training[valid]
    global_fallback = float(np.median(global_vals)) if global_vals.size else 0.0

    node_fallback = np.empty(n, float)
    bin_medians = np.empty((n, 12), float)
    for node in range(n):
        vals = training[:, node]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        node_fallback[node] = float(np.median(vals)) if vals.size else global_fallback
        for b in range(12):
            bvals = training[train_bins == b, node]
            bvals = bvals[np.isfinite(bvals) & (bvals > 0)]
            bin_medians[node, b] = float(np.median(bvals)) if bvals.size else node_fallback[node]

    imputed = selected.copy()
    invalid = ~np.isfinite(imputed) | (imputed <= 0)
    fallback = bin_medians[:, all_bins].T
    imputed[invalid] = fallback[invalid]

    training_imputed = imputed[:train_end]
    descriptors = np.column_stack([
        training_imputed[train_bins == b].mean(axis=0) for b in range(12)
    ])
    scaler = StandardScaler().fit(descriptors)
    node_features = scaler.transform(descriptors)

    all_windows = np.lib.stride_tricks.sliding_window_view(imputed, window_shape=12, axis=0)
    # shape [rows-11, nodes, 12], window ending at train_end is index train_end-11
    signal_windows = np.asarray(all_windows[train_end - 11:], float).copy()
    timestamps = [pd.Timestamp(v) for v in development.index[train_end:]]
    if signal_windows.shape != (len(development) - train_end, n, 12):
        raise RuntimeError(f'Unexpected signal window shape {signal_windows.shape}')
    return node_features.astype(float), signal_windows, timestamps

def all_offdiagonal_support(n: int = 20) -> np.ndarray:
    support = np.ones((n, n), dtype=bool)
    np.fill_diagonal(support, False)
    return support


def make_candidate_masks(n: int, seed: int, ratio: float, validation_fraction: float):
    support = all_offdiagonal_support(n)
    edges = np.argwhere(support)
    rng = np.random.default_rng(stable_seed(seed, 'td-candidate-order'))
    ordered = edges[rng.permutation(len(edges))]
    observed_count = int(round(ratio * len(edges)))
    observed = np.zeros_like(support)
    chosen = ordered[:observed_count]
    observed[chosen[:, 0], chosen[:, 1]] = True

    obs_edges = np.argwhere(observed)
    split_rng = np.random.default_rng(stable_seed(seed, 'td-gat-val', ratio))
    shuffled = obs_edges[split_rng.permutation(len(obs_edges))]
    val_count = min(len(obs_edges) - 1, max(1, int(round(validation_fraction * len(obs_edges)))))
    validation = np.zeros_like(support)
    val_edges = shuffled[:val_count]
    validation[val_edges[:, 0], val_edges[:, 1]] = True
    fit = observed & ~validation
    missing = support & ~observed
    assert not np.any(fit & validation)
    assert not np.any(fit & missing)
    assert not np.any(validation & missing)
    assert np.array_equal(fit | validation | missing, support)
    return observed, fit, validation, missing


def masked_mae(prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = torch.abs(prediction[mask] - truth[mask])
    if vals.numel() == 0:
        raise RuntimeError('Empty loss mask')
    return vals.mean()


def train_gat_candidate(
    node_features: np.ndarray,
    visible_truth: np.ndarray,
    fit_candidates: np.ndarray,
    validation_candidates: np.ndarray,
    seed: int,
    ratio: float,
):
    """Train with observed labels only; hidden candidate labels are unavailable."""
    observed_candidates = fit_candidates | validation_candidates
    if np.any(~np.isfinite(visible_truth[observed_candidates])):
        raise RuntimeError('Observed candidate labels must be finite')
    if np.any(np.isfinite(visible_truth[~observed_candidates])):
        raise RuntimeError('Hidden candidate labels unexpectedly exposed to GAT')

    run_seed = stable_seed(seed, 'td-gat-init', ratio)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.use_deterministic_algorithms(True)

    x = torch.tensor(node_features[None, :, :], dtype=torch.float32)
    target_np = np.zeros_like(visible_truth, float)
    target_np[observed_candidates] = visible_truth[observed_candidates]
    y = torch.tensor(target_np[None, :, :], dtype=torch.float32)
    fit_loss = torch.tensor(fit_candidates[None, :, :], dtype=torch.bool)
    val_loss = torch.tensor(validation_candidates[None, :, :], dtype=torch.bool)

    # Message edges are determined only from FIT-observed candidate labels.
    fit_message_np = np.zeros_like(fit_candidates, dtype=bool)
    fit_message_np[fit_candidates] = visible_truth[fit_candidates] > 0
    fit_message = torch.tensor(fit_message_np[None, :, :], dtype=torch.bool)
    observed_values_np = np.zeros_like(visible_truth, float)
    observed_values_np[fit_message_np] = visible_truth[fit_message_np]
    observed_values = torch.tensor(observed_values_np[None, :, :], dtype=torch.float32)

    model = DenseGAT(
        input_dim=GAT_CFG['input_dim'],
        hidden=GAT_CFG['hidden_per_head'],
        heads=GAT_CFG['heads'],
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=GAT_CFG['learning_rate'],
        weight_decay=GAT_CFG['weight_decay'],
    )
    best = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, GAT_CFG['max_epochs'] + 1):
        model.train()
        opt.zero_grad()
        pred = model(x, observed_values, fit_message)
        loss = masked_mae(pred, y, fit_loss)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(x, observed_values, fit_message)
            val_mae = float(masked_mae(pred_val, y, val_loss))
        if val_mae < best - GAT_CFG['min_delta']:
            best = val_mae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= GAT_CFG['patience']:
            break
    if best_state is None:
        raise RuntimeError('No GAT checkpoint selected')
    model.load_state_dict(best_state)
    return model, {
        'best_epoch': best_epoch,
        'epochs_run': epoch,
        'validation_candidate_mae': best,
        'fit_candidate_count': int(fit_candidates.sum()),
        'fit_positive_message_count': int(fit_message_np.sum()),
        'validation_candidate_count': int(validation_candidates.sum()),
        'validation_positive_count': int(np.sum(visible_truth[validation_candidates] > 0)),
        'hidden_labels_exposed_to_gat': False,
    }


def predict_completed_candidate(
    model: DenseGAT,
    node_features: np.ndarray,
    visible_truth: np.ndarray,
    observed_candidates: np.ndarray,
):
    support = all_offdiagonal_support(visible_truth.shape[0])
    missing = support & ~observed_candidates
    if np.any(~np.isfinite(visible_truth[observed_candidates])):
        raise RuntimeError('Observed labels unavailable at completion time')
    if np.any(np.isfinite(visible_truth[missing])):
        raise RuntimeError('Hidden labels exposed at completion time')

    message_np = np.zeros_like(observed_candidates, dtype=bool)
    message_np[observed_candidates] = visible_truth[observed_candidates] > 0
    x = torch.tensor(node_features[None, :, :], dtype=torch.float32)
    observed_values_np = np.zeros_like(visible_truth, float)
    observed_values_np[message_np] = visible_truth[message_np]
    observed_values = torch.tensor(observed_values_np[None, :, :], dtype=torch.float32)
    message_mask = torch.tensor(message_np[None, :, :], dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        pred = model(x, observed_values, message_mask).numpy()[0].astype(float)
    completed = np.zeros_like(visible_truth, float)
    completed[observed_candidates] = visible_truth[observed_candidates]
    completed[missing] = pred[missing]
    np.fill_diagonal(completed, 0.0)
    return completed


def candidate_features(visible_truth, observed, completed):
    """Build ASGC inputs from observed labels and predictions only."""
    n = visible_truth.shape[0]
    support = all_offdiagonal_support(n)
    missing = support & ~observed
    if np.any(~np.isfinite(visible_truth[observed])):
        raise RuntimeError('Observed labels unavailable for ASGC features')
    if np.any(np.isfinite(visible_truth[missing])):
        raise RuntimeError('Hidden labels exposed to ASGC features')
    return in_strength_features(visible_truth, observed, missing, completed)


@dataclass
class Calibrator:
    scaler: StandardScaler
    ridge: Ridge
    alpha: float

    def apply(self, features):
        x = features['x']
        correction = self.ridge.predict(self.scaler.transform(x))
        return project_in_strength(features['raw_total'] + correction, x)


def fit_calibrator(scenarios, keys, alpha):
    X = np.vstack([scenarios[k]['features']['x'] for k in keys])
    y = np.concatenate([
        scenarios[k]['true_total'] - scenarios[k]['features']['raw_total'] for k in keys
    ])
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(scaler.transform(X), y)
    return Calibrator(scaler, ridge, alpha)


def edge_metrics(completed, truth, missing):
    ea_all = missing_edge_mae(completed, truth, missing)
    pos = missing & (truth > 0)
    ea_pos = missing_edge_mae(completed, truth, pos) if np.any(pos) else float('nan')
    return ea_all, ea_pos, int(pos.sum())


def graph_mask_seed(graph_id: int, rep: int) -> int:
    return 310000 + graph_id * 100 + rep


def prepare_graph_data(groups, sensor_ids, adjacency, prefix, train_end):
    id_to_idx = {s: i for i, s in enumerate(sensor_ids)}
    graphs = {}
    for gid, ids in enumerate(groups):
        idx = [id_to_idx[s] for s in ids]
        truth = adjacency[np.ix_(idx, idx)].copy()
        np.fill_diagonal(truth, 0.0)
        node_features, windows, timestamps = preprocess_signals_fast(prefix, train_end, ids)
        graphs[gid] = {
            'sensor_ids': ids,
            'indices': idx,
            'truth': truth,
            'node_features': node_features,
            'windows': windows,
            'timestamps': timestamps,
            'positive_edge_count': int(((truth > 0) & all_offdiagonal_support(20)).sum()),
        }
    return graphs


def train_scenarios(graphs, group_ids, phase, out_dir):
    scenarios = {}
    records = []
    for gid in group_ids:
        g = graphs[gid]
        for rep in MASK_REPS:
            seed = graph_mask_seed(gid, rep)
            for ratio in OBS_RATIOS:
                observed, fit, validation, missing = make_candidate_masks(
                    20, seed, ratio, INTERNAL_VALIDATION_FRACTION
                )
                t0 = time.time()
                visible_truth = np.full_like(g['truth'], np.nan, dtype=float)
                visible_truth[observed] = g['truth'][observed]
                model, record = train_gat_candidate(
                    g['node_features'], visible_truth, fit, validation, seed, ratio
                )
                completed = predict_completed_candidate(
                    model, g['node_features'], visible_truth, observed
                )
                features = candidate_features(visible_truth, observed, completed)
                key = (gid, rep, ratio)
                scenarios[key] = {
                    'features': features,
                    'completed': completed,
                    'observed': observed,
                    'missing': missing,
                }
                if phase == 'development':
                    scenarios[key]['true_total'] = g['truth'].sum(axis=0)
                record.update({
                    'phase': phase,
                    'graph_id': gid,
                    'mask_rep': rep,
                    'mask_seed': seed,
                    'observation_ratio': ratio,
                    'candidate_count': 380,
                    'observed_candidate_count': int(observed.sum()),
                    'missing_candidate_count': int(missing.sum()),
                    'missing_positive_count': int((missing & (g['truth'] > 0)).sum()),
                    'runtime_seconds': time.time() - t0,
                })
                records.append(record)
                print(
                    f'[{phase}] graph={gid} rep={rep} ratio={ratio:.1f} '
                    f'pos={g["positive_edge_count"]} best_ep={record["best_epoch"]} '
                    f'val={record["validation_candidate_mae"]:.5f} '
                    f't={record["runtime_seconds"]:.2f}s',
                    flush=True,
                )
    pd.DataFrame(records).to_csv(out_dir / f'{phase}_gat_training.csv', index=False)
    return scenarios


def select_lambda(dev_scenarios, out_dir):
    rows = []
    for alpha in LAMBDA_GRID:
        graph_scores = []
        for held_gid in DEV_GROUPS:
            train_keys = [k for k in dev_scenarios if k[0] != held_gid]
            held_keys = [k for k in dev_scenarios if k[0] == held_gid]
            cal = fit_calibrator(dev_scenarios, train_keys, alpha)
            scenario_scores = []
            for k in held_keys:
                s = dev_scenarios[k]
                c_hat = cal.apply(s['features'])
                scenario_scores.append(weight_error(c_hat, s['true_total']))
            graph_mean = float(np.mean(scenario_scores))
            graph_scores.append(graph_mean)
            rows.append({
                'lambda': alpha,
                'heldout_development_graph': held_gid,
                'mean_E_w_over_4_scenarios': graph_mean,
            })
        rows.append({
            'lambda': alpha,
            'heldout_development_graph': 'ALL_MEAN',
            'mean_E_w_over_4_scenarios': float(np.mean(graph_scores)),
        })
    cv = pd.DataFrame(rows)
    cv.to_csv(out_dir / 'lambda_graph_cv.csv', index=False)
    summary = cv[cv['heldout_development_graph'].astype(str) == 'ALL_MEAN'].copy()
    summary['lambda_numeric'] = summary['lambda'].astype(float)
    best = summary.sort_values(['mean_E_w_over_4_scenarios', 'lambda_numeric']).iloc[0]
    return float(best['lambda']), cv


def evaluate_test(graphs, test_scenarios, calibrator, out_dir):
    rows = []
    for key, s in sorted(test_scenarios.items()):
        gid, rep, ratio = key
        g = graphs[gid]
        raw = s['features']['raw_total']
        asgc = calibrator.apply(s['features'])
        true = g['truth'].sum(axis=0)  # evaluation-only reference
        ea_all, ea_pos, npos = edge_metrics(s['completed'], g['truth'], s['missing'])
        for method, c_hat in [('Raw-GAT', raw), ('ASGC', asgc)]:
            rows.append({
                'graph_id': gid,
                'mask_rep': rep,
                'mask_seed': graph_mask_seed(gid, rep),
                'observation_ratio': ratio,
                'Method': method,
                'E_a_all': ea_all,
                'E_a_positive': ea_pos,
                'hidden_positive_count': npos,
                'E_c': in_strength_error(c_hat, true),
                'E_w': weight_error(c_hat, true),
                'E_x2_METR': float(
                    metr_la_temporal_l2_signal_errors(
                        c_hat, true, g['windows'], EPSILON
                    ).mean()
                ),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / 'test_scenario_metrics.csv', index=False)
    return metrics


def summarize(metrics, out_dir):
    metric_cols = ['E_a_all', 'E_a_positive', 'E_c', 'E_w', 'E_x2_METR']
    summary_rows = []
    for ratio in OBS_RATIOS:
        sub = metrics[metrics.observation_ratio == ratio]
        for method in ['Raw-GAT', 'ASGC']:
            m = sub[sub.Method == method]
            for metric in metric_cols:
                vals = m[metric].to_numpy(float)
                summary_rows.append({
                    'observation_ratio': ratio,
                    'Method': method,
                    'metric': metric,
                    'scenario_count': len(vals),
                    'mean': float(np.nanmean(vals)),
                    'std_scenario': float(np.nanstd(vals, ddof=1)),
                })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / 'test_summary.csv', index=False)

    # Paired Raw vs ASGC, with graph as the resampling unit. Within each graph/ratio,
    # average the two masks first.
    boot_rows = []
    rng_seed = 20270916
    for ratio in OBS_RATIOS:
        sub = metrics[metrics.observation_ratio == ratio]
        for metric in ['E_c', 'E_w', 'E_x2_METR']:
            pivot = sub.pivot_table(
                index=['graph_id', 'mask_rep'], columns='Method', values=metric
            ).reset_index()
            pivot['diff'] = pivot['ASGC'] - pivot['Raw-GAT']
            graph_diff = pivot.groupby('graph_id')['diff'].mean().sort_index().to_numpy(float)
            rng = np.random.default_rng(stable_seed(rng_seed, ratio, metric))
            idx = rng.integers(0, len(graph_diff), size=(5000, len(graph_diff)))
            means = graph_diff[idx].mean(axis=1)
            lo, hi = np.quantile(means, [0.025, 0.975])
            boot_rows.append({
                'observation_ratio': ratio,
                'metric': metric,
                'unit': 'test_graph',
                'test_graph_count': len(graph_diff),
                'masks_per_graph': 2,
                'mean_difference_ASGC_minus_Raw': float(graph_diff.mean()),
                'median_graph_difference': float(np.median(graph_diff)),
                'ci95_lower': float(lo),
                'ci95_upper': float(hi),
                'graphs_improved': int(np.sum(graph_diff < 0)),
                'bootstrap_repetitions': 5000,
            })
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out_dir / 'graph_cluster_bootstrap.csv', index=False)

    signflip_rows = []
    reduction_rows = []
    for ratio in OBS_RATIOS:
        sub = metrics[metrics.observation_ratio == ratio]
        for metric in ['E_c', 'E_w', 'E_x2_METR']:
            pivot = sub.pivot_table(
                index=['graph_id', 'mask_rep'], columns='Method', values=metric
            ).reset_index()
            pivot['diff'] = pivot['ASGC'] - pivot['Raw-GAT']
            graph_diff = (
                pivot.groupby('graph_id')['diff']
                .mean()
                .sort_index()
                .to_numpy(float)
            )
            if len(graph_diff) != 6:
                raise RuntimeError('Exact sign-flip requires six test graphs')
            signs = np.asarray(
                [
                    [1.0 if (bits >> j) & 1 else -1.0 for j in range(6)]
                    for bits in range(2**6)
                ]
            )
            permuted_means = np.mean(signs * graph_diff[None, :], axis=1)
            observed_mean = float(np.mean(graph_diff))
            tolerance = 1.0e-15
            two_sided = float(
                np.mean(
                    np.abs(permuted_means)
                    >= abs(observed_mean) - tolerance
                )
            )
            one_sided = float(
                np.mean(permuted_means <= observed_mean + tolerance)
            )
            signflip_rows.append({
                'observation_ratio': ratio,
                'metric': metric,
                'unit': 'test_graph',
                'test_graph_count': len(graph_diff),
                'observed_mean_difference_ASGC_minus_Raw': observed_mean,
                'exact_two_sided_signflip_p': two_sided,
                'exact_one_sided_ASGC_lower_p': one_sided,
                'graphs_improved': int(np.sum(graph_diff < 0)),
                'total_sign_patterns': len(permuted_means),
            })

            raw_mean = float(sub[sub.Method == 'Raw-GAT'][metric].mean())
            asgc_mean = float(sub[sub.Method == 'ASGC'][metric].mean())
            reduction_rows.append({
                'observation_ratio': ratio,
                'metric': metric,
                'Raw-GAT': raw_mean,
                'ASGC': asgc_mean,
                'relative_reduction_percent':
                    100.0 * (raw_mean - asgc_mean) / raw_mean,
            })

    pd.DataFrame(signflip_rows).to_csv(
        out_dir / 'graph_exact_signflip.csv', index=False
    )
    pd.DataFrame(reduction_rows).to_csv(
        out_dir / 'asgc_relative_reductions.csv', index=False
    )
    return summary, boot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/METR-LA')
    parser.add_argument('--output-dir', default='results/topology_disjoint_metr_la')
    args = parser.parse_args()
    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Crucial ordering: the node split is fixed from locations before adjacency is loaded.
    groups, unused, split_manifest = location_only_groups(data_dir)

    sensor_ids, _, adjacency = load_metr_la_graph(data_dir)
    if set(sensor_ids) != set(s for g in groups for s in g) | set(unused):
        raise RuntimeError('Location and graph sensor ID sets differ')

    prefix, temporal = read_development_prefix(data_dir, sensor_ids, 0.70, 0.80)
    graphs = prepare_graph_data(
        groups,
        sensor_ids,
        adjacency,
        prefix,
        int(temporal['train_stop_index_exclusive']),
    )

    group_rows = []
    for gid, g in graphs.items():
        group_rows.append({
            'graph_id': gid,
            'split': 'development' if gid in DEV_GROUPS else 'test',
            'sensor_count': 20,
            'candidate_count': 380,
            'positive_truth_edge_count': g['positive_edge_count'],
            'sensor_ids': ' '.join(g['sensor_ids']),
        })
    pd.DataFrame(group_rows).to_csv(out_dir / 'node_split.csv', index=False)

    manifest = {
        **split_manifest,
        'candidate_rule': 'all directed non-self node pairs (20*19=380)',
        'candidate_rule_uses_truth': False,
        'development_graphs': len(DEV_GROUPS),
        'test_graphs': len(TEST_GROUPS),
        'mask_repetitions_per_graph': len(MASK_REPS),
        'observation_ratios': OBS_RATIOS,
        'test_scenario_count': len(TEST_GROUPS) * len(MASK_REPS) * len(OBS_RATIOS),
        'gat_training': 'graph-specific for every graph/mask/ratio; fit-observed candidates enter the training loss; validation-observed candidates enter validation loss; only observed positive relations are used for message passing',
        'calibration_training': 'development graphs only; test scenarios do not store c_star during GAT/calibration input construction',
        'lambda_grid': LAMBDA_GRID,
        'lambda_selection': 'leave-one-development-graph-out CV minimizing mean E_w; expanded grid to verify boundary optimum',
        'signal_scope': '0-70% node descriptors; 70-80% temporal fusion evaluation only',
        'temporal_manifest': temporal,
    }
    (out_dir / 'protocol_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    print('Node split fixed from graph_sensor_locations.csv only.', flush=True)
    for row in group_rows:
        print(
            f"graph={row['graph_id']} split={row['split']} positive_edges={row['positive_truth_edge_count']}",
            flush=True,
        )

    dev_scenarios = train_scenarios(graphs, DEV_GROUPS, 'development', out_dir)
    best_lambda, cv = select_lambda(dev_scenarios, out_dir)
    print(f'Selected lambda={best_lambda:g} by development-graph CV', flush=True)
    final_cal = fit_calibrator(dev_scenarios, list(dev_scenarios), best_lambda)
    np.savez_compressed(
        out_dir / 'development_calibrator.npz',
        scaler_mean=final_cal.scaler.mean_,
        scaler_scale=final_cal.scaler.scale_,
        ridge_intercept=np.asarray([final_cal.ridge.intercept_]),
        ridge_coefficients=final_cal.ridge.coef_,
        ridge_alpha=np.asarray([best_lambda]),
    )

    test_scenarios = train_scenarios(graphs, TEST_GROUPS, 'test', out_dir)
    metrics = evaluate_test(graphs, test_scenarios, final_cal, out_dir)
    summary, boot = summarize(metrics, out_dir)

    manifest['selected_lambda'] = best_lambda
    manifest['raw_asgc_share_same_completion'] = True
    (out_dir / 'protocol_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    print('\n=== SUMMARY ===')
    print(summary.to_string(index=False))
    print('\n=== GRAPH-LEVEL PAIRED BOOTSTRAP ===')
    print(boot.to_string(index=False))


if __name__ == '__main__':
    main()
