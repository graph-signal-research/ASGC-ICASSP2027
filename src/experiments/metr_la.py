from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..metrics import (
    in_strength_error,
    metr_la_temporal_l2_signal_errors,
    missing_edge_mae,
    weight_error,
)
from ..models.asgc import in_strength_features, project_in_strength
from ..models.gat import DenseGAT
from ..utils import stable_seed


def masked_positive_mae(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return MAE on the selected positive/support edges."""
    values = torch.abs(prediction[mask] - truth[mask])
    if values.numel() == 0:
        raise RuntimeError('Empty edge loss mask')
    return values.mean()


def train_one_gat(
    node_features: np.ndarray,
    truth: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    seed: int,
    ratio: float,
    cfg: dict,
):
    """Train one METR-LA GAT with disjoint fit and validation edges."""
    run_seed = stable_seed(seed, 'metr-la-gat-init', ratio)
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    torch.use_deterministic_algorithms(True)

    features_tensor = torch.tensor(
        node_features[None, :, :], dtype=torch.float32
    )
    truth_tensor = torch.tensor(truth[None, :, :], dtype=torch.float32)
    fit_tensor = torch.tensor(fit[None, :, :], dtype=torch.bool)
    validation_tensor = torch.tensor(
        validation[None, :, :], dtype=torch.bool
    )
    observed_tensor = torch.tensor(
        np.where(fit, truth, 0.0)[None, :, :],
        dtype=torch.float32,
    )

    model = DenseGAT(
        input_dim=int(cfg['input_dim']),
        hidden=int(cfg['hidden_per_head']),
        heads=int(cfg['heads']),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg['learning_rate']),
        weight_decay=float(cfg['weight_decay']),
    )

    best_validation_mae = float('inf')
    best_state = None
    stale_epochs = 0
    best_epoch = 0

    for epoch in range(1, int(cfg['max_epochs']) + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(features_tensor, observed_tensor, fit_tensor)
        loss = masked_positive_mae(prediction, truth_tensor, fit_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_mae = float(
                masked_positive_mae(
                    model(features_tensor, observed_tensor, fit_tensor),
                    truth_tensor,
                    validation_tensor,
                )
            )

        if validation_mae < best_validation_mae - float(cfg['min_delta']):
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= int(cfg['patience']):
            break

    if best_state is None:
        raise RuntimeError('GAT early stopping produced no checkpoint')

    model.load_state_dict(best_state)
    training_record = {
        'best_epoch': best_epoch,
        'best_observed_validation_edge_MAE': best_validation_mae,
        'epochs_run': epoch,
    }
    return model, training_record


def predict_completed(
    model: DenseGAT,
    node_features: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Keep observed edges exact and predict only unobserved support edges."""
    features_tensor = torch.tensor(
        node_features[None, :, :], dtype=torch.float32
    )
    observed_tensor = torch.tensor(
        np.where(observed, truth, 0.0)[None, :, :],
        dtype=torch.float32,
    )
    mask_tensor = torch.tensor(observed[None, :, :], dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        prediction = model(
            features_tensor,
            observed_tensor,
            mask_tensor,
        ).numpy()[0].astype(float)

    completed = np.zeros_like(truth, float)
    completed[support & ~observed] = prediction[support & ~observed]
    completed[observed] = truth[observed]
    np.fill_diagonal(completed, 0.0)
    return completed


def load_checkpoint(path, cfg: dict) -> DenseGAT:
    """Load a METR-LA GAT state dictionary."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    architecture = checkpoint.get('architecture', {})
    expected = {
        'input_dim': int(cfg['input_dim']),
        'hidden_per_head': int(cfg['hidden_per_head']),
        'heads': int(cfg['heads']),
    }
    for key, value in expected.items():
        if key in architecture and int(architecture[key]) != value:
            raise RuntimeError(
                f'Checkpoint architecture mismatch for {key}: '
                f'{architecture[key]} != {value}'
            )
    model = DenseGAT(
        input_dim=int(cfg['input_dim']),
        hidden=int(cfg['hidden_per_head']),
        heads=int(cfg['heads']),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def evaluate_scenario(
    truth: np.ndarray,
    support: np.ndarray,
    observed: np.ndarray,
    completed: np.ndarray,
    signal_windows: np.ndarray,
    asgc_model,
    epsilon: float = 0.01,
) -> list[dict]:
    """Evaluate Raw-GAT and ASGC on one graph/mask scenario."""
    features = in_strength_features(
        truth,
        support,
        observed,
        completed,
    )
    raw_strength = features['raw_total']
    asgc_strength = asgc_model.apply(features)
    true_strength = features['true_total']
    missing = features['missing']
    edge_mae = missing_edge_mae(completed, truth, missing)

    rows = []
    for method, strength in [
        ('Raw-GAT', raw_strength),
        ('ASGC', asgc_strength),
    ]:
        signal_errors = metr_la_temporal_l2_signal_errors(
            strength,
            true_strength,
            signal_windows,
            epsilon,
        )
        rows.append(
            {
                'Method': method,
                'missing_edge_MAE': edge_mae,
                'E_c': in_strength_error(strength, true_strength),
                'E_w': weight_error(strength, true_strength, epsilon),
                'E_x': float(signal_errors.mean()),
            }
        )
    return rows


def crossfit_ridge(
    scenarios: dict,
    development_seeds: list[int],
    ratios: list[float],
    alpha: float = 1.0,
):
    """Cross-fit Ridge by mask seed and refit once on all development seeds."""
    crossfit_outputs = {}

    for held_seed in development_seeds:
        training_keys = [
            key for key in scenarios if key[0] != held_seed
        ]
        x = np.vstack(
            [scenarios[key]['features']['x'] for key in training_keys]
        )
        y = np.concatenate(
            [
                scenarios[key]['features']['true_total']
                - scenarios[key]['features']['raw_total']
                for key in training_keys
            ]
        )
        scaler = StandardScaler().fit(x)
        ridge = Ridge(alpha=alpha, fit_intercept=True).fit(
            scaler.transform(x), y
        )

        for ratio in ratios:
            features = scenarios[held_seed, ratio]['features']
            correction = ridge.predict(scaler.transform(features['x']))
            crossfit_outputs[held_seed, ratio] = project_in_strength(
                features['raw_total'] + correction,
                features['x'],
            )

    all_keys = list(scenarios)
    x = np.vstack([scenarios[key]['features']['x'] for key in all_keys])
    y = np.concatenate(
        [
            scenarios[key]['features']['true_total']
            - scenarios[key]['features']['raw_total']
            for key in all_keys
        ]
    )
    final_scaler = StandardScaler().fit(x)
    final_ridge = Ridge(alpha=alpha, fit_intercept=True).fit(
        final_scaler.transform(x), y
    )
    return crossfit_outputs, final_scaler, final_ridge
