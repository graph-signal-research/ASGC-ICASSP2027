from __future__ import annotations

import numpy as np

from .models.asgc import normalized_weights


def missing_edge_mae(
    matrix: np.ndarray,
    truth: np.ndarray,
    missing: np.ndarray,
) -> float:
    """Mean absolute error on unobserved support edges."""
    if not np.any(missing):
        return 0.0
    return float(
        np.mean(
            np.abs(np.asarray(matrix)[missing] - np.asarray(truth)[missing])
        )
    )


def in_strength_error(c_hat: np.ndarray, c_star: np.ndarray) -> float:
    """Return graph-level L1 in-strength error E_c."""
    return float(np.abs(np.asarray(c_hat) - np.asarray(c_star)).sum())


def weight_error(
    c_hat: np.ndarray,
    c_star: np.ndarray,
    epsilon: float = 0.01,
) -> float:
    """Return L1 error E_w between normalized aggregation weights."""
    w_hat = normalized_weights(c_hat, epsilon)
    w_star = normalized_weights(c_star, epsilon)
    return float(np.abs(w_hat - w_star).sum())


def weighted_l1_fused_signal_error(
    weights_hat: np.ndarray,
    weights_star: np.ndarray,
    node_signals: np.ndarray,
    channel_weights: np.ndarray | None = None,
) -> float:
    """Return the channel-weighted L1 fused-signal error in paper Eq. (3).

    `node_signals` has shape [N, M, P]. The default channel weights are
    uniform, nu_p = 1/P.
    """
    signals = np.asarray(node_signals, float)
    if signals.ndim != 3:
        raise ValueError(
            f'Expected node_signals with shape [N,M,P], got {signals.shape}'
        )

    delta_weights = np.asarray(weights_hat, float) - np.asarray(
        weights_star, float
    )
    fused_delta = np.einsum('i,irp->rp', delta_weights, signals)
    channel_count = signals.shape[2]

    if channel_weights is None:
        nu = np.full(channel_count, 1.0 / channel_count)
    else:
        nu = np.asarray(channel_weights, float)

    if nu.shape != (channel_count,) or not np.isclose(nu.sum(), 1.0):
        raise ValueError('channel_weights must have shape [P] and sum to one')

    return float(
        np.sum(np.abs(fused_delta) * nu[None, :]) / signals.shape[1]
    )


def metr_la_temporal_l2_signal_errors(
    c_hat: np.ndarray,
    c_star: np.ndarray,
    signal_windows: np.ndarray,
    epsilon: float = 0.01,
) -> np.ndarray:
    """Return the per-window temporal L2 signal deviation for METR-LA.

    `signal_windows` has shape [T, N, 12]. The reported METR-LA value is
    the mean of these per-window deviations. Synthetic Eq. (3) uses the
    weighted-L1 metric implemented above.
    """
    delta_weights = normalized_weights(
        c_hat, epsilon
    ) - normalized_weights(c_star, epsilon)
    temporal_delta = np.einsum(
        'i,tij->tj', delta_weights, np.asarray(signal_windows, float)
    )
    return np.linalg.norm(temporal_delta, axis=1)
