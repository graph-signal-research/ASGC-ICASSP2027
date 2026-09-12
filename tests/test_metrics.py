import numpy as np

from src.metrics import (
    in_strength_error,
    metr_la_temporal_l2_signal_errors,
    missing_edge_mae,
    weight_error,
    weighted_l1_fused_signal_error,
)


def test_metrics_basic():
    c = np.array([1.0, 2.0])
    star = np.array([1.5, 1.5])
    assert in_strength_error(c, star) == 1.0
    assert weight_error(c, star) > 0

    signals = np.ones((2, 2, 3))
    errors = metr_la_temporal_l2_signal_errors(c, star, signals)
    assert errors.shape == (2,)

    matrix = np.array([[0.0, 0.2], [0.4, 0.0]])
    truth = np.array([[0.0, 0.3], [0.4, 0.0]])
    missing = np.array([[0, 1], [0, 0]], bool)
    assert abs(missing_edge_mae(matrix, truth, missing) - 0.1) < 1e-12


def test_weighted_l1_fused_signal_error_matches_eq3():
    w_hat = np.array([0.75, 0.25])
    w_star = np.array([0.5, 0.5])
    signals = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    got = weighted_l1_fused_signal_error(w_hat, w_star, signals)
    delta = np.einsum('i,irp->rp', w_hat - w_star, signals)
    expected = np.sum(np.abs(delta) * np.array([0.5, 0.5])[None, :]) / 2
    assert np.isclose(got, expected)
