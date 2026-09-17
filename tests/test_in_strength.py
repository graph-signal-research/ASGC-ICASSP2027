import numpy as np
from src.models.asgc import in_strength_features

def test_eight_features_and_column_sums():
    t = np.array([[0.0, 0.2, 0.3], [0.4, 0.0, 0.5], [0.6, 0.7, 0.0]])
    obs = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]], bool)
    candidates = ~np.eye(3, dtype=bool)
    missing = candidates & ~obs
    visible = np.full_like(t, np.nan)
    visible[obs] = t[obs]
    completed = np.where(obs, t, 0.25)
    np.fill_diagonal(completed, 0)
    f = in_strength_features(visible, obs, missing, completed)
    assert f['x'].shape == (3, 8)
    np.testing.assert_allclose(f['raw_total'], [0.65, 0.9, 0.5])
    assert 'true_total' not in f
