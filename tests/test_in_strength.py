import numpy as np
from src.models.asgc import in_strength_features

def test_eight_features_and_column_sums():
    t = np.array([[0.0, 0.2, 0.3], [0.4, 0.0, 0.5], [0.6, 0.7, 0.0]])
    support = t > 0
    obs = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]], bool)
    completed = np.where(obs, t, 0.25)
    np.fill_diagonal(completed, 0)
    f = in_strength_features(t, support, obs, completed)
    assert f['x'].shape == (3, 8)
    np.testing.assert_allclose(f['true_total'], t.sum(axis=0))
