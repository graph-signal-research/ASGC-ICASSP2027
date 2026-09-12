import numpy as np
from src.data.metr_la import in_strength

def test_direction_and_in_strength():
    A = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]])
    np.testing.assert_allclose(in_strength(A), A.T @ np.ones(3))
    np.testing.assert_allclose(in_strength(A), [4.0, 2.0, 3.0])
