import numpy as np
from src.models.asgc import project_in_strength

def test_projection_interval():
    x = np.zeros((3, 8))
    x[:, 1] = [1, 2, 3]
    x[:, 4] = [2, 1, 4]
    u = np.array([-5, 2.5, 99.0])
    np.testing.assert_allclose(project_in_strength(u, x), [1, 2.5, 7])
