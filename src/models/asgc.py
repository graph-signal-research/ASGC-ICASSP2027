from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    'tilde_c_i',
    'o_i',
    'p_i',
    'n_i_o',
    'n_i_m',
    'mu_i_o',
    'mu_i_m',
    'sigma_i_m',
)


def in_strength_features(
    truth: np.ndarray,
    support: np.ndarray,
    observed: np.ndarray,
    completed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the eight target-node statistics from [source, target] matrices."""
    missing = support & ~observed

    observed_mass = np.sum(np.where(observed, truth, 0.0), axis=0)
    predicted_mass = np.sum(np.where(missing, completed, 0.0), axis=0)
    preliminary_total = observed_mass + predicted_mass
    true_total = np.sum(np.where(support, truth, 0.0), axis=0)

    observed_count = observed.sum(axis=0).astype(float)
    missing_count = missing.sum(axis=0).astype(float)
    observed_mean = observed_mass / np.maximum(observed_count, 1.0)
    missing_mean = predicted_mass / np.maximum(missing_count, 1.0)

    missing_std = np.asarray(
        [
            np.std(completed[:, j][missing[:, j]])
            if np.any(missing[:, j])
            else 0.0
            for j in range(truth.shape[0])
        ]
    )

    features = np.column_stack(
        [
            preliminary_total,
            observed_mass,
            predicted_mass,
            observed_count,
            missing_count,
            observed_mean,
            missing_mean,
            missing_std,
        ]
    )
    return {
        'x': features,
        'raw_total': preliminary_total,
        'true_total': true_total,
        'observed': observed,
        'missing': missing,
    }


def project_in_strength(
    unprojected: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """Project each target strength onto [o_i, o_i + n_i^m]."""
    lower = features[:, 1]
    upper = lower + features[:, 4]
    return np.clip(np.asarray(unprojected, float), lower, upper)


def normalized_weights(c: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Normalize in-strengths using the additive epsilon in the paper."""
    shifted = np.asarray(c, float) + epsilon
    return shifted / shifted.sum()


@dataclass(frozen=True)
class ASGCModel:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    ridge_intercept: float
    ridge_coefficients: np.ndarray
    ridge_alpha: float = 1.0

    @classmethod
    def load(cls, path) -> 'ASGCModel':
        """Load a saved StandardScaler/Ridge parameter bundle."""
        values = np.load(path, allow_pickle=False)
        return cls(
            values['scaler_mean'],
            values['scaler_scale'],
            float(np.ravel(values['ridge_intercept'])[0]),
            values['ridge_coefficients'],
            float(np.ravel(values['ridge_alpha'])[0]),
        )

    def apply(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Apply residual calibration and the node-wise feasible projection."""
        x = np.asarray(features['x'], float)
        standardized = (x - self.scaler_mean) / self.scaler_scale
        correction = (
            self.ridge_intercept + standardized @ self.ridge_coefficients
        )
        unprojected = np.asarray(features['raw_total']) + correction
        return project_in_strength(unprojected, x)


def fit_asgc(
    feature_blocks: list[np.ndarray],
    residual_blocks: list[np.ndarray],
    alpha: float = 1.0,
) -> tuple[StandardScaler, Ridge]:
    """Fit the Ridge residual model on development samples."""
    x = np.vstack(feature_blocks)
    y = np.concatenate(residual_blocks)
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=alpha, fit_intercept=True).fit(
        scaler.transform(x), y
    )
    return scaler, ridge
