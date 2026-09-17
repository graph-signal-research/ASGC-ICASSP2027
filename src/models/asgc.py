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
    observed_labels: np.ndarray,
    observed: np.ndarray,
    missing: np.ndarray,
    completed: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build ASGC features using only observable candidate-pair quantities.

    Matrices use the convention ``[source, target]``. ``observed_labels`` may
    contain placeholders (for example NaN) outside ``observed``; those entries
    are never read. ``missing`` defines the candidate pairs supplied by the
    completion model and is independent of the unknown ground-truth support.
    """
    observed_labels = np.asarray(observed_labels, float)
    observed = np.asarray(observed, bool)
    missing = np.asarray(missing, bool)
    completed = np.asarray(completed, float)
    if not (
        observed_labels.shape == observed.shape == missing.shape == completed.shape
    ):
        raise ValueError('ASGC feature matrices must have identical shapes')
    if observed_labels.ndim != 2 or observed_labels.shape[0] != observed_labels.shape[1]:
        raise ValueError('ASGC feature matrices must be square')
    if np.any(observed & missing):
        raise ValueError('Observed and missing candidate masks must be disjoint')
    if np.any(~np.isfinite(observed_labels[observed])):
        raise ValueError('Observed candidate labels must be finite')
    if np.any(~np.isfinite(completed[missing])):
        raise ValueError('Predicted missing-candidate values must be finite')

    observed_mass = np.sum(np.where(observed, observed_labels, 0.0), axis=0)
    predicted_mass = np.sum(np.where(missing, completed, 0.0), axis=0)
    preliminary_total = observed_mass + predicted_mass

    observed_count = observed.sum(axis=0).astype(float)
    missing_count = missing.sum(axis=0).astype(float)
    observed_mean = observed_mass / np.maximum(observed_count, 1.0)
    missing_mean = predicted_mass / np.maximum(missing_count, 1.0)

    missing_std = np.asarray(
        [
            np.std(completed[:, j][missing[:, j]])
            if np.any(missing[:, j])
            else 0.0
            for j in range(observed_labels.shape[0])
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
