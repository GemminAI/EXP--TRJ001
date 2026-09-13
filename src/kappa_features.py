"""κ-derived and baseline feature engineering for EXP-2026-NVS-001 (plan
v1.4 §4).

All features are length-normalized (densities/rates), per §4.2 -- no raw
sums or counts. `theta` (the spike threshold) is a single fixed number
computed once, from the Train split's pooled kappa(t) distribution (§4.1),
and then applied unchanged to Val/Test -- callers must compute it once via
`compute_theta` and pass it into `kappa_features` for every split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KappaFeatures:
    kappa_mean: float
    kappa_max: float
    kappa_p95: float
    kappa_std: float
    spike_rate: float
    kappa_auc_density: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.kappa_mean, self.kappa_max, self.kappa_p95, self.kappa_std, self.spike_rate, self.kappa_auc_density]
        )

    @staticmethod
    def names() -> list[str]:
        return ["kappa_mean", "kappa_max", "kappa_p95", "kappa_std", "spike_rate", "kappa_auc_density"]


def _valid_kappa(kappa: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    kappa = np.asarray(kappa, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    vals = kappa[valid_mask]
    if vals.size == 0:
        raise ValueError("no valid (non-endpoint) kappa values -- sequence too short")
    return vals


def compute_theta(kappa_arrays: list[np.ndarray], valid_masks: list[np.ndarray], percentile: float = 90.0) -> float:
    """§4.1: theta = the Train split's pooled kappa(t) 90th percentile.
    Pools every valid (internal-token, non-Class-E) kappa(t) value across
    all Train-split samples into one distribution before taking the
    percentile -- not a per-sample or per-prompt percentile."""
    pooled = np.concatenate([_valid_kappa(k, m) for k, m in zip(kappa_arrays, valid_masks)])
    return float(np.percentile(pooled, percentile))


def kappa_features(kappa: np.ndarray, valid_mask: np.ndarray, theta: float) -> KappaFeatures:
    vals = _valid_kappa(kappa, valid_mask)
    t_minus_2 = vals.size  # number of valid (internal) positions == T-2
    spike_rate = float(np.mean(vals > theta))
    return KappaFeatures(
        kappa_mean=float(np.mean(vals)),
        kappa_max=float(np.max(vals)),
        kappa_p95=float(np.percentile(vals, 95)),
        kappa_std=float(np.std(vals)),
        spike_rate=spike_rate,
        kappa_auc_density=float(np.sum(vals) / t_minus_2),
    )


def norm_features(z: np.ndarray) -> np.ndarray:
    """BL-NORM (§4.3): mean/max/std of ||z(t)|| across all T token
    positions (endpoints included -- unlike kappa, the norm is defined
    everywhere)."""
    z = np.asarray(z, dtype=np.float64)
    norms = np.linalg.norm(z, axis=-1)
    return np.array([np.mean(norms), np.max(norms), np.std(norms)])
