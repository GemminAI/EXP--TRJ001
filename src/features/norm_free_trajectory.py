"""Norm/direction decoupling for TRJ002b's re-analysis
(ANALYSIS-TEMPORAL-TRIANGULATION-001 task 1).

TRJ002b's first pass showed BL-NORM (mean/max/std of ||z(t)||, no temporal
structure at all) scoring almost identically to M4/M5 -- consistent with
P_opt (a classification-trained Supervised Autoencoder whose loss acts on
the *mean-pooled* 64D encoding) having baked label-predictive information
directly into the latent norm. This module explicitly splits z(t) into:

    magnitude(t) = ||z(t)||_2                      (kept ONLY as the
                                                      unchanged BL-NORM
                                                      control feature)
    direction(t) = z(t) / ||z(t)||_2  (unit vector, lies on S^63)

so M4-NF / M5-NF never see magnitude at all -- only the direction
trajectory ẑ(t) -- closing off the norm-leakage path identified in the
first TRJ002b run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from compute_kappa import compute_kappa  # noqa: E402

EPS = 1e-8


def decompose(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z: (T, 64) -> (magnitude (T,), direction (T, 64) unit vectors)."""
    magnitude = np.linalg.norm(z, axis=1)
    direction = z / np.clip(magnitude, EPS, None)[:, None]
    return magnitude, direction


def kappa_cos_features(direction: np.ndarray, theta: float) -> np.ndarray:
    """kappa^cos: the same Gram-determinant curvature (compute_kappa.py,
    frozen in EXP--TRJ001 S3) applied to the *direction* trajectory ẑ(t)
    instead of the raw (norm-carrying) z(t) -- i.e. curvature of the purely
    angular/directional path, with magnitude information already removed
    before curvature is ever computed. Same 6-feature, length-normalized
    summary convention as TRJ002a S4.1 (kappa_mean/max/p95/std/spike_rate/
    auc_density) so it's directly comparable to TRJ002a's PROP-kappa."""
    T = direction.shape[0]
    res = compute_kappa(direction)
    vals = res.kappa[res.valid_mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.zeros(6)
    kappa_mean = float(np.mean(vals))
    kappa_max = float(np.max(vals))
    kappa_p95 = float(np.percentile(vals, 95))
    kappa_std = float(np.std(vals))
    spike_rate = float(np.mean(vals > theta)) if (T - 2) > 0 else 0.0
    kappa_auc_density = float(np.sum(vals) / max(T - 2, 1))
    return np.array(
        [kappa_mean, kappa_max, kappa_p95, kappa_std, spike_rate, kappa_auc_density]
    )


def kappa_cos_theta_from_train(train_directions: list[np.ndarray]) -> float:
    """90th percentile of kappa^cos(t) pooled over Train direction
    trajectories -- same fixed-from-Train convention as TRJ002a S4.1."""
    all_kappa = []
    for direction in train_directions:
        if direction.shape[0] < 3:
            continue
        res = compute_kappa(direction)
        vals = res.kappa[res.valid_mask]
        vals = vals[~np.isnan(vals)]
        all_kappa.append(vals)
    pooled = np.concatenate(all_kappa) if all_kappa else np.array([0.0])
    return float(np.percentile(pooled, 90))


def direction_arc_length(direction: np.ndarray) -> np.ndarray:
    """Cumulative chord length along the unit-sphere direction trajectory
    (the M4-NF input) -- purely angular by construction since
    ||direction(t)|| == 1 for all t, so magnitude cannot leak in here."""
    diffs = np.linalg.norm(np.diff(direction, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])
