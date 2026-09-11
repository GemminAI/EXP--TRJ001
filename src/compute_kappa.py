"""kappa (generalized curvature) computation for EXP-2026-NVS-001 (plan v1.3 §3).

Implements §3.1's Gram-determinant generalization of curvature (ordinary
3D cross-product curvature is undefined for d != 3, 7) on a discrete
trajectory z(t) in R^D (D=64 for the main PC1-64 analysis; the same
formula applies unchanged to the 4096D ABL-1 ablation and the 64D PC4-67
ABL-2 ablation, §6.3 -- no separate code path is needed for those).

    v(t) = ( z(t+1) - z(t-1) ) / 2
    a(t) = z(t+1) - 2 z(t) + z(t-1)
    G(t)     = [[ <v,v>, <v,a> ], [ <a,v>, <a,a> ]]
    det G(t) = ||v||^2 ||a||^2 - <v,a>^2
    kappa(t) = sqrt( det G(t) ) / ( ||v(t)||^3 + eps )

Parameterization is by index time (token position), not arc length -- this
is an explicit, non-negotiable pre-registered choice (§3.2): arc-length
reparameterization would remove the very token-to-token speed variation
the plan wants kept as signal, and the plan states this choice may not be
changed post hoc. Do not add an arc-length mode here.

Endpoints (t=1, t=T) are undefined by construction -- central differences
need both neighbors -- and are always excluded, independent of the
separate T < 20 Class-E exclusion (§3.2/§5.2), which is an analysis-level
rule about whole short sequences, not about the two endpoint positions of
an otherwise-long one. `MIN_SEQUENCE_LENGTH` is defined here (rather than
duplicated) and `eval_hallucination.py` imports it, so the two modules
cannot silently drift out of sync on this frozen threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-8  # §3.1, frozen
MIN_SEQUENCE_LENGTH = 20  # §3.2 / §5.2: T < 20 -> Class E, excluded from analysis


@dataclass
class KappaResult:
    kappa: np.ndarray  # shape (T,); NaN at undefined (endpoint) positions
    velocity: np.ndarray  # shape (T, D); NaN at endpoints
    acceleration: np.ndarray  # shape (T, D); NaN at endpoints
    valid_mask: np.ndarray  # shape (T,) bool; True where kappa is defined


def compute_velocity_acceleration(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """§3.1 central differences.

    z: shape (T, D). Returns (v, a), each shape (T, D). Rows 0 and T-1
    (0-indexed; token positions t=1 and t=T) are NaN -- central
    differences need z(t-1) and z(t+1), which do not exist there.
    """
    z = np.asarray(z, dtype=np.float64)
    T = z.shape[0]
    v = np.full_like(z, np.nan)
    a = np.full_like(z, np.nan)
    if T < 3:
        return v, a
    v[1:-1] = (z[2:] - z[:-2]) / 2.0
    a[1:-1] = z[2:] - 2 * z[1:-1] + z[:-2]
    return v, a


def compute_kappa(z: np.ndarray, epsilon: float = EPSILON) -> KappaResult:
    """§3.1: generalized curvature via the 2x2 Gram determinant.

    z: shape (T, D), a single trajectory (any D -- see module docstring).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"z must be 2D (T, D), got shape {z.shape}")

    T = z.shape[0]
    v, a = compute_velocity_acceleration(z)

    valid_mask = np.zeros(T, dtype=bool)
    if T >= 3:
        valid_mask[1:-1] = True

    vv = np.einsum("td,td->t", v, v, optimize=True)
    aa = np.einsum("td,td->t", a, a, optimize=True)
    va = np.einsum("td,td->t", v, a, optimize=True)

    det_g = vv * aa - va**2
    # Numerical guard only, not a modeling choice: det_g = ||v||^2||a||^2 -
    # <v,a>^2 is a Gram determinant, which Cauchy-Schwarz guarantees is >= 0
    # for real vectors. Floating-point cancellation can push it a hair below
    # zero for near-zero curvature; clip before sqrt so that case reads as
    # kappa ~ 0, not NaN (which would be indistinguishable from a real
    # endpoint/undefined position).
    det_g = np.clip(det_g, 0.0, None)

    speed = np.sqrt(vv)
    with np.errstate(invalid="ignore"):
        kappa = np.sqrt(det_g) / (speed**3 + epsilon)
    kappa = np.where(valid_mask, kappa, np.nan)

    return KappaResult(kappa=kappa, velocity=v, acceleration=a, valid_mask=valid_mask)


def is_analysis_eligible(token_count: int, min_length: int = MIN_SEQUENCE_LENGTH) -> bool:
    """§3.2 / §5.2: sequences with T < min_length are Class E, excluded from
    analysis. This does not compute or touch kappa -- it only reports
    eligibility so callers (e.g. eval_hallucination.py) share one threshold.
    """
    return token_count >= min_length


def compute_kappa_batch(sequences: list[np.ndarray], epsilon: float = EPSILON) -> list[KappaResult]:
    """Convenience wrapper for a batch of (possibly ragged-length) trajectories."""
    return [compute_kappa(z, epsilon=epsilon) for z in sequences]
