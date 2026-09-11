import numpy as np
import pytest

from compute_kappa import (
    EPSILON,
    MIN_SEQUENCE_LENGTH,
    compute_kappa,
    compute_kappa_batch,
    compute_velocity_acceleration,
    is_analysis_eligible,
)


def test_endpoints_are_undefined():
    z = np.random.default_rng(0).normal(size=(10, 4))
    result = compute_kappa(z)
    assert np.isnan(result.kappa[0])
    assert np.isnan(result.kappa[-1])
    assert not result.valid_mask[0]
    assert not result.valid_mask[-1]
    assert result.valid_mask[1:-1].all()


def test_straight_line_has_zero_curvature():
    # A straight line has zero acceleration everywhere -> det G = 0 -> kappa = 0.
    T, D = 15, 8
    t = np.arange(T).reshape(-1, 1).astype(np.float64)
    direction = np.ones((1, D))
    z = t * direction  # constant velocity, zero acceleration
    result = compute_kappa(z)
    valid = result.kappa[result.valid_mask]
    assert np.allclose(valid, 0.0, atol=1e-10)


def test_constant_sequence_uses_epsilon_not_division_by_zero():
    # Zero velocity AND zero acceleration everywhere: det G = 0, ||v|| = 0.
    # kappa = sqrt(0) / (0 + eps) = 0, not NaN/inf.
    z = np.zeros((10, 4))
    result = compute_kappa(z)
    valid = result.kappa[result.valid_mask]
    assert np.all(np.isfinite(valid))
    assert np.allclose(valid, 0.0)


def test_curved_path_has_positive_curvature():
    # A circular arc has nonzero, well-defined curvature everywhere interior.
    T = 30
    theta = np.linspace(0, np.pi, T)
    z = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    result = compute_kappa(z)
    valid = result.kappa[result.valid_mask]
    assert np.all(valid > 0)
    assert np.all(np.isfinite(valid))


def test_generic_dimension_ablation_reuse():
    # Same function must work unmodified for D=4096 (ABL-1) and D=64 (main
    # analysis / ABL-2 PC4-67) -- §6.3 requires no separate code path.
    rng = np.random.default_rng(1)
    for D in (64, 4096):
        z = rng.normal(size=(25, D))
        result = compute_kappa(z)
        assert result.kappa.shape == (25,)
        assert result.velocity.shape == (25, D)


def test_short_sequence_below_three_is_all_nan():
    z = np.zeros((2, 4))
    result = compute_kappa(z)
    assert np.isnan(result.kappa).all()
    assert not result.valid_mask.any()


def test_epsilon_value_is_frozen():
    assert EPSILON == 1e-8


def test_is_analysis_eligible_matches_min_sequence_length():
    assert MIN_SEQUENCE_LENGTH == 20
    assert not is_analysis_eligible(19)
    assert is_analysis_eligible(20)


def test_compute_velocity_acceleration_matches_manual_central_difference():
    z = np.array([[0.0], [1.0], [3.0], [6.0], [10.0]])
    v, a = compute_velocity_acceleration(z)
    # v(t) = (z(t+1) - z(t-1)) / 2 at index 1: (3 - 0)/2 = 1.5
    assert v[1, 0] == pytest.approx(1.5)
    # a(t) = z(t+1) - 2z(t) + z(t-1) at index 1: 3 - 2*1 + 0 = 1
    assert a[1, 0] == pytest.approx(1.0)
    assert np.isnan(v[0]).all() and np.isnan(v[-1]).all()


def test_compute_kappa_batch_handles_ragged_lengths():
    rng = np.random.default_rng(2)
    sequences = [rng.normal(size=(T, 4)) for T in (5, 20, 3)]
    results = compute_kappa_batch(sequences)
    assert len(results) == 3
    assert results[0].kappa.shape == (5,)
    assert results[2].kappa.shape == (3,)
