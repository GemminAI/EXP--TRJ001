import numpy as np
import pytest

from kappa_features import compute_theta, kappa_features, norm_features


def test_kappa_features_basic_shapes_and_values():
    kappa = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
    valid_mask = np.array([False, True, True, True, False])
    feats = kappa_features(kappa, valid_mask, theta=2.5)
    assert feats.kappa_mean == pytest.approx(2.0)
    assert feats.kappa_max == pytest.approx(3.0)
    assert feats.kappa_std == pytest.approx(np.std([1.0, 2.0, 3.0]))
    # spike_rate: fraction of valid values > theta (2.5) -> only 3.0 -> 1/3
    assert feats.spike_rate == pytest.approx(1 / 3)
    # kappa_auc_density: sum / (T-2) = 6 / 3
    assert feats.kappa_auc_density == pytest.approx(2.0)


def test_kappa_features_raises_on_all_invalid():
    kappa = np.array([np.nan, np.nan])
    valid_mask = np.array([False, False])
    with pytest.raises(ValueError):
        kappa_features(kappa, valid_mask, theta=1.0)


def test_compute_theta_pools_across_samples():
    kappas = [np.array([np.nan, 1.0, 5.0, np.nan]), np.array([np.nan, 2.0, 3.0, np.nan])]
    masks = [np.array([False, True, True, False])] * 2
    theta = compute_theta(kappas, masks, percentile=90.0)
    expected = np.percentile([1.0, 5.0, 2.0, 3.0], 90.0)
    assert theta == pytest.approx(expected)


def test_norm_features_shape_and_values():
    z = np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])  # norms: 5, 0, 10
    feats = norm_features(z)
    assert feats.shape == (3,)
    assert feats[0] == pytest.approx(5.0)  # mean
    assert feats[1] == pytest.approx(10.0)  # max
    assert feats[2] == pytest.approx(np.std([5.0, 0.0, 10.0]))  # std
