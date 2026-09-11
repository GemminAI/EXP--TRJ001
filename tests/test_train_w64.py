import numpy as np
import pytest

from train_w64 import N_COMPONENTS, SOURCE_DIMENSION, fit_pca_projection, project


@pytest.fixture
def synthetic_hidden_states():
    rng = np.random.default_rng(0)
    return rng.standard_normal((600, SOURCE_DIMENSION)).astype(np.float32)


def test_shapes(synthetic_hidden_states):
    mu, W, diagnostics = fit_pca_projection(synthetic_hidden_states)
    assert mu.shape == (SOURCE_DIMENSION,)
    assert W.shape == (SOURCE_DIMENSION, N_COMPONENTS)
    assert diagnostics.total_samples == 600


def test_diagnostics_are_monotonic_and_bounded(synthetic_hidden_states):
    _, _, diagnostics = fit_pca_projection(synthetic_hidden_states)
    assert 0.0 <= diagnostics.pc1_3_explained_variance
    assert diagnostics.pc1_3_explained_variance <= diagnostics.pc1_64_explained_variance
    assert diagnostics.pc1_64_explained_variance <= diagnostics.pc1_128_explained_variance
    assert diagnostics.pc1_128_explained_variance <= 1.0


def test_mu_matches_column_means(synthetic_hidden_states):
    mu, _, _ = fit_pca_projection(synthetic_hidden_states)
    assert np.allclose(mu, synthetic_hidden_states.mean(axis=0), atol=1e-4)


def test_project_output_shape_and_centering(synthetic_hidden_states):
    mu, W, _ = fit_pca_projection(synthetic_hidden_states)
    z = project(synthetic_hidden_states, mu, W)
    assert z.shape == (600, N_COMPONENTS)

    # Projecting mu itself must land at the origin (mu is subtracted first).
    z_mu = project(mu, mu, W)
    assert np.allclose(z_mu, 0.0, atol=1e-3)


def test_wrong_dimension_raises():
    bad = np.zeros((100, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="4096"):
        fit_pca_projection(bad)


def test_too_few_samples_raises():
    rng = np.random.default_rng(1)
    too_few = rng.standard_normal((50, SOURCE_DIMENSION)).astype(np.float32)
    with pytest.raises(ValueError, match="more samples"):
        fit_pca_projection(too_few, n_components=128)
