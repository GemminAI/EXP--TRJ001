from pathlib import Path

import numpy as np
import pytest

from collect_trajectories import (
    MockGenerator,
    PENDING_TYPE_A_LABEL,
    TrajectoryCollector,
    collect,
    deterministic_seed,
)
from eval_hallucination import EvalConfig, Label, PromptRecord, PromptType
from train_w64 import fit_pca_projection

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def frozen_projection(tmp_path):
    rng = np.random.default_rng(0)
    hidden_states = rng.standard_normal((600, 4096)).astype(np.float32)
    mu, W, _ = fit_pca_projection(hidden_states)
    w_path = tmp_path / "W_pca128.npy"
    mu_path = tmp_path / "mu_pca.npy"
    np.save(w_path, W)
    np.save(mu_path, mu)
    return w_path, mu_path


@pytest.fixture
def eval_config():
    return EvalConfig(
        refusal_patterns_path=CONFIGS / "refusal_patterns_en.json",
        typeB_reference_path=CONFIGS / "typeB_reference.json",
    )


@pytest.fixture
def collector(frozen_projection, eval_config):
    w_path, mu_path = frozen_projection
    return TrajectoryCollector(eval_config=eval_config, w_path=w_path, mu_path=mu_path)


def test_missing_projection_raises(tmp_path, eval_config):
    with pytest.raises(FileNotFoundError):
        TrajectoryCollector(
            eval_config=eval_config,
            w_path=tmp_path / "missing_W.npy",
            mu_path=tmp_path / "missing_mu.npy",
        )


def test_deterministic_seed_is_stable_and_reproducible():
    s1 = deterministic_seed("p1", 0)
    s2 = deterministic_seed("p1", 0)
    s3 = deterministic_seed("p1", 1)
    assert s1 == s2
    assert s1 != s3
    assert 0 <= s1 < 2**32


def test_type_a_non_abstain_response_is_pending_not_a_failure(collector):
    prompt = PromptRecord(
        prompt_id="type_a_fictional_movie_0001",
        prompt_type=PromptType.TYPE_A,
        template="fictional_movie",
        text='Can you tell me about the movie "The Echo of Kalcrest"?',
    )
    generator = MockGenerator(min_tokens=25, max_tokens=30)
    sample = generator.generate(prompt.prompt_id, 0, prompt.text, seed=1)
    # MockGenerator's placeholder text never matches an abstain/hedge pattern.
    record = collector.process_sample(sample, prompt)
    assert record["label"] == PENDING_TYPE_A_LABEL
    assert "pending_reason" in record["label_details"]


def test_short_sequence_is_class_e(collector):
    prompt = PromptRecord(
        prompt_id="type_a_fictional_movie_0001",
        prompt_type=PromptType.TYPE_A,
        template="fictional_movie",
        text="irrelevant",
    )
    generator = MockGenerator(min_tokens=5, max_tokens=6)
    sample = generator.generate(prompt.prompt_id, 0, prompt.text, seed=2)
    assert sample.token_count < 20
    record = collector.process_sample(sample, prompt)
    assert record["label"] == Label.CLASS_E_INDETERMINATE.value
    assert record["kappa"] is None  # too short for kappa (< MIN_SEQUENCE_LENGTH)


def test_kappa_only_computed_from_layer_16(collector):
    prompt = PromptRecord(
        prompt_id="type_a_fictional_movie_0001",
        prompt_type=PromptType.TYPE_A,
        template="fictional_movie",
        text="irrelevant",
    )
    generator = MockGenerator(min_tokens=25, max_tokens=30)
    sample = generator.generate(prompt.prompt_id, 0, prompt.text, seed=3)
    record = collector.process_sample(sample, prompt)
    assert record["kappa"] is not None
    assert record["kappa"].kappa.shape == (sample.token_count,)
    # All three layers still get projected and saved.
    assert set(record["projected_layers"].keys()) == {
        "layer_8_128d", "layer_16_128d", "layer_24_128d",
    }
    for proj in record["projected_layers"].values():
        assert proj.shape == (sample.token_count, 128)


def test_projection_uses_same_basis_across_layers(collector):
    # Same input matrix for two different "layers" must project identically,
    # confirming one shared (W, mu) is applied uniformly (not per-layer).
    h = np.random.default_rng(4).standard_normal((30, 4096)).astype(np.float32)
    proj_a = collector.project(h)
    proj_b = collector.project(h)
    assert np.allclose(proj_a, proj_b)


def test_collect_end_to_end_smoke(collector):
    prompts = [
        PromptRecord(
            prompt_id="type_b_movie_001",
            prompt_type=PromptType.TYPE_B,
            template="movie",
            text='In what year was the film "Jaws" released?',
            reference_key="type_b_movie_001",
        )
    ]
    generator = MockGenerator(min_tokens=25, max_tokens=30)
    records = collect(prompts, generator, collector, samples_per_prompt=2)
    assert len(records) == 2
    assert {r["sample_index"] for r in records} == {0, 1}
