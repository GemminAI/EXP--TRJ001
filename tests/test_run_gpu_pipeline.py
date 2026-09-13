from pathlib import Path

import numpy as np
import pytest

import collect_trajectories
import run_gpu_pipeline as pipeline


def test_log_gpu_environment_writes_file(tmp_path):
    out = tmp_path / "env_info.json"
    info = pipeline.log_gpu_environment(output_path=out)
    assert out.exists()
    assert "python_version" in info
    assert "cuda_available" in info


def test_compute_file_sha256_matches_known_hash(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello world")
    import hashlib

    expected = hashlib.sha256(b"hello world").hexdigest()
    assert pipeline.compute_file_sha256(f) == expected


def test_extract_hidden_states_refuses_without_gpu(monkeypatch):
    # No-fabrication contract (preserved from before real vLLM support
    # existed): without a GPU/vLLM available, extract_hidden_states must
    # raise loudly, never synthesize data.
    monkeypatch.setattr(pipeline, "vllm", None)
    monkeypatch.setattr(pipeline, "_ENGINE", None)
    with pytest.raises(RuntimeError):
        pipeline.extract_hidden_states(["some prompt"])


@pytest.mark.skipif(not pipeline._gpu_available(), reason="requires a real GPU + vLLM install")
def test_extract_hidden_states_produces_real_hidden_states():
    # Real end-to-end check (this environment has a GPU): exercises the
    # actual vLLM extraction path (see _VLLMHiddenStateEngine), not a mock.
    # Slow (~model load) by nature of testing real hardware-backed behavior.
    result = pipeline.extract_hidden_states(["What is the capital of France?"], layers=(16,))
    assert len(result) == 1
    arr = result[0][16]
    assert arr.ndim == 2
    assert arr.shape[1] == pipeline.SOURCE_DIMENSION
    assert arr.shape[0] > 0


def test_phase0_calibration_fails_loudly_without_fabricating(monkeypatch, tmp_path):
    # Confirms it still does NOT fall back to writing synthetic data into
    # the real configs/W_pca128.npy / configs/mu_pca.npy paths, now that
    # extract_hidden_states is real: stub it (fast, hardware-independent)
    # with too few tokens to fit a 128-component PCA, so the real
    # fit_pca_projection() raises ValueError -- that loud failure, not a
    # silent synthetic fallback, is exactly the contract under test.
    def fake_extract(prompt_texts, *, layers=(16,)):
        return [{16: np.zeros((3, pipeline.SOURCE_DIMENSION), dtype=np.float32)} for _ in prompt_texts]

    monkeypatch.setattr(pipeline, "extract_hidden_states", fake_extract)

    calib_path = tmp_path / "calibration_prompts_v1.json"
    calib_path.write_text('[{"prompt_id": "c1", "text": "hello"}]', encoding="utf-8")
    with pytest.raises(ValueError):
        pipeline.execute_phase0_calibration(calibration_prompts_path=calib_path)
    assert not (tmp_path / "W_pca128.npy").exists()


def test_phase1_pilot_fails_loudly_without_real_generation(monkeypatch, tmp_path):
    # Frozen projection missing -> FileNotFoundError, must fail loudly and
    # never write a fabricated results file. Pin the collector's lookup
    # paths to guaranteed-missing tmp locations rather than relying on this
    # repo's real configs/W_pca128.npy happening not to exist yet (it will,
    # after a real Phase 0 run).
    monkeypatch.setattr(collect_trajectories, "DEFAULT_W_PATH", tmp_path / "missing_W.npy")
    monkeypatch.setattr(collect_trajectories, "DEFAULT_MU_PATH", tmp_path / "missing_mu.npy")

    with pytest.raises(FileNotFoundError):
        pipeline.execute_phase1_pilot(output_path=tmp_path / "phase1_pilot_trajectories.json")
    assert not (tmp_path / "phase1_pilot_trajectories.json").exists()
