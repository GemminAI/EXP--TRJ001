from pathlib import Path

import pytest

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


def test_extract_hidden_states_refuses_rather_than_fabricates():
    # Must raise, never silently return synthetic data for the real path.
    with pytest.raises(NotImplementedError):
        pipeline.extract_hidden_states(["some prompt"])


def test_phase0_calibration_fails_loudly_without_real_extraction(monkeypatch, tmp_path):
    # Confirms it does NOT fall back to writing synthetic data into the real
    # configs/W_pca128.npy / configs/mu_pca.npy paths.
    calib_path = tmp_path / "calibration_prompts_v1.json"
    calib_path.write_text('[{"prompt_id": "c1", "text": "hello"}]', encoding="utf-8")
    with pytest.raises(NotImplementedError):
        pipeline.execute_phase0_calibration(calibration_prompts_path=calib_path)


def test_phase1_pilot_fails_loudly_without_real_generation(tmp_path):
    # Either the frozen projection doesn't exist yet (FileNotFoundError, if
    # Phase 0 calibration hasn't produced real W/mu), or it does and
    # RealGenerator.generate() correctly refuses to fabricate a generation
    # (NotImplementedError). Either way, this must fail loudly, not write a
    # fabricated results file.
    with pytest.raises((FileNotFoundError, NotImplementedError)):
        pipeline.execute_phase1_pilot(output_path=tmp_path / "phase1_pilot_trajectories.json")
    assert not (tmp_path / "phase1_pilot_trajectories.json").exists()
