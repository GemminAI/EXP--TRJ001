from setup_phase0_configs import build_generation_config, build_mask_protocol


def test_generation_config_matches_plan_2_1():
    cfg = build_generation_config()
    assert cfg["model"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert cfg["sampling"] == {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 200,
    }
    assert cfg["layers_saved"] == [8, 16, 24]
    assert cfg["primary_layer"] == 16
    # Appendix B item 1: explicitly unconfirmed, must not be fabricated.
    assert cfg["model_commit_hash"] is None
    assert cfg["inference_engine_version"] is None


def test_mask_protocol_matches_plan_7_4():
    proto = build_mask_protocol()
    assert proto["mask_seed"] == 20260912
    assert proto["interpolation"]["levels"] == [0.20, 0.35, 0.50]
    assert proto["extrapolation"]["fraction"] == 0.20
