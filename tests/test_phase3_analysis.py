import numpy as np
import pytest

from phase3_analysis import (
    cluster_bootstrap_auroc,
    cluster_bootstrap_delta_auroc,
    filter_mixed_prompts,
    holm_correct,
)


def make_record(prompt_id, sample_index, label, token_count=50):
    return {
        "prompt_id": prompt_id,
        "sample_index": sample_index,
        "label": label,
        "token_count": token_count,
    }


class TestFilterMixedPrompts:
    def test_drops_all_positive_prompt(self):
        records = [make_record("p1", i, "type_a_positive") for i in range(8)]
        assert filter_mixed_prompts(records) == []

    def test_drops_all_negative_prompt(self):
        records = [make_record("p1", i, "type_c_abstain") for i in range(8)]
        assert filter_mixed_prompts(records) == []

    def test_keeps_mixed_prompt(self):
        records = [make_record("p1", i, "type_a_positive" if i < 4 else "type_c_abstain") for i in range(8)]
        kept = filter_mixed_prompts(records)
        assert len(kept) == 8

    def test_drops_class_e_samples_but_keeps_rest_of_mixed_prompt(self):
        records = [
            make_record("p1", 0, "type_a_positive"),
            make_record("p1", 1, "type_c_abstain"),
            make_record("p1", 2, "class_e_indeterminate"),
        ]
        kept = filter_mixed_prompts(records)
        assert len(kept) == 2
        assert all(r["label"] != "class_e_indeterminate" for r in kept)

    def test_hedge_counts_as_negative(self):
        records = [
            make_record("p1", 0, "type_a_positive"),
            make_record("p1", 1, "type_d_hedge"),
        ]
        assert len(filter_mixed_prompts(records)) == 2

    def test_only_class_e_prompt_fully_dropped(self):
        records = [make_record("p1", i, "class_e_indeterminate") for i in range(8)]
        assert filter_mixed_prompts(records) == []

    def test_independent_prompts_evaluated_separately(self):
        mixed = [make_record("p1", i, "type_a_positive" if i < 4 else "type_c_abstain") for i in range(8)]
        all_pos = [make_record("p2", i, "type_a_positive") for i in range(8)]
        kept = filter_mixed_prompts(mixed + all_pos)
        assert {r["prompt_id"] for r in kept} == {"p1"}


def test_holm_correct_matches_manual_two_comparison_case():
    # Two p-values: smaller gets multiplied by 2, larger by 1, then
    # monotonicity enforced (Holm step-down).
    p = [0.01, 0.04]
    adj = holm_correct(p)
    assert adj[0] == pytest.approx(0.02)
    assert adj[1] == pytest.approx(0.04)


def test_holm_correct_enforces_monotonicity():
    # If the smaller-p's adjustment would exceed the larger-p's raw*1, the
    # running max must be carried forward.
    p = [0.03, 0.02]
    adj = holm_correct(p)
    # sorted order: 0.02 (idx1) *2=0.04, then 0.03 (idx0) *1=0.03 -> must be
    # raised to running max 0.04
    assert adj[1] == pytest.approx(0.04)
    assert adj[0] == pytest.approx(0.04)


def test_cluster_bootstrap_auroc_perfect_separation():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.1, 0.2, 0.2, 0.9, 0.9, 0.8, 0.8])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    result = cluster_bootstrap_auroc(y, scores, groups, n_resamples=200, seed=1)
    assert result["point_estimate"] == pytest.approx(1.0)
    assert result["ci95_lower"] > 0.9
    assert result["p_value_le_0.5"] == pytest.approx(0.0)


def test_cluster_bootstrap_delta_auroc_zero_when_identical_scores():
    y = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    result = cluster_bootstrap_delta_auroc(y, scores, scores, groups, n_resamples=100, seed=1)
    assert result["point_estimate"] == pytest.approx(0.0)
