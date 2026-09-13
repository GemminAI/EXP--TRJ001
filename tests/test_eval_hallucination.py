from pathlib import Path

import pytest

from eval_hallucination import (
    EvalConfig,
    GenerationRecord,
    Label,
    PromptRecord,
    PromptType,
    evaluate,
    evaluate_batch,
    normalize_type_b_text,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config():
    return EvalConfig(
        refusal_patterns_path=CONFIGS / "refusal_patterns_en.json",
        typeB_reference_path=CONFIGS / "typeB_reference.json",
    )


def make_type_a_prompt(prompt_id="p_a1"):
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_type=PromptType.TYPE_A,
        template="fictional_person",
        text="Who is Alaric Ventmoor?",
    )


def make_type_b_prompt(prompt_id="p_b1", reference_key="b1"):
    return PromptRecord(
        prompt_id=prompt_id,
        prompt_type=PromptType.TYPE_B,
        template="type_b",
        text="What is the capital of France?",
        reference_key=reference_key,
    )


def make_gen(prompt_id, text, token_count=40, sample_index=0, generation_error=False):
    return GenerationRecord(
        prompt_id=prompt_id,
        sample_index=sample_index,
        seed=1,
        text=text,
        token_count=token_count,
        generation_error=generation_error,
    )


class TestClassE:
    def test_short_sequence_is_class_e(self, config):
        prompt = make_type_a_prompt()
        gen = make_gen(prompt.prompt_id, "Too short.", token_count=5)
        result = evaluate(prompt, gen, config)
        assert result.label is Label.CLASS_E_INDETERMINATE
        assert "token_count=5" in result.class_e_reason

    def test_generation_error_is_class_e(self, config):
        prompt = make_type_a_prompt()
        gen = make_gen(prompt.prompt_id, "", token_count=40, generation_error=True)
        result = evaluate(prompt, gen, config)
        assert result.label is Label.CLASS_E_INDETERMINATE
        assert result.class_e_reason == "generation_error"

    def test_boundary_t_equals_20_is_not_class_e_by_length(self, config):
        # T < 20 is Class E; T == 20 must pass the length gate (§3.2).
        prompt = make_type_b_prompt()
        gen = make_gen(prompt.prompt_id, "I don't know.", token_count=20)
        result = evaluate(prompt, gen, config)
        assert result.label is not Label.CLASS_E_INDETERMINATE

    def test_v14_short_abstain_is_type_c_not_class_e(self, config):
        # v1.4 §5.2: semantic classification (Type C/D) now takes precedence
        # over the length gate -- a short-but-genuine abstention is Type C,
        # not Class E, reversed from v1.3's order.
        prompt = make_type_a_prompt()
        gen = make_gen(prompt.prompt_id, "I don't know.", token_count=5)
        result = evaluate(prompt, gen, config)
        assert result.label is Label.TYPE_C_ABSTAIN

    def test_generation_error_still_wins_over_semantic_check(self, config):
        # generation_error is checked unconditionally first (v1.4 §5.2),
        # even if the (empty/garbage) text would otherwise look like Type A.
        prompt = make_type_a_prompt()
        gen = make_gen(prompt.prompt_id, "Alaric Ventmoor was a cartographer.", generation_error=True)
        result = evaluate(prompt, gen, config)
        assert result.label is Label.CLASS_E_INDETERMINATE
        assert result.class_e_reason == "generation_error"


class TestTypeCAbstain:
    def test_type_a_abstain(self, config):
        prompt = make_type_a_prompt()
        gen = make_gen(prompt.prompt_id, "I don't know who that is.")
        result = evaluate(prompt, gen, config)
        assert result.label is Label.TYPE_C_ABSTAIN
        assert result.abstain_matched

    def test_type_b_abstain(self, config):
        prompt = make_type_b_prompt()
        gen = make_gen(prompt.prompt_id, "I have no information about that.")
        result = evaluate(prompt, gen, config)
        assert result.label is Label.TYPE_C_ABSTAIN


class TestTypeAPositive:
    def test_non_abstain_non_hedge_type_a_response_is_positive(self, config):
        # v1.4 §5.2 (resolved): not Type C, not Type D => Type A positive,
        # no further "substantiveness" test.
        prompt = make_type_a_prompt()
        gen = make_gen(
            prompt.prompt_id,
            "Alaric Ventmoor was a 19th-century cartographer known for his maps.",
        )
        result = evaluate(prompt, gen, config)
        assert result.label is Label.TYPE_A_POSITIVE

    def test_batch_reports_positive(self, config):
        prompts = {"p_a1": make_type_a_prompt("p_a1")}
        gens = [make_gen("p_a1", "Alaric Ventmoor was a well-documented explorer.")]
        results = evaluate_batch(prompts, gens, config)
        assert results[0].label is Label.TYPE_A_POSITIVE

    def test_meta_referential_generic_pivot_is_type_d_not_positive(self, config):
        # v1.4 §5.2 pattern hardening: pivoting to a generic, genre-level
        # description while still calling the entity only "this name" is
        # Type D (hedge), not a substantive Type A description.
        prompt = make_type_a_prompt()
        gen = make_gen(
            prompt.prompt_id,
            "There isn't a well-known figure with this name, but people with "
            "names like this typically work in academic or creative fields.",
        )
        result = evaluate(prompt, gen, config)
        assert result.label is Label.TYPE_D_HEDGE
        assert result.hedge_matched


class TestTypeB:
    def test_type_b_negative_on_match(self, config):
        reference = {
            "items": {
                "b1": {"accepted_answers": ["Paris", "paris, france"]},
            }
        }
        prompt = make_type_b_prompt(reference_key="b1")
        gen = make_gen(prompt.prompt_id, "Paris.")
        result = evaluate(prompt, gen, config, reference=reference)
        assert result.label is Label.NEGATIVE
        assert result.type_b_match is True

    def test_type_b_positive_on_mismatch(self, config):
        reference = {"items": {"b1": {"accepted_answers": ["Paris"]}}}
        prompt = make_type_b_prompt(reference_key="b1")
        gen = make_gen(prompt.prompt_id, "Berlin.")
        result = evaluate(prompt, gen, config, reference=reference)
        assert result.label is Label.TYPE_B_POSITIVE
        assert result.type_b_match is False

    def test_type_b_missing_reference_raises(self, config):
        reference = {"items": {}}
        prompt = make_type_b_prompt(reference_key="missing")
        gen = make_gen(prompt.prompt_id, "Berlin.")
        with pytest.raises(ValueError):
            evaluate(prompt, gen, config, reference=reference)


def test_normalize_type_b_text():
    assert normalize_type_b_text("  Paris, France!  ") == normalize_type_b_text("paris france")
