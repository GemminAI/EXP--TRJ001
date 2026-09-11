from pathlib import Path

import pytest

from eval_hallucination import (
    EvalConfig,
    GenerationRecord,
    Label,
    PromptRecord,
    PromptType,
    TypeADecisionPending,
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


class TestTypeAPending:
    def test_non_abstain_type_a_response_raises_pending(self, config):
        prompt = make_type_a_prompt()
        gen = make_gen(
            prompt.prompt_id,
            "Alaric Ventmoor was a 19th-century cartographer known for his maps.",
        )
        with pytest.raises(TypeADecisionPending):
            evaluate(prompt, gen, config)

    def test_batch_does_not_swallow_pending(self, config):
        prompts = {"p_a1": make_type_a_prompt("p_a1")}
        gens = [make_gen("p_a1", "Alaric Ventmoor was a well-documented explorer.")]
        with pytest.raises(TypeADecisionPending):
            evaluate_batch(prompts, gens, config)


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
