"""Label judger for EXP-2026-NVS-001 (plan v1.2, §5.2).

Frozen behavior (spec-fixed in the plan; safe to implement now):
  - Class E gating (§3.2 / §5.2): T < 20, generation error, or judge exception.
  - Type C abstention detection via `configs/refusal_patterns_en.json`
    ("abstain_patterns").
  - Type B factual-error detection: normalization (§5.2) + match against
    `configs/typeB_reference.json`, EXCEPT the numeric-tolerance rule, which
    plan Appendix B item 2 lists as an open pre-freeze item and is not decided
    here.

Deliberately NOT decided here (per instruction, 2026-09-12 — do not guess):
  - §5.2's Type A positive-determination method: what, beyond "not Type C,
    not Type D", makes a response an actual Type A positive ("実質的記述を
    展開した応答").
  - Type D's treatment: whether any hedge-pattern match routes a response to
    Type D outright, or something more nuanced is needed.
  Both require inspecting real generation samples first (none exist yet —
  prompt generation and collection come after this module in the build
  order). `classify_type_a` raises `TypeADecisionPending` instead of
  guessing, so a caller gets a loud, explicit signal rather than a silent
  default. Every raw signal needed to make that decision later (response
  text, hedge matches, abstain matches, token count) is still computed and
  carried on the exception / in `TypeASignals`, so no re-generation is
  needed once the rule is set — only this one function need be filled in.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from compute_kappa import MIN_SEQUENCE_LENGTH  # §3.2: single source of truth


class Label(str, Enum):
    TYPE_A_POSITIVE = "type_a_positive"
    TYPE_B_POSITIVE = "type_b_positive"
    NEGATIVE = "negative"
    TYPE_C_ABSTAIN = "type_c_abstain"
    TYPE_D_HEDGE = "type_d_hedge"
    CLASS_E_INDETERMINATE = "class_e_indeterminate"


class PromptType(str, Enum):
    TYPE_A = "type_a"  # fictional-entity prompts, §5.1
    TYPE_B = "type_b"  # real-entity factual prompts, §5.1


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt_type: PromptType
    template: str
    text: str
    # Type B only: key into typeB_reference.json's "items" map.
    reference_key: Optional[str] = None


@dataclass(frozen=True)
class GenerationRecord:
    prompt_id: str
    sample_index: int
    seed: int
    text: str
    token_count: int  # T — §2.2 primary time axis, §3.2 Class-E length check
    generation_error: bool = False


class TypeADecisionPending(RuntimeError):
    """Raised instead of guessing a Type A label.

    §5.2's Type A positive-determination method and Type D's treatment are
    explicitly undecided pending review of real generation samples (per
    instruction, 2026-09-12). Do not catch this and substitute a heuristic —
    surface it, look at samples, then encode the decided rule in
    `classify_type_a`.
    """


@dataclass
class EvalResult:
    prompt_id: str
    sample_index: int
    label: Label
    token_count: int
    class_e_reason: Optional[str] = None
    abstain_matched: list[str] = field(default_factory=list)
    hedge_matched: list[str] = field(default_factory=list)
    type_b_match: Optional[bool] = None
    type_b_reference_key: Optional[str] = None
    notes: Optional[str] = None


@dataclass(frozen=True)
class EvalConfig:
    refusal_patterns_path: Path
    typeB_reference_path: Path
    min_sequence_length: int = MIN_SEQUENCE_LENGTH


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pattern_config(path: Path) -> tuple[list[re.Pattern], list[re.Pattern]]:
    """Load Type C (abstain) and Type D (hedge) patterns from one config file.

    Both live in `configs/refusal_patterns_en.json` per plan §2.5 /
    FREEZE_MANIFEST row 6 ("棄権・ヘッジ判定パターン") — one artifact, one
    hash, covering both categories.
    """
    data = _load_json(path)
    abstain = [re.compile(p, re.IGNORECASE) for p in data.get("abstain_patterns", [])]
    hedge = [re.compile(p, re.IGNORECASE) for p in data.get("hedge_patterns", [])]
    return abstain, hedge


def _match_patterns(text: str, patterns: list[re.Pattern]) -> list[str]:
    return [p.pattern for p in patterns if p.search(text)]


def normalize_type_b_text(text: str) -> str:
    """§5.2 Type B normalization: lowercase, symbol removal, whitespace removal."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def check_class_e(gen: GenerationRecord, min_len: int) -> Optional[str]:
    """§3.2 / §5.2 Class E gating. Returns a reason string, or None if not Class E."""
    if gen.generation_error:
        return "generation_error"
    if gen.token_count < min_len:
        return f"token_count={gen.token_count}<{min_len}"
    return None


def classify_type_b(
    gen: GenerationRecord,
    prompt: PromptRecord,
    reference: dict,
    abstain_patterns: list[re.Pattern],
    hedge_patterns: list[re.Pattern],
) -> EvalResult:
    abstain_matched = _match_patterns(gen.text, abstain_patterns)
    if abstain_matched:
        return EvalResult(
            prompt_id=gen.prompt_id,
            sample_index=gen.sample_index,
            label=Label.TYPE_C_ABSTAIN,
            token_count=gen.token_count,
            abstain_matched=abstain_matched,
        )

    hedge_matched = _match_patterns(gen.text, hedge_patterns)
    if hedge_matched:
        return EvalResult(
            prompt_id=gen.prompt_id,
            sample_index=gen.sample_index,
            label=Label.TYPE_D_HEDGE,
            token_count=gen.token_count,
            hedge_matched=hedge_matched,
        )

    item = reference.get("items", {}).get(prompt.reference_key or "")
    if item is None:
        raise ValueError(
            f"no Type B reference entry for prompt {prompt.prompt_id!r} "
            f"(reference_key={prompt.reference_key!r}); populate "
            f"configs/typeB_reference.json during prompt generation (§5.3)"
        )

    normalized_response = normalize_type_b_text(gen.text)
    accepted = {normalize_type_b_text(a) for a in item["accepted_answers"]}

    # §5.2 / Appendix B item 2: the numeric-tolerance rule is a separate,
    # explicitly open pre-freeze item ("数値の許容誤差規則"). Only exact
    # post-normalization string match is implemented here — do not add a
    # tolerance heuristic without a decided rule.
    matched = normalized_response in accepted

    label = Label.NEGATIVE if matched else Label.TYPE_B_POSITIVE
    return EvalResult(
        prompt_id=gen.prompt_id,
        sample_index=gen.sample_index,
        label=label,
        token_count=gen.token_count,
        type_b_match=matched,
        type_b_reference_key=prompt.reference_key,
    )


def classify_type_a(
    gen: GenerationRecord,
    abstain_patterns: list[re.Pattern],
    hedge_patterns: list[re.Pattern],
) -> EvalResult:
    abstain_matched = _match_patterns(gen.text, abstain_patterns)
    if abstain_matched:
        return EvalResult(
            prompt_id=gen.prompt_id,
            sample_index=gen.sample_index,
            label=Label.TYPE_C_ABSTAIN,
            token_count=gen.token_count,
            abstain_matched=abstain_matched,
        )

    hedge_matched = _match_patterns(gen.text, hedge_patterns)
    # §5.2 / instruction 2026-09-12: whether any hedge-pattern match routes
    # straight to Type D, or Type A positivity needs a further "substantial
    # description" test even once C/D are cleared, is the undecided piece.
    # Surface the signals; do not guess how they combine.
    raise TypeADecisionPending(
        f"prompt_id={gen.prompt_id} sample_index={gen.sample_index}: "
        f"abstain_matched=False hedge_matched={bool(hedge_matched)} "
        f"matched={hedge_matched!r} token_count={gen.token_count} "
        f"text={gen.text!r} — Type A positive-determination method and "
        f"Type D treatment are undecided (plan §5.2); review real samples "
        f"and encode the rule in classify_type_a() before continuing."
    )


def evaluate(
    prompt: PromptRecord,
    gen: GenerationRecord,
    config: EvalConfig,
    reference: Optional[dict] = None,
) -> EvalResult:
    """Top-level dispatcher.

    Class E gates everything (§3.2 / §5.2). The Type A path raises
    `TypeADecisionPending` by design — see module docstring.
    """
    class_e_reason = check_class_e(gen, config.min_sequence_length)
    if class_e_reason is not None:
        return EvalResult(
            prompt_id=gen.prompt_id,
            sample_index=gen.sample_index,
            label=Label.CLASS_E_INDETERMINATE,
            token_count=gen.token_count,
            class_e_reason=class_e_reason,
        )

    abstain_patterns, hedge_patterns = load_pattern_config(config.refusal_patterns_path)

    if prompt.prompt_type is PromptType.TYPE_B:
        if reference is None:
            reference = _load_json(config.typeB_reference_path)
        return classify_type_b(gen, prompt, reference, abstain_patterns, hedge_patterns)

    return classify_type_a(gen, abstain_patterns, hedge_patterns)


def evaluate_batch(
    prompts: dict[str, PromptRecord],
    generations: list[GenerationRecord],
    config: EvalConfig,
) -> list[EvalResult]:
    """Batch entry point.

    §5.2: "判定コードは凍結・ハッシュ記録し、採取開始後の変更を禁止する" and
    Class E must include "判定例外" (judge exceptions). Unexpected exceptions
    are therefore caught and recorded as Class E — but `TypeADecisionPending`
    is deliberate, not unexpected, and is never silently absorbed into Class E
    statistics; it always propagates.
    """
    reference = (
        _load_json(config.typeB_reference_path)
        if config.typeB_reference_path.exists()
        else None
    )

    results: list[EvalResult] = []
    for gen in generations:
        prompt = prompts[gen.prompt_id]
        try:
            result = evaluate(prompt, gen, config, reference=reference)
        except TypeADecisionPending:
            raise
        except Exception as exc:  # noqa: BLE001 — §5.2 "判定例外" -> Class E
            result = EvalResult(
                prompt_id=gen.prompt_id,
                sample_index=gen.sample_index,
                label=Label.CLASS_E_INDETERMINATE,
                token_count=gen.token_count,
                class_e_reason=f"judge_exception:{type(exc).__name__}:{exc}",
            )
        results.append(result)
    return results
