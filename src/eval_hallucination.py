"""Label judger for EXP-2026-NVS-001 (plan v1.4, §5.2).

v1.3 -> v1.4 (2026-09-13, decided from the real Phase 1 pilot's 800 samples --
see `results/phase1_pilot_trajectories.json`, collected under v1.3): the two
items v1.3 deliberately left undecided are now resolved.

  - **Type A positive-determination**: resolved to exactly what v1.3 §5.2
    already stated as the target rule -- "not Type C, not Type D" -- with no
    further "substantiveness" test. Once C/D are ruled out, a Type A response
    is positive. `classify_type_a` returns this outcome directly instead of
    raising `TypeADecisionPending` (removed; it never fires anymore).
  - **Type D treatment**: any hedge-pattern match routes a response to Type D
    outright (no further combination logic) -- unchanged from what v1.3
    already implied, now confirmed rather than deferred.

**Order of judgment (v1.4 §5.2 change)**: semantic classification (Type C
abstain / Type D hedge) now runs BEFORE the length/truncation gate (Class E),
reversed from v1.3. A short-but-genuine abstention ("I don't know.", T=5) is
therefore Type C, not Class E -- semantics take precedence over length.
`generation_error` remains an unconditional, first-priority Class E trigger
independent of this reordering: it is a pipeline failure, not a
length/truncation judgment, and there is no text to classify. Hitting
`max_new_tokens` (200) without an EOS is likewise NOT a separate Class E
trigger (v1.4 §5.2) -- a truncated response is classified normally like any
other, gated only by `T < 20` same as before.

**Type D pattern hardening (v1.4 §5.2)**: `configs/refusal_patterns_en.json`
gained a syntactic-skeleton pattern class for a specific evasive move seen in
real Phase 1 samples -- pivoting from the specific (fictional) entity to a
generic, genre-level description while still referring back to it only as
"this name"/"this title" (never asserting it exists under that description).
This is a deterministic conjunction of a meta-referential subject phrase
("this name", "this title", "a name/title like/matching this") and a
generalizing predicate (typically/generally/usually/often/commonly, or
"refers to") within a bounded distance, in either order -- not a simple
phrase list entry, since neither half alone is hedging language on its own.

Frozen behavior (spec-fixed in the plan; unaffected by v1.4):
  - Type B factual-error detection: normalization (§5.2) + match against
    `configs/typeB_reference.json`, EXCEPT the numeric-tolerance rule, which
    plan Appendix B item 2 lists as an open pre-freeze item and is not decided
    here.
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


def check_generation_error(gen: GenerationRecord) -> Optional[str]:
    """Unconditional, first-priority Class E trigger (v1.4 §5.2): a pipeline
    failure, not a length/truncation judgment, so it is not reordered behind
    semantic classification the way the length gate is."""
    return "generation_error" if gen.generation_error else None


def check_length_gate(gen: GenerationRecord, min_len: int) -> Optional[str]:
    """§3.2 / §5.2 length gate: `T < 20` -> Class E. v1.4 §5.2: this runs
    AFTER semantic (Type C/D) classification, not before -- see `evaluate`.
    Hitting `max_new_tokens` without an EOS is not a separate trigger here;
    a truncated response is gated only by this same length check."""
    if gen.token_count < min_len:
        return f"token_count={gen.token_count}<{min_len}"
    return None


def classify_type_b_reference_match(gen: GenerationRecord, prompt: PromptRecord, reference: dict) -> EvalResult:
    """Type B positive-determination (§5.2): only reached once Type C/D and
    the length gate are already ruled out by `evaluate`."""
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


def classify_type_a(gen: GenerationRecord) -> EvalResult:
    """Type A positive-determination (v1.4 §5.2, resolved): only reached
    once Type C/D and the length gate are already ruled out by `evaluate` --
    at that point the response is Type A positive, no further test."""
    return EvalResult(
        prompt_id=gen.prompt_id,
        sample_index=gen.sample_index,
        label=Label.TYPE_A_POSITIVE,
        token_count=gen.token_count,
    )


def evaluate(
    prompt: PromptRecord,
    gen: GenerationRecord,
    config: EvalConfig,
    reference: Optional[dict] = None,
) -> EvalResult:
    """Top-level dispatcher (v1.4 §5.2): semantic classification (Type C
    abstain / Type D hedge) takes precedence over the length/truncation gate
    (Class E) -- reversed from v1.3. `generation_error` is still checked
    first, unconditionally, since it is a pipeline failure rather than a
    length/truncation judgment and there is no text to classify."""
    generation_error_reason = check_generation_error(gen)
    if generation_error_reason is not None:
        return EvalResult(
            prompt_id=gen.prompt_id,
            sample_index=gen.sample_index,
            label=Label.CLASS_E_INDETERMINATE,
            token_count=gen.token_count,
            class_e_reason=generation_error_reason,
        )

    abstain_patterns, hedge_patterns = load_pattern_config(config.refusal_patterns_path)

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

    length_gate_reason = check_length_gate(gen, config.min_sequence_length)
    if length_gate_reason is not None:
        return EvalResult(
            prompt_id=gen.prompt_id,
            sample_index=gen.sample_index,
            label=Label.CLASS_E_INDETERMINATE,
            token_count=gen.token_count,
            class_e_reason=length_gate_reason,
        )

    if prompt.prompt_type is PromptType.TYPE_B:
        if reference is None:
            reference = _load_json(config.typeB_reference_path)
        return classify_type_b_reference_match(gen, prompt, reference)

    return classify_type_a(gen)


def evaluate_batch(
    prompts: dict[str, PromptRecord],
    generations: list[GenerationRecord],
    config: EvalConfig,
) -> list[EvalResult]:
    """Batch entry point.

    §5.2: "判定コードは凍結・ハッシュ記録し、採取開始後の変更を禁止する" and
    Class E must include "判定例外" (judge exceptions). Unexpected exceptions
    are therefore caught and recorded as Class E.
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
