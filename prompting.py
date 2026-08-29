"""Prompt assembly (T037).

The prompt carries two very different things, and they are not equal:

* the **motion description**, which is advisory and may be truncated;
* the **speech script**, which is the output's content and is never truncated,
  dropped, or reordered.

When the assembled prompt exceeds the profile's capacity, motion gives way — all
of it if necessary. Speech never does. Cutting a script to fit would silently
change what the person on screen says, which is the one failure mode this
application must never produce quietly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from domain import AssembledPrompt, DialogueSegment, ModelProfile, MotionTruncation
from errors import LanguageError, ValidationError

STRUCTURING_VERSION = "1"
"""Locally built prompt structuring. H3-Context-IR is a hosted service we do not
call, so the structuring is ours and is versioned so bundles stay attributable."""

Tokenizer = Callable[[str], Sequence[object]]


def count_tokens(text: str, *, tokenizer: Tokenizer | None = None) -> int:
    """Measure prompt length.

    With no tokenizer, use a whitespace-and-punctuation approximation. The
    approximation is deliberately generous so the offline suite never *under*
    counts and lets an over-capacity prompt through.
    """
    if tokenizer is not None:
        return len(tokenizer(text))
    if not text:
        return 0
    words = text.split()
    return sum(max(1, (len(word) + 3) // 4) for word in words)


def _segment_script(script: str, language: str) -> list[DialogueSegment]:
    """One segment carrying the whole script, in order.

    Kept as a list because a profile may later declare multi-segment dialogue,
    but nothing here may split, drop, or reorder what the operator typed.
    """
    return [DialogueSegment(language=language, text=script)]


def render(motion_text: str, segments: Sequence[DialogueSegment], profile: ModelProfile) -> str:
    tags = [
        profile.dialogue_tag_form.format(language=seg.language, text=seg.text) for seg in segments
    ]
    parts = [part for part in (motion_text.strip(), *tags) if part]
    return " ".join(parts)


def assemble_prompt(
    *,
    motion_prompt: str,
    speech_script: str,
    language: str,
    profile: ModelProfile,
    tokenizer: Tokenizer | None = None,
) -> AssembledPrompt:
    """Build the prompt actually submitted to the adapter.

    Never refuses on length. Motion gives way first, all of it if necessary; if
    the script alone still exceeds capacity the prompt is returned with
    `over_capacity` set and the full script intact, and the adapter decides. A
    hard-bound option deliberately does not exist here, because using one would
    terminate a request for its script length.
    """
    motion_text = (motion_prompt or "").strip()
    script = (speech_script or "").strip()

    if not script:
        raise ValidationError("speech script must not be empty")
    if language not in profile.dialogue_languages:
        raise LanguageError(
            f"{language!r} is not a supported language for this model. "
            f"Supported: {', '.join(profile.dialogue_languages)}"
        )

    segments = _segment_script(script, language)
    capacity = profile.prompt_capacity_tokens
    original_motion_length = len(motion_text)

    rendered = render(motion_text, segments, profile)
    token_count = count_tokens(rendered, tokenizer=tokenizer)
    truncation: MotionTruncation | None = None

    if token_count > capacity and motion_text:
        # Give way with motion only. Binary search the longest motion prefix that
        # fits, so we discard as little as possible rather than dropping it all.
        low, high, best = 0, len(motion_text), 0
        while low <= high:
            mid = (low + high) // 2
            candidate = render(motion_text[:mid].strip(), segments, profile)
            if count_tokens(candidate, tokenizer=tokenizer) <= capacity:
                best, low = mid, mid + 1
            else:
                high = mid - 1

        retained = motion_text[:best].strip()
        if best < original_motion_length:
            truncation = MotionTruncation(
                original_length=original_motion_length,
                retained_length=len(retained),
                discarded_length=original_motion_length - len(retained),
            )
        motion_text = retained
        rendered = render(motion_text, segments, profile)
        token_count = count_tokens(rendered, tokenizer=tokenizer)

    # An over-capacity script reaches AssembledPrompt as-is; `over_capacity`
    # records the fact and nothing here refuses it.
    return AssembledPrompt(
        motion_text=motion_text,
        dialogue_segments=segments,
        rendered=rendered,
        token_count=token_count,
        token_capacity=capacity,
        motion_truncation=truncation,
        structuring_version=STRUCTURING_VERSION,
    )
