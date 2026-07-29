"""Consistent three-state answerability classification."""

from __future__ import annotations

from typing import Literal

AnswerStatus = Literal["answered", "partially_answered", "unanswerable"]


def classify_answer_status(
    answer: str,
    evidence: list[object],
    *,
    needs_clarification: bool = False,
) -> AnswerStatus:
    if not evidence:
        return "unanswerable"
    normalized = answer.casefold()
    partial_markers = (
        "확인 불가",
        "확인할 수 없",
        "확인되지 않",
        "명시되어 있지 않",
        "추가 확인",
    )
    if needs_clarification or any(marker in normalized for marker in partial_markers):
        return "partially_answered"
    return "answered"


def is_answerable_status(status: AnswerStatus) -> bool:
    return status != "unanswerable"
