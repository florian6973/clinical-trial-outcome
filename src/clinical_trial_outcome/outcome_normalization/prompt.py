"""Retrieval-augmented prompt construction for Qwen group assignment."""

from __future__ import annotations

from .retrieval import Candidate, RetrievalContext


SYSTEM_MESSAGE = "You select the best normalized clinical outcome group from retrieved context."


def _format_candidates(candidates: tuple[Candidate, ...]) -> str:
    return ", ".join(f"{candidate.label} ({candidate.score:.2f})" for candidate in candidates)


def build_group_prompt(term: str, context: RetrievalContext) -> str:
    clean_term = term.strip()
    if not clean_term:
        raise ValueError("Outcome term must not be empty")
    if len(context.terms) != 5 or len(context.groups) != 5:
        raise ValueError("Qwen prompts require exactly five retrieved terms and five groups")
    instruction = (
        "Task: Normalize the Clinical Trial Outcome Object into one standardized group. "
        "Select an existing candidate group when it represents the concept; create a concise "
        "new group only when none is suitable."
    )
    return f'''{instruction}

IMPORTANT: Respond only with `GROUP: [group name]` and no explanation.

Main term: {clean_term}

Five similar terms:
{_format_candidates(context.terms)}

Five candidate groups:
{_format_candidates(context.groups)}

Required response:
GROUP: [group name]'''
