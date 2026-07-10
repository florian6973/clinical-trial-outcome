"""Dependency-free cosine retrieval for outcome terms and groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .config import RetrievalConfig


Vector = Sequence[float]


@dataclass(frozen=True)
class Candidate:
    label: str
    score: float


@dataclass(frozen=True)
class RetrievalContext:
    terms: tuple[Candidate, ...]
    groups: tuple[Candidate, ...]


def _cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Cosine vectors must have equal, non-zero dimensions")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(value) ** 2 for value in left))
    norm_right = math.sqrt(sum(float(value) ** 2 for value in right))
    if norm_left == 0 or norm_right == 0:
        raise ValueError("Cosine vectors must have non-zero magnitude")
    return dot / (norm_left * norm_right)


def top_k_cosine(
    query: Vector,
    labels: Sequence[str],
    vectors: Sequence[Vector],
    *,
    k: int,
) -> tuple[Candidate, ...]:
    if len(labels) != len(vectors):
        raise ValueError("Labels and vectors must have equal lengths")
    if k <= 0:
        raise ValueError("k must be positive")
    scored = [
        Candidate(label=label, score=_cosine(query, vector))
        for label, vector in zip(labels, vectors)
    ]
    scored.sort(key=lambda candidate: candidate.score, reverse=True)
    return tuple(scored[:k])


def retrieve_context(
    query: Vector,
    *,
    term_labels: Sequence[str],
    term_vectors: Sequence[Vector],
    group_labels: Sequence[str],
    group_vectors: Sequence[Vector],
    config: RetrievalConfig | None = None,
) -> RetrievalContext:
    config = config or RetrievalConfig()
    return RetrievalContext(
        terms=top_k_cosine(query, term_labels, term_vectors, k=config.top_k_terms),
        groups=top_k_cosine(query, group_labels, group_vectors, k=config.top_k_groups),
    )
