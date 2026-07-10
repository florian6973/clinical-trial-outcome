"""FAISS candidate retrieval for L2-normalized SNOMED CT embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .contract import SnomedCandidate


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    index_type: str = "ivf_flat_inner_product"
    nlist_max: int = 4096
    nlist_divisor: int = 30
    nprobe: int = 64

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.index_type != "ivf_flat_inner_product":
            raise ValueError("paper contract requires ivf_flat_inner_product")
        if min(self.nlist_max, self.nlist_divisor, self.nprobe) < 1:
            raise ValueError("FAISS index settings must be positive")


class FaissSnomedRetriever:
    """Retrieve SNOMED candidates with FAISS inner-product search.

    Optional heavy dependencies are imported only when this class is
    instantiated. The default pipeline dry run never imports them.
    """

    def __init__(
        self,
        candidate_embeddings: Any,
        metadata: Sequence[dict[str, Any]],
        config: RetrievalConfig | None = None,
    ) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "FAISS and NumPy are required only for --execute condition retrieval"
            ) from exc

        self._np = np
        self.config = config or RetrievalConfig()
        embeddings = np.asarray(candidate_embeddings, dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("candidate_embeddings must be a non-empty 2D array")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("metadata rows must match candidate embeddings")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("SNOMED embeddings must be L2-normalized")

        self._metadata = tuple(metadata)
        dimension = int(embeddings.shape[1])
        nlist = min(
            self.config.nlist_max,
            max(1, embeddings.shape[0] // self.config.nlist_divisor),
        )
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(
            quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(self.config.nprobe, nlist)
        self._index = index

    def retrieve(self, condition_text: str, query_embedding: Any) -> tuple[SnomedCandidate, ...]:
        if not condition_text.strip():
            raise ValueError("condition_text must not be empty")
        query = self._np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        norm = float(self._np.linalg.norm(query))
        if not self._np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError("condition query embedding must be L2-normalized")
        k = min(self.config.top_k, len(self._metadata))
        scores, indices = self._index.search(query, k)
        candidates: list[SnomedCandidate] = []
        seen_concept_ids: set[str] = set()
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if int(index) < 0:
                continue
            row = self._metadata[int(index)]
            concept_id = str(row["concept_id"])
            if concept_id in seen_concept_ids:
                continue
            seen_concept_ids.add(concept_id)
            candidates.append(
                SnomedCandidate(
                    concept_id=concept_id,
                    term=str(row["term"]),
                    disease_area=str(row["disease_area"]),
                    similarity=max(-1.0, min(1.0, float(score))),
                    rank=rank,
                )
            )
        if not candidates:
            raise RuntimeError("FAISS returned no SNOMED candidates")
        return tuple(candidates)
