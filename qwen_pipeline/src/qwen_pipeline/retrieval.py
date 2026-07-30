"""NV-Embed-v2 encoding and cosine retrieval in FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import config_digest, load_configs
from .io import atomic_write_json, read_jsonl, sha256_file
from .validation import validate_schema


def vocabulary_text(task: str, item: dict[str, Any]) -> str:
    if task == "outcome":
        return item["term"]
    if task == "condition":
        return item["preferred_term"]
    raise ValueError(f"unsupported task: {task}")


def inspect_vocabulary(task: str, path: str | Path) -> dict[str, Any]:
    records = list(read_jsonl(path))
    if not records:
        raise ValueError("vocabulary is empty")
    ids: set[str] = set()
    for record in records:
        validate_schema(record, "vocabulary.schema.json")
        if record["vocabulary_id"] in ids:
            raise ValueError(f"duplicate vocabulary_id: {record['vocabulary_id']}")
        ids.add(record["vocabulary_id"])
        vocabulary_text(task, record)
    return {
        "task": task,
        "records": len(records),
        "sha256": sha256_file(path),
        "sample_ids": sorted(ids)[:3],
    }


def build_index(task: str, vocabulary_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Build an exact inner-product index over L2-normalized embeddings."""
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("full retrieval dependencies are not installed; run `pip install -e .`") from exc

    configs = load_configs()
    config = configs["pipeline"]
    items = list(read_jsonl(vocabulary_path))
    inspect_vocabulary(task, vocabulary_path)
    texts = [vocabulary_text(task, item) for item in items]
    model = SentenceTransformer(config["embedding"]["model_id"], trust_remote_code=True)
    embeddings = model.encode(
        texts,
        batch_size=config["embedding"]["batch_size"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    if embeddings.ndim != 2:
        raise RuntimeError("embedding model did not return a two-dimensional array")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destination / "index.faiss"))
    with (destination / "vocabulary.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    group_count = None
    if task == "outcome":
        groups = sorted({item["normalized_group"] for item in items})
        group_embeddings = model.encode(
            groups,
            batch_size=config["embedding"]["batch_size"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
        faiss.normalize_L2(group_embeddings)
        group_index = faiss.IndexFlatIP(group_embeddings.shape[1])
        group_index.add(group_embeddings)
        faiss.write_index(group_index, str(destination / "groups.faiss"))
        atomic_write_json(destination / "groups.json", groups)
        group_count = len(groups)
    manifest = {
        "status": "executed",
        "task": task,
        "records": len(items),
        "normalized_groups": group_count,
        "dimensions": int(embeddings.shape[1]),
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "configuration_sha256": config_digest(config),
        "embedding_model_id": config["embedding"]["model_id"],
        "index_type": type(index).__name__,
    }
    atomic_write_json(destination / "manifest.json", manifest)
    return manifest


class Retriever:
    """Load a previously built index and retain exact candidate provenance."""

    def __init__(self, task: str, index_dir: str | Path):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("full retrieval dependencies are not installed") from exc
        self._faiss = faiss
        self.task = task
        self.path = Path(index_dir)
        self.items = list(read_jsonl(self.path / "vocabulary.jsonl"))
        self.index = faiss.read_index(str(self.path / "index.faiss"))
        model_id = load_configs()["pipeline"]["embedding"]["model_id"]
        self.model = SentenceTransformer(model_id, trust_remote_code=True)
        self.groups = []
        self.group_index = None
        if task == "outcome" and (self.path / "groups.faiss").exists():
            self.groups = json.loads((self.path / "groups.json").read_text(encoding="utf-8"))
            self.group_index = faiss.read_index(str(self.path / "groups.faiss"))

    def search(self, text: str, k: int) -> list[dict[str, Any]]:
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        self._faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, min(k, len(self.items)))
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            candidate = dict(self.items[int(index)])
            candidate["similarity"] = float(score)
            results.append(candidate)
        return results

    def search_groups(self, text: str, k: int) -> list[dict[str, Any]]:
        if self.task != "outcome" or self.group_index is None:
            raise RuntimeError("a separate outcome-group index is not available")
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        self._faiss.normalize_L2(vector)
        scores, indices = self.group_index.search(vector, min(k, len(self.groups)))
        return [
            {"group": self.groups[int(index)], "similarity": float(score)}
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]
