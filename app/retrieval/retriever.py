from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import os

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Force transformers to skip TensorFlow/Flax optional imports on Windows.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    score: float
    record: dict[str, Any]


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                records.append(payload)
    return tuple(records)


@lru_cache(maxsize=8)
def _load_faiss_index(index_path: str):
    return faiss.read_index(index_path)


@lru_cache(maxsize=8)
def _load_records(records_path: str) -> tuple[dict[str, Any], ...]:
    return _read_jsonl(Path(records_path))


@lru_cache(maxsize=8)
def _load_manifest(manifest_path: str) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    return {}


@lru_cache(maxsize=4)
def _load_sentence_model(embed_model: str) -> SentenceTransformer:
    return SentenceTransformer(embed_model, local_files_only=True)


@lru_cache(maxsize=16)
def _load_hashing_vectorizer(
    n_features: int,
    analyzer: str,
    ngram_start: int,
    ngram_end: int,
) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        analyzer=analyzer,
        ngram_range=(ngram_start, ngram_end),
    )


class Retriever:
    def __init__(self, index_dir: Path, embed_model: str) -> None:
        self.index_dir = index_dir
        self.embed_model = embed_model

        index_path = index_dir / "faiss.index"
        records_path = index_dir / "records.jsonl"
        manifest_path = index_dir / "manifest.json"

        if not index_path.exists() or not records_path.exists():
            raise FileNotFoundError("Index files not found. Build index first.")

        self.index = _load_faiss_index(str(index_path))
        self.records = _load_records(str(records_path))
        self.embedding_backend = "sentence-transformers"
        self.hashing_n_features = 4096
        self.hashing_analyzer = "char_wb"
        self.hashing_ngram_range = (3, 5)

        manifest = _load_manifest(str(manifest_path))
        if manifest:
            self.embedding_backend = str(manifest.get("embedding_backend", "sentence-transformers"))

            hashing_cfg = manifest.get("hashing_backend", {})
            if isinstance(hashing_cfg, dict):
                self.hashing_n_features = int(hashing_cfg.get("n_features", 4096))
                self.hashing_analyzer = str(hashing_cfg.get("analyzer", "char_wb"))
                ng = hashing_cfg.get("ngram_range", [3, 5])
                if isinstance(ng, list) and len(ng) == 2:
                    self.hashing_ngram_range = (int(ng[0]), int(ng[1]))

        if self.embedding_backend == "sentence-transformers":
            try:
                self.model = _load_sentence_model(embed_model)
            except Exception as exc:
                self.model = None
                self.model_load_error = exc
            if self.model is None:
                raise RuntimeError(
                    "SentenceTransformer 模型未在本地缓存，无法使用当前索引查询。"
                    "请先下载模型缓存，或者重新构建哈希索引。"
                ) from self.model_load_error
        else:
            self.model = None
            self.hashing_vectorizer = _load_hashing_vectorizer(
                self.hashing_n_features,
                self.hashing_analyzer,
                self.hashing_ngram_range[0],
                self.hashing_ngram_range[1],
            )

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        q = query.strip()
        if not q:
            return []

        if self.embedding_backend == "sentence-transformers":
            if self.model is None:
                raise RuntimeError(
                    "SentenceTransformer 模型不可用，无法执行查询。"
                )
            q_vec = self.model.encode(
                [q],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            q_vec = np.asarray(q_vec, dtype=np.float32)
        else:
            sparse = self.hashing_vectorizer.transform([q])
            q_vec = sparse.astype(np.float32).toarray()
            norms = np.linalg.norm(q_vec, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            q_vec = q_vec / norms

        k = max(1, min(top_k, len(self.records)))
        scores, indices = self.index.search(q_vec, k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            results.append(RetrievedChunk(score=float(score), record=self.records[idx]))
        return results
