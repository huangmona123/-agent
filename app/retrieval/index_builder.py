from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
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
class IndexBuildResult:
    chunk_count: int
    embedding_dim: int
    index_path: Path
    records_path: Path
    manifest_path: Path


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                yield payload


def _embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _hashing_embed_texts(
    texts: list[str],
    *,
    n_features: int = 4096,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (3, 5),
) -> np.ndarray:
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        analyzer=analyzer,
        ngram_range=ngram_range,
    )
    sparse = vectorizer.transform(texts)
    dense = sparse.astype(np.float32).toarray()
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return dense / norms


def build_faiss_index(
    chunks_path: Path,
    index_dir: Path,
    *,
    embed_model: str,
    batch_size: int = 64,
    max_chunks: int = 0,
) -> IndexBuildResult:
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {chunks_path}")

    records: list[dict[str, Any]] = []
    texts: list[str] = []

    for row in read_jsonl(chunks_path):
        text = str(row.get("text", "")).strip()
        if not text:
            continue

        records.append(row)
        texts.append(text)
        if max_chunks > 0 and len(records) >= max_chunks:
            break

    if not records:
        raise ValueError("No chunk records available to index.")

    backend = "sentence-transformers"
    fallback_reason = ""

    try:
        model = SentenceTransformer(embed_model, local_files_only=True)
        embeddings = _embed_texts(model, texts, batch_size=batch_size)
    except Exception as exc:
        backend = "hashing"
        fallback_reason = str(exc)
        print(
            "WARN embedding model load failed, switching to offline hashing backend. "
            f"reason={type(exc).__name__}"
        )
        embeddings = _hashing_embed_texts(texts)

    if embeddings.ndim != 2:
        raise ValueError("Embedding output shape is invalid.")

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "faiss.index"
    records_path = index_dir / "records.jsonl"
    manifest_path = index_dir / "manifest.json"

    faiss.write_index(index, str(index_path))

    with records_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "embed_model": embed_model,
        "embedding_backend": backend,
        "fallback_reason": fallback_reason,
        "hashing_backend": {
            "n_features": 4096,
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
        },
        "chunk_count": len(records),
        "embedding_dim": dim,
        "index_path": str(index_path),
        "records_path": str(records_path),
        "source_chunks_path": str(chunks_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return IndexBuildResult(
        chunk_count=len(records),
        embedding_dim=dim,
        index_path=index_path,
        records_path=records_path,
        manifest_path=manifest_path,
    )
