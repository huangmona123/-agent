from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid optional TensorFlow import path from transformers on Windows.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from app.config import get_settings
from app.retrieval.index_builder import build_faiss_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index from processed chunk corpus.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl",
        help="Path to chunk JSONL file.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Index output directory. Defaults to INDEX_DIR from settings.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--max-chunks", type=int, default=0, help="Optional cap for chunks to index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    index_dir = args.index_dir or settings.index_dir
    result = build_faiss_index(
        chunks_path=args.chunks,
        index_dir=index_dir,
        embed_model=settings.embed_model,
        batch_size=args.batch_size,
        max_chunks=args.max_chunks,
    )

    print(
        f"INDEX_OK chunks={result.chunk_count} dim={result.embedding_dim} "
        f"index={result.index_path}"
    )


if __name__ == "__main__":
    main()
