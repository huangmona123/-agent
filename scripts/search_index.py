from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.retriever import Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search local FAISS index.")
    parser.add_argument("--query", type=str, required=True, help="Query text.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Index directory. Defaults to INDEX_DIR from settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    index_dir = args.index_dir or settings.index_dir
    try:
        retriever = Retriever(index_dir=index_dir, embed_model=settings.embed_model)
        results = retriever.search(args.query, top_k=args.top_k)
    except Exception as exc:
        print(f"SEARCH_FAIL {type(exc).__name__}: {exc}")
        return

    print(f"SEARCH_OK hits={len(results)}")
    for idx, item in enumerate(results, start=1):
        title = str(item.record.get("title", ""))
        url = str(item.record.get("url", ""))
        text = str(item.record.get("text", ""))
        preview = (text[:180] + "...") if len(text) > 180 else text
        print(f"[{idx}] score={item.score:.4f} title={title}")
        print(f"    url={url}")
        print(f"    preview={preview}")


if __name__ == "__main__":
    main()
