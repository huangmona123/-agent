from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge raw medical datasets, normalize records, deduplicate, and split into chunks."
    )
    parser.add_argument(
        "--medlineplus-in",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "medlineplus_topics_en.jsonl",
        help="MedlinePlus JSONL input file.",
    )
    parser.add_argument(
        "--pubmed-in",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pubmed_abstracts.jsonl",
        help="PubMed JSONL input file.",
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "corpus_docs.jsonl",
        help="Normalized document-level JSONL output.",
    )
    parser.add_argument(
        "--chunks-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "corpus_chunks.jsonl",
        help="Chunk-level JSONL output for retrieval/indexing.",
    )
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum document content length.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Chunk size measured by characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Overlap between adjacent chunks in characters.",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=120,
        help="Discard chunks shorter than this size.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap for processed docs (0 means no cap).",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                print(f"WARN skip invalid json line file={path} line={line_no}")
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def sanitize_record(record: dict[str, Any], min_chars: int) -> dict[str, Any] | None:
    raw_title = str(record.get("title", ""))
    raw_content = str(record.get("content", ""))
    title = normalize_text(raw_title)
    content = normalize_text(raw_content)

    if not title or not content:
        return None
    if len(content) < min_chars:
        return None

    rec_id = normalize_text(str(record.get("id", ""))) or f"auto:{hashlib.sha1(title.encode('utf-8')).hexdigest()[:12]}"
    source = normalize_text(str(record.get("source", "unknown"))) or "unknown"
    url = normalize_text(str(record.get("url", "")))
    language = normalize_text(str(record.get("language", "en"))) or "en"
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "id": rec_id,
        "source": source,
        "title": title,
        "content": content,
        "url": url,
        "language": language,
        "metadata": metadata,
    }


def dedup_key(record: dict[str, Any]) -> tuple[str, str]:
    url = record.get("url", "")
    if isinstance(url, str) and url:
        return ("url", url)

    fingerprint = hashlib.sha1(
        f"{record.get('title', '')}\n{record.get('content', '')}".encode("utf-8")
    ).hexdigest()
    return ("hash", fingerprint)


def split_text(content: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    windows: list[tuple[int, int, str]] = []
    text = normalize_text(content)
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        segment = text[start:end].strip()
        if segment:
            windows.append((start, end, segment))
        if end >= text_len:
            break
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return windows


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()

    input_paths = [args.medlineplus_in, args.pubmed_in]
    dedup_seen: set[tuple[str, str]] = set()
    docs: list[dict[str, Any]] = []
    source_stats: dict[str, int] = {}
    duplicate_count = 0
    skipped_count = 0

    for path in input_paths:
        for raw in read_jsonl(path):
            record = sanitize_record(raw, min_chars=args.min_chars)
            if record is None:
                skipped_count += 1
                continue

            key = dedup_key(record)
            if key in dedup_seen:
                duplicate_count += 1
                continue
            dedup_seen.add(key)

            docs.append(record)
            source = str(record.get("source", "unknown"))
            source_stats[source] = source_stats.get(source, 0) + 1

            if args.max_docs > 0 and len(docs) >= args.max_docs:
                break

        if args.max_docs > 0 and len(docs) >= args.max_docs:
            break

    chunk_records: list[dict[str, Any]] = []
    for doc in docs:
        windows = split_text(
            str(doc["content"]),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        for idx, (start, end, chunk_text) in enumerate(windows, start=1):
            if len(chunk_text) < args.min_chunk_chars:
                continue

            chunk_records.append(
                {
                    "id": f"{doc['id']}#chunk-{idx:04d}",
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "title": doc["title"],
                    "text": chunk_text,
                    "url": doc["url"],
                    "language": doc["language"],
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": idx,
                        "start_char": start,
                        "end_char": end,
                    },
                }
            )

    docs_count = write_jsonl(docs, args.docs_out)
    chunks_count = write_jsonl(chunk_records, args.chunks_out)

    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "inputs": [str(p) for p in input_paths],
        "docs_out": str(args.docs_out),
        "chunks_out": str(args.chunks_out),
        "doc_count": docs_count,
        "chunk_count": chunks_count,
        "duplicate_count": duplicate_count,
        "skipped_count": skipped_count,
        "source_stats": source_stats,
        "chunking": {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "min_chunk_chars": args.min_chunk_chars,
        },
    }
    manifest_path = args.chunks_out.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"CORPUS_OK docs={docs_count} chunks={chunks_count} "
        f"dedup={duplicate_count} skipped={skipped_count}"
    )


if __name__ == "__main__":
    main()
