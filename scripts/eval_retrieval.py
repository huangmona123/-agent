from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.retriever import Retriever


@dataclass
class EvalRow:
    query: str
    relevant_ids: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retriever with Recall@K and MRR.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "eval" / "retrieval_eval_sample.jsonl",
        help="JSONL dataset with query and relevant_ids.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values, e.g. 1,3,5,10.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "index",
        help="Index directory.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name (used when index backend is sentence-transformers).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "retrieval_eval_report.json",
        help="Evaluation output report path.",
    )
    return parser.parse_args()


def resolve_dataset_path(path: Path) -> Path:
    if path.exists():
        return path

    if path.is_absolute():
        raise FileNotFoundError(f"Dataset not found: {path}")

    candidates = [
        PROJECT_ROOT / path,
        PROJECT_ROOT / "eval" / path,
        PROJECT_ROOT / "eval" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Dataset not found: {path}")


def load_dataset(path: Path) -> list[EvalRow]:
    path = resolve_dataset_path(path)

    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                continue

            query = str(payload.get("query", "")).strip()
            rel = payload.get("relevant_ids", [])
            if not query or not isinstance(rel, list):
                print(f"WARN skip invalid sample line={line_no}")
                continue

            rel_ids = {str(item).strip() for item in rel if str(item).strip()}
            if not rel_ids:
                print(f"WARN skip sample with empty relevant_ids line={line_no}")
                continue

            rows.append(EvalRow(query=query, relevant_ids=rel_ids))

    if not rows:
        raise ValueError("No valid evaluation samples found.")
    return rows


def parse_ks(ks_text: str) -> list[int]:
    values: list[int] = []
    for part in ks_text.split(","):
        p = part.strip()
        if not p:
            continue
        k = int(p)
        if k <= 0:
            continue
        values.append(k)
    values = sorted(set(values))
    if not values:
        raise ValueError("No valid K values.")
    return values


def candidate_ids(record: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    rid = str(record.get("id", "")).strip()
    if rid:
        ids.add(rid)

    doc_id = str(record.get("doc_id", "")).strip()
    if doc_id:
        ids.add(doc_id)

    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        pmid = str(metadata.get("pmid", "")).strip()
        if pmid:
            ids.add(f"pubmed:{pmid}")

        topic_id = str(metadata.get("topic_id", "")).strip()
        if topic_id:
            ids.add(f"medlineplus:{topic_id}")

    return ids


def first_relevant_rank(relevant: set[str], retrieved_records: list[dict[str, Any]]) -> int | None:
    for rank, rec in enumerate(retrieved_records, start=1):
        if candidate_ids(rec) & relevant:
            return rank
    return None


def evaluate(rows: list[EvalRow], retriever: Retriever, ks: list[int]) -> dict[str, Any]:
    max_k = max(ks)
    recall_hits = {k: 0 for k in ks}
    reciprocal_rank_sum = 0.0
    details: list[dict[str, Any]] = []

    for row in rows:
        hits = retriever.search(row.query, top_k=max_k)
        retrieved_records = [hit.record for hit in hits]

        rank = first_relevant_rank(row.relevant_ids, retrieved_records)
        if rank is not None:
            reciprocal_rank_sum += 1.0 / rank

        item = {
            "query": row.query,
            "first_relevant_rank": rank,
            "hit_at": {},
        }

        for k in ks:
            topk = retrieved_records[:k]
            matched = any((candidate_ids(rec) & row.relevant_ids) for rec in topk)
            if matched:
                recall_hits[k] += 1
            item["hit_at"][f"{k}"] = matched

        details.append(item)

    n = len(rows)
    recall_at_k = {f"recall@{k}": recall_hits[k] / n for k in ks}
    mrr = reciprocal_rank_sum / n

    return {
        "sample_count": n,
        "metrics": {
            **recall_at_k,
            "mrr": mrr,
        },
        "details": details,
    }


def main() -> None:
    args = parse_args()
    ks = parse_ks(args.ks)
    rows = load_dataset(args.dataset)

    try:
        retriever = Retriever(index_dir=args.index_dir, embed_model=args.embed_model)
    except Exception as exc:
        print(f"EVAL_FAIL {type(exc).__name__}: {exc}")
        return

    report = evaluate(rows, retriever, ks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("EVAL_OK")
    print(f"samples={report['sample_count']}")
    for key, value in report["metrics"].items():
        print(f"{key}={value:.4f}")
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
