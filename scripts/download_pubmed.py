from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PubMed abstracts through E-utilities.")
    parser.add_argument(
        "--query",
        type=str,
        default="(hypertension[Title/Abstract]) OR (diabetes[Title/Abstract])",
        help="PubMed query string.",
    )
    parser.add_argument("--max-results", type=int, default=200, help="Maximum number of PubMed IDs.")
    parser.add_argument("--batch-size", type=int, default=100, help="EFetch batch size.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "pubmed_abstracts.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument("--email", type=str, default="", help="Contact email for NCBI requests.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("NCBI_API_KEY", ""),
        help="Optional NCBI API key.",
    )
    parser.add_argument("--tool", type=str, default="medqa-agent-mvp", help="Tool name reported to NCBI.")
    parser.add_argument("--sleep", type=float, default=0.35, help="Sleep seconds between EFetch calls.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of retry attempts for transient HTTP errors.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Base seconds for exponential backoff between retries.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def article_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_text("".join(element.itertext()))


def common_params(tool: str, email: str, api_key: str) -> dict[str, str]:
    params = {"tool": tool}
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def request_with_retry(
    session: requests.Session,
    *,
    url: str,
    params: dict[str, str | int],
    timeout: int,
    retries: int,
    retry_backoff: float,
    action_name: str,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as exc:
            last_error = exc
            if attempt >= retries:
                break

            wait_seconds = max(retry_backoff, 0.1) * (2**attempt)
            print(
                f"WARN retrying {action_name} attempt={attempt + 1}/{retries} "
                f"wait={wait_seconds:.1f}s reason={type(exc).__name__}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to {action_name} after {retries + 1} attempts. "
        f"Try increasing --timeout and reducing --max-results. Last error: {last_error}"
    )


def search_pubmed(
    session: requests.Session,
    query: str,
    max_results: int,
    timeout: int,
    retries: int,
    retry_backoff: float,
    tool: str,
    email: str,
    api_key: str,
) -> tuple[list[str], int]:
    params: dict[str, str | int] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        "sort": "relevance",
    }
    params.update(common_params(tool, email, api_key))

    response = request_with_retry(
        session,
        url=ESEARCH_URL,
        params=params,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
        action_name="search PubMed IDs",
    )
    data = response.json().get("esearchresult", {})

    id_list = data.get("idlist", [])
    count = int(data.get("count", 0))
    return id_list, count


def fetch_abstract_xml(
    session: requests.Session,
    pmids: list[str],
    timeout: int,
    retries: int,
    retry_backoff: float,
    tool: str,
    email: str,
    api_key: str,
) -> bytes:
    params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    params.update(common_params(tool, email, api_key))

    response = request_with_retry(
        session,
        url=EFETCH_URL,
        params=params,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
        action_name="fetch PubMed abstracts",
    )
    return response.content


def extract_pub_date(article: ET.Element) -> str:
    pub_date = article.find(".//JournalIssue/PubDate")
    if pub_date is None:
        return ""

    year = normalize_text(pub_date.findtext("Year", default=""))
    month = normalize_text(pub_date.findtext("Month", default=""))
    day = normalize_text(pub_date.findtext("Day", default=""))
    medline_date = normalize_text(pub_date.findtext("MedlineDate", default=""))

    parts = [part for part in (year, month, day) if part]
    if parts:
        return "-".join(parts)
    return medline_date


def parse_pubmed_xml(xml_payload: bytes, query: str) -> Iterable[dict[str, object]]:
    root = ET.fromstring(xml_payload)

    for article in root.findall(".//PubmedArticle"):
        pmid = normalize_text(article.findtext(".//MedlineCitation/PMID", default=""))
        title = article_text(article.find(".//ArticleTitle"))

        abstract_parts: list[str] = []
        for abstract_node in article.findall(".//Abstract/AbstractText"):
            label = normalize_text(abstract_node.attrib.get("Label", ""))
            text = article_text(abstract_node)
            if not text:
                continue
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)

        abstract = "\n".join(abstract_parts).strip()
        if not pmid or not title or not abstract:
            continue

        journal = normalize_text(article.findtext(".//Journal/Title", default=""))
        pub_date = extract_pub_date(article)

        yield {
            "id": f"pubmed:{pmid}",
            "source": "pubmed",
            "title": title,
            "content": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "language": "en",
            "metadata": {
                "pmid": pmid,
                "journal": journal,
                "pub_date": pub_date,
                "query": query,
            },
        }


def write_jsonl(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(
    output_path: Path,
    query: str,
    pubmed_hit_count: int,
    requested_ids: int,
    abstract_records: int,
) -> None:
    manifest_path = output_path.with_suffix(".manifest.json")
    payload = {
        "source": "pubmed",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "query": query,
        "pubmed_hit_count": pubmed_hit_count,
        "requested_ids": requested_ids,
        "abstract_records": abstract_records,
        "output_jsonl": str(output_path),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    session = requests.Session()
    session.headers.update({"User-Agent": f"{args.tool}/1.0"})

    id_list, total_hits = search_pubmed(
        session=session,
        query=args.query,
        max_results=args.max_results,
        timeout=args.timeout,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
        tool=args.tool,
        email=args.email,
        api_key=args.api_key,
    )

    if not id_list:
        write_jsonl([], args.out)
        write_manifest(
            output_path=args.out,
            query=args.query,
            pubmed_hit_count=total_hits,
            requested_ids=0,
            abstract_records=0,
        )
        print("PUBMED_OK records=0")
        return

    records: list[dict[str, object]] = []
    batches = list(chunked(id_list, args.batch_size))
    for batch_index, pmid_batch in enumerate(batches):
        xml_payload = fetch_abstract_xml(
            session=session,
            pmids=pmid_batch,
            timeout=args.timeout,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            tool=args.tool,
            email=args.email,
            api_key=args.api_key,
        )
        records.extend(parse_pubmed_xml(xml_payload, query=args.query))

        if batch_index < len(batches) - 1:
            time.sleep(max(args.sleep, 0.0))

    write_jsonl(records, args.out)
    write_manifest(
        output_path=args.out,
        query=args.query,
        pubmed_hit_count=total_hits,
        requested_ids=len(id_list),
        abstract_records=len(records),
    )

    print(f"PUBMED_OK records={len(records)} out={args.out}")


if __name__ == "__main__":
    main()
