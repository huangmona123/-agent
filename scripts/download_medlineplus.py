from __future__ import annotations

import argparse
import html
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_URL = "https://medlineplus.gov/xml.html"
COMPRESSED_PATTERN = re.compile(
    r"https://medlineplus\.gov/xml/mplus_topics_compressed_\d{4}-\d{2}-\d{2}\.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and parse MedlinePlus health topics.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "medlineplus_topics_en.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional maximum number of records (0 means no limit).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--keep-xml",
        action="store_true",
        help="Save extracted XML next to output for debugging.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_text("".join(element.itertext()))


def clean_summary(raw_value: str) -> str:
    if not raw_value:
        return ""

    unescaped = html.unescape(raw_value)
    no_tags = re.sub(r"<[^>]+>", " ", unescaped)
    return normalize_text(no_tags)


def discover_latest_zip_url(index_html: str) -> str:
    matches = COMPRESSED_PATTERN.findall(index_html)
    if not matches:
        raise RuntimeError("Could not locate MedlinePlus compressed topic XML URL.")
    return matches[0]


def download_latest_zip(session: requests.Session, timeout: int) -> tuple[str, bytes]:
    index_response = session.get(INDEX_URL, timeout=timeout)
    index_response.raise_for_status()
    zip_url = discover_latest_zip_url(index_response.text)

    zip_response = session.get(zip_url, timeout=timeout)
    zip_response.raise_for_status()
    return zip_url, zip_response.content


def extract_xml(zip_payload: bytes) -> tuple[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
        xml_names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if not xml_names:
            raise RuntimeError("Compressed MedlinePlus file does not contain XML.")
        xml_name = xml_names[0]
        return xml_name, archive.read(xml_name)


def parse_topics(xml_payload: bytes, max_records: int) -> list[dict[str, object]]:
    root = ET.fromstring(xml_payload)
    records: list[dict[str, object]] = []

    for topic in root.findall(".//health-topic"):
        language = normalize_text(topic.attrib.get("language", "") or topic.findtext("language", default=""))
        if language and "english" not in language.lower():
            continue

        title = normalize_text(topic.attrib.get("title", "") or topic.findtext("title", default=""))
        summary = clean_summary(element_text(topic.find("full-summary")))
        if not title or not summary:
            continue

        topic_id = normalize_text(topic.attrib.get("id", "") or topic.findtext("id", default=""))
        topic_url = normalize_text(topic.attrib.get("url", "") or topic.findtext("url", default=""))
        date_created = normalize_text(
            topic.attrib.get("date-created", "") or topic.findtext("date-created", default="")
        )

        groups: list[str] = []
        for group in topic.findall("group"):
            group_name = normalize_text("".join(group.itertext()))
            if group_name:
                groups.append(group_name)

        if not topic_id:
            topic_id = f"auto-{len(records) + 1}"

        records.append(
            {
                "id": f"medlineplus:{topic_id}",
                "source": "medlineplus",
                "title": title,
                "content": summary,
                "url": topic_url,
                "language": "en",
                "metadata": {
                    "topic_id": topic_id,
                    "date_created": date_created,
                    "groups": groups,
                },
            }
        )

        if max_records > 0 and len(records) >= max_records:
            break

    return records


def write_jsonl(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(
    output_path: Path,
    zip_url: str,
    xml_name: str,
    record_count: int,
    xml_saved_path: Path | None,
) -> None:
    manifest_path = output_path.with_suffix(".manifest.json")
    payload: dict[str, object] = {
        "source": "medlineplus",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "zip_url": zip_url,
        "xml_file_in_zip": xml_name,
        "record_count": record_count,
        "output_jsonl": str(output_path),
    }
    if xml_saved_path is not None:
        payload["xml_saved_path"] = str(xml_saved_path)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    session = requests.Session()

    zip_url, zip_payload = download_latest_zip(session, timeout=args.timeout)
    xml_name, xml_payload = extract_xml(zip_payload)
    records = parse_topics(xml_payload, max_records=args.max_records)
    write_jsonl(records, args.out)

    xml_saved_path: Path | None = None
    if args.keep_xml:
        xml_saved_path = args.out.with_suffix(".xml")
        xml_saved_path.write_bytes(xml_payload)

    write_manifest(
        output_path=args.out,
        zip_url=zip_url,
        xml_name=xml_name,
        record_count=len(records),
        xml_saved_path=xml_saved_path,
    )

    print(f"MEDLINEPLUS_OK records={len(records)} out={args.out}")


if __name__ == "__main__":
    main()
