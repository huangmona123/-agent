from __future__ import annotations


def dedupe_answer_text(text: str) -> str:
    """Remove duplicate lines while preserving order."""
    raw = str(text or "").strip()
    if not raw:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    for line in (part.strip() for part in raw.splitlines()):
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return "\n".join(deduped)
