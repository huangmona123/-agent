from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from app.agent.graph import run_agent


@dataclass
class QAResponse:
    answer: str
    citations: list[dict[str, Any]]
    logs: list[str]
    engine: str
    used_llm: bool
    thread_id: str
    total_ms: float
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    error: str = ""

    def __getitem__(self, key: str):
        return asdict(self)[key]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_question(
    question: str | list[dict[str, Any]],
    *,
    user_id: str = "default-user",
    thread_id: str | None = None,
    top_k: int = 3,
    response_style: str = "empathetic",
) -> QAResponse:
    result = run_agent(
        question,
        user_id=user_id,
        thread_id=thread_id,
        top_k=top_k,
        response_style=response_style,
    )

    return QAResponse(
        answer=str(result.get("answer", "")),
        citations=list(result.get("citations", [])),
        logs=list(result.get("logs", [])),
        engine=str(result.get("engine", "langgraph")),
        used_llm=bool(result.get("used_llm", False)),
        thread_id=str(result.get("thread_id", thread_id or "")),
        total_ms=float(result.get("total_ms", 0.0)),
        retrieval_ms=float(result.get("retrieval_ms", 0.0)),
        generation_ms=float(result.get("generation_ms", 0.0)),
        error=str(result.get("error", "")),
    )
