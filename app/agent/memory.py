from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import re
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

try:
    from langgraph.checkpoint.sqlite import SQLiteCheckpointer  # type: ignore
except Exception:
    SQLiteCheckpointer = None

try:
    from langgraph.store.sqlite import SQLiteStore  # type: ignore
except Exception:
    SQLiteStore = None

from app.config import get_settings


PROFILE_PATTERNS = (
    re.compile(r"(我有[^。！；\n]{2,60})"),
    re.compile(r"(我对[^。！；\n]{2,60}过敏)"),
    re.compile(r"(我喜欢[^。！；\n]{2,60})"),
    re.compile(r"(请以后[^。！；\n]{2,60})"),
)


def new_thread_id() -> str:
    return uuid4().hex


@dataclass
class SqlMemoryRow:
    created_at: str
    payload: str


class AgentMemoryService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.memory_backend = "memory"
        self.memory_backend_reason = ""
        self.checkpointer, self.store = self._build_langgraph_backends()
        self._ensure_schema()

    def _build_langgraph_backends(self):
        if SQLiteCheckpointer is not None and SQLiteStore is not None:
            try:
                checkpointer = SQLiteCheckpointer(str(self.db_path))
                store = SQLiteStore(str(self.db_path))
                self.memory_backend = "sqlite"
                return checkpointer, store
            except Exception as exc:
                self.memory_backend_reason = (
                    f"sqlite backend unavailable: {type(exc).__name__}: {exc}"
                )
        else:
            self.memory_backend_reason = "sqlite backend packages are not installed"

        return InMemorySaver(), InMemoryStore()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    response_style TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_turns_user_created
                ON conversation_turns (user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_user_memories_user_created
                ON user_memories (user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS reminder_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    phone_number TEXT NOT NULL,
                    remind_time_raw TEXT NOT NULL,
                    remind_at TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    provider_payload_json TEXT NOT NULL DEFAULT '{}',
                    last_error TEXT NOT NULL DEFAULT '',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    next_retry_at TEXT,
                    sent_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reminder_tasks_due
                ON reminder_tasks (status, remind_at, next_retry_at);
                """
            )

    def record_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        question: str,
        answer: str,
        response_style: str,
        citations: list[dict[str, Any]],
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "question": question,
            "answer": answer,
            "response_style": response_style,
            "citations": citations,
            "created_at": timestamp,
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns (
                    user_id,
                    thread_id,
                    question,
                    answer,
                    response_style,
                    citations_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    thread_id,
                    question,
                    answer,
                    response_style,
                    json.dumps(citations, ensure_ascii=False),
                    timestamp,
                ),
            )

        self.store.put((user_id, "turns"), uuid4().hex, payload)

    def save_profile_memory(self, *, user_id: str, note: str, memory_type: str = "profile") -> None:
        clean_note = note.strip()
        if not clean_note:
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT 1
                FROM user_memories
                WHERE user_id = ? AND memory_type = ? AND note = ?
                LIMIT 1
                """,
                (user_id, memory_type, clean_note),
            ).fetchone()
            if existing:
                return

            conn.execute(
                """
                INSERT INTO user_memories (user_id, memory_type, note, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, memory_type, clean_note, timestamp),
            )

        self.store.put(
            (user_id, "profile"),
            uuid4().hex,
            {
                "note": clean_note,
                "memory_type": memory_type,
                "created_at": timestamp,
            },
        )

    def extract_profile_memories(self, text: str) -> list[str]:
        notes: list[str] = []
        for pattern in PROFILE_PATTERNS:
            for match in pattern.findall(text):
                item = str(match).strip()
                if item and item not in notes:
                    notes.append(item)

        # Simple fallback for common chronic-condition phrases.
        lower_text = text.strip().lower()
        if "我有" in text and any(k in lower_text for k in ["高血压", "糖尿病", "痛风", "哮喘"]):
            for disease in ("高血压", "糖尿病", "痛风", "哮喘"):
                if disease in text and f"我有{disease}" not in notes:
                    notes.append(f"我有{disease}")
        return notes

    def search_turns(self, *, user_id: str, query: str, limit: int = 5) -> list[dict[str, str]]:
        stripped_query = query.strip()
        with self._connect() as conn:
            if stripped_query:
                like = f"%{stripped_query}%"
                rows = conn.execute(
                    """
                    SELECT question, answer, created_at
                    FROM conversation_turns
                    WHERE user_id = ?
                      AND (question LIKE ? OR answer LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, like, like, max(1, limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT question, answer, created_at
                    FROM conversation_turns
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, max(1, limit)),
                ).fetchall()

            # Chinese matching fallback: if direct LIKE misses, return recent turns.
            if not rows and stripped_query:
                rows = conn.execute(
                    """
                    SELECT question, answer, created_at
                    FROM conversation_turns
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (user_id, max(1, limit)),
                ).fetchall()

        return [
            {
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def list_profile_memories(self, *, user_id: str, limit: int = 10) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT note
                FROM user_memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, max(1, limit)),
            ).fetchall()
        return [str(row["note"]) for row in rows]

    def load_relevant_memories(self, *, user_id: str, query: str, limit: int = 5) -> list[str]:
        results: list[str] = []

        try:
            profile_items = self.store.search((user_id, "profile"), query=query, limit=limit)
        except Exception:
            try:
                profile_items = self.store.search((user_id, "profile"), limit=limit)
            except Exception:
                profile_items = []

        for item in profile_items:
            value = getattr(item, "value", {}) or {}
            note = str(value.get("note", "")).strip()
            if note and note not in results:
                results.append(note)

        # Keep long-term profile memories clean; avoid injecting old answer text.
        # Chat turn recall is still available via SQL context when needed.

        for note in self.list_profile_memories(user_id=user_id, limit=limit):
            if query.strip() and query.strip() not in note:
                continue
            if note and note not in results:
                results.append(note)

        return results[:limit]

    def build_memory_context(self, *, user_id: str, query: str, limit: int = 5) -> str:
        memories = self.load_relevant_memories(user_id=user_id, query=query, limit=limit)
        if not memories:
            return ""
        return "\n".join(f"- {item}" for item in memories)

    def build_sql_context(self, *, user_id: str, query: str, limit: int = 3) -> str:
        rows = self.search_turns(user_id=user_id, query=query, limit=limit)
        if not rows:
            return ""
        lines = []
        for row in rows:
            lines.append(
                f"- {row['created_at']}: Q={row['question']} | A={row['answer'][:100]}"
            )
        return "\n".join(lines)

    def run_readonly_sql(self, sql: str) -> str:
        statement = sql.strip().rstrip(";")
        lowered = statement.lower()

        if not lowered.startswith("select"):
            return "Only SELECT statements are allowed."

        blocked_tokens = [
            " insert ",
            " update ",
            " delete ",
            " drop ",
            " alter ",
            " pragma ",
            " attach ",
            " detach ",
        ]
        padded = f" {lowered} "
        if any(token in padded for token in blocked_tokens):
            return "Unsafe SQL blocked. Use read-only SELECT queries."

        allowed_tables = {"conversation_turns", "user_memories", "reminder_tasks"}
        referenced_tables = {
            part.strip(",")
            for part in re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", lowered)
        }
        if referenced_tables and not referenced_tables.issubset(allowed_tables):
            return (
                "Only these tables are available: conversation_turns, user_memories, reminder_tasks."
            )

        with self._connect() as conn:
            rows = conn.execute(statement).fetchall()

        payload = [dict(row) for row in rows[:20]]
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_thread_messages(self, thread_id: str) -> list[dict[str, str]]:
        checkpoint = self.checkpointer.get({"configurable": {"thread_id": thread_id}})
        history: list[dict[str, str]] = []

        if checkpoint:
            if isinstance(checkpoint, dict):
                channel_values = checkpoint.get("channel_values") or {}
            else:
                channel_values = getattr(checkpoint, "channel_values", {}) or {}
            messages = channel_values.get("messages", []) or []
            for msg in messages:
                if isinstance(msg, dict):
                    role = str(msg.get("role", "")).strip()
                    content = str(msg.get("content", "")).strip()
                    if content and role in {"user", "assistant"}:
                        history.append({"role": role, "content": content})
                    continue
                content = getattr(msg, "content", "")
                if not content:
                    continue
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": str(content)})
                elif isinstance(msg, AIMessage):
                    history.append({"role": "assistant", "content": str(content)})

        if history:
            return history

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT question, answer
                FROM conversation_turns
                WHERE thread_id = ?
                ORDER BY created_at ASC
                """,
                (thread_id,),
            ).fetchall()

        for row in rows:
            history.append({"role": "user", "content": str(row["question"])})
            history.append({"role": "assistant", "content": str(row["answer"])})

        return history

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)
        with self._connect() as conn:
            conn.execute("DELETE FROM conversation_turns WHERE thread_id = ?", (thread_id,))

    def create_reminder_task(
        self,
        *,
        user_id: str,
        thread_id: str,
        phone_number: str,
        remind_time_raw: str,
        remind_at: str,
        content: str,
    ) -> int:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO reminder_tasks (
                    user_id,
                    thread_id,
                    phone_number,
                    remind_time_raw,
                    remind_at,
                    content,
                    status,
                    provider_payload_json,
                    last_error,
                    retry_count,
                    next_retry_at,
                    sent_at,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', '{}', '', 0, NULL, NULL, ?, ?)
                """,
                (
                    user_id,
                    thread_id,
                    phone_number,
                    remind_time_raw,
                    remind_at,
                    content,
                    timestamp,
                    timestamp,
                ),
            )
            return int(cursor.lastrowid or 0)

    def fetch_due_reminder_tasks(self, *, now_iso: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, thread_id, phone_number, remind_time_raw, remind_at, content,
                       retry_count, next_retry_at
                FROM reminder_tasks
                WHERE status = 'pending'
                  AND remind_at <= ?
                  AND (next_retry_at IS NULL OR next_retry_at <= ?)
                ORDER BY remind_at ASC
                LIMIT ?
                """,
                (now_iso, now_iso, max(1, limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_reminder_sent(self, task_id: int, provider_payload: dict[str, Any] | None = None) -> None:
        payload = json.dumps(provider_payload or {}, ensure_ascii=False)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE reminder_tasks
                SET status = 'sent',
                    provider_payload_json = ?,
                    last_error = '',
                    sent_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (payload, timestamp, timestamp, int(task_id)),
            )

    def mark_reminder_failed(self, task_id: int, error: str, *, retry_after_seconds: int = 300) -> None:
        now = datetime.now(timezone.utc)
        next_retry = now.timestamp() + max(60, int(retry_after_seconds))
        next_retry_iso = datetime.fromtimestamp(next_retry, tz=timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE reminder_tasks
                SET status = 'pending',
                    last_error = ?,
                    retry_count = retry_count + 1,
                    next_retry_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (str(error)[:500], next_retry_iso, now.isoformat(), int(task_id)),
            )

    def mark_reminder_abandoned(self, task_id: int, error: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE reminder_tasks
                SET status = 'failed',
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (str(error)[:500], timestamp, int(task_id)),
            )


@lru_cache(maxsize=1)
def get_memory_service() -> AgentMemoryService:
    settings = get_settings()
    return AgentMemoryService(settings.memory_db_path)
