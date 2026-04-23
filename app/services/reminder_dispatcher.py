from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.agent.memory import get_memory_service
from app.tools.langgraph_tools import send_followup_reminder


def _parse_send_result(raw: str) -> tuple[bool, dict[str, Any], str]:
    try:
        payload = json.loads(raw)
    except Exception:
        return False, {}, f"短信工具返回非JSON: {raw[:300]}"

    status = str(payload.get("status", "")).strip().lower()
    if status == "sent":
        return True, payload, ""

    if "error" in payload:
        return False, payload, str(payload.get("error", ""))[:500]
    return False, payload, str(payload.get("provider_message", "短信发送失败"))[:500]


def dispatch_due_reminders(*, max_tasks: int = 20, max_retry: int = 3) -> dict[str, Any]:
    service = get_memory_service()
    now_iso = datetime.now(timezone.utc).isoformat()
    tasks = service.fetch_due_reminder_tasks(now_iso=now_iso, limit=max_tasks)

    sent = 0
    failed = 0
    retried = 0
    details: list[dict[str, Any]] = []

    for task in tasks:
        task_id = int(task.get("id", 0))
        retry_count = int(task.get("retry_count", 0))
        phone_number = str(task.get("phone_number", "")).strip()
        remind_time_raw = str(task.get("remind_time_raw", "")).strip()
        content = str(task.get("content", "")).strip()

        raw_result = send_followup_reminder.invoke(
            {
                "phone_number": phone_number,
                "remind_time": remind_time_raw,
                "content": content,
            }
        )
        ok, payload, error = _parse_send_result(str(raw_result))
        if ok:
            service.mark_reminder_sent(task_id, provider_payload=payload)
            sent += 1
            details.append({"task_id": task_id, "status": "sent"})
            continue

        if retry_count + 1 >= max_retry:
            service.mark_reminder_abandoned(task_id, error or "超过最大重试次数")
            failed += 1
            details.append({"task_id": task_id, "status": "failed", "error": error})
        else:
            service.mark_reminder_failed(task_id, error or "发送失败，待重试")
            retried += 1
            details.append({"task_id": task_id, "status": "retry", "error": error})

    return {
        "checked_at": now_iso,
        "picked": len(tasks),
        "sent": sent,
        "failed": failed,
        "retried": retried,
        "details": details,
    }
