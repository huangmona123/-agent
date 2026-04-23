from __future__ import annotations

import json
from typing import Any

from app.tools.langgraph_tools import get_agent_tools


_ALIAS = {
    "find_hospital": "find_hospital_guidance",
    "hospital_search": "find_hospital_guidance",
    "diagnosis": "search_medical_knowledge",
    "search": "search_medical_knowledge",
}


def _parse_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except Exception:
            return {"result": raw}
    return {"result": raw}


def execute_tool(name: str, query: str = "", **kwargs):
    tool_name = _ALIAS.get(name.strip(), name.strip())
    tool_map = {tool.name: tool for tool in get_agent_tools()}
    tool = tool_map.get(tool_name)
    if tool is None:
        return {"error": f"Unknown tool: {name}"}

    payload = dict(kwargs)
    if query and not payload:
        if tool_name == "find_hospital_guidance":
            payload = {"location": query, "keyword": "医院", "limit": 5}
        elif tool_name in {"search_medical_knowledge", "web_search_medical"}:
            payload = {"query": query}
        else:
            payload = {"query": query}

    try:
        result = tool.invoke(payload)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return _parse_payload(result)
