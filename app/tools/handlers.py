from __future__ import annotations

import json
from typing import Any, Dict

from app.tools.langgraph_tools import (
    find_hospital_guidance,
    search_medical_knowledge,
    web_search_medical,
)


_DEPRECATED_MESSAGE = (
    "handlers.py 已废弃。请优先使用 LangGraph ReAct Agent，让大模型自主决策并调用工具。"
)


def _text(params: Dict[str, Any], key: str) -> str:
    return str(params.get(key, "")).strip()


def _to_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except Exception:
            return {"result": payload}
    return {"result": payload}


def recommend_department_handler(params: Dict[str, Any]):
    return {
        "error": _DEPRECATED_MESSAGE,
        "next_step": "请调用 app.agent.graph.run_agent，由模型结合上下文和工具给出分诊建议。",
        "received": _text(params, "symptom"),
    }


def find_nearby_hospitals_handler(params: Dict[str, Any]):
    location = _text(params, "location") or _text(params, "query")
    if not location:
        return {"error": "缺少 location 参数。", "next_step": "例如：{'location': '北京朝阳区'}"}
    result = find_hospital_guidance.invoke({"location": location, "keyword": "医院", "limit": 5})
    return _to_dict(result)


def diet_advice_handler(params: Dict[str, Any]):
    query = _text(params, "query")
    if not query:
        return {"error": "缺少 query 参数。"}
    result = search_medical_knowledge.invoke({"query": query, "top_k": 3})
    return {
        "notice": _DEPRECATED_MESSAGE,
        "hint": "饮食建议请由 ReAct Agent 基于检索结果综合生成。",
        "knowledge_hits": _to_dict(result),
    }


def symptom_diagnosis_handler(params: Dict[str, Any]):
    query = _text(params, "query")
    if not query:
        return {"error": "缺少 query 参数。"}

    local_hits = _to_dict(search_medical_knowledge.invoke({"query": query, "top_k": 3}))
    web_hits = _to_dict(web_search_medical.invoke({"query": query, "max_results": 3}))
    return {
        "notice": _DEPRECATED_MESSAGE,
        "hint": "诊断结论必须由模型结合上下文与工具结果生成，不在 handlers 中硬编码。",
        "local_knowledge": local_hits,
        "web_results": web_hits,
    }


def recommend_emergency_handler(params: Dict[str, Any]):
    return {
        "emergency": True,
        "advice": "如有胸痛、呼吸困难、意识改变或大出血等高危症状，请立即急诊。",
        "notice": _DEPRECATED_MESSAGE,
        "received": params,
    }
