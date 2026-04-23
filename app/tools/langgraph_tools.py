from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from langchain_core.tools import tool

from app.agent.memory import get_memory_service
from app.config import get_settings
from app.retrieval.retriever import Retriever


def _to_json(payload: dict[str, Any] | list[Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_hits(hits) -> str:
    items = []
    for idx, hit in enumerate(hits, start=1):
        record = hit.record or {}
        items.append(
            {
                "rank": idx,
                "score": round(float(hit.score), 4),
                "title": str(record.get("title", "")),
                "url": str(record.get("url", "")),
                "snippet": str(record.get("text", ""))[:300],
            }
        )
    return _to_json(items)


def _clip_limit(limit: int, *, low: int = 1, high: int = 20) -> int:
    return max(low, min(int(limit), high))


def _clip_top_k(top_k: int) -> int:
    return _clip_limit(top_k, low=1, high=10)


def _parse_remind_time_to_utc(remind_time: str) -> tuple[str, str]:
    raw = remind_time.strip()
    if not raw:
        raise ValueError("提醒时间不能为空")

    now_local = datetime.now()
    date_text = raw
    if raw.startswith("今天"):
        date_text = raw.replace("今天", now_local.strftime("%Y-%m-%d"), 1)
    elif raw.startswith("明天"):
        date_text = raw.replace("明天", (now_local + timedelta(days=1)).strftime("%Y-%m-%d"), 1)
    elif raw.startswith("后天"):
        date_text = raw.replace("后天", (now_local + timedelta(days=2)).strftime("%Y-%m-%d"), 1)

    normalized = (
        date_text.replace("年", "-")
        .replace("月", "-")
        .replace("日", "")
        .replace("/", "-")
        .replace("：", ":")
        .strip()
    )
    normalized = normalized.replace("点半", ":30").replace("时半", ":30")
    normalized = normalized.replace("点", ":00").replace("时", ":00")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"^(\d{4}-\d{1,2}-\d{1,2})(\d{1,2}:\d{2})$", r"\1 \2", normalized)
    normalized = re.sub(r"^(\d{4}-\d{1,2}-\d{1,2})(\d{1,2})$", r"\1 \2:00", normalized)
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", normalized):
        normalized = f"{normalized} 09:00"
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}$", normalized):
        normalized = f"{normalized}:00"
    normalized = normalized.replace("T", " ")

    dt_local = None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d %H:%M:%S"):
        try:
            dt_local = datetime.strptime(normalized, fmt)
            break
        except ValueError:
            continue
    if dt_local is None:
        raise ValueError("提醒时间格式无法识别，请使用如 2026-04-25 09:00 的格式")
    if dt_local < now_local:
        raise ValueError("提醒时间不能早于当前时间")

    dt_utc = dt_local.replace(tzinfo=timezone(timedelta(hours=8))).astimezone(timezone.utc)
    return normalized, dt_utc.isoformat()


@tool
def search_medical_knowledge(query: str, top_k: int = 3) -> str:
    """Search the local medical knowledge index and return top matches."""
    try:
        settings = get_settings()
        retriever = Retriever(index_dir=settings.index_dir, embed_model=settings.embed_model)
        hits = retriever.search(query, top_k=_clip_top_k(top_k))
        if not hits:
            return "[]"
        return _format_hits(hits)
    except Exception as exc:
        return _to_json({"error": f"{type(exc).__name__}: {exc}"})


@tool
def web_search_medical(query: str, max_results: int = 5) -> str:
    """Search web by Tavily and return summarized medical-relevant results."""
    settings = get_settings()
    api_key = settings.tavily_api_key.strip()
    if not api_key:
        return _to_json(
            {
                "error": "未配置 TAVILY_API_KEY，无法执行 web 搜索。",
                "next_step": "请在 .env 中设置 TAVILY_API_KEY 后重试。",
            }
        )

    payload = {
        "api_key": api_key,
        "query": query.strip(),
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": _clip_limit(max_results, low=1, high=8),
        "topic": "general",
    }
    try:
        resp = requests.post(
            settings.tavily_base_url,
            json=payload,
            timeout=settings.tavily_timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return _to_json({"error": f"Tavily 请求失败: {type(exc).__name__}: {exc}"})

    results = data.get("results") or []
    compact = []
    for idx, item in enumerate(results, start=1):
        compact.append(
            {
                "rank": idx,
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", ""))[:320],
                "score": item.get("score"),
            }
        )
    return _to_json({"query": query, "count": len(compact), "results": compact})


@tool
def find_hospital_guidance(location: str, keyword: str = "医院", limit: int = 5) -> str:
    """Search nearby hospitals by AMap API and return practical guidance."""
    settings = get_settings()
    api_key = settings.amap_api_key.strip()
    if not api_key:
        return _to_json(
            {
                "error": "未配置 AMAP_API_KEY，暂时无法在线检索附近医院。",
                "next_step": "请在 .env 中设置 AMAP_API_KEY 后重试。",
            }
        )

    query = f"{location.strip()} {keyword.strip()}".strip()
    params = {
        "key": api_key,
        "keywords": query or "医院",
        "city": location.strip(),
        "offset": _clip_limit(limit),
        "page": 1,
        "extensions": "base",
    }
    url = f"{settings.amap_base_url.rstrip('/')}/v3/place/text"

    try:
        resp = requests.get(url, params=params, timeout=settings.amap_timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return _to_json({"error": f"高德地图请求失败: {type(exc).__name__}: {exc}"})

    pois = payload.get("pois") or []
    hospitals: list[dict[str, Any]] = []
    for item in pois[: _clip_limit(limit)]:
        hospitals.append(
            {
                "name": str(item.get("name", "")),
                "address": str(item.get("address", "")),
                "tel": str(item.get("tel", "")),
                "type": str(item.get("type", "")),
                "distance": str(item.get("distance", "")),
                "location": str(item.get("location", "")),
            }
        )

    if not hospitals:
        return _to_json(
            {
                "notice": "未检索到匹配医院",
                "query": query,
                "suggestion": "请尝试更具体的地区词，例如“朝阳区 三甲医院”或“浦东新区 急诊”。",
            }
        )

    return _to_json(
        {
            "query": query,
            "count": len(hospitals),
            "hospitals": hospitals,
            "safety_notice": "如出现胸痛、呼吸困难、意识障碍等急症，请立即急诊。",
        }
    )


@tool
def schedule_registration(
    patient_name: str,
    id_number: str,
    phone_number: str,
    hospital_name: str,
    department: str,
    visit_date: str,
    symptom: str = "",
) -> str:
    """
    Submit registration request to external hospital registration API.
    This tool needs HOSPITAL_REG_API_BASE_URL and HOSPITAL_REG_API_KEY.
    """
    settings = get_settings()
    base_url = (settings.hospital_reg_api_base_url or "").strip()
    api_key = settings.hospital_reg_api_key.strip()
    if not base_url or not api_key:
        return _to_json(
            {
                "error": "未配置挂号 API。",
                "required_env": ["HOSPITAL_REG_API_BASE_URL", "HOSPITAL_REG_API_KEY"],
                "demo_payload": {
                    "patient_name": patient_name,
                    "id_number_masked": f"{id_number[:3]}****{id_number[-4:]}" if len(id_number) >= 8 else "***",
                    "phone_tail": phone_number[-4:] if phone_number else "",
                    "hospital_name": hospital_name,
                    "department": department,
                    "visit_date": visit_date,
                    "symptom": symptom,
                },
            }
        )

    request_payload = {
        "patient_name": patient_name,
        "id_number": id_number,
        "phone_number": phone_number,
        "hospital_name": hospital_name,
        "department": department,
        "visit_date": visit_date,
        "symptom": symptom,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/appointments",
            headers=headers,
            json=request_payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
    except Exception as exc:
        return _to_json({"error": f"挂号请求失败: {type(exc).__name__}: {exc}"})

    return _to_json({"status": "submitted", "provider_response": data})


@tool
def send_followup_reminder(phone_number: str, remind_time: str, content: str) -> str:
    """Send follow-up reminder by Aliyun SMS API."""
    settings = get_settings()
    missing = [
        name
        for name, value in [
            ("ALIYUN_ACCESS_KEY_ID", settings.aliyun_access_key_id),
            ("ALIYUN_ACCESS_KEY_SECRET", settings.aliyun_access_key_secret),
            ("ALIYUN_SMS_SIGN_NAME", settings.aliyun_sms_sign_name),
            ("ALIYUN_SMS_TEMPLATE_CODE", settings.aliyun_sms_template_code),
        ]
        if not str(value).strip()
    ]
    if missing:
        return _to_json(
            {
                "error": "短信配置不完整。",
                "missing_env": missing,
                "tip": "先补齐 .env，再重启服务。",
            }
        )

    try:
        from aliyunsdkcore.client import AcsClient
        from aliyunsdkcore.request import CommonRequest
    except Exception:
        return _to_json({"error": "缺少阿里云短信 SDK 依赖。", "install": "pip install aliyun-python-sdk-core"})

    params = {"time": remind_time, "content": content}
    request = CommonRequest()
    request.set_accept_format("json")
    request.set_domain(settings.aliyun_sms_endpoint)
    request.set_version("2017-05-25")
    request.set_action_name("SendSms")
    request.set_method("POST")
    request.add_query_param("RegionId", settings.aliyun_sms_region)
    request.add_query_param("PhoneNumbers", phone_number)
    request.add_query_param("SignName", settings.aliyun_sms_sign_name)
    request.add_query_param("TemplateCode", settings.aliyun_sms_template_code)
    request.add_query_param("TemplateParam", json.dumps(params, ensure_ascii=False))

    client = AcsClient(
        settings.aliyun_access_key_id,
        settings.aliyun_access_key_secret,
        settings.aliyun_sms_region,
    )
    try:
        raw = client.do_action_with_exception(request)
        response = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return _to_json({"error": f"短信发送失败: {type(exc).__name__}: {exc}"})

    code = str(response.get("Code", "")).strip()
    ok = code == "OK"
    return _to_json(
        {
            "status": "sent" if ok else "failed",
            "provider_code": code,
            "provider_message": str(response.get("Message", "")),
            "biz_id": str(response.get("BizId", "")),
            "request_id": str(response.get("RequestId", "")),
        }
    )


@tool
def schedule_followup_reminder(user_id: str, thread_id: str, phone_number: str, remind_time: str, content: str) -> str:
    """Create a reminder task that will be sent automatically at remind_time."""
    phone = phone_number.strip()
    if not re.fullmatch(r"1[3-9]\d{9}", phone):
        return _to_json({"error": "手机号格式不正确，请提供11位中国大陆手机号。"})

    try:
        remind_time_local, remind_at_utc = _parse_remind_time_to_utc(remind_time)
    except Exception as exc:
        return _to_json({"error": f"提醒时间解析失败: {type(exc).__name__}: {exc}"})

    service = get_memory_service()
    task_id = service.create_reminder_task(
        user_id=user_id.strip() or "default-user",
        thread_id=thread_id.strip() or "thread",
        phone_number=phone,
        remind_time_raw=remind_time_local,
        remind_at=remind_at_utc,
        content=content.strip() or "请按时复诊",
    )
    return _to_json(
        {
            "status": "scheduled",
            "task_id": task_id,
            "phone_tail": phone[-4:],
            "remind_time_local": remind_time_local,
            "remind_at_utc": remind_at_utc,
            "content": content.strip() or "请按时复诊",
        }
    )


@tool
def query_sqlite_memory(sql: str) -> str:
    """Run a read-only SELECT query against the local SQLite memory database."""
    return get_memory_service().run_readonly_sql(sql)


def get_agent_tools():
    return [
        search_medical_knowledge,
        web_search_medical,
        find_hospital_guidance,
        schedule_registration,
        schedule_followup_reminder,
        send_followup_reminder,
        query_sqlite_memory,
    ]
