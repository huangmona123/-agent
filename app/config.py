from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            value = value.strip().strip('"').strip("'")
            values[key] = value

    return values


def _coerce_int(value: object, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _coerce_float(value: object, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    if parsed < 0:
        return 0.0
    if parsed > 1.5:
        return 1.5
    return parsed


def _coerce_positive_float(value: object, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _coerce_path(value: object, default: Path) -> Path:
    raw = str(value).strip() if value is not None else ""
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings loaded from .env files and environment."""

    llm_api_key: str = ""
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 320
    llm_temperature: float = 0.1
    memory_llm_api_key: str = ""
    memory_llm_base_url: Optional[str] = None
    memory_llm_model: str = "gpt-4o-mini"
    memory_llm_max_tokens: int = 160
    memory_llm_temperature: float = 0.1
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 0
    vision_llm_api_key: str = ""
    vision_llm_base_url: Optional[str] = None
    vision_llm_model: str = "qwen-vl-plus"
    tavily_api_key: str = ""
    tavily_base_url: str = "https://api.tavily.com/search"
    tavily_timeout_seconds: float = 12.0
    tavily_max_results: int = 5
    amap_api_key: str = ""
    amap_base_url: str = "https://restapi.amap.com"
    amap_timeout_seconds: float = 10.0
    aliyun_access_key_id: str = ""
    aliyun_access_key_secret: str = ""
    aliyun_sms_sign_name: str = ""
    aliyun_sms_template_code: str = ""
    aliyun_sms_endpoint: str = "dysmsapi.aliyuncs.com"
    aliyun_sms_region: str = "cn-hangzhou"
    aliyun_sms_timeout_seconds: float = 10.0
    hospital_reg_api_base_url: Optional[str] = None
    hospital_reg_api_key: str = ""
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    data_dir: Path = PROJECT_ROOT / "data" / "processed"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    memory_db_path: Path = PROJECT_ROOT / "data" / "agent_memory.sqlite"

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    def ensure_runtime_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)

    def validate_for_generation(self) -> None:
        if not self.llm_ready:
            raise ValueError(
                "Missing required configuration for generation: LLM_BASE_URL and LLM_MODEL."
            )


def _build_settings() -> Settings:
    merged: dict[str, str] = {}
    for source in (
        _load_env_file(PROJECT_ROOT / ".env.example"),
        _load_env_file(PROJECT_ROOT / ".env"),
        dict(os.environ),
    ):
        merged.update(source)

    settings = Settings(
        llm_api_key=str(merged.get("LLM_API_KEY", "")).strip(),
        llm_base_url=str(merged.get("LLM_BASE_URL", "")).strip() or None,
        llm_model=str(merged.get("LLM_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini",
        llm_max_tokens=_coerce_int(merged.get("LLM_MAX_TOKENS", 320), 320),
        llm_temperature=_coerce_float(merged.get("LLM_TEMPERATURE", 0.1), 0.1),
        llm_timeout_seconds=_coerce_positive_float(merged.get("LLM_TIMEOUT_SECONDS", 20.0), 20.0),
        llm_max_retries=max(0, int(str(merged.get("LLM_MAX_RETRIES", "0")).strip() or "0")),
        vision_llm_api_key=str(merged.get("VISION_LLM_API_KEY", "")).strip(),
        vision_llm_base_url=str(merged.get("VISION_LLM_BASE_URL", "")).strip() or None,
        vision_llm_model=str(merged.get("VISION_LLM_MODEL", "qwen-vl-plus")).strip() or "qwen-vl-plus",
        tavily_api_key=str(merged.get("TAVILY_API_KEY", "")).strip(),
        tavily_base_url=str(merged.get("TAVILY_BASE_URL", "https://api.tavily.com/search")).strip()
        or "https://api.tavily.com/search",
        tavily_timeout_seconds=_coerce_positive_float(merged.get("TAVILY_TIMEOUT_SECONDS", 12.0), 12.0),
        tavily_max_results=_coerce_int(merged.get("TAVILY_MAX_RESULTS", 5), 5),
        amap_api_key=str(merged.get("AMAP_API_KEY", "")).strip(),
        amap_base_url=str(merged.get("AMAP_BASE_URL", "https://restapi.amap.com")).strip()
        or "https://restapi.amap.com",
        amap_timeout_seconds=_coerce_positive_float(merged.get("AMAP_TIMEOUT_SECONDS", 10.0), 10.0),
        aliyun_access_key_id=str(merged.get("ALIYUN_ACCESS_KEY_ID", "")).strip(),
        aliyun_access_key_secret=str(merged.get("ALIYUN_ACCESS_KEY_SECRET", "")).strip(),
        aliyun_sms_sign_name=str(merged.get("ALIYUN_SMS_SIGN_NAME", "")).strip(),
        aliyun_sms_template_code=str(merged.get("ALIYUN_SMS_TEMPLATE_CODE", "")).strip(),
        aliyun_sms_endpoint=str(merged.get("ALIYUN_SMS_ENDPOINT", "dysmsapi.aliyuncs.com")).strip()
        or "dysmsapi.aliyuncs.com",
        aliyun_sms_region=str(merged.get("ALIYUN_SMS_REGION", "cn-hangzhou")).strip()
        or "cn-hangzhou",
        aliyun_sms_timeout_seconds=_coerce_positive_float(
            merged.get("ALIYUN_SMS_TIMEOUT_SECONDS", 10.0), 10.0
        ),
        hospital_reg_api_base_url=str(merged.get("HOSPITAL_REG_API_BASE_URL", "")).strip() or None,
        hospital_reg_api_key=str(merged.get("HOSPITAL_REG_API_KEY", "")).strip(),
        memory_llm_api_key=str(merged.get("MEMORY_LLM_API_KEY", "")).strip(),
        memory_llm_base_url=str(merged.get("MEMORY_LLM_BASE_URL", "")).strip() or None,
        memory_llm_model=str(merged.get("MEMORY_LLM_MODEL", "gpt-4o-mini")).strip() or "gpt-4o-mini",
        memory_llm_max_tokens=_coerce_int(merged.get("MEMORY_LLM_MAX_TOKENS", 160), 160),
        memory_llm_temperature=_coerce_float(merged.get("MEMORY_LLM_TEMPERATURE", 0.1), 0.1),
        embed_model=str(
            merged.get(
                "EMBED_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
        ).strip()
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        data_dir=_coerce_path(
            merged.get("DATA_DIR", PROJECT_ROOT / "data" / "processed"),
            PROJECT_ROOT / "data" / "processed",
        ),
        index_dir=_coerce_path(
            merged.get("INDEX_DIR", PROJECT_ROOT / "data" / "index"),
            PROJECT_ROOT / "data" / "index",
        ),
        memory_db_path=_coerce_path(
            merged.get("MEMORY_DB_PATH", PROJECT_ROOT / "data" / "agent_memory.sqlite"),
            PROJECT_ROOT / "data" / "agent_memory.sqlite",
        ),
    )
    settings.ensure_runtime_dirs()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _build_settings()
