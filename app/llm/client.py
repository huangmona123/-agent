from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI

from app.config import Settings, get_settings


_PLACEHOLDER_API_KEYS = {"", "ollama", "local", "dummy", "none"}


@dataclass
class ModelResult:
    message: AIMessage
    used_llm: bool
    error: str | None = None


class LLMClient:
    _availability_cache: dict[tuple[str, str], tuple[float, bool, str | None]] = {}
    _endpoint_cache: dict[str, tuple[float, bool, str | None]] = {}
    _cache_ttl_seconds = 30.0

    def __init__(self, settings: Settings | None = None, *, use_memory_model: bool = False):
        self.settings = settings or get_settings()
        self.use_memory_model = use_memory_model

        if use_memory_model:
            base_url = self.settings.memory_llm_base_url or self.settings.llm_base_url
            model = self.settings.memory_llm_model or self.settings.llm_model
            api_key = self.settings.memory_llm_api_key or self.settings.llm_api_key
            default_temperature = self.settings.memory_llm_temperature
            default_max_tokens = self.settings.memory_llm_max_tokens
        else:
            base_url = self.settings.llm_base_url
            model = self.settings.llm_model
            api_key = self.settings.llm_api_key
            default_temperature = self.settings.llm_temperature
            default_max_tokens = self.settings.llm_max_tokens

        self.base_url = (base_url or "").strip() or None
        self.model = model.strip()
        self.api_key = api_key.strip() or "not-needed"
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self._client = (
            OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
            if self.base_url
            else None
        )

    @staticmethod
    def has_image_content(messages: list[BaseMessage]) -> bool:
        for message in messages:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type", "")).strip().lower()
                if item_type in {"image_url", "input_image", "image"}:
                    return True
        return False

    @property
    def is_available(self) -> bool:
        return bool(self._client and self.model)

    def _cache_key(self) -> tuple[str, str]:
        return (self.base_url or "", self.model)

    def _headers_hint(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key.lower() not in _PLACEHOLDER_API_KEYS:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _serialize_message(self, message: BaseMessage) -> dict[str, Any]:
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": self._message_content(message.content)}
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": self._message_content(message.content)}
        if isinstance(message, ToolMessage):
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": self._message_content(message.content),
            }
        if isinstance(message, AIMessage):
            payload: dict[str, Any] = {
                "role": "assistant",
                "content": self._message_content(message.content),
            }
            if message.tool_calls:
                tool_calls = []
                for call in message.tool_calls:
                    arguments = call.get("args", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    tool_calls.append(
                        {
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": call.get("name"),
                                "arguments": arguments,
                            },
                        }
                    )
                payload["tool_calls"] = tool_calls
            return payload
        return {"role": "user", "content": self._message_content(message.content)}

    def _message_content(self, content: Any) -> Any:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            normalized: list[dict[str, Any]] = []
            has_multimodal = False
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = str(item.get("type", "")).strip().lower()
                    if item_type in {"text", "input_text"}:
                        has_multimodal = True
                        text = item.get("text", item.get("content", ""))
                        normalized.append({"type": "text", "text": str(text)})
                        parts.append(str(text))
                    elif item_type in {"image_url", "input_image", "image"}:
                        has_multimodal = True
                        image_payload = item.get("image_url", item.get("image", ""))
                        if isinstance(image_payload, dict):
                            url = str(image_payload.get("url", "")).strip()
                        else:
                            url = str(image_payload or item.get("url", "")).strip()
                        if url:
                            normalized.append({"type": "image_url", "image_url": {"url": url}})
                    elif "text" in item:
                        parts.append(str(item["text"]))
                    elif "content" in item:
                        parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            if has_multimodal and normalized:
                return normalized
            return "\n".join(parts)
        return str(content)

    def invoke(
        self,
        messages: list[BaseMessage],
        *,
        tools: list[BaseTool] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResult:
        if not self.is_available:
            return ModelResult(
                message=AIMessage(content=""),
                used_llm=False,
                error="LLM is not configured. Set LLM_BASE_URL and LLM_MODEL in .env.",
            )

        cache_key = self._cache_key()
        endpoint_key = self.base_url or ""
        now = time.time()
        endpoint_cached = self._endpoint_cache.get(endpoint_key)
        if endpoint_cached:
            cached_at, cached_available, cached_error = endpoint_cached
            if not cached_available and (now - cached_at) <= self._cache_ttl_seconds:
                return ModelResult(
                    message=AIMessage(content=""),
                    used_llm=False,
                    error=cached_error or "LLM temporarily unavailable.",
                )

        cached = self._availability_cache.get(cache_key)
        if cached:
            cached_at, cached_available, cached_error = cached
            if not cached_available and (now - cached_at) <= self._cache_ttl_seconds:
                return ModelResult(
                    message=AIMessage(content=""),
                    used_llm=False,
                    error=cached_error or "LLM temporarily unavailable.",
                )

        payload_messages = [self._serialize_message(message) for message in messages]
        tool_specs = [convert_to_openai_tool(tool) for tool in tools] if tools else None

        try:
            request_kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": payload_messages,
                "temperature": self.default_temperature if temperature is None else temperature,
                "max_tokens": self.default_max_tokens if max_tokens is None else max_tokens,
            }
            if tool_specs:
                request_kwargs["tools"] = tool_specs

            response = self._client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            message_payload = choice.message
            content = message_payload.content or ""
            tool_calls = []

            for call in message_payload.tool_calls or []:
                raw_args = call.function.arguments or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = {"raw": raw_args}
                tool_calls.append(
                    {
                        "name": call.function.name,
                        "args": parsed_args,
                        "id": call.id,
                        "type": "tool_call",
                    }
                )

            message = AIMessage(
                content=content,
                tool_calls=tool_calls,
                response_metadata={
                    "model": self.model,
                    "finish_reason": choice.finish_reason,
                },
            )

            self._availability_cache[cache_key] = (now, True, None)
            self._endpoint_cache[endpoint_key] = (now, True, None)
            return ModelResult(message=message, used_llm=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._availability_cache[cache_key] = (now, False, error)
            self._endpoint_cache[endpoint_key] = (now, False, error)
            return ModelResult(message=AIMessage(content=""), used_llm=False, error=error)


class VisionLLMClient(LLMClient):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.use_memory_model = False
        self.base_url = (self.settings.vision_llm_base_url or self.settings.llm_base_url or "").strip() or None
        self.model = (self.settings.vision_llm_model or self.settings.llm_model).strip()
        self.api_key = (
            self.settings.vision_llm_api_key
            or self.settings.llm_api_key
            or "not-needed"
        ).strip() or "not-needed"
        self.default_temperature = self.settings.llm_temperature
        self.default_max_tokens = self.settings.llm_max_tokens
        self._client = (
            OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
            if self.base_url
            else None
        )
