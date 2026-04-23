from __future__ import annotations

import json
from functools import lru_cache
from time import perf_counter
from typing import Any

import langsmith
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.memory import get_memory_service, new_thread_id
from app.agent.prompts import (
    MEMORY_SUMMARY_SYSTEM_PROMPT,
    build_memory_summary_prompt,
    build_system_prompt,
)
from app.llm.client import LLMClient, VisionLLMClient
from app.llm.postprocess import dedupe_answer_text
from app.tools import get_agent_tools


MAX_TOOL_ROUNDS = 4


memory_service = get_memory_service()
llm_client = LLMClient()
vision_llm_client = VisionLLMClient()
memory_llm_client = LLMClient(use_memory_model=True)
agent_tools = get_agent_tools()


class AgentState(MessagesState):
    user_id: str
    thread_id: str
    response_style: str
    top_k: int
    knowledge_context: str
    memory_context: str
    sql_context: str
    citations: list[dict[str, Any]]
    logs: list[str]
    used_llm: bool
    engine: str
    error: str
    answer: str
    started_at: float
    retrieval_ms: float
    generation_ms: float
    tool_rounds: int


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message.content).strip()
    return ""


def _latest_assistant_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return _message_text(message.content).strip()
    return ""


def _message_has_image(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).strip().lower()
        if item_type in {"image_url", "image", "input_image"}:
            return True
    return False


def _latest_user_has_image(messages: list[BaseMessage]) -> bool:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_has_image(message.content)
    return False


def _parse_json_maybe(content: Any) -> Any | None:
    text = _message_text(content).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_citations(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _append(title: str, url: str, score: Any = "") -> None:
        t = title.strip() or "Untitled"
        u = url.strip()
        key = (t, u)
        if key in seen:
            return
        seen.add(key)
        citations.append({"title": t, "url": u, "score": score})

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        payload = _parse_json_maybe(message.content)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    _append(str(item.get("title", "")), str(item.get("url", "")), item.get("score", ""))
            continue
        if not isinstance(payload, dict):
            continue

        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    _append(str(item.get("title", "")), str(item.get("url", "")), item.get("score", ""))

        hospitals = payload.get("hospitals")
        if isinstance(hospitals, list):
            for item in hospitals:
                if isinstance(item, dict):
                    _append(str(item.get("name", "")), "")

    return citations[:8]


def prepare_context(state: AgentState) -> AgentState:
    logs = list(state.get("logs", []))
    user_id = str(state.get("user_id", "default-user")).strip() or "default-user"
    question = _latest_user_message(state["messages"])

    memory_context = ""
    sql_context = ""
    try:
        memory_context = memory_service.build_memory_context(user_id=user_id, query=question, limit=3)
        if memory_context:
            logs.append("Long-term memory loaded")
    except Exception as exc:
        logs.append(f"Memory loading failed: {type(exc).__name__}: {exc}")

    try:
        sql_context = memory_service.build_sql_context(user_id=user_id, query=question, limit=3)
        if sql_context:
            logs.append("SQLite history loaded")
    except Exception as exc:
        logs.append(f"SQLite history loading failed: {type(exc).__name__}: {exc}")

    return {
        **state,
        "memory_context": memory_context,
        "sql_context": sql_context,
        "logs": logs,
    }


def agent_step(state: AgentState) -> AgentState:
    with langsmith.trace(
        "agent_step",
        run_type="llm",
        inputs={"question": _latest_user_message(state["messages"]), "tool_rounds": int(state.get("tool_rounds", 0))},
    ):
        started = perf_counter()
        logs = list(state.get("logs", []))
        user_id = str(state.get("user_id", "default-user")).strip() or "default-user"
        thread_id = str(state.get("thread_id", "")).strip() or "thread"
        response_style = str(state.get("response_style", "empathetic")).strip() or "empathetic"
        has_user_image = _latest_user_has_image(state["messages"])

        system_prompt = build_system_prompt(
            response_style=response_style,
            user_id=user_id,
            thread_id=thread_id,
            knowledge_context=state.get("knowledge_context", ""),
            memory_context=state.get("memory_context", ""),
            sql_context=state.get("sql_context", ""),
        )
        conversation: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        conversation.extend(state["messages"][-16:])

        active_llm = llm_client
        if has_user_image and vision_llm_client.is_available:
            active_llm = vision_llm_client
            logs.append("Vision model selected for image turn")

        if not active_llm.is_available:
            fallback = "当前模型服务不可用，请先在 .env 中配置 LLM_BASE_URL / LLM_MODEL 并重启。"
            return {
                **state,
                "messages": [AIMessage(content=fallback)],
                "answer": fallback,
                "used_llm": False,
                "error": "LLM is not configured.",
                "logs": logs,
                "generation_ms": float(state.get("generation_ms", 0.0)) + (perf_counter() - started) * 1000.0,
            }

        result = active_llm.invoke(conversation, tools=agent_tools)
        if result.error:
            fallback = "我暂时无法访问模型服务，请稍后重试。若持续失败，请检查 .env 的 API 配置。"
            logs.append(f"LLM invoke failed: {result.error}")
            return {
                **state,
                "messages": [AIMessage(content=fallback)],
                "answer": fallback,
                "used_llm": False,
                "error": result.error,
                "logs": logs,
                "generation_ms": float(state.get("generation_ms", 0.0)) + (perf_counter() - started) * 1000.0,
            }

        ai_message = result.message
        if not _message_text(ai_message.content).strip() and not ai_message.tool_calls:
            ai_message = AIMessage(content="我已收到你的问题。请补充更多细节，我会继续帮你分析。")

        tool_rounds = int(state.get("tool_rounds", 0))
        if ai_message.tool_calls:
            tool_rounds += 1
            names = [str(call.get("name", "")).strip() for call in ai_message.tool_calls if call.get("name")]
            if names:
                logs.append(f"Tool calls: {', '.join(names)}")

        return {
            **state,
            "messages": [ai_message],
            "used_llm": True,
            "error": "",
            "logs": logs,
            "generation_ms": float(state.get("generation_ms", 0.0)) + (perf_counter() - started) * 1000.0,
            "tool_rounds": tool_rounds,
        }


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls and int(state.get("tool_rounds", 0)) <= MAX_TOOL_ROUNDS:
        return "tools"
    return "finalize"


def finalize(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    logs = list(state.get("logs", []))

    answer = _latest_assistant_message(messages) or str(state.get("answer", "")).strip()
    append_message = False
    if not answer:
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and last.tool_calls:
            answer = "我已完成多轮工具检索，但仍缺少关键信息。请补充更具体的症状、地点或时间，我继续帮你。"
        else:
            answer = "当前信息不足以给出可靠建议。请补充症状细节或尽快线下就医。"
        append_message = True

    answer = dedupe_answer_text(answer)
    citations = _extract_citations(messages)
    tool_count = sum(1 for item in messages if isinstance(item, ToolMessage))
    if tool_count:
        logs.append(f"Tool rounds executed: {tool_count}")

    updated: AgentState = {**state, "answer": answer, "citations": citations, "logs": logs}
    if append_message:
        updated["messages"] = [AIMessage(content=answer)]
    return updated


def save_memory(state: AgentState) -> AgentState:
    with langsmith.trace(
        "save_memory",
        run_type="chain",
        inputs={"thread_id": str(state.get("thread_id", "")), "user_id": str(state.get("user_id", ""))},
    ):
        logs = list(state.get("logs", []))
        user_id = str(state.get("user_id", "default-user")).strip() or "default-user"
        thread_id = str(state.get("thread_id", "")).strip() or "thread"
        response_style = str(state.get("response_style", "empathetic")).strip() or "empathetic"
        question = _latest_user_message(state["messages"])
        answer = _latest_assistant_message(state["messages"]) or str(state.get("answer", "")).strip()

        try:
            memory_service.record_turn(
                user_id=user_id,
                thread_id=thread_id,
                question=question,
                answer=answer,
                response_style=response_style,
                citations=list(state.get("citations", [])),
            )

            summary_note = ""
            if memory_llm_client.is_available and question and answer:
                summary_result = memory_llm_client.invoke(
                    [
                        SystemMessage(content=MEMORY_SUMMARY_SYSTEM_PROMPT),
                        HumanMessage(content=build_memory_summary_prompt(question=question, answer=answer)),
                    ],
                    temperature=0.0,
                    max_tokens=memory_llm_client.default_max_tokens,
                )
                if not summary_result.error:
                    summary_note = _message_text(summary_result.message.content).strip()

            if summary_note and summary_note != "无长期记忆":
                memory_service.save_profile_memory(user_id=user_id, note=summary_note, memory_type="summary")
            for note in memory_service.extract_profile_memories(question):
                memory_service.save_profile_memory(user_id=user_id, note=note)

            logs.append("Memory saved")
        except Exception as exc:
            logs.append(f"Memory save failed: {type(exc).__name__}: {exc}")

        return {**state, "logs": logs}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("agent_step", agent_step)
    graph.add_node("tools", ToolNode(agent_tools))
    graph.add_node("finalize", finalize)
    graph.add_node("save_memory", save_memory)

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "agent_step")
    graph.add_conditional_edges("agent_step", route_after_agent, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent_step")
    graph.add_edge("finalize", "save_memory")
    graph.add_edge("save_memory", END)
    return graph


@lru_cache(maxsize=1)
def get_graph():
    graph = _build_graph()
    return graph.compile(checkpointer=memory_service.checkpointer, store=memory_service.store)


@lru_cache(maxsize=1)
def get_platform_graph():
    graph = _build_graph()
    return graph.compile()


def run_agent(
    question: str | list[dict[str, Any]],
    *,
    user_id: str = "default-user",
    thread_id: str | None = None,
    top_k: int = 3,
    response_style: str = "empathetic",
) -> dict[str, Any]:
    with langsmith.trace(
        "run_agent",
        run_type="chain",
        inputs={
            "question": question,
            "user_id": user_id,
            "thread_id": thread_id,
            "top_k": top_k,
            "response_style": response_style,
        },
    ):
        graph = get_graph()
        thread = thread_id or new_thread_id()
        started = perf_counter()

        state_input: AgentState = {
            "messages": [HumanMessage(content=question)],
            "user_id": user_id,
            "thread_id": thread,
            "response_style": response_style,
            "top_k": top_k,
            "knowledge_context": "",
            "memory_context": "",
            "sql_context": "",
            "citations": [],
            "logs": [],
            "used_llm": False,
            "engine": "langgraph-react",
            "error": "",
            "answer": "",
            "started_at": started,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "tool_rounds": 0,
        }

        result = graph.invoke(state_input, config={"configurable": {"thread_id": thread}})
        answer = _latest_assistant_message(result["messages"]) or str(result.get("answer", "")).strip()
        total_ms = (perf_counter() - started) * 1000.0
        return {
            **result,
            "answer": answer,
            "thread_id": thread,
            "total_ms": total_ms,
            "engine": "langgraph-react",
        }


agent = get_platform_graph()
