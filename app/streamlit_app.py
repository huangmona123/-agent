from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import streamlit as st

from app.agent.graph import new_thread_id
from app.agent.memory import get_memory_service
from app.pipeline import answer_question


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&family=Noto+Serif+SC:wght@500;700&display=swap');

        :root {
          --bg-a: #f7fbff;
          --bg-b: #eef8f3;
          --bg-c: #fff9f0;
          --card: #ffffff;
          --text: #14263a;
          --muted: #4a6074;
          --primary: #0f766e;
          --primary-strong: #0a5e58;
          --accent: #e17a36;
          --border: #d7e5ef;
          --shadow: 0 12px 32px rgba(18, 40, 65, 0.1);
        }

        .stApp {
          background:
            radial-gradient(1100px 520px at 8% -10%, rgba(15, 118, 110, 0.12), transparent 60%),
            radial-gradient(950px 500px at 92% -6%, rgba(225, 122, 54, 0.12), transparent 58%),
            linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 52%, var(--bg-c) 100%);
          color: var(--text);
          font-family: "Manrope", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
        }

        h1, h2, h3, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
          font-family: "Noto Serif SC", "STKaiti", "KaiTi", serif;
          color: var(--text);
          letter-spacing: 0.2px;
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(242,250,248,0.95) 100%);
          border-right: 1px solid rgba(15, 118, 110, 0.14);
        }

        [data-testid="stSidebar"] > div:first-child {
          padding-top: 1.2rem;
        }

        .hero-panel {
          background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(245,255,252,0.9));
          border: 1px solid var(--border);
          box-shadow: var(--shadow);
          border-radius: 18px;
          padding: 1.05rem 1.1rem 0.95rem 1.1rem;
          margin-bottom: 0.9rem;
          animation: fade-up 0.32s ease-out both;
        }

        .hero-badge {
          display: inline-block;
          border-radius: 999px;
          padding: 0.24rem 0.7rem;
          background: rgba(15, 118, 110, 0.12);
          border: 1px solid rgba(15, 118, 110, 0.25);
          color: var(--primary-strong);
          font-size: 0.82rem;
          font-weight: 700;
          margin-bottom: 0.46rem;
        }

        .hero-title {
          margin: 0;
          font-size: 1.62rem;
          line-height: 1.3;
        }

        .hero-subtitle {
          margin: 0.35rem 0 0 0;
          color: var(--muted);
          font-size: 0.95rem;
        }

        div[data-testid="stChatMessage"] {
          border: 1px solid var(--border);
          background: rgba(255, 255, 255, 0.9);
          border-radius: 16px;
          box-shadow: 0 8px 20px rgba(18, 40, 65, 0.08);
          margin-bottom: 0.72rem;
          animation: fade-up 0.25s ease-out both;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
          line-height: 1.68;
        }

        .stButton > button {
          border: 0;
          border-radius: 12px;
          background: linear-gradient(135deg, var(--primary), #14a89f);
          color: #ffffff;
          font-weight: 700;
          box-shadow: 0 7px 18px rgba(15, 118, 110, 0.28);
          transition: all 0.16s ease;
        }

        .stButton > button:hover {
          transform: translateY(-1px);
          box-shadow: 0 10px 22px rgba(15, 118, 110, 0.32);
          background: linear-gradient(135deg, var(--primary-strong), #12948c);
        }

        div[data-testid="stExpander"] {
          border: 1px solid var(--border);
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.82);
        }

        .stTextInput > div > div,
        .stSelectbox > div > div,
        .stSlider {
          border-radius: 10px;
        }

        @keyframes fade-up {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 900px) {
          .hero-title { font-size: 1.36rem; }
          .hero-subtitle { font-size: 0.9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sources_logs(citations: list[dict], logs: list[str]) -> None:
    if not citations and not logs:
        return
    with st.expander("来源与日志", expanded=False):
        if citations:
            st.write("来源：")
            for cite in citations:
                title = str(cite.get("title", "")).strip() or "未命名来源"
                st.write(f"- {title}")
                url = str(cite.get("url", "")).strip()
                if url:
                    st.write(url)
        if logs:
            st.write("日志：")
            for log in logs:
                st.write(f"- {log}")


st.set_page_config(
    page_title="医疗健康 Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)
_inject_styles()

st.markdown(
    """
    <div class="hero-panel">
      <div class="hero-badge">Medical Health Assistant · LangGraph ReAct</div>
      <h1 class="hero-title">医疗健康 Agent</h1>
      <p class="hero-subtitle">用于健康科普与就医信息整理，不替代医生诊断。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
if "history" not in st.session_state:
    st.session_state.history = []
if "user_id" not in st.session_state:
    st.session_state.user_id = "default-user"
if "uploaded_image_url" not in st.session_state:
    st.session_state.uploaded_image_url = ""

with st.sidebar:
    st.header("会话设置")

    user_id = st.text_input("用户 ID", value=st.session_state.user_id)

    style_map = {
        "同理关怀": "empathetic",
        "专业简洁": "professional",
        "健康教练": "coach",
    }
    style_label = st.selectbox("回答风格", list(style_map.keys()), index=0)
    response_style = style_map[style_label]

    top_k = st.slider("检索条数", min_value=1, max_value=8, value=3, help="数值越大，检索内容越多。")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("清空会话", use_container_width=True):
            get_memory_service().delete_thread(st.session_state.thread_id)
            st.session_state.thread_id = new_thread_id()
            st.session_state.history = []
            st.rerun()
    with col_b:
        if st.button("新建会话", use_container_width=True):
            st.session_state.thread_id = new_thread_id()
            st.session_state.history = []
            st.session_state.uploaded_image_url = ""
            st.rerun()

    st.subheader("图片问诊（可选）")
    image_mode = st.selectbox("图片输入方式", ["不使用图片", "图片 URL"], index=0)
    if image_mode == "图片 URL":
        st.session_state.uploaded_image_url = st.text_input(
            "图片 URL",
            value=st.session_state.uploaded_image_url,
            placeholder="https://...",
        ).strip()

    with st.expander("查看会话历史", expanded=False):
        history = get_memory_service().get_thread_messages(st.session_state.thread_id)
        if not history:
            st.write("暂无历史")
        else:
            for item in history:
                role = "用户" if item.get("role") == "user" else "助手"
                st.write(f"{role}: {item.get('content', '')}")

st.session_state.user_id = user_id

for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.write(item["content"])
        if item["role"] == "assistant":
            _render_sources_logs(item.get("citations", []), item.get("logs", []))

q = st.chat_input("请输入你的问题，例如：我腹痛三天该挂什么科？")

if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        with st.spinner("分析中..."):
            image_url = st.session_state.uploaded_image_url.strip()
            question_payload: str | list[dict] = q
            if image_url:
                question_payload = [
                    {"type": "text", "text": q},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]

            res = answer_question(
                question_payload,
                user_id=user_id,
                thread_id=st.session_state.thread_id,
                top_k=top_k,
                response_style=response_style,
            )

            st.session_state.thread_id = res.thread_id
            st.write(res.answer)
            if res.error:
                st.warning(res.error)
            _render_sources_logs(res.citations, res.logs)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": res.answer,
            "citations": res.citations,
            "logs": res.logs,
        }
    )
