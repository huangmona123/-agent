from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agent.graph import new_thread_id
from app.pipeline import answer_question


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LangGraph medical QA agent.")
    parser.add_argument("--question", type=str, required=True, help="User question.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks.")
    parser.add_argument("--style", type=str, default="empathetic", choices=["professional", "empathetic", "coach"])
    parser.add_argument("--user-id", type=str, default="default-user", help="User ID for long-term memory.")
    parser.add_argument("--thread-id", type=str, default="", help="Conversation thread ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thread_id = args.thread_id.strip() or new_thread_id()
    response = answer_question(
        question=args.question,
        user_id=args.user_id,
        thread_id=thread_id,
        top_k=args.top_k,
        response_style=args.style,
    )

    print(f"QA_OK engine={response.engine} used_llm={response.used_llm}")
    print(f"thread_id={response.thread_id}")
    print(
        "LATENCY "
        f"total_ms={response.total_ms:.1f} "
        f"retrieval_ms={response.retrieval_ms:.1f} "
        f"generation_ms={response.generation_ms:.1f}"
    )
    print("=" * 80)
    print(response.answer)
    print("=" * 80)
    if response.error:
        print(f"NOTE: {response.error}")
    print("Sources:")
    for idx, item in enumerate(response.citations, start=1):
        print(f"[{idx}] {item.get('title', '')}")
        print(f"    {item.get('url', '')}")


if __name__ == "__main__":
    main()
