from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

from app.agent.graph import new_thread_id
from app.pipeline import answer_question

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical health LangGraph agent.")
    parser.add_argument("--question", type=str, required=True, help="User question.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks.")
    parser.add_argument(
        "--style",
        type=str,
        default="empathetic",
        choices=["professional", "empathetic", "coach"],
        help="Answer style.",
    )
    parser.add_argument("--user-id", type=str, default="default-user", help="User ID for memory.")
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
    print(response.answer)


if __name__ == "__main__":
    main()
