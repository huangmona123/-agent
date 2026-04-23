from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.services.reminder_dispatcher import dispatch_due_reminders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch due reminder tasks and send SMS.")
    parser.add_argument("--max-tasks", type=int, default=20, help="Max tasks per run.")
    parser.add_argument("--max-retry", type=int, default=3, help="Max retry count before marking failed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = dispatch_due_reminders(max_tasks=max(1, args.max_tasks), max_retry=max(1, args.max_retry))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
