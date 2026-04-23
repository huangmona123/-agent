import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    print("CONFIG_OK")
    print(f"llm_model={settings.llm_model}")
    print(f"embed_model={settings.embed_model}")
    print(f"data_dir={settings.data_dir}")
    print(f"index_dir={settings.index_dir}")


if __name__ == "__main__":
    main()
