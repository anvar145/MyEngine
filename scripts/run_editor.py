"""Запуск редактора MyEngine из корня репозитория.

Запуск::

    .venv\\Scripts\\python.exe scripts\\run_editor.py
    .venv\\Scripts\\python.exe scripts\\run_editor.py --language en
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from editor.app import main  # noqa: E402 - путь настраивается выше

if __name__ == "__main__":
    raise SystemExit(main())
