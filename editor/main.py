"""Точка входа редактора MyEngine.

Запуск::

    .venv\\Scripts\\python.exe editor\\main.py
    .venv\\Scripts\\python.exe editor\\main.py --language en --quality high
    .venv\\Scripts\\python.exe editor\\main.py --screenshot _work/editor.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# Редактор — отдельное приложение: он импортирует движок, но не наоборот.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from editor.app import main  # noqa: E402 - путь настраивается выше

if __name__ == "__main__":
    raise SystemExit(main())
