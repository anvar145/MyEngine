"""Точка входа собранной игры (PyInstaller): тонкая обёртка над плеером.

Запуск из исходников: ``python build/packaging/run_player.py --project ...``.
"""

from __future__ import annotations

import multiprocessing
import sys

# Проверка ошибок PyOpenGL заметно дороже самих вызовов на слабой iGPU; флаг
# нужно выставить до первого импорта ``OpenGL.GL`` (как в ``engine.player``).
if "--gl-debug" not in sys.argv:
    import OpenGL

    OpenGL.ERROR_CHECKING = False

from engine.player.app import main  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
