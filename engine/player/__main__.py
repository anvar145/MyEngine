"""Запуск плеера: ``python -m engine.player --project <каталог> ...``."""

from __future__ import annotations

import sys

# Проверка ошибок PyOpenGL (glGetError после каждого вызова) заметно дороже
# самих вызовов на слабой iGPU. Флаг нужно выставить до первого импорта
# ``OpenGL.GL``, поэтому он стоит раньше импорта приложения.
if "--gl-debug" not in sys.argv:
    import OpenGL

    OpenGL.ERROR_CHECKING = False

from engine.player.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
