"""MyEngine — модульный 3D-движок (ядро, рендеринг, физика, скриптинг, редактор).

Пакет ``engine`` — точка входа репозитория. Ядро движка живёт в
:mod:`engine.core` и не зависит ни от PySide6, ни от OpenGL.
"""

from __future__ import annotations

__all__ = ["__version__", "core"]

__version__ = "0.1.0"
