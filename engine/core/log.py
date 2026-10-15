"""Логирование движка: единая точка настройки и получения логгеров.

Модуль намеренно тонкий: он не заменяет :mod:`logging`, а лишь выдаёт
согласованно названные логгеры и умеет один раз настроить вывод.
"""

from __future__ import annotations

import logging
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_LOG_FORMAT = DEFAULT_FORMAT
_configured = False

__all__ = ["get_logger", "setup_logging", "DEFAULT_FORMAT"]


def get_logger(name: str) -> logging.Logger:
    """Вернуть логгер с префиксом движка (``engine.<name>``).

    Args:
        name: Короткое имя подсистемы, например ``"core.scene"``.

    Returns:
        Логгер, пригодный для модуля движка.
    """
    if name.startswith("engine."):
        return logging.getLogger(name)
    return logging.getLogger(f"engine.{name}")


def setup_logging(level: int | str = logging.INFO, *, stream=None) -> None:
    """Один раз настроить корневой логгер движка.

    Повторные вызовы не добавляют обработчиков (идемпотентно), но позволяют
    сменить уровень логирования.

    Args:
        level: Уровень логирования (``logging.DEBUG`` или строка ``"DEBUG"``).
        stream: Поток вывода; по умолчанию ``sys.stderr``.
    """
    global _configured  # noqa: PLW0603 - однократная настройка на процесс

    root = logging.getLogger("engine")
    root.setLevel(level)
    if not _configured:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)
        _configured = True
