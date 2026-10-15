"""Ошибки подсистемы рендеринга."""

from __future__ import annotations

__all__ = [
    "MeshUploadError",
    "RenderError",
    "ShaderCompileError",
    "ShaderLinkError",
    "UnsupportedFeatureError",
]


class RenderError(Exception):
    """Базовая ошибка рендеринга."""


class ShaderCompileError(RenderError):
    """Шейдер не скомпилировался; в сообщении — лог драйвера."""


class ShaderLinkError(RenderError):
    """Программа не слинковалась; в сообщении — лог драйвера."""


class MeshUploadError(RenderError):
    """Меш не удалось загрузить в GPU (нет данных или неизвестный атрибут)."""


class UnsupportedFeatureError(RenderError):
    """Запрошенная возможность не поддерживается драйвером."""
