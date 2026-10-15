"""Порт графического контекста и сведения о драйвере.

Рендерер **не знает**, кто владеет контекстом: редактор (QOpenGLWidget),
плеер (GLFW/Win32) или headless-режим (offscreen surface). Всё, что ему нужно,
описано протоколом :class:`IRenderContextProvider` — это и есть точка
расширения, позволяющая собирать игру без PySide6 (см. §3 архитектуры).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from engine.rendering.settings import RenderSettings

__all__ = [
    "ContextInfo",
    "IRenderContextProvider",
    "IRenderSurface",
    "query_context_info",
]


@runtime_checkable
class IRenderSurface(Protocol):
    """Поверхность, в которую рисует движок."""

    def size(self) -> tuple[int, int]:
        """Размер поверхности в пикселях (ширина, высота)."""
        ...

    def dpi_scale(self) -> float:
        """Масштаб DPI (1.0 для обычного экрана)."""
        ...

    def is_offscreen(self) -> bool:
        """``True`` для рендера без окна (тесты, превью, golden-тесты)."""
        ...


@runtime_checkable
class IRenderContextProvider(Protocol):
    """Владелец OpenGL-контекста."""

    @property
    def settings(self) -> RenderSettings:
        """Настройки, с которыми создан контекст."""
        ...

    def create_context(self, surface: IRenderSurface | None = None) -> None:
        """Создать (или подготовить) контекст под указанную поверхность."""
        ...

    def make_current(self, surface: IRenderSurface | None = None) -> bool:
        """Сделать контекст текущим; ``False``, если не удалось."""
        ...

    def done_current(self) -> None:
        """Освободить контекст (``doneCurrent``)."""
        ...

    def swap_buffers(self, surface: IRenderSurface | None = None) -> None:
        """Показать готовый кадр."""
        ...

    def set_swap_interval(self, interval: int) -> None:
        """Включить/выключить вертикальную синхронизацию."""
        ...

    @property
    def is_valid(self) -> bool:
        """``True``, если контекст создан и готов к работе."""
        ...


@dataclass(slots=True)
class ContextInfo:
    """Сведения о работающем контексте (для логов и панели статистики)."""

    version: str = ""
    renderer: str = ""
    vendor: str = ""
    glsl: str = ""
    major: int = 0
    minor: int = 0

    @classmethod
    def parse(cls, version: str, renderer: str, vendor: str, glsl: str) -> ContextInfo:
        """Разобрать строки драйвера в структуру.

        Args:
            version: Строка ``GL_VERSION``.
            renderer: Строка ``GL_RENDERER``.
            vendor: Строка ``GL_VENDOR``.
            glsl: Строка ``GL_SHADING_LANGUAGE_VERSION``.

        Returns:
            Сведения о контексте с выделенными major/minor.
        """
        major = minor = 0
        head = version.split(" ", 1)[0]
        parts = head.split(".")
        if parts and parts[0].isdigit():
            major = int(parts[0])
        if len(parts) > 1 and parts[1].strip().isdigit():
            minor = int(parts[1])
        return cls(version=version, renderer=renderer, vendor=vendor, glsl=glsl,
                   major=major, minor=minor)

    def describe(self) -> str:
        """Однострочное описание для лога."""
        return f"{self.renderer} | GL {self.version} | GLSL {self.glsl}"


def query_context_info() -> ContextInfo:
    """Прочитать сведения о текущем контексте.

    Требует активного (текущего) контекста OpenGL.

    Returns:
        Сведения о драйвере и версии.
    """
    from OpenGL import GL

    def read(name: int) -> str:
        value = GL.glGetString(name)
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    return ContextInfo.parse(
        version=read(GL.GL_VERSION),
        renderer=read(GL.GL_RENDERER),
        vendor=read(GL.GL_VENDOR),
        glsl=read(GL.GL_SHADING_LANGUAGE_VERSION),
    )
