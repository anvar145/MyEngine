"""Проверки ошибок OpenGL и отладочный вывод драйвера.

На «слабом» железе ошибки GL легко остаются незамеченными и превращаются в
чёрный экран, поэтому в режиме разработки вызовы проверяются явно.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging

from engine.rendering.errors import RenderError

__all__ = [
    "check_gl_error",
    "disable_debug_output",
    "enable_debug_output",
    "get_gl",
    "gl_string",
]

_log = logging.getLogger("engine.rendering.gl")

_gl_module = None


def get_gl():
    """Возвращает модуль ``OpenGL.GL`` (импорт один раз).

    Повторный ``from OpenGL import GL`` в горячих путях дорог: PySide6 ставит
    хук на ``__import__`` (≈4.5 мкс на вызов, ~160 вызовов за кадр).
    """
    global _gl_module
    if _gl_module is None:
        from OpenGL import GL

        _gl_module = GL
    return _gl_module


#: Человекочитаемые имена кодов ошибок OpenGL.
_ERROR_NAMES = {
    0x0500: "GL_INVALID_ENUM",
    0x0501: "GL_INVALID_VALUE",
    0x0502: "GL_INVALID_OPERATION",
    0x0503: "GL_STACK_OVERFLOW",
    0x0504: "GL_STACK_UNDERFLOW",
    0x0505: "GL_OUT_OF_MEMORY",
    0x0506: "GL_INVALID_FRAMEBUFFER_OPERATION",
}


def check_gl_error(where: str = "") -> None:
    """Проверить очередь ошибок OpenGL и упасть с понятным сообщением.

    Args:
        where: Описание места вызова (для сообщения об ошибке).

    Raises:
        RenderError: Если драйвер вернул ошибку.
    """
    from OpenGL import GL

    errors: list[str] = []
    while True:
        code = GL.glGetError()
        if not code:
            break
        errors.append(f"{_ERROR_NAMES.get(code, hex(code))}")
        if len(errors) > 8:  # защита от бесконечного цикла на «залипшей» ошибке
            break
    if errors:
        location = f" ({where})" if where else ""
        raise RenderError(f"Ошибка OpenGL{location}: {', '.join(errors)}")


def gl_string(name: int) -> str:
    """Прочитать строку GL по имени константы (``GL_VERSION`` и т. п.)."""
    from OpenGL import GL

    value = GL.glGetString(name)
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def enable_debug_output() -> bool:
    """Включить отладочный вывод драйвера (``KHR_debug``), если он доступен.

    Returns:
        ``True``, если отладочный вывод включён.
    """
    from OpenGL import GL

    if not hasattr(GL, "glDebugMessageCallback"):
        return False
    try:
        GL.glEnable(GL.GL_DEBUG_OUTPUT)
        GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS)

        def _callback(source, message_type, msg_id, severity, length, message, user):
            text = message if isinstance(message, str) else str(message)
            _log.debug("GL debug [%s] %s", severity, text.strip())

        GL.glDebugMessageCallback(_callback, None)
        return True
    except Exception as exc:  # noqa: BLE001 - отладка не обязательна
        _log.debug("Отладочный вывод GL недоступен: %s", exc)
        return False


def disable_debug_output() -> None:
    """Выключить отладочный вывод драйвера."""
    from OpenGL import GL

    if hasattr(GL, "glDebugMessageCallback"):
        with contextlib.suppress(Exception):
            GL.glDisable(GL.GL_DEBUG_OUTPUT)


def void_pointer(offset: int) -> ctypes.c_void_p:
    """Обернуть смещение в ``c_void_p`` — так его ожидает PyOpenGL."""
    return ctypes.c_void_p(offset)
