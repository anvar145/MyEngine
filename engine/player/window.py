"""Окно плеера без Qt: протокол ``IWindow`` и реализация на GLFW."""

from __future__ import annotations

import logging
from typing import Protocol

from engine.core.input import MouseState

__all__ = ["GlfwWindow", "HiddenGlfwContext", "IWindow", "create_window"]

_log = logging.getLogger("engine.player.window")

#: Коды кнопок GLFW (0 — левая, 1 — правая, 2 — средняя).
_BUTTON_NAMES = {0: "left", 1: "right", 2: "middle"}


class IWindow(Protocol):
    """Окно с GL-контекстом."""

    def create(self, title: str, width: int, height: int) -> None:
        """Создать окно и контекст OpenGL 3.3 Core."""

    def poll_events(self) -> bool:
        """Обработать события; ``False`` — окно закрыто."""

    def make_context_current(self) -> None:
        """Сделать контекст окна текущим."""

    def swap_buffers(self) -> None:
        """Показать нарисованный кадр."""

    def size(self) -> tuple[int, int]:
        """Размер области кадра в пикселях."""

    def dpi_scale(self) -> float:
        """Масштаб экрана (1.0 = 96 dpi)."""

    def close(self) -> None:
        """Уничтожить окно."""


class GlfwWindow:
    """Реализация :class:`IWindow` поверх GLFW (extra ``[player]``).

    Args:
        vsync: Включить вертикальную синхронизацию.
        fullscreen: Полноэкранный режим на основном мониторе.
    """

    def __init__(self, vsync: bool = True, fullscreen: bool = False) -> None:
        self.vsync = vsync
        self.fullscreen = fullscreen
        self._glfw = None
        self._handle = None
        self.keys_pressed: set[int] = set()
        #: Нажатия с прошлого кадра (для хоткеев вроде F2); сбрасывает приложение.
        self.key_events: list[int] = []
        self.mouse = MouseState()
        #: Приёмник событий ввода (``GlfwInputBackend``); ``None`` - не слать.
        self.input_backend = None

    def create(self, title: str, width: int, height: int) -> None:
        """Создать окно; ошибка, если GLFW недоступен."""
        try:
            import glfw
        except ImportError as error:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Для окна плеера нужен пакет glfw (pip install .[player])"
            ) from error
        if not glfw.init():
            raise RuntimeError("Не удалось инициализировать GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        monitor = glfw.get_primary_monitor() if self.fullscreen else None
        handle = glfw.create_window(width, height, title, monitor, None)
        if not handle:
            glfw.terminate()
            raise RuntimeError("Не удалось создать окно OpenGL 3.3")
        self._glfw, self._handle = glfw, handle
        glfw.make_context_current(handle)
        glfw.swap_interval(1 if self.vsync else 0)
        glfw.set_key_callback(handle, self._on_key)
        glfw.set_cursor_pos_callback(handle, self._on_cursor)
        glfw.set_mouse_button_callback(handle, self._on_mouse_button)
        glfw.set_cursor_enter_callback(handle, self._on_cursor_enter)

    def _pixel_scale(self) -> tuple[float, float]:
        """Отношение размера framebuffer к размеру окна (HiDPI)."""
        win_w, win_h = self._glfw.get_window_size(self._handle)
        fb_w, fb_h = self._glfw.get_framebuffer_size(self._handle)
        return (fb_w / win_w if win_w else 1.0, fb_h / win_h if win_h else 1.0)

    def _on_cursor(self, _window: object, x: float, y: float) -> None:
        scale_x, scale_y = self._pixel_scale()
        self.mouse.move(x * scale_x, y * scale_y)
        if self.input_backend is not None:
            self.input_backend.on_cursor(x * scale_x, y * scale_y)

    def _on_mouse_button(
        self, _window: object, button: int, action: int, _mods: int
    ) -> None:
        name = _BUTTON_NAMES.get(button)
        if name is not None and action in (0, 1):
            self.mouse.button(name, action == 1)
            if self.input_backend is not None:
                self.input_backend.on_mouse_button(name, action == 1)

    def _on_cursor_enter(self, _window: object, entered: int) -> None:
        if not entered:
            self.mouse.leave()
            if self.input_backend is not None:
                self.input_backend.on_cursor_leave()

    def _on_key(
        self, _window: object, key: int, _scan: int, action: int, _mods: int
    ) -> None:
        if action == 1:
            self.keys_pressed.add(key)
            self.key_events.append(key)
        elif action == 0:
            self.keys_pressed.discard(key)
        if self.input_backend is not None and action in (0, 1):
            self.input_backend.on_key(key, action == 1)

    def poll_events(self) -> bool:
        """Обработать события; ``False`` — окно закрыто."""
        if self._handle is None:
            return False
        self._glfw.poll_events()
        return not self._glfw.window_should_close(self._handle)

    def request_close(self) -> None:
        """Попросить закрыть окно на следующем опросе."""
        if self._handle is not None:
            self._glfw.set_window_should_close(self._handle, True)

    def is_iconified(self) -> bool:
        """Свёрнуто ли окно (рендер можно приостановить)."""
        if self._handle is None:
            return False
        return bool(self._glfw.get_window_attrib(self._handle, self._glfw.ICONIFIED))

    def make_context_current(self) -> None:
        """Сделать контекст окна текущим."""
        self._glfw.make_context_current(self._handle)

    def swap_buffers(self) -> None:
        """Показать нарисованный кадр."""
        self._glfw.swap_buffers(self._handle)

    def size(self) -> tuple[int, int]:
        """Размер области кадра в пикселях."""
        width, height = self._glfw.get_framebuffer_size(self._handle)
        return int(width), int(height)

    def dpi_scale(self) -> float:
        """Масштаб экрана."""
        scale_x, _ = self._glfw.get_window_content_scale(self._handle)
        return float(scale_x)

    def close(self) -> None:
        """Уничтожить окно и освободить GLFW."""
        if self._handle is not None:
            self._glfw.destroy_window(self._handle)
            self._glfw.terminate()
            self._handle = None

    def __enter__(self) -> GlfwWindow:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class HiddenGlfwContext:
    """GL-контекст на невидимом окне GLFW для ``--headless`` без Qt.

    Собранная игра не содержит PySide6, поэтому offscreen-контекст Qt недоступен.
    Рисование идёт в ``RenderTarget``, окно нужно только ради контекста.
    """

    def __init__(self, width: int = 64, height: int = 64) -> None:
        self.width = width
        self.height = height
        self._glfw = None
        self._handle = None

    def __enter__(self) -> HiddenGlfwContext:
        try:
            import glfw
        except ImportError as error:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "Для --headless нужен PySide6 или пакет glfw (pip install .[player])"
            ) from error
        if not glfw.init():
            raise RuntimeError("Не удалось инициализировать GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        handle = glfw.create_window(self.width, self.height, "headless", None, None)
        if not handle:
            glfw.terminate()
            raise RuntimeError("Не удалось создать скрытый контекст OpenGL 3.3")
        glfw.make_context_current(handle)
        self._glfw, self._handle = glfw, handle
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            self._glfw.destroy_window(self._handle)
            self._glfw.terminate()
            self._handle = None


def create_window(
    title: str, width: int, height: int, *, fullscreen: bool = False
) -> GlfwWindow:
    """Создать окно по умолчанию (GLFW).

    Args:
        title: Заголовок.
        width: Ширина клиентской области.
        height: Высота клиентской области.
        fullscreen: Полноэкранный режим.

    Returns:
        Созданное окно.
    """
    window = GlfwWindow(fullscreen=fullscreen)
    window.create(title, width, height)
    return window
