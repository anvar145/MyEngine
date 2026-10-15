"""Qt-адаптер графического контекста.

Единственное место, где рендеринг встречается с PySide6:

* :func:`apply_surface_format` настраивает формат поверхности (версия, профиль,
  буферы, MSAA) до создания первого окна;
* :class:`QtOffscreenContext` даёт контекст без окна — им пользуются тесты,
  превью ассетов, генерация иконок и скриншоты;
* :class:`QtRenderSurface` реализует порт поверхности для headless-рендера.

Игре (плееру) этот модуль не нужен: там контекст создаёт оконный бэкенд
(GLFW/Win32). Так ядро и рендерер остаются свободными от Qt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from engine.rendering.settings import RenderSettings

__all__ = [
    "QtOffscreenContext",
    "QtRenderSurface",
    "apply_surface_format",
    "make_surface_format",
]

_log = logging.getLogger("engine.rendering.qt")


def make_surface_format(settings: RenderSettings):
    """Собрать ``QSurfaceFormat`` по настройкам движка.

    Args:
        settings: Настройки рендеринга.

    Returns:
        Настроенный формат поверхности Qt.
    """
    from PySide6.QtGui import QSurfaceFormat

    surface_format = QSurfaceFormat()
    surface_format.setVersion(settings.major, settings.minor)
    surface_format.setProfile(
        QSurfaceFormat.CoreProfile
        if settings.core_profile
        else QSurfaceFormat.CompatibilityProfile
    )
    if settings.forward_compatible:
        surface_format.setOption(QSurfaceFormat.DeprecatedFunctions, False)
    surface_format.setDepthBufferSize(settings.depth_bits)
    surface_format.setStencilBufferSize(settings.stencil_bits)
    surface_format.setSwapInterval(settings.swap_interval)
    if settings.samples > 0:
        surface_format.setSamples(settings.samples)
    return surface_format


def apply_surface_format(settings: RenderSettings) -> None:
    """Сделать формат поверхности форматом по умолчанию для всего приложения.

    Вызывать **до** создания ``QApplication``/первого окна.

    Args:
        settings: Настройки рендеринга.
    """
    from PySide6.QtGui import QSurfaceFormat

    QSurfaceFormat.setDefaultFormat(make_surface_format(settings))
    _log.debug("Формат поверхности Qt: %s", settings.describe())


@dataclass(slots=True)
class QtRenderSurface:
    """Поверхность без окна (offscreen), реализующая порт ``IRenderSurface``.

    Attributes:
        width: Ширина буфера.
        height: Высота буфера.
        scale: Масштаб DPI (у offscreen всегда 1.0).
        offscreen: Признак отсутствия окна.
    """

    width: int = 1280
    height: int = 720
    scale: float = 1.0
    offscreen: bool = True

    def size(self) -> tuple[int, int]:
        """Размер поверхности в пикселях."""
        return self.width, self.height

    def dpi_scale(self) -> float:
        """Масштаб DPI."""
        return self.scale

    def is_offscreen(self) -> bool:
        """``True`` — рендер идёт без окна."""
        return self.offscreen

    def resize(self, width: int, height: int) -> None:
        """Изменить размер поверхности."""
        self.width = max(1, int(width))
        self.height = max(1, int(height))


@dataclass(slots=True)
class QtOffscreenContext:
    """Контекст OpenGL на offscreen-поверхности Qt.

    Пример:
        >>> with QtOffscreenContext((640, 360)) as context:
        ...     renderer.initialize()
        ...     stats = renderer.render(camera, items, context.surface.size())

    Attributes:
        settings: Настройки рендеринга.
        surface: Описание поверхности (размер и признак offscreen).
    """

    size: tuple[int, int] = (640, 360)
    settings: RenderSettings = field(default_factory=RenderSettings)
    surface: QtRenderSurface = field(init=False)
    _context: object = field(default=None, init=False, repr=False)
    _qt_surface: object = field(default=None, init=False, repr=False)
    _owns_application: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.surface = QtRenderSurface(width=self.size[0], height=self.size[1])

    # ------------------------------------------------------------------ #
    # Создание контекста
    # ------------------------------------------------------------------ #
    def create(self) -> bool:
        """Создать offscreen-поверхность и контекст OpenGL.

        Returns:
            ``True``, если контекст создан и готов к работе.
        """
        from PySide6.QtGui import QOffscreenSurface, QOpenGLContext
        from PySide6.QtWidgets import QApplication

        if QApplication.instance() is None:
            self._owns_application = True
            QApplication([])

        surface_format = make_surface_format(self.settings)
        qt_surface = QOffscreenSurface()
        qt_surface.setFormat(surface_format)
        qt_surface.create()
        if not qt_surface.isValid():
            _log.error("Offscreen-поверхность Qt не создана")
            return False

        context = QOpenGLContext()
        context.setFormat(surface_format)
        if not context.create():
            _log.error(
                "Не удалось создать контекст OpenGL %d.%d. Если платформа Qt — "
                "'offscreen', она не поддерживает GL: задайте QT_QPA_PLATFORM=windows",
                self.settings.major,
                self.settings.minor,
            )
            return False

        self._qt_surface = qt_surface
        self._context = context
        return True

    def make_current(self) -> bool:
        """Сделать контекст текущим (создавая его при необходимости)."""
        if self._context is None and not self.create():
            return False
        return bool(self._context.makeCurrent(self._qt_surface))

    def done_current(self) -> None:
        """Освободить текущий контекст."""
        if self._context is not None:
            self._context.doneCurrent()

    @property
    def is_valid(self) -> bool:
        """``True``, если контекст создан."""
        return self._context is not None and bool(self._context.isValid())

    @property
    def handle(self) -> object:
        """Внутренний ``QOpenGLContext`` (для интеграции с Qt)."""
        return self._context

    def __enter__(self) -> QtOffscreenContext:
        if not self.make_current():
            raise RuntimeError(
                "Не удалось активировать offscreen-контекст OpenGL: "
                "проверьте драйвер и QT_QPA_PLATFORM"
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        self.done_current()

    def dispose(self) -> None:
        """Освободить контекст и поверхность."""
        self.done_current()
        self._qt_surface = None
        self._context = None

    def __repr__(self) -> str:
        state = "valid" if self.is_valid else "no-context"
        return (
            f"<QtOffscreenContext {self.surface.width}x{self.surface.height}"
            f" {state}>"
        )
