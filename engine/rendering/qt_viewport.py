"""Вьюпорт редактора на ``QOpenGLWidget``.

Виджет — тонкая оболочка: он владеет Qt-контекстом, раздаёт события мыши
орбитальной камере и вызывает :class:`~engine.rendering.renderer.Renderer`.
Всё, что касается рендеринга, живёт в рендерере и потому одинаково работает и
в редакторе, и в плеере, и в headless-режиме.

Содержимое сцены задаётся извне двумя способами:

* :meth:`ViewportWidget.set_draw_items` — статический список объектов;
* :meth:`ViewportWidget.set_draw_provider` — функция, возвращающая список на
  каждый кадр (так вьюпорт связывается с ECS-сценой редактора);
* :meth:`ViewportWidget.set_instance_groups` — группы инстансной отрисовки.

Example:
    >>> widget = ViewportWidget()
    >>> widget.set_draw_items([(cube_mesh, red_material, identity_matrix())])
    >>> widget.start_animation(60)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from engine.core.math3d import Vector3
from engine.rendering.camera import Camera, OrbitCameraController
from engine.rendering.material import Material
from engine.rendering.mesh import GpuMesh
from engine.rendering.renderer import InstanceGroup, Renderer, RenderStats
from engine.rendering.screenshot import capture_framebuffer, save_png
from engine.rendering.settings import RenderSettings

__all__ = ["ViewportWidget"]

_log = logging.getLogger("engine.rendering.viewport")

#: Сдвиг курсора (пиксели), в пределах которого нажатие считается кликом.
CLICK_SLOP_PX = 4.0

DrawItemTuple = tuple[GpuMesh, Material, np.ndarray]
DrawProvider = Callable[[], Iterable[DrawItemTuple]]


def _entry_mode(entry: Sequence[object]) -> int | None:
    """Взять примитив OpenGL из записи очереди (4-й элемент, если он есть)."""
    if len(entry) > 3:
        mode = entry[3]
        return int(mode) if mode is not None else None  # type: ignore[arg-type]
    return None


def _entry_shadow(entry: Sequence[object]) -> tuple[bool, bool]:
    """Взять флаги ``(cast, receive)`` теней из записи (5-й элемент)."""
    if len(entry) > 4 and entry[4] is not None:
        cast, receive = entry[4]  # type: ignore[misc]
        return bool(cast), bool(receive)
    return True, True


def _submit_entries(
    renderer: Renderer,
    entries: Sequence[Sequence[object]],
    min_group: int,
) -> None:
    """Поставить объекты кадра в очередь, склеивая одинаковые в инстансы.

    Непрозрачные объекты с общими мешем и материалом (а их в сцене обычно
    большинство) уходят одним инстансным вызовом вместо сотен отдельных:
    для Python-рендерера на слабой iGPU это главный выигрыш по времени кадра.

    Args:
        renderer: Рендерер с открытым кадром.
        entries: Записи ``(меш, материал, матрица[, примитив[, (cast, receive)]])``.
        min_group: Минимальный размер группы для инстансинга; ``0`` — выкл.
    """
    groups: dict[tuple[int, int, bool, bool], list[Sequence[object]]] = {}
    if min_group > 0:
        for entry in entries:
            material = entry[1]
            if _entry_mode(entry) is None and not getattr(
                material, "transparent", False
            ):
                key = (id(entry[0]), id(material), *_entry_shadow(entry))
                groups.setdefault(key, []).append(entry)
    batched: set[int] = set()
    for members in groups.values():
        if len(members) < min_group:
            continue
        matrices = np.stack([np.asarray(entry[2]) for entry in members])
        cast, receive = _entry_shadow(members[0])
        renderer.draw_instanced(
            members[0][0],  # type: ignore[arg-type]
            members[0][1],  # type: ignore[arg-type]
            matrices,
            cast_shadow=cast,
            receive_shadow=receive,
        )
        batched.update(id(entry) for entry in members)
    for entry in entries:
        if id(entry) not in batched:
            cast, receive = _entry_shadow(entry)
            renderer.submit(
                entry[0],  # type: ignore[arg-type]
                entry[1],  # type: ignore[arg-type]
                entry[2],  # type: ignore[arg-type]
                mode=_entry_mode(entry),
                cast_shadow=cast,
                receive_shadow=receive,
            )


def _create_widget_base():
    """Импортировать базовый класс Qt лениво, чтобы модуль был безопасен вне GUI."""
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    return QOpenGLWidget


def _make_widget_class():
    """Собрать класс виджета после того, как стал доступен Qt."""
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QMouseEvent, QWheelEvent
    from PySide6.QtOpenGLWidgets import QOpenGLWidget

    class ViewportWidget(QOpenGLWidget):
        """Интерактивный вьюпорт: рендер, орбитальная камера, скриншоты."""

        frame_rendered = Signal(float)  # FPS последнего кадра

        def __init__(
            self,
            parent: object = None,
            *,
            settings: RenderSettings | None = None,
            camera: Camera | None = None,
            renderer: Renderer | None = None,
            shader_root: Path | str | None = None,
            on_ready: Callable[[Renderer], None] | None = None,
        ) -> None:
            super().__init__(parent)
            self.settings = settings or RenderSettings()
            self.camera = camera or Camera(position=Vector3(4.0, 3.0, 4.0))
            self.controller = OrbitCameraController(distance=6.0)
            self.controller.apply(self.camera)
            self.renderer = renderer or Renderer(
                settings=self.settings, shader_root=shader_root
            )
            self._on_ready = on_ready
            self._items: list[DrawItemTuple] = []
            self._instances: list[InstanceGroup] = []
            self._provider: DrawProvider | None = None
            self._pick_provider: Callable[[float, float], bool] | None = None
            self._drag_handler: object | None = None
            self._click_provider: Callable[[float, float], None] | None = None
            self._drag_active = False
            self._press_position: tuple[float, float] | None = None
            self._last_mouse: tuple[float, float] | None = None
            self._button = Qt.NoButton
            self._timer = QTimer(self)
            self._timer.timeout.connect(self.update)
            self.fps = 0.0
            self.stats: RenderStats | None = None
            #: С какого числа одинаковых объектов включать инстансинг (0 — выкл.);
            #: 4 выбрано замером на 1000 кубов (группы по 5–7 при 8 не склеивались).
            self.auto_instance_min = 4
            #: Прогонять кадр через пост-обработку (нужен ``hdr`` в настройках).
            self.post_enabled = False

        # -------------------------------------------------------------- #
        # Содержимое
        # -------------------------------------------------------------- #
        def set_draw_items(self, items: Sequence[DrawItemTuple]) -> None:
            """Задать статический список объектов для отрисовки."""
            self._items = list(items)
            self.update()

        def set_draw_provider(self, provider: DrawProvider | None) -> None:
            """Задать функцию, возвращающую объекты на каждый кадр."""
            self._provider = provider
            self.update()

        def set_pick_provider(
            self, provider: Callable[[float, float], bool] | None
        ) -> None:
            """Задать обработчик клика по гизмо.

            Провайдер вызывается при нажатии левой кнопки и получает координаты
            курсора. Он возвращает ``True``, если клик «поглощён» гизмо —
            тогда орбита не начинается: так выбирают объекты по иконке.
            """
            self._pick_provider = provider

        def set_drag_handler(self, handler: object | None) -> None:
            """Задать обработчик перетаскивания гизмо (оси трансформа).

            Обработчик имеет методы ``press(x, y) -> bool`` (``True`` — гизмо
            захвачено, камера не вращается), ``move(x, y)`` и ``release()``.
            """
            self._drag_handler = handler

        def set_click_provider(
            self, provider: Callable[[float, float], None] | None
        ) -> None:
            """Задать обработчик «чистого» клика (без перетаскивания).

            Вызывается при отпускании левой кнопки, если курсор почти не
            сдвинулся: так выбирают объекты, а орбита остаётся доступной.
            """
            self._click_provider = provider

        def set_instance_groups(self, groups: Sequence[InstanceGroup]) -> None:
            """Задать группы инстансной отрисовки (много одинаковых объектов)."""
            self._instances = list(groups)
            self.update()

        # -------------------------------------------------------------- #
        # Управление камерой
        # -------------------------------------------------------------- #
        def reset_view(self) -> None:
            """Вернуть камеру в исходное положение."""
            self.controller.reset()
            self.controller.apply(self.camera)
            self.update()

        def focus_on(self, center: Vector3, radius: float = 1.0) -> None:
            """Навести камеру на объект."""
            self.controller.focus_on(center, radius)
            self.controller.apply(self.camera)
            self.update()

        # -------------------------------------------------------------- #
        # Qt: инициализация и кадр
        # -------------------------------------------------------------- #
        def initializeGL(self) -> None:  # noqa: N802 - имя из Qt API
            """Инициализировать рендерер в созданном Qt-контексте."""
            self.renderer.initialize()
            _log.info("Вьюпорт готов: %s", self.renderer.info.describe())
            if self._on_ready is not None:
                self._on_ready(self.renderer)

        def resizeGL(self, width: int, height: int) -> None:  # noqa: N802 - имя из Qt API
            """Пересчитать размеры области вывода с учётом DPI."""
            scale = self.devicePixelRatioF()
            pixel_width = max(1, int(width * scale))
            pixel_height = max(1, int(height * scale))
            self.renderer.resize(pixel_width, pixel_height)
            self.camera.set_aspect(pixel_width, pixel_height)

        def paintGL(self) -> None:  # noqa: N802 - имя из Qt API
            """Отрисовать кадр."""
            import time

            started = time.perf_counter()
            scale = self.devicePixelRatioF()
            width = max(1, int(self.width() * scale))
            height = max(1, int(self.height() * scale))

            # FBO виджета не равен 0: к нему возвращаются после теневого прохода.
            self.renderer.state.set_default_framebuffer(
                self.defaultFramebufferObject()
            )
            self.renderer.begin_frame(
                self.camera, (width, height), post=self.post_enabled
            )
            _submit_entries(
                self.renderer, self._current_items(), self.auto_instance_min
            )
            for group in self._instances:
                if group.count:
                    self.renderer.draw_instanced(
                        group.mesh, group.material, group.matrices
                    )
            self.stats = self.renderer.end_frame()

            elapsed = time.perf_counter() - started
            if elapsed > 0.0:
                instant = 1.0 / elapsed
                self.fps = (
                    instant if self.fps == 0.0 else self.fps * 0.9 + instant * 0.1
                )
            self.frame_rendered.emit(self.fps)

        def _current_items(self) -> list[DrawItemTuple]:
            """Собрать объекты кадра (статический список или провайдер)."""
            if self._provider is not None:
                return list(self._provider())
            return self._items

        # -------------------------------------------------------------- #
        # Ввод
        # -------------------------------------------------------------- #
        def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
            """Запомнить кнопку и позицию для орбиты/панорамы."""
            self._button = event.button()
            self._last_mouse = (event.position().x(), event.position().y())
            self._press_position = self._last_mouse
            if (
                self._button == Qt.LeftButton
                and self._drag_handler is not None
                and not (event.modifiers() & Qt.AltModifier)
                and self._drag_handler.press(*self._last_mouse)
            ):
                self._drag_active = True
                self.update()
                event.accept()
                return
            # Гизмо получает приоритет: клик по иконке камеры — это выбор, а не
            # начало орбиты, иначе камера «уезжала» бы при выборе объекта.
            # Alt+ЛКМ всегда остаётся орбитой и гизмо не перехватывает.
            if (
                self._button == Qt.LeftButton
                and self._pick_provider is not None
                and not (event.modifiers() & Qt.AltModifier)
                and self._pick_provider(*self._last_mouse)
            ):
                self._last_mouse = None
                self._button = Qt.NoButton
                self.update()
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
            """Завершить перетаскивание (и обработать клик без сдвига)."""
            position = (event.position().x(), event.position().y())
            if self._drag_active:
                self._drag_active = False
                if self._drag_handler is not None:
                    self._drag_handler.release()
            elif (
                self._button == Qt.LeftButton
                and self._click_provider is not None
                and self._press_position is not None
                and abs(position[0] - self._press_position[0]) <= CLICK_SLOP_PX
                and abs(position[1] - self._press_position[1]) <= CLICK_SLOP_PX
                and not (event.modifiers() & Qt.AltModifier)
            ):
                self._click_provider(*position)
            self._button = Qt.NoButton
            self._last_mouse = None
            self._press_position = None
            self.update()
            super().mouseReleaseEvent(event)

        def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
            """Орбита (левая кнопка) или панорама (средняя/правая) камеры."""
            if self._drag_active:
                if self._drag_handler is not None:
                    self._drag_handler.move(event.position().x(), event.position().y())
                self.update()
                return
            if self._last_mouse is None:
                return
            x, y = event.position().x(), event.position().y()
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self._last_mouse = (x, y)

            if self._button == Qt.LeftButton:
                self.controller.orbit(dx, dy)
            elif self._button in (Qt.MiddleButton, Qt.RightButton):
                self.controller.pan(dx, dy, self.camera)
            self.controller.apply(self.camera)
            self.update()

        def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
            """Зум колесом мыши."""
            steps = event.angleDelta().y() / 120.0
            self.controller.zoom(steps)
            self.controller.apply(self.camera)
            self.update()

        def keyPressEvent(self, event) -> None:  # noqa: N802 - тип из Qt
            """Горячие клавиши: ``R`` — сброс вида."""
            if event.key() == Qt.Key_R:
                self.reset_view()
                return
            super().keyPressEvent(event)

        # -------------------------------------------------------------- #
        # Анимация и скриншоты
        # -------------------------------------------------------------- #
        def start_animation(self, fps: int = 60) -> None:
            """Запустить перерисовку по таймеру."""
            interval = max(1, int(1000 / max(1, fps)))
            self._timer.start(interval)

        def stop_animation(self) -> None:
            """Остановить перерисовку по таймеру."""
            self._timer.stop()

        def grab_frame(self) -> np.ndarray:
            """Прочитать текущий кадр вьюпорта.

            Returns:
                Массив ``(H, W, 4)`` uint8 (сверху вниз).
            """
            self.makeCurrent()
            try:
                self.renderer.begin_frame(
                    self.camera,
                    (max(1, int(self.width())), max(1, int(self.height()))),
                    post=self.post_enabled,
                )
                for entry in self._current_items():
                    self.renderer.submit(
                        entry[0], entry[1], entry[2], mode=_entry_mode(entry)
                    )
                for group in self._instances:
                    if group.count:
                        self.renderer.draw_instanced(
                            group.mesh, group.material, group.matrices
                        )
                self.renderer.end_frame()
                pixels = capture_framebuffer(
                    max(1, int(self.width())), max(1, int(self.height()))
                )
            finally:
                self.doneCurrent()
            return pixels

        def save_screenshot(self, path: Path | str) -> Path:
            """Сохранить текущий кадр в PNG."""
            return save_png(path, self.grab_frame(), flip_vertical=False)

    return ViewportWidget


_WIDGET_CLASS = None


def ViewportWidget(*args: object, **kwargs: object):  # noqa: N802 - фабрика класса
    """Создать вьюпорт (фабрика: Qt импортируется только при первом вызове).

    Returns:
        Экземпляр виджета :class:`QOpenGLWidget`.
    """
    global _WIDGET_CLASS  # noqa: PLW0603 - однократная сборка класса
    if _WIDGET_CLASS is None:
        _WIDGET_CLASS = _make_widget_class()
    return _WIDGET_CLASS(*args, **kwargs)
