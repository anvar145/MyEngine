"""Секции камеры для инспектора объекта.

Камера — обычный компонент объекта (:class:`~engine.rendering.CameraComponent`),
поэтому её свойства живут в инспекторе рядом с трансформом, а не в отдельной
панели. Камера объекта никак не связана с камерой редактора: она целиком
определяется трансформом своего объекта.

Значения применяются к камере сразу; об изменении сообщает сигнал ``changed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from editor.widgets import ColorButton, FormSection, VectorEditor
from engine.core.math3d import Vector3
from engine.rendering.camera import (
    ASPECT_FIXED,
    ASPECT_VIEWPORT,
    PROJECTION_MODES,
    Camera,
    ProjectionMode,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext
    from editor.viewport.editor_viewport import EditorViewport
    from engine.core.game_object import GameObject

__all__ = ["CameraSection"]

#: Максимум поля маски слоёв: Qt хранит значение как знаковый 32-битный int.
_MASK_MAX = 2_147_483_647


class CameraSection(QWidget):
    """Свойства камеры объекта: проекция, вьюпорт, отсечение.

    Args:
        context: Контекст редактора.
        camera: Редактируемая камера.
        game_object: Объект-владелец камеры.
        viewport: Вьюпорт редактора (размер области, перерисовка, фокус).
        parent: Родительский виджет.

    Signals:
        changed: Параметр камеры изменён (нужно перерисовать кадр).
    """

    changed = Signal()

    def __init__(
        self,
        context: EditorContext,
        camera: Camera,
        game_object: GameObject | None = None,
        viewport: EditorViewport | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.context = context
        self._camera = camera
        self._game_object = game_object
        self._viewport = viewport

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_projection())
        layout.addWidget(self._build_viewport())
        layout.addWidget(self._build_culling())

    def tr(self, key: str, **kwargs: object) -> str:  # noqa: A003 - привычное имя Qt
        """Перевести строку через переводчик редактора."""
        return self.context.tr(key, **kwargs)

    @property
    def camera(self) -> Camera:
        """Редактируемая камера."""
        return self._camera

    # ------------------------------------------------------------------ #
    # Построение
    # ------------------------------------------------------------------ #
    def _build_projection(self) -> FormSection:
        """Секция «Камера»: включённость, порядок, проекция, плоскости."""
        camera = self._camera
        section = FormSection(self.tr("camera.section.projection"), self)
        section.add_check(
            self.tr("camera.enabled"), camera.enabled, self._apply_enabled
        )
        section.add_int(
            self.tr("camera.depth"),
            camera.depth,
            self._apply_depth,
            minimum=-100,
            maximum=100,
            tooltip=self.tr("camera.depth_hint"),
        )
        section.add_combo(
            self.tr("camera.projection"),
            [(self.tr(f"camera.mode.{mode}"), mode) for mode in PROJECTION_MODES],
            camera.mode,
            self._apply_mode,
        )
        self._fov = section.add_spin(
            self.tr("camera.fov"),
            camera.fov,
            self._apply_fov,
            minimum=1.0,
            maximum=179.0,
            step=1.0,
            decimals=1,
            suffix="°",
        )
        self._fov.setEnabled(camera.mode == ProjectionMode.PERSPECTIVE)
        self._ortho = section.add_spin(
            self.tr("camera.ortho_height"),
            camera.ortho_height,
            self._apply_ortho_height,
            minimum=0.01,
            maximum=10_000.0,
            step=0.5,
        )
        self._ortho.setEnabled(camera.is_orthographic)
        section.add_spin(
            self.tr("camera.near"),
            camera.near,
            self._apply_near,
            minimum=0.001,
            maximum=10_000.0,
            decimals=3,
        )
        section.add_spin(
            self.tr("camera.far"),
            camera.far,
            self._apply_far,
            minimum=0.01,
            maximum=100_000.0,
            step=10.0,
        )
        return section

    def _build_viewport(self) -> FormSection:
        """Секция «Вьюпорт»: прямоугольник, aspect, очистка."""
        camera = self._camera
        section = FormSection(self.tr("camera.section.viewport"), self)

        rect = VectorEditor(
            minimum=0.0,
            maximum=10_000.0,
            step=0.05,
            decimals=3,
            labels=("X", "Y", "W"),
            parent=section,
        )
        x, y, width, height = camera.viewport_rect
        rect.set_value(Vector3(x, y, width))
        rect.value_changed.connect(self._apply_viewport_rect)
        rect.setToolTip(self.tr("camera.viewport_hint"))
        section.form.addRow(self.tr("camera.viewport"), rect)
        section.add_spin(
            self.tr("camera.viewport_height"),
            height,
            self._apply_viewport_height,
            maximum=100_000.0,
            step=0.05,
            decimals=3,
        )
        section.add_combo(
            self.tr("camera.aspect_mode"),
            [
                (self.tr("camera.aspect.viewport"), ASPECT_VIEWPORT),
                (self.tr("camera.aspect.fixed"), ASPECT_FIXED),
            ],
            camera.aspect_mode,
            self._apply_aspect_mode,
            tooltip=self.tr("camera.aspect_hint"),
        )
        section.add_spin(
            self.tr("camera.aspect"),
            camera.effective_aspect,
            self._apply_aspect,
            minimum=0.01,
            maximum=100.0,
            step=0.05,
            decimals=3,
        )
        color = ColorButton(Vector3(*camera.clear_color[:3]), section)
        color.set_label(self.tr("camera.clear_color"))
        color.color_changed.connect(self._apply_clear_color)
        section.form.addRow(self.tr("camera.clear_color"), color)
        section.add_check(
            self.tr("camera.clear_color_flag"),
            camera.clears_color,
            lambda value: self._apply_clear_flag("color", value),
        )
        section.add_check(
            self.tr("camera.clear_depth_flag"),
            camera.clears_depth,
            lambda value: self._apply_clear_flag("depth", value),
        )
        return section

    def _build_culling(self) -> FormSection:
        """Секция «Отсечение»: маска слоёв и переход к камере."""
        camera = self._camera
        section = FormSection(self.tr("camera.section.culling"), self)
        section.add_int(
            self.tr("camera.culling_mask"),
            min(max(0, camera.culling_mask), _MASK_MAX),
            self._apply_culling_mask,
            maximum=_MASK_MAX,
            tooltip=self.tr("camera.culling_mask_hint"),
        )
        self._layers_label = QLabel(
            self.tr("camera.culling_layers", layers=_describe_mask(camera)), section
        )
        self._layers_label.setProperty("role", "hint")
        section.form.addRow("", self._layers_label)

        focus = QPushButton(self.tr("camera.focus"), section)
        focus.clicked.connect(self._focus_on_camera)
        section.add_widget(focus)
        return section

    # ------------------------------------------------------------------ #
    # Применение
    # ------------------------------------------------------------------ #
    def _set(self, field: str, value: object) -> None:
        """Записать поле камеры командой с undo."""
        self.context.set_attr(
            self._camera, field, value, game_object=self._game_object
        )

    def _touch(self, *, invalidate: bool = False) -> None:
        """Сообщить редактору об изменении камеры."""
        if invalidate:
            self._camera.invalidate()
        self._layers_label.setText(
            self.tr("camera.culling_layers", layers=_describe_mask(self._camera))
        )
        self.changed.emit()

    def _apply_enabled(self, value: bool) -> None:
        self._set("enabled", bool(value))
        self._touch()

    def _apply_depth(self, value: int) -> None:
        self._set("depth", int(value))
        self._touch()

    def _apply_mode(self, mode: object) -> None:
        if mode is None:
            return
        self._set("mode", str(mode))
        self._fov.setEnabled(self._camera.mode == ProjectionMode.PERSPECTIVE)
        self._ortho.setEnabled(self._camera.is_orthographic)
        self._touch(invalidate=True)

    def _apply_fov(self, value: float) -> None:
        self._set("fov", float(value))
        self._touch(invalidate=True)

    def _apply_ortho_height(self, value: float) -> None:
        self._set("ortho_height", float(value))
        self._touch(invalidate=True)

    def _apply_near(self, value: float) -> None:
        self._set("near", float(value))
        self._touch(invalidate=True)

    def _apply_far(self, value: float) -> None:
        self._set("far", float(value))
        self._touch(invalidate=True)

    def _apply_viewport_rect(self, value: Vector3) -> None:
        height = self._camera.viewport_rect[3]
        self._set(
            "viewport_rect",
            (float(value.x), float(value.y), float(value.z), height),
        )
        self._touch()

    def _apply_viewport_height(self, value: float) -> None:
        x, y, width, _height = self._camera.viewport_rect
        self._set("viewport_rect", (x, y, width, float(value)))
        self._touch()

    def _apply_aspect_mode(self, mode: object) -> None:
        if mode is None:
            return
        self._set("aspect_mode", str(mode))
        if self._camera.aspect_mode == ASPECT_VIEWPORT:
            self._camera.set_aspect(*self._viewport_size())
        self._touch()

    def _apply_aspect(self, value: float) -> None:
        self._set("aspect", float(value))
        self._touch(invalidate=True)

    def _apply_clear_color(self, color: Vector3) -> None:
        alpha = self._camera.clear_color[3]
        self._set("clear_color", (color.x, color.y, color.z, alpha))
        self._touch()

    def _apply_clear_flag(self, name: str, value: bool) -> None:
        flags = [flag for flag in self._camera.clear_flags if flag != name]
        if value:
            flags.append(name)
        self._set("clear_flags", tuple(flags))
        self._touch()

    def _apply_culling_mask(self, value: int) -> None:
        self._set("culling_mask", int(value))
        self._touch()

    def _focus_on_camera(self) -> None:
        """Навести камеру редактора на камеру объекта."""
        if self._viewport is None:
            return
        self._viewport.viewport.focus_on(self._camera.position, 1.5)
        self._viewport.refresh()

    def _viewport_size(self) -> tuple[int, int]:
        """Размер области вывода редактора (для автоматического aspect)."""
        if self._viewport is None:
            return (1280, 720)
        widget = self._viewport.viewport
        return (max(1, int(widget.width())), max(1, int(widget.height())))

    def __repr__(self) -> str:
        return f"<CameraSection камера='{self._camera.name}'>"


def _describe_mask(camera: Camera) -> str:
    """Строка с перечнем слоёв, которые пропускает маска камеры."""
    from engine.rendering.camera import ALL_LAYERS_MASK, LAYER_COUNT

    mask = int(camera.culling_mask) & ALL_LAYERS_MASK
    if mask == ALL_LAYERS_MASK:
        return "all"
    layers = [str(index) for index in range(LAYER_COUNT) if mask & (1 << index)]
    return ", ".join(layers) if layers else "—"
