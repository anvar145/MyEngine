"""Секции источника света для инспектора объекта.

Свет — компонент объекта (:class:`~engine.rendering.lighting.LightComponent`):
положение и направление источника задаются трансформом объекта (свет смотрит
вдоль локальной оси +Z), остальные свойства редактируются здесь. Значения
применяются сразу; об изменении сообщает сигнал ``changed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from editor.widgets import ColorButton, FormSection
from engine.core.math3d import Vector3
from engine.rendering.lighting import (
    DirectionalLight,
    Light,
    PointLight,
    SpotLight,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext
    from editor.viewport.editor_viewport import EditorViewport
    from engine.core.game_object import GameObject

__all__ = ["LIGHT_TYPE_KEYS", "LightSection"]

#: Подписи типов источников (ключи перевода).
LIGHT_TYPE_KEYS: dict[str, str] = {
    "directional": "light.kind.directional",
    "point": "light.kind.point",
    "spot": "light.kind.spot",
    "ambient": "light.kind.ambient",
}

#: Максимум поля маски слоёв: Qt хранит значение как знаковый 32-битный int.
_MASK_MAX = 2_147_483_647


class LightSection(QWidget):
    """Свойства источника света: яркость, радиус, конус, тени, слои.

    Args:
        context: Контекст редактора.
        light: Редактируемый источник.
        viewport: Вьюпорт редактора (менеджер света, перерисовка).
        parent: Родительский виджет.

    Signals:
        changed: Параметр источника изменён (нужно перерисовать кадр).
    """

    changed = Signal()

    def __init__(
        self,
        context: EditorContext,
        light: Light,
        game_object: GameObject | None = None,
        viewport: EditorViewport | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.context = context
        self._light = light
        self._game_object = game_object
        self._viewport = viewport

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_main())
        if isinstance(light, (DirectionalLight, SpotLight)):
            hint = QLabel(self.tr("light.direction_from_rotation"), self)
            hint.setWordWrap(True)
            hint.setProperty("role", "hint")
            layout.addWidget(hint)
        if isinstance(light, SpotLight):
            layout.addWidget(self._build_cone())
        if isinstance(light, DirectionalLight):
            layout.addWidget(self._build_shadows())
        layout.addWidget(self._build_culling())

    def tr(self, key: str, **kwargs: object) -> str:  # noqa: A003 - привычное имя Qt
        """Перевести строку через переводчик редактора."""
        return self.context.tr(key, **kwargs)

    @property
    def light(self) -> Light:
        """Редактируемый источник."""
        return self._light

    # ------------------------------------------------------------------ #
    # Построение
    # ------------------------------------------------------------------ #
    def _build_main(self) -> FormSection:
        """Секция «Источник»: тип, цвет, яркость, радиус."""
        light = self._light
        title = self.tr(LIGHT_TYPE_KEYS.get(light.type_name, "light.section.identity"))
        section = FormSection(title, self)
        section.add_check(self.tr("light.enabled"), light.enabled, self._apply_enabled)
        color = ColorButton(light.color, section)
        color.set_label(self.tr("light.color"))
        color.color_changed.connect(self._apply_color)
        section.form.addRow(self.tr("light.color"), color)
        section.add_spin(
            self.tr("light.intensity"),
            light.intensity,
            self._apply_intensity,
            maximum=100.0,
        )
        if isinstance(light, (PointLight, SpotLight)):
            section.add_spin(
                self.tr("light.range"),
                light.range,
                self._apply_range,
                minimum=0.01,
                maximum=10_000.0,
                step=0.5,
                tooltip=self.tr("light.range_hint"),
            )
            section.add_spin(
                self.tr("light.falloff"),
                light.falloff,
                self._apply_falloff,
                minimum=0.1,
                maximum=16.0,
                tooltip=self.tr("light.falloff_hint"),
            )
        return section

    def _build_cone(self) -> FormSection:
        """Секция «Конус»: углы прожектора."""
        assert isinstance(self._light, SpotLight)
        section = FormSection(self.tr("light.section.cone"), self)
        self._inner = section.add_spin(
            self.tr("light.inner_angle"),
            self._light.inner_angle,
            self._apply_inner_angle,
            maximum=89.0,
            step=1.0,
            decimals=1,
            suffix="°",
            tooltip=self.tr("light.inner_hint"),
        )
        self._outer = section.add_spin(
            self.tr("light.outer_angle"),
            self._light.outer_angle,
            self._apply_outer_angle,
            minimum=0.1,
            maximum=89.9,
            step=1.0,
            decimals=1,
            suffix="°",
            tooltip=self.tr("light.outer_hint"),
        )
        return section

    def _build_shadows(self) -> FormSection:
        """Секция «Тени»: карта теней направленного источника."""
        assert isinstance(self._light, DirectionalLight)
        light = self._light
        section = FormSection(self.tr("light.section.shadows"), self)
        section.add_check(
            self.tr("light.cast_shadow"),
            light.cast_shadow,
            self._apply_cast_shadow,
            tooltip=self.tr("light.cast_shadow_hint"),
        )
        section.add_combo(
            self.tr("light.shadow_size"),
            [(f"{value}×{value}", value) for value in (256, 512, 1024, 2048)],
            light.shadow_map_size,
            self._apply_shadow_size,
            tooltip=self.tr("light.shadow_size_hint"),
        )
        section.add_spin(
            self.tr("light.shadow_distance"),
            light.shadow_distance,
            self._apply_shadow_distance,
            minimum=1.0,
            maximum=1000.0,
            step=5.0,
            decimals=1,
            tooltip=self.tr("light.shadow_distance_hint"),
        )
        section.add_spin(
            self.tr("light.shadow_bias"),
            light.shadow_bias,
            self._apply_shadow_bias,
            maximum=0.1,
            step=0.0005,
            decimals=5,
            tooltip=self.tr("light.bias_hint"),
        )
        section.add_spin(
            self.tr("light.normal_bias"),
            light.normal_bias,
            self._apply_normal_bias,
            maximum=1.0,
            step=0.01,
            decimals=3,
            tooltip=self.tr("light.normal_bias_hint"),
        )
        return section

    def _build_culling(self) -> FormSection:
        """Секция «Отсечение»: слои и динамическое освещение."""
        section = FormSection(self.tr("light.section.culling"), self)
        section.add_int(
            self.tr("light.layers"),
            min(max(0, self._light.layers), _MASK_MAX),
            self._apply_layers,
            maximum=_MASK_MAX,
            tooltip=self.tr("light.layers_hint"),
        )
        self._layers_label = QLabel(
            self.tr("light.layers_list", layers=_describe_layers(self._light)), section
        )
        self._layers_label.setProperty("role", "hint")
        section.form.addRow("", self._layers_label)

        # Динамическое освещение — главный переключатель производительности
        # на слабых устройствах, поэтому он под рукой.
        manager = self._viewport.light_manager() if self._viewport else None
        dynamic = section.add_check(
            self.tr("light.dynamic"),
            manager.dynamic if manager is not None else True,
            self._apply_dynamic,
            tooltip=self.tr("light.dynamic_hint"),
        )
        # Без GL-контекста менеджера ещё нет: переключатель выключаем.
        dynamic.setEnabled(manager is not None)
        return section

    # ------------------------------------------------------------------ #
    # Применение
    # ------------------------------------------------------------------ #
    def _set(self, field: str, value: object) -> None:
        """Записать поле источника командой с undo."""
        self.context.set_attr(
            self._light, field, value, game_object=self._game_object
        )

    def _touch(self) -> None:
        """Сообщить редактору об изменении источника."""
        self._layers_label.setText(
            self.tr("light.layers_list", layers=_describe_layers(self._light))
        )
        self.changed.emit()

    def _apply_enabled(self, value: bool) -> None:
        self._set("enabled", bool(value))
        self._touch()

    def _apply_color(self, color: Vector3) -> None:
        self._set("color", Vector3(color.x, color.y, color.z))
        self._touch()

    def _apply_intensity(self, value: float) -> None:
        self._set("intensity", float(value))
        self._touch()

    def _apply_range(self, value: float) -> None:
        if isinstance(self._light, (PointLight, SpotLight)):
            self._set("range", float(value))
            self._touch()

    def _apply_falloff(self, value: float) -> None:
        if isinstance(self._light, (PointLight, SpotLight)):
            self._set("falloff", float(value))
            self._touch()

    def _apply_inner_angle(self, value: float) -> None:
        if not isinstance(self._light, SpotLight):
            return
        inner = float(value)
        # Внешний угол обязан остаться больше внутреннего.
        if self._light.outer_angle <= inner:
            name = self.context.tr("undo.edit", name=self._light.type_name)
            with self.context.transaction(name):
                self._set("inner_angle", inner)
                self._set("outer_angle", min(89.9, inner + 1.0))
        else:
            self._set("inner_angle", inner)
        self._outer.blockSignals(True)
        self._outer.setValue(self._light.outer_angle)
        self._outer.blockSignals(False)
        self._touch()

    def _apply_outer_angle(self, value: float) -> None:
        if isinstance(self._light, SpotLight):
            self._set("outer_angle", max(float(value), self._light.inner_angle + 0.1))
            self._touch()

    def _apply_cast_shadow(self, value: bool) -> None:
        if isinstance(self._light, DirectionalLight):
            self._set("cast_shadow", bool(value))
            self._touch()

    def _apply_shadow_size(self, value: object) -> None:
        if isinstance(self._light, DirectionalLight) and value is not None:
            self._set("shadow_map_size", int(value))  # type: ignore[arg-type]
            self._touch()

    def _apply_shadow_distance(self, value: float) -> None:
        if isinstance(self._light, DirectionalLight):
            self._set("shadow_distance", float(value))
            self._touch()

    def _apply_shadow_bias(self, value: float) -> None:
        if isinstance(self._light, DirectionalLight):
            self._set("shadow_bias", float(value))
            self._touch()

    def _apply_normal_bias(self, value: float) -> None:
        if isinstance(self._light, DirectionalLight):
            self._set("normal_bias", float(value))
            self._touch()

    def _apply_layers(self, value: int) -> None:
        self._set("layers", int(value))
        self._touch()

    def _apply_dynamic(self, value: bool) -> None:
        """Включить или выключить динамическое освещение в рендерере."""
        manager = self._viewport.light_manager() if self._viewport else None
        if manager is None:
            return
        manager.dynamic = bool(value)
        manager.clear_cache()
        self._touch()

    def __repr__(self) -> str:
        return f"<LightSection источник='{self._light.type_name}'>"


def _describe_layers(light: Light) -> str:
    """Строка с перечнем слоёв, на которые действует свет."""
    from engine.rendering.lighting import ALL_LAYERS_MASK, LAYER_COUNT

    mask = int(light.layers) & ALL_LAYERS_MASK
    if mask == ALL_LAYERS_MASK:
        return "all"
    layers = [str(index) for index in range(LAYER_COUNT) if mask & (1 << index)]
    return ", ".join(layers) if layers else "—"
