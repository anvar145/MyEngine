"""Секции физических компонентов для инспектора объекта.

Три секции — ``RigidBodyComponent``, ``ColliderComponent`` и
``CharacterControllerComponent``. Значения применяются сразу (компонент сам
помечает тело на пересоздание); об изменении сообщает сигнал ``changed``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QRegularExpression, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from editor.panels.auto_section import ComponentSection, quat_to_euler
from editor.widgets import FocusDoubleSpinBox, VectorEditor
from engine.core.math3d import Quaternion, Vector3
from engine.core.physics_components import (
    BODY_TYPES,
    COMBINE_MODES,
    SHAPE_TYPES,
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
    has_character_rigidbody_conflict,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = [
    "PHYSICS_COMPONENTS",
    "CharacterControllerSection",
    "ColliderSection",
    "RigidBodySection",
]

#: Физические компоненты: ключ перевода меню «Добавить» → класс.
PHYSICS_COMPONENTS: dict[str, type] = {
    "rigidbody": RigidBodyComponent,
    "collider": ColliderComponent,
    "character": CharacterControllerComponent,
}

_MASK_ALL = 0xFFFFFFFF


class MaskEditor(QWidget):
    """Редактор 32-битной маски: hex-поле и кнопки «Все» / «Ничего».

    ``QSpinBox`` ограничен 31 битом, поэтому маска вводится строкой
    (``FFFFFFFF`` или ``0xFFFFFFFF``) и применяется по Enter или потере фокуса.

    Args:
        value: Начальное значение.
        all_text: Подпись кнопки «все биты».
        none_text: Подпись кнопки «ни одного бита».
        parent: Родительский виджет.

    Signals:
        value_changed: Значение изменено пользователем (аргумент — ``int``).
    """

    value_changed = Signal(object)

    def __init__(
        self,
        value: int,
        all_text: str,
        none_text: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.line_edit = QLineEdit(self)
        self.line_edit.setValidator(
            QRegularExpressionValidator(
                QRegularExpression(r"(0[xX])?[0-9a-fA-F]{0,8}"), self.line_edit
            )
        )
        self.all_button = QPushButton(all_text, self)
        self.none_button = QPushButton(none_text, self)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.all_button)
        layout.addWidget(self.none_button)
        self._value = int(value) & _MASK_ALL
        self._show()
        self.line_edit.editingFinished.connect(self._on_edited)
        self.all_button.clicked.connect(lambda: self._apply(_MASK_ALL))
        self.none_button.clicked.connect(lambda: self._apply(0))

    def value(self) -> int:
        """Текущее значение маски (0..0xFFFFFFFF)."""
        return self._value

    def set_value(self, value: int) -> None:
        """Установить значение без сигнала."""
        self._value = int(value) & _MASK_ALL
        self._show()

    def _show(self) -> None:
        self.line_edit.setText(f"0x{self._value:08X}")

    def _on_edited(self) -> None:
        text = self.line_edit.text().strip()
        try:
            value = int(text, 16)
        except ValueError:  # пустая строка или «0x»: вернуть прежнее значение
            self._show()
            return
        self._apply(value)

    def _apply(self, value: int) -> None:
        self._value = value & _MASK_ALL
        self._show()
        self.value_changed.emit(self._value)


class _PhysicsSection(ComponentSection):
    """База: хранит контекст и компонент, присваивает поля с сигналом.

    Заголовок с «Включён»/«Удалить» строит :class:`ComponentSection`.

    Секции контроллера и твёрдого тела (``shows_conflict``) под заголовком
    показывают красную подпись, когда у объекта активны оба компонента: движок не
    берёт такой объект в мир. Подпись только сообщает - правка полей не
    блокируется.
    """

    #: Показывать подпись конфликта CharacterController + RigidBody.
    shows_conflict = False

    #: Ключ подсказки галочки «Включён».
    enabled_hint_key = "inspector.physics.enabled_hint"

    def __init__(
        self,
        context: EditorContext,
        component: object,
        title_key: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(
            context,
            component,
            context.tr(title_key),
            parent,
            enabled_hint_key=self.enabled_hint_key,
        )
        if self.shows_conflict:
            self.conflict_label = QLabel(
                context.tr("inspector.physics.conflict.cc_rb"), self
            )
            self.conflict_label.setProperty("role", "error")
            self.conflict_label.setWordWrap(True)
            self._layout.insertWidget(1, self.conflict_label)  # сразу под заголовком
            self.refresh_conflict()

    def refresh_conflict(self) -> None:
        """Показать или спрятать подпись конфликта по текущему состоянию объекта."""
        if not self.shows_conflict:
            return
        owner = self._component.game_object  # type: ignore[attr-defined]
        self.conflict_label.setVisible(
            owner is not None and has_character_rigidbody_conflict(owner)
        )

    def _set(self, name: str, value: object) -> None:
        if getattr(self._component, name) != value:
            self.context.set_attr(self._component, name, value)
            self.changed.emit()

    def _add_mask(self, field: str, tooltip_key: str | None = None) -> MaskEditor:
        """Строка 32-битной маски для поля ``field`` (``layer`` или ``mask``)."""
        tr = self.context.tr
        editor = MaskEditor(
            getattr(self._component, field),
            tr("inspector.physics.mask_all"),
            tr("inspector.physics.mask_none"),
            self,
        )
        if tooltip_key is not None:
            editor.setToolTip(tr(tooltip_key))
        editor.value_changed.connect(lambda value: self._set(field, int(value)))
        self.form.addRow(tr(f"inspector.physics.{field}"), editor)
        return editor

    def _vector(
        self, label: str, name: str, *, minimum: float, maximum: float
    ) -> VectorEditor:
        editor = VectorEditor(minimum=minimum, maximum=maximum, step=0.1, parent=self)
        editor.set_value(Vector3(*getattr(self._component, name)))
        editor.value_changed.connect(lambda v: self._set(name, (v.x, v.y, v.z)))
        self.form.addRow(label, editor)
        return editor


class RigidBodySection(_PhysicsSection):
    """Свойства твёрдого тела: тип, масса, затухание, гравитация, сон."""

    shows_conflict = True

    def __init__(
        self,
        context: EditorContext,
        component: RigidBodyComponent,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, component, "inspector.physics.rigidbody", parent)
        tr = context.tr
        self.body_type = self.add_combo(
            tr("inspector.physics.body_type"),
            [(tr(f"inspector.physics.body_type.{t}"), t) for t in BODY_TYPES],
            component.body_type,
            lambda value: self._set("body_type", str(value)),
        )
        self.mass = self.add_spin(
            tr("inspector.physics.mass"),
            component.mass,
            lambda value: self._set("mass", float(value)),
            minimum=0.001,
            maximum=1_000_000.0,
            step=0.5,
            decimals=3,
        )
        self.linear_damping = self.add_spin(
            tr("inspector.physics.linear_damping"),
            component.linear_damping,
            lambda value: self._set("linear_damping", float(value)),
            maximum=100.0,
            step=0.01,
        )
        self.angular_damping = self.add_spin(
            tr("inspector.physics.angular_damping"),
            component.angular_damping,
            lambda value: self._set("angular_damping", float(value)),
            maximum=100.0,
            step=0.01,
        )
        self.gravity_scale = self.add_spin(
            tr("inspector.physics.gravity_scale"),
            component.gravity_scale,
            lambda value: self._set("gravity_scale", float(value)),
            minimum=-100.0,
            maximum=100.0,
            step=0.1,
        )
        self.use_gravity = self.add_check(
            tr("inspector.physics.use_gravity"),
            component.use_gravity,
            lambda value: self._set("use_gravity", bool(value)),
        )
        self.freeze_rotation = self.add_check(
            tr("inspector.physics.freeze_rotation"),
            component.freeze_rotation,
            lambda value: self._set("freeze_rotation", bool(value)),
        )
        self.can_sleep = self.add_check(
            tr("inspector.physics.can_sleep"),
            component.can_sleep,
            lambda value: self._set("can_sleep", bool(value)),
        )


class ColliderSection(_PhysicsSection):
    """Свойства коллайдера: форма, размеры, смещение, материал, слои."""

    def __init__(
        self,
        context: EditorContext,
        component: ColliderComponent,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, component, "inspector.physics.collider", parent)
        tr = context.tr
        self.shape = self.add_combo(
            tr("inspector.physics.shape"),
            [(tr(f"inspector.physics.shape.{s}"), s) for s in SHAPE_TYPES],
            component.shape,
            self._on_shape,
        )
        self.size = self._vector(
            tr("inspector.physics.size"), "size", minimum=0.001, maximum=10_000.0
        )
        self.radius = self.add_spin(
            tr("inspector.physics.radius"),
            component.radius,
            lambda value: self._set("radius", float(value)),
            minimum=0.001,
            maximum=10_000.0,
            step=0.05,
            decimals=3,
        )
        self.height = self.add_spin(
            tr("inspector.physics.height"),
            component.height,
            lambda value: self._set("height", float(value)),
            minimum=0.001,
            maximum=10_000.0,
            step=0.1,
            decimals=3,
        )
        self.offset = self._vector(
            tr("inspector.physics.offset"),
            "offset",
            minimum=-10_000.0,
            maximum=10_000.0,
        )
        self.rotation_offset = VectorEditor(
            minimum=-360.0, maximum=360.0, step=1.0, decimals=2, parent=self
        )
        self.rotation_offset.setToolTip(tr("inspector.physics.rotation_offset_hint"))
        self.rotation_offset.set_value(quat_to_euler(component.rotation_offset))
        self.rotation_offset.value_changed.connect(self._on_rotation_offset)
        self.form.addRow(tr("inspector.physics.rotation_offset"), self.rotation_offset)
        self.is_trigger = self.add_check(
            tr("inspector.physics.is_trigger"),
            component.is_trigger,
            lambda value: self._set("is_trigger", bool(value)),
            tooltip=tr("inspector.physics.is_trigger_hint"),
        )
        self.friction = self.add_spin(
            tr("inspector.physics.friction"),
            component.friction,
            lambda value: self._set("friction", float(value)),
            maximum=10.0,
            step=0.05,
        )
        self.restitution = self.add_spin(
            tr("inspector.physics.restitution"),
            component.restitution,
            lambda value: self._set("restitution", float(value)),
            maximum=1.0,
            step=0.05,
        )
        combine_items = [
            (tr(f"inspector.physics.combine.{m}"), m) for m in COMBINE_MODES
        ]
        self.friction_combine = self.add_combo(
            tr("inspector.physics.friction_combine"),
            combine_items,
            component.friction_combine,
            lambda value: self._set("friction_combine", str(value)),
            tooltip=tr("inspector.physics.combine_hint"),
        )
        self.restitution_combine = self.add_combo(
            tr("inspector.physics.restitution_combine"),
            combine_items,
            component.restitution_combine,
            lambda value: self._set("restitution_combine", str(value)),
            tooltip=tr("inspector.physics.combine_hint"),
        )
        self.layer = self._add_mask("layer", "inspector.physics.layer_hint")
        self.mask = self._add_mask("mask", "inspector.physics.mask_hint")
        self._update_shape_fields(component.shape)

    def _on_rotation_offset(self, value: Vector3) -> None:
        quat = Quaternion.from_euler(value.x, value.y, value.z)
        self._set("rotation_offset", quat.to_tuple())

    def _on_shape(self, value: object) -> None:
        self._set("shape", str(value))
        self._update_shape_fields(str(value))

    def _update_shape_fields(self, shape: str) -> None:
        """Включить только поля, нужные текущей форме."""
        self.size.setEnabled(shape == "box")
        self.radius.setEnabled(shape in ("sphere", "capsule"))
        self.height.setEnabled(shape == "capsule")


class CharacterControllerSection(_PhysicsSection):
    """Свойства контроллера персонажа.

    Выключенный контроллер «заморожен»: двойник остаётся в мире, но не двигается
    (``docs/CORE.md`` §7.2), поэтому у галочки своя подсказка.
    """

    shows_conflict = True
    enabled_hint_key = "inspector.physics.character.enabled_hint"

    def __init__(
        self,
        context: EditorContext,
        component: CharacterControllerComponent,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(context, component, "inspector.physics.character", parent)
        tr = context.tr
        self.radius = self.add_spin(
            tr("inspector.physics.radius"),
            component.radius,
            lambda value: self._set("radius", float(value)),
            minimum=0.01,
            maximum=100.0,
            step=0.05,
            decimals=3,
        )
        self.height = self.add_spin(
            tr("inspector.physics.height"),
            component.height,
            lambda value: self._set("height", float(value)),
            minimum=0.02,
            maximum=100.0,
            step=0.1,
            decimals=3,
        )
        self.step_offset = self.add_spin(
            tr("inspector.physics.step_offset"),
            component.step_offset,
            lambda value: self._set("step_offset", float(value)),
            maximum=10.0,
            step=0.05,
            decimals=3,
            tooltip=tr("inspector.physics.step_offset_hint"),
        )
        self.slope_limit = self.add_spin(
            tr("inspector.physics.slope_limit"),
            component.slope_limit,
            lambda value: self._set("slope_limit", float(value)),
            maximum=90.0,
            step=1.0,
            decimals=1,
            suffix="°",
        )
        self.skin_width = self.add_spin(
            tr("inspector.physics.skin_width"),
            component.skin_width,
            lambda value: self._set("skin_width", float(value)),
            minimum=0.001,
            maximum=1.0,
            step=0.005,
            decimals=3,
        )
        self._add_movement_fields(component)
        self.layer = self._add_mask("layer", "inspector.physics.layer_hint")
        self.mask = self._add_mask("mask", "inspector.physics.mask_hint")

    def _add_movement_fields(self, component: CharacterControllerComponent) -> None:
        """Скорости, прыжок, гравитация и привязка к земле.

        Границы виджетов совпадают с допустимыми значениями компонента
        (``clamp_character_field``): ввод вне диапазона до компонента не доходит.

        Args:
            component: Редактируемый компонент.
        """
        tr = self.context.tr

        def spin(name: str, **limits: float) -> FocusDoubleSpinBox:
            return self.add_spin(
                tr(f"inspector.physics.character.{name}"),
                getattr(component, name),
                lambda value: self._set(name, float(value)),
                tooltip=tr(f"inspector.physics.character.{name}_hint"),
                **limits,
            )

        def check(name: str) -> QCheckBox:
            return self.add_check(
                tr(f"inspector.physics.character.{name}"),
                getattr(component, name),
                lambda value: self._set(name, bool(value)),
                tooltip=tr(f"inspector.physics.character.{name}_hint"),
            )

        self.move_speed = spin("move_speed", maximum=100.0, step=0.25)
        self.sprint_multiplier = spin(
            "sprint_multiplier", minimum=1.0, maximum=10.0, step=0.1
        )
        self.jump_velocity = spin(
            "jump_velocity", minimum=0.01, maximum=100.0, step=0.25
        )
        self.use_gravity = check("use_gravity")
        self.gravity_scale = spin("gravity_scale", maximum=100.0, step=0.1)
        self.ground_check_distance = spin(
            "ground_check_distance", minimum=0.001, maximum=1.0, step=0.005,
            decimals=3,
        )
        self.air_control = spin("air_control", maximum=1.0, step=0.05)
        self.snap_to_ground = check("snap_to_ground")
        self.alignment_to_ground = check("alignment_to_ground")
