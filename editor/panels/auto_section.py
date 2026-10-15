"""Авто-секция инспектора: виджеты строятся по дескрипторам полей.

Компонент, зарегистрированный через ``component_fields``, получает форму без
ручного кода: вид виджета выбирает ``FieldDescriptor.kind``, границы и шаг берутся
из ``range``/``step``, подсказка - из ``tooltip``. Секции с особым поведением
(физика, камера, свет) остаются ручными: авто-секция строится только для классов,
у которых своей секции нет.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from editor.theme import icon
from editor.widgets import FocusDoubleSpinBox, FormSection, VectorEditor
from engine.core.math3d import Quaternion, Vector3, quaternion_to_euler
from engine.core.reflection import (
    FieldDescriptor,
    FieldKind,
    evaluate_visible_if,
    registry,
)

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = [
    "AutoSection",
    "ComponentSection",
    "ListEditor",
    "has_auto_section",
    "quat_to_euler",
]

_BIG = 1_000_000.0  # замена бесконечной границы диапазона у спинбокса
_ASSET_SUFFIXES: dict[str, tuple[str, ...]] = {
    "audio": (".wav", ".ogg"),
    "sound_cue": (".cue.json",),
    "music": (".music.json",),
    "mesh": (".obj", ".glb", ".gltf", ".fbx"),
}
_OPEN_MIN = 1e-3  # нижняя граница поля с открытым минимумом (как у реестра)


def has_auto_section(component_type: type) -> bool:
    """Есть ли у класса дескрипторы, по которым можно построить секцию."""
    return bool(registry.fields(component_type))


def quat_to_euler(quat: object) -> Vector3:
    """Кватернион xyzw -> углы Эйлера в градусах для редактора."""
    values = tuple(quat)  # type: ignore[call-overload]
    if len(values) != 4 or not any(values):
        return Vector3(0.0, 0.0, 0.0)
    return quaternion_to_euler(Quaternion(*values))


def _pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


class ComponentSection(FormSection):
    """База секции компонента: заголовок с галочкой «Включён» и кнопкой «Удалить».

    Галочка применяется сразу (``component.enabled``), у выключенного компонента
    заголовок получает серую полосу. Удаление секция сама не делает, а шлёт
    ``remove_requested``: инспектор спросит подтверждение. Общая для ручных
    физических секций и :class:`AutoSection`.

    Args:
        context: Контекст редактора (переводы, уведомление о правке).
        component: Редактируемый компонент.
        title: Заголовок секции.
        parent: Родительский виджет.
        header: Показывать ли галочку и кнопку в заголовке.
        enabled_hint_key: Ключ перевода подсказки галочки.

    Signals:
        changed: Значение изменено пользователем.
    """

    changed = Signal()

    def __init__(
        self,
        context: EditorContext,
        component: object,
        title: str,
        parent: QWidget | None = None,
        *,
        header: bool = True,
        enabled_hint_key: str = "inspector.physics.enabled_hint",
    ) -> None:
        super().__init__(title, parent)
        self.context = context
        self._component = component
        if header and hasattr(component, "enabled"):
            self.add_header_controls(
                enabled=bool(component.enabled),  # type: ignore[attr-defined]
                enabled_text=context.tr("inspector.physics.enabled"),
                remove_text=context.tr("inspector.physics.remove"),
                enabled_tooltip=context.tr(enabled_hint_key),
                remove_tooltip=context.tr("inspector.physics.remove_hint"),
            )
            self.enabled_toggled.connect(self._on_enabled_toggled)

    @property
    def component(self) -> object:
        """Редактируемый компонент."""
        return self._component

    def _on_enabled_toggled(self, value: bool) -> None:
        """Галочка «Включён»: применить, показать серую полосу, пометить сцену."""
        self.context.set_attr(self._component, "enabled", bool(value))
        self.set_frozen(not value)
        self.changed.emit()


def _sample_kind(items: list) -> str:
    """Вид элементов списка по первому элементу: bool/int/float/str/vec3."""
    if not items:
        return "str"
    first = items[0]
    if isinstance(first, bool):
        return "bool"
    if isinstance(first, int):
        return "int"
    if isinstance(first, float):
        return "float"
    if isinstance(first, (tuple, list)) and len(first) == 3:
        return "vec3"
    return "str"


_LIST_DEFAULTS: dict[str, Any] = {
    "bool": False, "int": 0, "float": 0.0, "str": "", "vec3": (0.0, 0.0, 0.0),
}


class ListEditor(QWidget):
    """Редактор списка примитивов или ``Vector3`` с кнопками +, -, вверх, вниз.

    Вид элементов определяется по первому элементу (для пустого списка берётся
    ``kind``). Список после каждой правки отдаётся сигналом ``value_changed``.

    Args:
        value: Начальный список.
        kind: Вид элементов для пустого списка (``bool/int/float/str/vec3``).
        parent: Родительский виджет.

    Signals:
        value_changed: Список изменён (новый ``list``).
    """

    value_changed = Signal(object)

    def __init__(
        self,
        value: list,
        kind: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._items: list = list(value)
        self._kind = _sample_kind(self._items) if self._items else (kind or "str")
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(2)
        self.rows: list[QWidget] = []
        self.add_button = QPushButton(icon("plus"), "", self)
        self.add_button.setObjectName("list_add")
        self.add_button.clicked.connect(lambda _c=False: self.add_item())
        self._rebuild()

    def value(self) -> list:
        """Текущий список (копия)."""
        return list(self._items)

    def add_item(self) -> None:
        """Добавить элемент со значением по умолчанию."""
        self._items.append(_LIST_DEFAULTS[self._kind])
        self._commit()

    def remove_item(self, index: int) -> None:
        """Удалить элемент."""
        del self._items[index]
        self._commit()

    def move_item(self, index: int, delta: int) -> None:
        """Сдвинуть элемент на ``delta`` позиций (за границы не выходит)."""
        target = index + delta
        if 0 <= target < len(self._items):
            self._items[index], self._items[target] = (
                self._items[target], self._items[index]
            )
            self._commit()

    def set_item(self, index: int, value: Any) -> None:
        """Записать значение элемента (без перестроения виджетов)."""
        self._items[index] = value
        self.value_changed.emit(self.value())

    def _commit(self) -> None:
        self._rebuild()
        self.value_changed.emit(self.value())

    def _element_widget(self, index: int) -> QWidget:
        item = self._items[index]
        kind = self._kind
        if kind == "bool":
            check = QCheckBox(self)
            check.setChecked(bool(item))
            check.toggled.connect(lambda v, i=index: self.set_item(i, bool(v)))
            return check
        if kind in ("int", "float"):
            is_int = kind == "int"
            spin = FocusDoubleSpinBox(self)
            spin.setRange(-_BIG, _BIG)
            spin.setDecimals(0 if is_int else 3)
            spin.setSingleStep(1.0 if is_int else 0.1)
            spin.setValue(float(item))
            spin.valueChanged.connect(
                lambda v, i=index, n=is_int: self.set_item(
                    i, round(v) if n else float(v)
                )
            )
            return spin
        if kind == "vec3":
            vec = VectorEditor(minimum=-_BIG, maximum=_BIG, step=0.1, parent=self)
            vec.set_value(Vector3(*item))
            vec.value_changed.connect(
                lambda v, i=index: self.set_item(i, (v.x, v.y, v.z))
            )
            return vec
        edit = QLineEdit(str(item), self)
        edit.editingFinished.connect(
            lambda i=index, w=edit: self.set_item(i, w.text())
        )
        return edit

    def _button(self, name: str, slot) -> QPushButton:
        button = QPushButton(icon(name), "", self)
        button.setIconSize(QSize(14, 14))
        button.setFixedWidth(26)
        button.clicked.connect(lambda _c=False: slot())
        return button

    def _rebuild(self) -> None:
        for row in self.rows:
            self._outer.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self.rows = []
        self._outer.removeWidget(self.add_button)
        for index in range(len(self._items)):
            row = QWidget(self)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            layout.addWidget(self._element_widget(index), 1)
            layout.addWidget(
                self._button("chevron-up", lambda i=index: self.move_item(i, -1))
            )
            layout.addWidget(
                self._button("chevron-down", lambda i=index: self.move_item(i, 1))
            )
            layout.addWidget(self._button("minus", lambda i=index: self.remove_item(i)))
            self._outer.addWidget(row)
            self.rows.append(row)
        self._outer.addWidget(self.add_button)


class AutoSection(ComponentSection):
    """Секция компонента, построенная по его дескрипторам полей.

    Правка поля присваивается компоненту сразу (через ``setattr``, поэтому
    сработают его собственные уведомления) и сообщается сигналом ``changed``.
    Поле с ``readonly`` недоступно для правки, с ``visible_if`` - прячется вместе
    с подписью, когда условие ложно.

    Args:
        context: Контекст редактора (переводы).
        component: Компонент с зарегистрированными полями.
        parent: Родительский виджет.
        header: Показывать ли в заголовке «Включён»/«Удалить».

    Signals:
        changed: Значение поля изменено пользователем.
    """

    def __init__(
        self,
        context: EditorContext,
        component: object,
        parent: QWidget | None = None,
        *,
        header: bool = True,
    ) -> None:
        cls = type(component)
        super().__init__(
            context,
            component,
            context.tr(f"inspector.auto.{cls.__name__}") or "",
            parent,
            header=header,
        )
        if self.title.text() == f"inspector.auto.{cls.__name__}":
            self.title.setText(cls.__name__)  # перевода нет - имя класса
        self.widgets: dict[str, QWidget] = {}
        self.labels: dict[str, QLabel] = {}
        self._descriptors = registry.fields(cls)
        for desc in self._descriptors:
            self._add_field(desc)
        self.inherit_button: QPushButton | None = None
        self._add_inherit_button()
        self._update_visibility()

    # ------------------------------------------------------------ построение

    def _add_inherit_button(self) -> None:
        """Кнопка «Унаследовать от MeshRenderer» для MeshCollider без меша."""
        inheritable = getattr(self._component, "inheritable_mesh", None)
        if inheritable is None or inheritable() is None:
            return
        prefix = f"inspector.auto.{type(self._component).__name__}"
        button = QPushButton(self.context.tr(f"{prefix}.inherit"), self)
        button.setToolTip(self.context.tr(f"{prefix}.inherit_tip"))
        button.clicked.connect(self._inherit_mesh)
        self.form.addRow("", button)
        self.inherit_button = button
        self._refresh_inherit()

    def _refresh_inherit(self) -> None:
        if self.inherit_button is not None:
            self.inherit_button.setVisible(
                getattr(self._component, "mesh", None) is None
            )

    def _inherit_mesh(self) -> None:
        ref = self._component.inheritable_mesh()  # type: ignore[attr-defined]
        if ref is None:
            return
        desc = next(d for d in self._descriptors if d.name == "mesh")
        self._set(desc, ref)
        field = self.widgets.get("mesh")
        if field is not None and hasattr(field, "set_asset"):
            db = self.context.asset_db
            field.set_asset(ref, (lambda r: r.resolve(db)) if db else None)

    def _label(self, desc: FieldDescriptor) -> str:
        key = f"inspector.auto.{type(self._component).__name__}.{desc.name}"
        text = self.context.tr(key)
        return _pretty(desc.name) if text == key else text

    def _add_field(self, desc: FieldDescriptor) -> None:
        value = getattr(self._component, desc.name)
        label = self._label(desc)
        tooltip = desc.tooltip or ""
        kind = desc.kind
        widget: QWidget
        if kind is FieldKind.BOOL:
            widget = self.add_check(
                label, bool(value), lambda v, d=desc: self._set(d, bool(v)),
                tooltip=tooltip,
            )
        elif kind in (FieldKind.INT, FieldKind.FLOAT):
            widget = self._add_number(desc, label, value, tooltip)
        elif kind is FieldKind.ENUM:
            items = [(c, c) for c in desc.choices or ()]
            widget = self.add_combo(
                label, items, value, lambda v, d=desc: self._set(d, str(v)),
                tooltip=tooltip,
            )
        elif kind is FieldKind.STR:
            widget = QLineEdit(str(value), self)
            widget.setToolTip(tooltip)
            widget.editingFinished.connect(
                lambda d=desc, w=widget: self._set(d, w.text())
            )
            self.form.addRow(label, widget)
        elif kind is FieldKind.VEC3:
            widget = VectorEditor(
                minimum=-_BIG, maximum=_BIG, step=desc.step or 0.1, parent=self
            )
            widget.set_value(Vector3(*value))
            widget.setToolTip(tooltip)
            widget.value_changed.connect(
                lambda v, d=desc: self._set(d, (v.x, v.y, v.z))
            )
            self.form.addRow(label, widget)
        elif kind is FieldKind.QUAT:
            widget = VectorEditor(
                minimum=-_BIG, maximum=_BIG, step=desc.step or 1.0, parent=self
            )
            widget.set_value(quat_to_euler(value))
            widget.setToolTip(tooltip)
            widget.value_changed.connect(lambda v, d=desc: self._set_euler(d, v))
            self.form.addRow(label, widget)
        elif kind is FieldKind.LIST:
            widget = ListEditor(list(value), parent=self)
            widget.setToolTip(tooltip)
            widget.value_changed.connect(lambda v, d=desc: self._set(d, v))
            self.form.addRow(label, widget)
        elif kind is FieldKind.MASK32:
            from editor.panels.physics_sections import MaskEditor  # цикл импорта

            widget = MaskEditor(
                int(value),
                self.context.tr("inspector.physics.mask_all"),
                self.context.tr("inspector.physics.mask_none"),
                self,
            )
            widget.setToolTip(tooltip)
            widget.value_changed.connect(lambda v, d=desc: self._set(d, int(v)))
            self.form.addRow(label, widget)
        elif kind is FieldKind.ASSET_REF:
            widget = self._add_asset_ref(desc, label, value, tooltip)
        else:  # виды без редактора (списки, ссылки): только показать
            widget = QLabel(str(value), self)
            widget.setToolTip(tooltip)
            self.form.addRow(label, widget)
        if desc.readonly:
            widget.setEnabled(False)
        self.widgets[desc.name] = widget
        row_label = self.form.labelForField(widget)
        if isinstance(row_label, QLabel):
            self.labels[desc.name] = row_label

    def _add_number(
        self, desc: FieldDescriptor, label: str, value: float, tooltip: str
    ) -> QWidget:
        is_int = desc.kind is FieldKind.INT
        low, high = desc.range if desc.range is not None else (-_BIG, _BIG)
        if desc.exclusive_min:
            low = _OPEN_MIN
        high = _BIG if not math.isfinite(high) else high
        return self.add_spin(
            label,
            value,
            lambda v, d=desc, i=is_int: self._set(d, round(v) if i else float(v)),
            minimum=low,
            maximum=high,
            step=desc.step or (1.0 if is_int else 0.1),
            decimals=0 if is_int else 3,
            tooltip=tooltip,
        )

    # ------------------------------------------------------------- поведение

    def _add_asset_ref(
        self, desc: FieldDescriptor, label: str, value: object, tooltip: str
    ) -> QWidget:
        """Поле ссылки на ассет: drop файла нужного вида, кнопка очистки."""
        from editor.panels.script_fields import ReferenceField, make_asset_ref

        suffixes = _ASSET_SUFFIXES.get(desc.asset_kind or "")
        db = self.context.asset_db
        resolver = (lambda ref: ref.resolve(db)) if db is not None else None
        field = ReferenceField("asset", self)
        field.setToolTip(tooltip)
        field.set_asset(value, resolver)  # type: ignore[arg-type]

        def dropped(path: str) -> None:
            if suffixes is not None and not path.lower().endswith(suffixes):
                return
            ref = make_asset_ref(db, path)
            self._set(desc, ref)
            field.set_asset(ref, resolver)

        def cleared() -> None:
            self._set(desc, None)
            field.set_asset(None, resolver)

        field.dropped.connect(dropped)
        field.cleared.connect(cleared)
        self.form.addRow(label, field)
        return field

    def _set(self, desc: FieldDescriptor, value: object) -> None:
        if getattr(self._component, desc.name) != value:
            self.context.set_attr(self._component, desc.name, value)
            self._update_visibility()
            self._refresh_inherit()
            self.changed.emit()

    def _set_euler(self, desc: FieldDescriptor, euler: Vector3) -> None:
        """Углы Эйлера (градусы) из редактора -> кватернион компонента."""
        quat = Quaternion.from_euler(euler.x, euler.y, euler.z)
        self._set(desc, quat.to_tuple())

    def _update_visibility(self) -> None:
        """Показать поля, у которых ``visible_if`` истинно."""
        for desc in self._descriptors:
            if desc.visible_if is None:
                continue
            visible = evaluate_visible_if(
                desc.visible_if, lambda n: getattr(self._component, n)
            )
            self.widgets[desc.name].setVisible(visible)
            label = self.labels.get(desc.name)
            if label is not None:
                label.setVisible(visible)
