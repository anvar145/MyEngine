"""Поля скрипта в инспекторе: переменные модуля, настраиваемые вручную.

Список полей берётся из исходника скрипта
(:func:`~engine.scripting.fields.scan_fields`), значение из кода — значение
по умолчанию, ручное — переопределяет его.
Ссылки на объекты и файлы задаются перетаскиванием в поле: объект — из иерархии,
файл — из панели ассетов.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QSize, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from editor.mime import OBJECT_MIME
from editor.theme import icon
from editor.widgets import FormSection, VectorEditor, fit_widths
from engine.core.game_object import GameObject
from engine.core.math3d import Vector3
from engine.core.resources import AssetRef
from engine.scripting.fields import ScriptField

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.scripting import ScriptComponent

__all__ = ["ReferenceField", "ScriptFieldsForm", "make_asset_ref"]

_INT_LIMIT = 2**31 - 1
_FLOAT_LIMIT = 1.0e9


def _same(a: Any, b: Any) -> bool:
    """Равенство значений поля; вектор сравнивается по числам."""

    def plain(v: Any) -> Any:
        if isinstance(v, Vector3):
            return (v.x, v.y, v.z)
        return tuple(v) if isinstance(v, list) else v
    return plain(a) == plain(b)


def make_asset_ref(db: Any, path: str) -> AssetRef:
    """``AssetRef`` для файла: GUID из базы, путь от ``Assets/`` (иначе как есть)."""
    if db is None:
        return AssetRef(path=path)
    guid = db.guid_of(path) or None
    try:
        rel = Path(path).resolve().relative_to(Path(db.root).resolve())
        return AssetRef(guid=guid, path=rel.as_posix())
    except ValueError:
        return AssetRef(guid=guid, path=path)


class ReferenceField(QFrame):
    """Поле-ссылка: «чип» с именем и кнопкой очистки, принимает drop.

    Args:
        kind: ``"object"`` (объект сцены) или ``"asset"`` (файл).
        parent: Родительский виджет.

    Signals:
        dropped: Брошено значение (id объекта или путь файла).
        cleared: Нажата кнопка очистки.
    """

    dropped = Signal(object)
    cleared = Signal()

    def __init__(self, kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 2, 0)
        layout.setSpacing(2)
        self.label = QLabel(self)
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.clear_button = QToolButton(self)
        self.clear_button.setText("×")
        self.clear_button.clicked.connect(self.cleared)
        layout.addWidget(self.label, 1)
        layout.addWidget(self.clear_button)
        self.setMinimumWidth(48)
        self.set_display("", broken=False, tooltip="")

    def set_display(self, text: str, *, broken: bool, tooltip: str) -> None:
        """Показать значение (пустая строка — «не задано»)."""
        self.label.setText(text)
        self.label.setProperty("role", "error" if broken else "")
        self.label.style().unpolish(self.label)
        self.label.style().polish(self.label)
        self.setToolTip(tooltip)
        self.clear_button.setVisible(bool(text))

    def set_asset(
        self,
        ref: AssetRef | None,
        resolve: Callable[[AssetRef], str | None] | None = None,
    ) -> bool:
        """Показать ``AssetRef``: имя файла, в подсказке путь и GUID.

        Args:
            ref: Ссылка (``None`` - не задано).
            resolve: ``ref -> guid | None``; без неё битость определяется по
                существованию файла.

        Returns:
            ``True``, если ссылка показана как битая (красная метка).
        """
        if ref is None:
            self.set_display("", broken=False, tooltip="")
            return False
        name = Path(ref.path).name if ref.path else (ref.guid or "")[:8]
        if ref.is_primitive:
            broken = False
        elif resolve is not None:
            broken = resolve(ref) is None
        else:
            broken = not (ref.path and Path(ref.path).is_file())
        lines = [ref.path or ""]
        if ref.guid:
            lines.append(f"GUID: {ref.guid}")
        self.set_display(name, broken=broken, tooltip="\n".join(lines))
        return broken

    # -- drop ------------------------------------------------------------ #
    def _payload(self, mime) -> Any:
        if self.kind == "object":
            if mime.hasFormat(OBJECT_MIME):
                try:
                    return int(bytes(mime.data(OBJECT_MIME)).decode("ascii"))
                except ValueError:
                    return None
            return None
        for url in mime.urls() if mime.hasUrls() else ():
            if url.isLocalFile() and Path(url.toLocalFile()).is_file():
                return str(Path(url.toLocalFile()))
        return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Принять только подходящий тип данных."""
        if self._payload(event.mimeData()) is not None:
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - API Qt
        """То же, пока курсор двигается над полем."""
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Принять значение и сообщить о нём."""
        payload = self._payload(event.mimeData())
        if payload is not None:
            event.acceptProposedAction()
            self.dropped.emit(payload)


class ScriptFieldsForm(FormSection):
    """Секция полей одного скрипта.

    Args:
        component: Компонент-скрипт.
        title: Заголовок секции.
        tr: Функция перевода строк интерфейса.
        on_modified: Вызывается после любой правки (например, ``notify_modified``).
        parent: Родительский виджет.
        asset_db: База ассетов проекта (GUID для брошенных файлов, проверка ссылок).
        on_changed: Вызывается после правки поля: ``(имя, было, стало)``, где
            ``None`` - нет переопределения (значение из кода).
    """

    reset_requested = Signal()

    def __init__(
        self,
        component: ScriptComponent,
        title: str,
        tr: Callable[..., str],
        on_modified: Callable[[], None],
        parent: QWidget | None = None,
        asset_db: Any = None,
        on_changed: Callable[[str, Any, Any], None] | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.component = component
        self._db = asset_db
        self._tr = tr
        self._on_modified = on_modified
        self._on_changed = on_changed
        self.widgets: dict[str, QWidget] = {}
        self._reset_buttons: dict[str, QPushButton] = {}
        self._fields = component.script_fields()
        for field in self._fields:
            self._add_row(field)
        fit_widths(self)

    @property
    def field_names(self) -> list[str]:
        """Имена показанных полей."""
        return [f.name for f in self._fields]

    # -- построение ------------------------------------------------------ #
    def _value(self, field: ScriptField) -> Any:
        return self.component.fields.get(field.name, field.default)

    @staticmethod
    def _bounds(field: ScriptField, limit: float) -> tuple[float, float]:
        """Границы спинбокса: ``range`` поля или общий предел."""
        low, high = field.range if field.range is not None else (-limit, limit)
        return max(low, -limit), min(high, limit)

    def _add_row(self, field: ScriptField) -> None:
        name = field.name
        value = self._value(field)
        tip = field.tooltip or ""
        if field.kind == "bool":
            widget: QWidget = self.add_check(
                "", bool(value), lambda v, n=name: self._set(n, bool(v)),
                label=name, tooltip=tip,
            )
        elif field.kind == "int":
            low, high = self._bounds(field, _INT_LIMIT)
            widget = self.add_int(
                name, int(value), lambda v, n=name: self._set(n, int(v)),
                minimum=int(low), maximum=int(high), tooltip=tip,
            )
        elif field.kind == "float":
            low, high = self._bounds(field, _FLOAT_LIMIT)
            widget = self.add_spin(
                name, float(value), lambda v, n=name: self._set(n, float(v)),
                minimum=low, maximum=high, step=field.step or 0.1, decimals=4,
                tooltip=tip,
            )
        elif field.kind == "str":
            edit = QLineEdit(str(value), self)
            edit.setMinimumWidth(0)
            edit.setToolTip(tip)
            edit.editingFinished.connect(lambda n=name, e=edit: self._set(n, e.text()))
            self.form.addRow(name, edit)
            widget = edit
        elif field.kind == "enum" and field.choices:
            widget = self.add_combo(
                name, [(c, c) for c in field.choices], value,
                lambda v, n=name: self._set(n, str(v)), tooltip=tip,
            )
        elif field.kind == "vec3":
            vec = VectorEditor(
                minimum=-_FLOAT_LIMIT, maximum=_FLOAT_LIMIT,
                step=field.step or 0.1, parent=self,
            )
            vec.set_value(Vector3(*(float(c) for c in value)))
            vec.setToolTip(tip)
            vec.value_changed.connect(
                lambda v, n=name: self._set(n, (v.x, v.y, v.z))
            )
            self.form.addRow(name, vec)
            widget = vec
        elif field.kind in ("object", "asset"):
            widget = self._add_reference(field)
        else:  # виды без редактора (цвет, кватернион, списки): только показать
            label = QLabel(str(value), self)
            label.setToolTip(tip)
            label.setEnabled(False)
            self.form.addRow(name, label)
            widget = label
        self.widgets[name] = widget
        reset = QPushButton(icon("rotate-ccw"), "", self)
        reset.setToolTip(self._tr("inspector.scripts.field_reset"))
        reset.setIconSize(QSize(14, 14))
        reset.setMaximumWidth(28)
        reset.clicked.connect(lambda _c=False, n=name: self._reset(n))
        self._reset_buttons[name] = reset
        self._place_reset(name, widget, reset)
        self._refresh_mark(name)

    def _place_reset(self, name: str, widget: QWidget, reset: QPushButton) -> None:
        """Положить кнопку сброса рядом с полем (в ту же ячейку формы)."""
        row, _role = self.form.getWidgetPosition(widget)
        self.form.removeWidget(widget)
        cell = QWidget(self)
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        widget.setParent(cell)
        layout.addWidget(widget, 1)
        layout.addWidget(reset)
        self.form.setWidget(row, QFormLayout.ItemRole.FieldRole, cell)

    def _add_reference(self, field: ScriptField) -> ReferenceField:
        name = field.name
        ref = ReferenceField(field.kind, self)
        ref.dropped.connect(lambda payload, n=name: self._on_drop(n, payload))
        ref.cleared.connect(lambda n=name: self._set(n, None))
        self.form.addRow(name, ref)
        self._show_reference(name, ref)
        return ref

    def _show_reference(self, name: str, ref: ReferenceField) -> None:
        value = self.component.fields.get(name)
        if ref.kind == "object":
            if isinstance(value, GameObject):
                broken = value.is_destroyed or value.scene is None
                ref.set_display(
                    f"{value.name} ({value.id})",
                    broken=broken,
                    tooltip=self._tr("inspector.scripts.field_object_hint"),
                )
            else:
                ref.set_display("", broken=False, tooltip="")
                ref.label.setText(self._tr("inspector.scripts.field_none"))
                ref.clear_button.hide()
        else:
            if isinstance(value, AssetRef):
                ref.set_asset(value, self._resolver())
            else:
                ref.set_display("", broken=False, tooltip="")
                ref.label.setText(self._tr("inspector.scripts.field_none"))
                ref.clear_button.hide()

    def _resolver(self) -> Callable[[AssetRef], str | None] | None:
        db = self._db
        return (lambda r: r.resolve(db)) if db is not None else None

    def _make_ref(self, path: str) -> AssetRef:
        return make_asset_ref(self._db, path)

    # -- правки ---------------------------------------------------------- #
    def _on_drop(self, name: str, payload: Any) -> None:
        if self.widgets[name].kind == "object":  # type: ignore[attr-defined]
            scene = self.component.scene
            target = scene.get_game_object(int(payload)) if scene else None
            if target is None:
                return
            self._set(name, target)
        else:
            field = next(f for f in self._fields if f.name == name)
            path = str(payload)
            if field.asset_kind == "prefab" and not path.lower().endswith(
                ".prefab.json"
            ):
                return  # в поле префаба принимаются только *.prefab.json
            self._set(name, self._make_ref(path))

    def _set(self, name: str, value: Any) -> None:
        """Применить значение к компоненту и обновить форму."""
        field = next(f for f in self._fields if f.name == name)
        if name not in self.component.fields and _same(value, field.default):
            return  # правка не изменила значение из кода — не переопределяем
        before = self.component.fields.get(name)
        if value is None:
            if not self.component.set_field(name, None):
                return
        elif not self.component.set_field(name, value):
            return
        if before is value or _same(before, value):
            return
        widget = self.widgets.get(name)
        if isinstance(widget, ReferenceField):
            self._show_reference(name, widget)
        self._refresh_mark(name)
        self._on_modified()
        if self._on_changed is not None:
            self._on_changed(name, before, self.component.fields.get(name))

    def _reset(self, name: str) -> None:
        """Вернуть значение из кода; форму пересобирает владелец."""
        before = self.component.fields.get(name)
        if self.component.clear_field(name):
            self._on_modified()
            if self._on_changed is not None:
                self._on_changed(name, before, None)
        self.reset_requested.emit()

    def _refresh_mark(self, name: str) -> None:
        overridden = name in self.component.fields
        button = self._reset_buttons[name]
        button.setVisible(overridden)
        widget = self.widgets[name]
        widget.setProperty("overridden", overridden)
        widget.setStyleSheet("font-weight: bold;" if overridden else "")

