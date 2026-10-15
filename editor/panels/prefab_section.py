"""Секция «Префаб» инспектора: экземпляр, его overrides и действия над ними."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from editor.widgets import FormSection
from engine.core.prefab_components import PrefabInstance

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from editor.context import EditorContext

__all__ = ["PrefabInstanceSection", "override_title"]


def override_title(rec: dict[str, Any]) -> str:
    """Короткая подпись записи override: ``L1 · MeshRenderer.color``."""
    field = rec.get("field")
    name = str(rec.get("component", ""))
    if rec.get("add"):
        name = f"+{name}"
    elif field:
        name = f"{name}.{field}"
    return f"{rec.get('target', '')} · {name}"


class PrefabInstanceSection(FormSection):
    """GUID, путь, состояние и список overrides экземпляра префаба.

    Список только для чтения; у каждой записи кнопки «Вернуть» (значение
    из префаба) и «Применить к префабу» (записать значение экземпляра в файл
    префаба). Устаревшие (stale) записи показываются отдельно, без кнопок.

    Args:
        context: Контекст редактора.
        component: Компонент-маркер корня экземпляра.
        parent: Родительский виджет.
        overrides: Живые overrides для показа (``None`` - не показывать).
        on_revert: Обработчик «Вернуть» (получает запись).
        on_apply: Обработчик «Применить к префабу» (получает запись).
    """

    def __init__(
        self,
        context: EditorContext,
        component: PrefabInstance,
        parent: QWidget | None = None,
        *,
        overrides: list[dict[str, Any]] | None = None,
        on_revert: Callable[[dict[str, Any]], None] | None = None,
        on_apply: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(context.tr("inspector.section.prefab"), parent)
        tr = context.tr
        live = overrides or []
        stale = [r for r in component.overrides if r.get("stale")]
        self.guid = self._value(component.prefab_guid or "-")
        self.path = self._value(component.prefab_path or "-")
        self.overrides_count = self._value(str(len(live)))
        self.state = self._value(
            tr("inspector.prefab.state.modified")
            if live
            else tr("inspector.prefab.state.clean")
        )
        self.form.addRow(tr("inspector.prefab.guid"), self.guid)
        self.form.addRow(tr("inspector.prefab.path"), self.path)
        self.form.addRow(tr("inspector.prefab.state"), self.state)
        self.form.addRow(tr("inspector.prefab.overrides"), self.overrides_count)
        self.revert_buttons: list[QPushButton] = []
        self.apply_buttons: list[QPushButton] = []
        self.rows: list[QLabel] = []
        for rec in live:
            self._add_row(context, rec, on_revert, on_apply)
        for rec in stale:
            text = f"{override_title(rec)} ({tr('inspector.prefab.stale')})"
            label = self._value(text)
            label.setProperty("role", "error")
            self.rows.append(label)
            self.form.addRow("", label)

    def _add_row(
        self,
        context: EditorContext,
        rec: dict[str, Any],
        on_revert: Callable[[dict[str, Any]], None] | None,
        on_apply: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        tr = context.tr
        label = self._value(override_title(rec))
        label.setToolTip(str(rec.get("value", "")))
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        self.rows.append(label)
        holder = QWidget(self)
        layout = QHBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        revert = QPushButton(tr("inspector.prefab.revert"), holder)
        apply = QPushButton(tr("inspector.prefab.apply"), holder)
        revert.clicked.connect(lambda _=False, r=rec: on_revert and on_revert(r))
        apply.clicked.connect(lambda _=False, r=rec: on_apply and on_apply(r))
        for button in (revert, apply):
            button.setMinimumWidth(button.sizeHint().width())
        layout.addWidget(revert)
        layout.addWidget(apply)
        layout.addStretch(1)
        self.revert_buttons.append(revert)
        self.apply_buttons.append(apply)
        self.form.addRow(label, holder)

    def _value(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label
