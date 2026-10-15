"""Форма настроек импорта выбранного ассета (строится по схеме импортёра)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from editor.widgets import FocusComboBox, FocusDoubleSpinBox, FocusSpinBox, SectionBox
from engine.core.resources import (
    AssetImportError,
    ImportState,
    SettingSpec,
    settings_schema,
)

__all__ = ["ImportSettingsSection"]

#: Граница для чисел без заданного диапазона.
_UNBOUNDED = 1_000_000_000


class ImportSettingsSection(SectionBox):
    """Секция инспектора: настройки импорта файла и кнопка «Применить».

    Args:
        context: Контекст редактора (даёт менеджер импортов и переводы).
        path: Исходный файл ассета.
        parent: Родительский виджет.
    """

    def __init__(self, context, path: Path, parent: QWidget | None = None) -> None:
        super().__init__(context.tr("import.settings.title"), parent)
        self.context = context
        self.path = Path(path)
        self.guid: str | None = None
        self.specs: list[SettingSpec] = []
        self._readers: dict[str, Callable[[], Any]] = {}
        self._fields: dict[str, QWidget] = {}
        self._build()

    # ------------------------------------------------------------------ #
    # Построение
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        """Собрать поля по схеме или показать причину, почему их нет."""
        manager = self.context.imports
        self.info = QLabel(self)
        self.info.setWordWrap(True)
        self.info.setProperty("role", "hint")
        self.add_widget(self.info)
        if manager is None:
            self.info.setText(self.context.tr("import.settings.disabled"))
            return
        guid = manager.db.guid_of(self.path)
        importer = manager.db.importers.importer_for(self.path)
        if guid is None or importer is None:
            self.info.setText(self.context.tr("import.settings.no_importer"))
            return
        self.guid = guid
        self.specs = settings_schema(importer)
        current = manager.db.settings_of(guid)
        self.info.setText(self._status_text())

        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.form.setHorizontalSpacing(12)
        for spec in self.specs:
            field = self._make_field(spec, current.get(spec.name, spec.default))
            self._fields[spec.name] = field
            self.form.addRow(spec.label or spec.name, field)
        self.add_layout(self.form)

        buttons = QHBoxLayout()
        self.apply_button = QPushButton(self.context.tr("import.settings.apply"), self)
        self.apply_button.clicked.connect(self.apply)
        self.reset_button = QPushButton(self.context.tr("import.settings.reset"), self)
        self.reset_button.clicked.connect(self.reset)
        self.reimport_button = QPushButton(self.context.tr("import.reimport"), self)
        self.reimport_button.clicked.connect(self.reimport)
        for button in (self.apply_button, self.reset_button, self.reimport_button):
            buttons.addWidget(button)
        self.add_layout(buttons)

    def _status_text(self) -> str:
        """Строка «состояние импорта» (с текстом ошибки, если он упал)."""
        manager = self.context.imports
        if manager is None or self.guid is None:
            return ""
        state = manager.state(self.guid)
        title = self.context.tr(f"import.state.{state.value}")
        text = self.context.tr("import.settings.state", state=title)
        error = manager.failures().get(self.guid)
        if state is ImportState.FAILED and error:
            text += f"\n{error}"
        return text

    def refresh_state(self) -> None:
        """Обновить строку состояния после события импорта."""
        if self.guid is not None:
            self.info.setText(self._status_text())

    def _make_field(self, spec: SettingSpec, value: Any) -> QWidget:
        """Создать поле нужного вида и запомнить способ чтения значения."""
        name = spec.name
        if spec.kind == "bool":
            box = QCheckBox(self)
            box.setChecked(bool(value))
            self._readers[name] = box.isChecked
            return box
        if spec.kind == "int":
            spin = FocusSpinBox(self)
            spin.setRange(
                int(spec.minimum if spec.minimum is not None else -_UNBOUNDED),
                int(spec.maximum if spec.maximum is not None else _UNBOUNDED),
            )
            spin.setValue(int(value))
            self._readers[name] = spin.value
            return spin
        if spec.kind == "float":
            dspin = FocusDoubleSpinBox(self)
            dspin.setDecimals(4)
            dspin.setSingleStep(0.1)
            dspin.setRange(
                spec.minimum if spec.minimum is not None else -float(_UNBOUNDED),
                spec.maximum if spec.maximum is not None else float(_UNBOUNDED),
            )
            dspin.setValue(float(value))
            self._readers[name] = dspin.value
            return dspin
        if spec.kind == "choice":
            combo = FocusComboBox(self)
            for choice in spec.choices:
                combo.addItem(str(choice), choice)
            combo.setCurrentIndex(max(0, combo.findData(value)))
            self._readers[name] = combo.currentData
            return combo
        # str, list, any — текстовое поле; списки и «любое» — как JSON.
        edit = QLineEdit(self)
        if spec.kind == "str":
            edit.setText("" if value is None else str(value))
            self._readers[name] = edit.text
        else:
            edit.setText(json.dumps(value, ensure_ascii=False))
            self._readers[name] = lambda: json.loads(edit.text() or "null")
        return edit

    # ------------------------------------------------------------------ #
    # Действия
    # ------------------------------------------------------------------ #
    def values(self) -> dict[str, Any]:
        """Значения полей формы.

        Raises:
            ValueError: В JSON-поле некорректный текст.
        """
        return {name: reader() for name, reader in self._readers.items()}

    def apply(self) -> bool:
        """Записать настройки в ``.meta`` и переимпортировать ассет.

        Returns:
            ``True``, если настройки приняты.
        """
        manager = self.context.imports
        if manager is None or self.guid is None:
            return False
        try:
            manager.set_settings(self.guid, self.values())
        except (AssetImportError, ValueError) as exc:
            self.info.setText(
                self.context.tr("import.settings.invalid", error=str(exc))
            )
            return False
        self.refresh_state()
        return True

    def reset(self) -> None:
        """Вернуть поля к значениям по умолчанию (без записи)."""
        for spec in self.specs:
            field = self._fields[spec.name]
            value = spec.default
            if isinstance(field, QCheckBox):
                field.setChecked(bool(value))
            elif isinstance(field, QSpinBox):
                field.setValue(int(value))
            elif isinstance(field, QDoubleSpinBox):
                field.setValue(float(value))
            elif isinstance(field, FocusComboBox):
                field.setCurrentIndex(max(0, field.findData(value)))
            elif isinstance(field, QLineEdit):
                field.setText(str(value) if spec.kind == "str" else json.dumps(value))

    def reimport(self) -> None:
        """Принудительно переимпортировать ассет с текущими настройками."""
        manager = self.context.imports
        if manager is not None and self.guid is not None:
            manager.reimport(self.guid, priority=True)
            self.refresh_state()
