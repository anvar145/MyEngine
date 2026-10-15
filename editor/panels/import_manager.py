"""Панель «Импорт»: состояние ассетов, ошибки и ручные операции."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from editor.panels.base import DockPanel
from engine.core.resources import ImportState

__all__ = ["ImportPanel"]

#: Колонки таблицы: ключи перевода заголовков.
COLUMNS = ("import.col.path", "import.col.importer", "import.col.state")

#: Задержка перед перестройкой таблицы после серии событий (мс).
REFRESH_DELAY_MS = 80

#: Порядок значений фильтра: ``None`` — все состояния.
_FILTERS: tuple[ImportState | None, ...] = (
    None,
    ImportState.FAILED,
    ImportState.QUEUED,
    ImportState.RUNNING,
    ImportState.OK,
)


class ImportPanel(DockPanel):
    """Таблица ассетов с состоянием импорта и кнопками управления."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.imports", context, parent)
        self._guids: list[str] = []
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        self.filter_combo = QComboBox(container)
        for _state in _FILTERS:
            self.filter_combo.addItem("")
        self.filter_combo.currentIndexChanged.connect(self.refresh)
        self.reimport_selected_button = QPushButton(container)
        self.reimport_selected_button.clicked.connect(self.reimport_selected)
        self.reimport_all_button = QPushButton(container)
        self.reimport_all_button.clicked.connect(self.reimport_all)
        self.clear_button = QPushButton(container)
        self.clear_button.clicked.connect(self.clear_library)
        controls.addWidget(self.filter_combo, 1)
        for button in (
            self.reimport_selected_button,
            self.reimport_all_button,
            self.clear_button,
        ):
            controls.addWidget(button)
        layout.addLayout(controls)

        self.table = QTableWidget(0, len(COLUMNS), container)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.itemSelectionChanged.connect(self._show_error)
        layout.addWidget(self.table, 1)

        self.error_view = QPlainTextEdit(container)
        self.error_view.setReadOnly(True)
        self.error_view.setMaximumHeight(90)
        layout.addWidget(self.error_view)
        bottom = QHBoxLayout()
        self.progress = QProgressBar(container)
        self.copy_button = QPushButton(container)
        self.copy_button.clicked.connect(self.copy_error)
        self.summary = QLabel(container)
        self.summary.setProperty("role", "hint")
        bottom.addWidget(self.summary, 1)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.copy_button)
        layout.addLayout(bottom)
        self.setWidget(container)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(REFRESH_DELAY_MS)
        self._timer.timeout.connect(self.refresh)
        context.imports_changed.connect(self._schedule_refresh)
        context.imports_restarted.connect(self._schedule_refresh)
        self.retranslate()

    # ------------------------------------------------------------------ #
    def retranslate(self, *_args: object) -> None:
        """Перевести подписи панели."""
        super().retranslate()
        self.table.setHorizontalHeaderLabels([self.tr(key) for key in COLUMNS])
        for index, state in enumerate(_FILTERS):
            name = "filter.all" if state is None else f"state.{state.value}"
            self.filter_combo.setItemText(index, self.tr(f"import.{name}"))
        self.reimport_selected_button.setText(self.tr("import.reimport_selected"))
        self.reimport_all_button.setText(self.tr("import.reimport_all"))
        self.clear_button.setText(self.tr("import.clear_library"))
        self.copy_button.setText(self.tr("import.copy_error"))
        self.refresh()

    def _schedule_refresh(self) -> None:
        """Отложить перестройку: пачка событий даёт одно обновление."""
        self._timer.start()

    def refresh(self, *_args: object) -> None:
        """Перестроить таблицу по состоянию менеджера."""
        manager = self.context.imports
        wanted = _FILTERS[max(0, self.filter_combo.currentIndex())]
        selected = self.selected_guid()
        rows: list[tuple[str, str, str, ImportState]] = []
        if manager is not None:
            root = manager.db.root
            states = manager.states()
            for guid in manager.db.guids:
                handle = manager.db.resolve(guid)
                state = states.get(guid, ImportState.IDLE)
                if wanted is not None and state is not wanted:
                    continue
                try:
                    shown = handle.path.relative_to(root).as_posix()
                except ValueError:
                    shown = str(handle.path)
                rows.append((shown, handle.importer, guid, state))
            rows.sort()
        self._guids = [row[2] for row in rows]
        self.table.setRowCount(len(rows))
        for row, (shown, importer, _guid, state) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(shown))
            self.table.setItem(row, 1, QTableWidgetItem(importer))
            self.table.setItem(
                row, 2, QTableWidgetItem(self.tr(f"import.state.{state.value}"))
            )
        if selected in self._guids:
            self.table.selectRow(self._guids.index(selected))
        self._update_progress()
        self._show_error()

    def _update_progress(self) -> None:
        """Обновить полосу прогресса и сводку."""
        manager = self.context.imports
        if manager is None:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.summary.setText(self.tr("import.disabled"))
            return
        done, total = manager.progress()
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done if total else 0)
        self.summary.setText(
            self.tr(
                "import.summary",
                total=len(manager.db.guids),
                failed=len(manager.failures()),
            )
        )

    # ------------------------------------------------------------------ #
    def selected_guid(self) -> str | None:
        """GUID выделенной строки таблицы."""
        row = self.table.currentRow()
        if 0 <= row < len(self._guids) and self.table.selectionModel().hasSelection():
            return self._guids[row]
        return None

    def _show_error(self) -> None:
        """Показать текст ошибки выделенного ассета."""
        manager = self.context.imports
        guid = self.selected_guid()
        text = ""
        if manager is not None and guid is not None:
            text = manager.failures().get(guid, "")
        self.error_view.setPlainText(text)

    def copy_error(self) -> None:
        """Скопировать текст ошибки в буфер обмена."""
        QGuiApplication.clipboard().setText(self.error_view.toPlainText())

    def reimport_selected(self) -> None:
        """Переимпортировать выделенный ассет."""
        guid = self.selected_guid()
        if self.context.imports is not None and guid is not None:
            self.context.imports.reimport(guid, priority=True)

    def reimport_all(self) -> None:
        """Переимпортировать все ассеты проекта."""
        if self.context.imports is not None:
            self.context.imports.reimport_all()

    def clear_library(self) -> None:
        """Очистить кэш ``Library`` (после подтверждения) и пересобрать его."""
        manager = self.context.imports
        if manager is None:
            return
        answer = QMessageBox.question(
            self,
            self.tr("import.clear_library"),
            self.tr("import.clear_confirm"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            manager.rebuild_library()
