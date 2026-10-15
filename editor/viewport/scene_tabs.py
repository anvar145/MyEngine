"""Вкладки открытых сцен над окном сцены."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QTabBar,
    QToolButton,
    QWidget,
)

from editor.context import EditorContext

#: Пометка несохранённой сцены в заголовке вкладки.
DIRTY_MARK = "*"


class SceneTabs(QWidget):
    """Полоса вкладок: одна вкладка на открытую сцену и кнопка «+».

    Args:
        context: Контекст редактора (список открытых сцен).
        parent: Родительский виджет.

    Attributes:
        bar: Сам ``QTabBar``.
        add_button: Кнопка создания новой сцены.

    Signals:
        close_requested: Нажат крестик на вкладке (номер вкладки).
        new_requested: Нажата кнопка «+».
    """

    close_requested = Signal(int)
    new_requested = Signal()

    def __init__(self, context: EditorContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.context = context
        self.setObjectName("scene_tabs")
        self.bar = QTabBar(self)
        self.bar.setObjectName("scene_tab_bar")
        self.bar.setTabsClosable(True)
        self.bar.setMovable(False)
        self.bar.setExpanding(False)
        self.bar.setDrawBase(False)
        self.bar.setElideMode(Qt.TextElideMode.ElideRight)
        self.add_button = QToolButton(self)
        self.add_button.setObjectName("scene_tab_add")
        self.add_button.setText("+")
        self.add_button.setAutoRaise(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.bar)
        layout.addWidget(self.add_button)
        layout.addStretch(1)

        self.bar.currentChanged.connect(self._on_current_changed)
        self.bar.tabCloseRequested.connect(self.close_requested)
        self.add_button.clicked.connect(
            lambda _checked=False: self.new_requested.emit()
        )
        context.documents_changed.connect(self.refresh)
        self.refresh()

    def tab_titles(self) -> list[str]:
        """Заголовки вкладок слева направо."""
        return [self.bar.tabText(i) for i in range(self.bar.count())]

    def _title_of(self, index: int) -> str:
        document = self.context.documents[index]
        name = document.display_name(self.context.tr("app.untitled_scene"))
        return f"{name}{DIRTY_MARK}" if document.dirty else name

    def refresh(self) -> None:
        """Привести вкладки в соответствие со списком открытых сцен."""
        documents = self.context.documents
        self.bar.blockSignals(True)
        try:
            while self.bar.count() > len(documents):
                self.bar.removeTab(self.bar.count() - 1)
            while self.bar.count() < len(documents):
                self.bar.addTab("")
            for index, document in enumerate(documents):
                self.bar.setTabText(index, self._title_of(index))
                self.bar.setTabToolTip(
                    index, str(document.path) if document.path else ""
                )
            self.bar.setCurrentIndex(self.context.active_index)
        finally:
            self.bar.blockSignals(False)

    def _on_current_changed(self, index: int) -> None:
        """Пользователь выбрал вкладку: переключить сцену."""
        self.context.activate_document(index)
