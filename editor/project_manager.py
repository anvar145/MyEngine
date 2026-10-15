"""Окно менеджера проектов: показывается при запуске до главного окна редактора.

Окно перечисляет проекты из ``<рабочий корень>/projects``, позволяет открыть
любой из них, создать новый (с проверкой имени), найти проект по имени,
отсортировать список, показать папку в проводнике и удалить проект. Файловая
логика целиком в :mod:`editor.project`, здесь только интерфейс.

Example:
    >>> choice = choose_project(Path("."), "ru", QSettings("MyEngine", "Editor"))
    >>> choice.project if choice else None
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRect, QSettings, QSize, Qt
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from editor.anim import fade_in
from editor.i18n import LANGUAGES, Translator
from editor.project import (
    PROJECT_TEMPLATES,
    create_project,
    list_projects,
    scenes_root_for,
    scripts_root_for,
    validate_project_name,
)
from editor.theme import PALETTE, icon
from engine import __version__

__all__ = [
    "LauncherChoice",
    "ProjectManagerDialog",
    "choose_project",
    "resolve_launch_project",
]

_log = logging.getLogger("editor.project_manager")

#: Ключи ``QSettings``.
KEY_LAST_PROJECT = "launcher/last_project"
KEY_AUTO_OPEN = "launcher/auto_open"
KEY_LANGUAGE = "launcher/language"

_SORT_MODES = ("recent", "modified", "name")


@dataclass(frozen=True)
class LauncherChoice:
    """Результат работы менеджера проектов.

    Attributes:
        project: Выбранный или созданный проект.
        language: Язык интерфейса, выбранный в окне.
    """

    project: Path
    language: str


@dataclass(frozen=True)
class _ProjectInfo:
    """Сводка по проекту для списка и карточки."""

    path: Path
    modified: float
    scenes: int
    scripts: int

    @property
    def name(self) -> str:
        return self.path.name


def _scan(path: Path) -> _ProjectInfo:
    """Собрать сводку по проекту (ошибки ФС не критичны)."""

    def count(folder: Path, pattern: str) -> int:
        try:
            return sum(1 for _ in folder.rglob(pattern))
        except OSError:
            return 0

    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0
    return _ProjectInfo(
        path,
        modified,
        count(scenes_root_for(path), "*.scene.json"),
        count(scripts_root_for(path), "*.py"),
    )


def _format_time(stamp: float) -> str:
    """Дата и время изменения в локальном формате."""
    return datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M")


_TEMPLATE_ICONS = {"empty": "file", "3d": "box", "2d": "layers"}

#: Высота строки проекта и размер плитки-иконки (px).
_ROW_HEIGHT = 68
_TILE = 40


class _ProjectDelegate(QStyledItemDelegate):
    """Рисует строку проекта карточкой: плитка-иконка, имя, путь и дата.

    Всё рисуется напрямую ``QPainter`` без виджетов-на-строку, поэтому список
    остаётся быстрым при любом числе проектов.
    """

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), _ROW_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        info = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(info, _ProjectInfo):
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        card = QRect(option.rect).adjusted(2, 3, -2, -3)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        fill = PALETTE["accent_soft"] if selected else PALETTE["surface"]
        if hovered and not selected:
            fill = PALETTE["hover"]
        border = PALETTE["accent"] if selected else PALETTE["border"]
        painter.setBrush(QColor(fill))
        painter.setPen(QPen(QColor(border), 1))
        painter.drawRoundedRect(card.adjusted(0, 0, -1, -1), 12, 12)

        tile = QRect(card.left() + 14, card.top() + (card.height() - _TILE) // 2,
                     _TILE, _TILE)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(PALETTE["surface_alt"]))
        painter.drawRoundedRect(tile, 10, 10)
        pixmap = icon("box", "accent").pixmap(22, 22)
        painter.drawPixmap(
            tile.center().x() - 11, tile.center().y() - 11, pixmap
        )

        text_left = tile.right() + 14
        date_width = 130
        name_rect = QRect(
            text_left, card.top() + 11, card.right() - text_left - date_width - 8, 22
        )
        font = QFont(option.font)
        font.setBold(True)
        font.setPixelSize(14)
        painter.setFont(font)
        painter.setPen(QColor(PALETTE["text"]))
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            painter.fontMetrics().elidedText(
                info.name, Qt.TextElideMode.ElideRight, name_rect.width()
            ),
        )
        small = QFont(option.font)
        small.setPixelSize(12)
        painter.setFont(small)
        painter.setPen(QColor(PALETTE["text_muted"]))
        path_rect = QRect(text_left, card.top() + 34, name_rect.width(), 20)
        painter.drawText(
            path_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            painter.fontMetrics().elidedText(
                str(info.path), Qt.TextElideMode.ElideMiddle, path_rect.width()
            ),
        )
        date_rect = QRect(
            card.right() - date_width - 14, card.top() + 11, date_width, 22
        )
        painter.drawText(
            date_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            _format_time(info.modified),
        )
        painter.restore()


class ProjectManagerDialog(QDialog):
    """Диалог выбора или создания проекта.

    Args:
        workspace_root: Рабочий корень, рядом с которым лежит ``projects/``.
        language: Начальный язык интерфейса.
        settings: Хранилище настроек (последний проект, автооткрытие).
        parent: Родительский виджет.

    Attributes:
        choice: Результат после закрытия диалога по «Открыть»/«Создать».
    """

    def __init__(
        self,
        workspace_root: Path,
        language: str,
        settings: QSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_root = Path(workspace_root)
        self.settings = settings
        self.translator = Translator(language, parent=self)
        self.choice: LauncherChoice | None = None
        self._infos: list[_ProjectInfo] = []
        self._build_ui()
        self.reload()
        self.retranslate()
        self.resize(820, 520)

    # ------------------------------------------------------------------ #
    # Построение
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_projects_page())
        self.pages.addWidget(self._build_create_page())
        root.addWidget(self.pages, 1)

        self.search.textChanged.connect(self._refill)
        self.sort_box.currentIndexChanged.connect(self._refill)
        self.list.currentItemChanged.connect(self._update_details)
        self.list.itemDoubleClicked.connect(lambda _item: self.open_selected())
        self.list.customContextMenuRequested.connect(self._show_menu)
        self.open_button.clicked.connect(self.open_selected)
        self.name_edit.textChanged.connect(self._validate_name)
        self.name_edit.returnPressed.connect(self.create_new)
        self.create_button.clicked.connect(self.create_new)
        self.empty_create_button.clicked.connect(lambda: self.show_page(1))
        self.nav_projects.clicked.connect(lambda: self.show_page(0))
        self.nav_create.clicked.connect(lambda: self.show_page(1))
        self.open_folder_button.clicked.connect(self.open_folder)
        self.language_box.currentIndexChanged.connect(self._on_language)
        self.exit_button.clicked.connect(self.reject)

    def _build_sidebar(self) -> QFrame:
        """Левая панель: бренд, разделы, «Открыть папку…», настройки."""
        sidebar = QFrame()
        sidebar.setObjectName("launcherSidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(4)

        brand = QHBoxLayout()
        logo = QLabel()
        logo.setObjectName("launcherLogo")
        logo.setPixmap(icon("box", "on_accent").pixmap(20, 20))
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setFixedSize(34, 34)
        self.title_label = QLabel("MyEngine")
        self.title_label.setObjectName("launcherTitle")
        brand.addWidget(logo)
        brand.addWidget(self.title_label, 1)
        layout.addLayout(brand)
        layout.addSpacing(18)

        group = QButtonGroup(self)
        self.nav_projects = QPushButton()
        self.nav_create = QPushButton()
        for button, icon_name in (
            (self.nav_projects, "folder"),
            (self.nav_create, "plus"),
        ):
            button.setProperty("role", "nav")
            button.setCheckable(True)
            button.setIcon(icon(icon_name, "text_muted"))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            group.addButton(button)
            layout.addWidget(button)
        self.nav_projects.setChecked(True)
        self.open_folder_button = QPushButton()
        self.open_folder_button.setProperty("role", "nav")
        self.open_folder_button.setIcon(icon("folder-open", "text_muted"))
        self.open_folder_button.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.open_folder_button)

        layout.addStretch(1)
        self.auto_open = QCheckBox()
        self.auto_open.setChecked(self.settings.value(KEY_AUTO_OPEN, False, bool))
        layout.addWidget(self.auto_open)
        layout.addSpacing(6)
        self.language_label = QLabel()
        self.language_label.setProperty("role", "hint")
        self.language_box = QComboBox()
        for code, label in self.translator.available_languages:
            self.language_box.addItem(label, code)
        self.language_box.setCurrentIndex(
            max(0, self.language_box.findData(self.translator.language))
        )
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_box)
        layout.addSpacing(6)
        self.exit_button = QPushButton()
        self.exit_button.setProperty("role", "nav")
        self.exit_button.setIcon(icon("x", "text_muted"))
        layout.addWidget(self.exit_button)
        self.version_label = QLabel()
        self.version_label.setProperty("role", "hint")
        layout.addWidget(self.version_label)
        return sidebar

    def _build_projects_page(self) -> QWidget:
        """Страница списка проектов: поиск, сортировка, список, «Открыть»."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(10)
        self.heading = QLabel()
        self.heading.setObjectName("launcherHeading")
        self.subtitle = QLabel()
        self.subtitle.setProperty("role", "hint")
        layout.addWidget(self.heading)
        layout.addWidget(self.subtitle)
        layout.addSpacing(6)

        self.search = QLineEdit()
        self.search.setClearButtonEnabled(True)
        self.sort_box = QComboBox()
        for mode in _SORT_MODES:
            self.sort_box.addItem("", mode)
        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.sort_box)
        layout.addLayout(top)

        self.list = QListWidget()
        self.list.setObjectName("projectList")
        self.list.setItemDelegate(_ProjectDelegate(self.list))
        self.list.setMouseTracking(True)
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.empty_state = self._build_empty_state()
        self.list_stack = QStackedWidget()
        self.list_stack.addWidget(self.list)
        self.list_stack.addWidget(self.empty_state)
        layout.addWidget(self.list_stack, 1)

        footer = QHBoxLayout()
        self.details = QLabel()
        self.details.setProperty("role", "hint")
        self.details.setWordWrap(True)
        self.open_button = QPushButton()
        self.open_button.setProperty("role", "primary")
        self.open_button.setDefault(True)
        self.open_button.setMinimumWidth(120)
        footer.addWidget(self.details, 1)
        footer.addWidget(self.open_button)
        layout.addLayout(footer)
        return page

    def _build_empty_state(self) -> QWidget:
        """Пустое состояние: иконка, пояснение и крупная кнопка создания."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)
        glyph = QLabel()
        glyph.setPixmap(icon("folder-plus", "text_dim").pixmap(48, 48))
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_title = QLabel()
        self.empty_title.setObjectName("launcherHeading")
        self.empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint = QLabel()
        self.empty_hint.setProperty("role", "hint")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setWordWrap(True)
        self.empty_create_button = QPushButton()
        self.empty_create_button.setProperty("role", "primary")
        self.empty_create_button.setMinimumWidth(180)
        layout.addWidget(glyph)
        layout.addWidget(self.empty_title)
        layout.addWidget(self.empty_hint)
        layout.addSpacing(10)
        layout.addWidget(self.empty_create_button, 0, Qt.AlignmentFlag.AlignCenter)
        return box

    def _build_create_page(self) -> QWidget:
        """Страница создания проекта: имя, шаблон, кнопка «Создать»."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(10)
        self.new_heading = QLabel()
        self.new_heading.setObjectName("launcherHeading")
        self.new_subtitle = QLabel()
        self.new_subtitle.setProperty("role", "hint")
        layout.addWidget(self.new_heading)
        layout.addWidget(self.new_subtitle)
        layout.addSpacing(10)
        self.name_edit = QLineEdit()
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setProperty("role", "error")
        self.error_label.setMinimumHeight(18)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.error_label)

        self.template_label = QLabel()
        self.template_label.setProperty("role", "section")
        layout.addWidget(self.template_label)
        cards = QHBoxLayout()
        cards.setSpacing(10)
        group = QButtonGroup(self)
        self.template_buttons: dict[str, QPushButton] = {}
        for name in PROJECT_TEMPLATES:
            button = QPushButton()
            button.setProperty("role", "template")
            button.setCheckable(True)
            button.setIcon(icon(_TEMPLATE_ICONS[name], "accent"))
            button.setIconSize(QSize(24, 24))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setMinimumHeight(84)
            group.addButton(button)
            cards.addWidget(button, 1)
            self.template_buttons[name] = button
        self.template_buttons["empty"].setChecked(True)
        layout.addLayout(cards)
        layout.addSpacing(12)

        self.create_button = QPushButton()
        self.create_button.setProperty("role", "primary")
        self.create_button.setMinimumHeight(34)
        row = QHBoxLayout()
        row.addWidget(self.create_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def show_page(self, index: int) -> None:
        """Переключить раздел: 0 — проекты, 1 — создание проекта."""
        self.pages.setCurrentIndex(index)
        (self.nav_projects, self.nav_create)[index].setChecked(True)
        if index == 1:
            self.name_edit.setFocus()

    def selected_template(self) -> str:
        """Выбранный шаблон нового проекта."""
        for name, button in self.template_buttons.items():
            if button.isChecked():
                return name
        return "empty"

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_shown_once", False):
            self._shown_once = True
            fade_in(self)

    # ------------------------------------------------------------------ #
    # Перевод
    # ------------------------------------------------------------------ #
    def retranslate(self) -> None:
        """Обновить все подписи под текущий язык."""
        t = self.translator.tr
        self.setWindowTitle(t("launcher.title"))
        self.heading.setText(t("launcher.heading"))
        self.subtitle.setText(t("launcher.subtitle"))
        self.nav_projects.setText("  " + t("launcher.nav.projects"))
        self.nav_create.setText("  " + t("launcher.nav.create"))
        self.open_folder_button.setText("  " + t("launcher.open_folder"))
        self.search.setPlaceholderText(t("launcher.search"))
        self.sort_box.setToolTip(t("launcher.sort"))
        for index, mode in enumerate(_SORT_MODES):
            self.sort_box.setItemText(index, t(f"launcher.sort.{mode}"))
        self.open_button.setText(t("launcher.open"))
        self.new_heading.setText(t("launcher.new.heading"))
        self.new_subtitle.setText(t("launcher.new.subtitle"))
        self.name_edit.setPlaceholderText(t("launcher.new.name"))
        self.create_button.setText(t("launcher.new.create"))
        self.template_label.setText(t("launcher.new.template").upper())
        for name, button in self.template_buttons.items():
            title = t(f"launcher.template.{name}")
            desc = t(f"launcher.template.{name}.desc")
            button.setText(f"{title}\n{desc}")
        self.empty_create_button.setText(t("launcher.empty.action"))
        self.auto_open.setText(t("launcher.auto_open"))
        self.auto_open.setToolTip(t("launcher.auto_open.tip"))
        self.language_label.setText(t("launcher.language"))
        self.exit_button.setText("  " + t("launcher.exit"))
        self.version_label.setText(t("launcher.version", version=__version__))
        self._validate_name()
        self._refill()

    def _on_language(self) -> None:
        self.translator.set_language(str(self.language_box.currentData()))
        self.retranslate()

    # ------------------------------------------------------------------ #
    # Список
    # ------------------------------------------------------------------ #
    @property
    def language(self) -> str:
        """Код выбранного языка."""
        return self.translator.language

    def reload(self) -> None:
        """Перечитать проекты с диска."""
        self._infos = [_scan(path) for path in list_projects(self.workspace_root)]
        self._refill()

    def _sorted(self, infos: list[_ProjectInfo]) -> list[_ProjectInfo]:
        mode = self.sort_box.currentData() or "recent"
        if mode == "name":
            return sorted(infos, key=lambda info: info.name.lower())
        by_time = sorted(infos, key=lambda info: info.modified, reverse=True)
        if mode == "recent":
            last = str(self.settings.value(KEY_LAST_PROJECT, "", str))
            by_time.sort(key=lambda info: info.name != last)
        return by_time

    def _refill(self, *_args: object) -> None:
        t = self.translator.tr
        wanted = self.search.text().strip().lower()
        previous = self.selected_path()
        self.list.clear()
        shown = [i for i in self._infos if wanted in i.name.lower()]
        for info in self._sorted(shown):
            item = QListWidgetItem(
                t(
                    "launcher.project_line",
                    name=info.name,
                    modified=_format_time(info.modified),
                    scenes=info.scenes,
                )
            )
            item.setData(Qt.ItemDataRole.UserRole, info)
            item.setSizeHint(QSize(0, _ROW_HEIGHT))
            self.list.addItem(item)
            if previous == info.path:
                self.list.setCurrentItem(item)
        if self.list.count() and self.list.currentItem() is None:
            self.list.setCurrentRow(0)
        empty = not self.list.count()
        self.list_stack.setCurrentWidget(self.empty_state if empty else self.list)
        self.empty_title.setText(
            t("launcher.empty.title" if not self._infos else "launcher.nothing_found")
        )
        self.empty_hint.setText(t("launcher.empty") if not self._infos else "")
        self.empty_create_button.setVisible(not self._infos)
        self._update_details()

    def selected_path(self) -> Path | None:
        """Путь выбранного проекта или ``None``."""
        item = self.list.currentItem()
        info = item.data(Qt.ItemDataRole.UserRole) if item else None
        return info.path if isinstance(info, _ProjectInfo) else None

    def _update_details(self, *_args: object) -> None:
        t = self.translator.tr
        item = self.list.currentItem()
        info = item.data(Qt.ItemDataRole.UserRole) if item else None
        if isinstance(info, _ProjectInfo):
            self.details.setText(
                t(
                    "launcher.details",
                    path=info.path,
                    modified=_format_time(info.modified),
                    scenes=info.scenes,
                    scripts=info.scripts,
                )
            )
        else:
            self.details.setText(t("launcher.details.none"))
        self.open_button.setEnabled(isinstance(info, _ProjectInfo))

    # ------------------------------------------------------------------ #
    # Действия
    # ------------------------------------------------------------------ #
    def _finish(self, project: Path, *, remember: bool = True) -> None:
        if remember:
            self.settings.setValue(KEY_LAST_PROJECT, project.name)
        self.settings.setValue(KEY_AUTO_OPEN, self.auto_open.isChecked())
        self.settings.setValue(KEY_LANGUAGE, self.language)
        self.choice = LauncherChoice(project, self.language)
        self.accept()

    def open_selected(self) -> None:
        """Открыть выбранный проект (если он ещё существует)."""
        path = self.selected_path()
        if path is None:
            return
        if not path.is_dir():
            self.error_label.setText(
                self.translator.tr("launcher.missing", name=path.name)
            )
            self.reload()
            return
        self._finish(path)

    def open_folder(self) -> None:
        """Открыть произвольную папку (вне ``projects/``) как проект."""
        chosen = QFileDialog.getExistingDirectory(
            self,
            self.translator.tr("launcher.open_folder.title"),
            str(self.workspace_root),
        )
        if chosen:
            self._finish(Path(chosen), remember=False)

    def _validate_name(self, *_args: object) -> bool:
        """Проверить введённое имя и показать причину отказа."""
        t = self.translator.tr
        text = self.name_edit.text()
        error = ""
        if text.strip():
            try:
                name = validate_project_name(text)
            except ValueError as exc:
                error = t("launcher.new.invalid", error=exc)
            else:
                if any(info.name.lower() == name.lower() for info in self._infos):
                    error = t("launcher.new.exists")
        self.error_label.setText(error)
        ok = bool(text.strip()) and not error
        self.create_button.setEnabled(ok)
        return ok

    def create_new(self) -> None:
        """Создать проект с введённым именем и открыть его."""
        if not self._validate_name():
            return
        try:
            project = create_project(
                self.workspace_root,
                self.name_edit.text(),
                template=self.selected_template(),
            )
        except (OSError, ValueError) as exc:
            self.error_label.setText(
                self.translator.tr("launcher.new.failed", error=exc)
            )
            return
        self._finish(project)

    # ------------------------------------------------------------------ #
    # Контекстное меню
    # ------------------------------------------------------------------ #
    def _show_menu(self, position: object) -> None:
        item = self.list.itemAt(position)  # type: ignore[arg-type]
        if item is None or not isinstance(
            item.data(Qt.ItemDataRole.UserRole), _ProjectInfo
        ):
            return
        self.list.setCurrentItem(item)
        t = self.translator.tr
        menu = QMenu(self)
        open_action = QAction(t("launcher.menu.open"), menu)
        open_action.triggered.connect(self.open_selected)
        reveal_action = QAction(t("launcher.menu.reveal"), menu)
        reveal_action.triggered.connect(self.reveal_selected)
        delete_action = QAction(t("launcher.menu.delete"), menu)
        delete_action.triggered.connect(self.delete_selected)
        menu.addActions([open_action, reveal_action])
        menu.addSeparator()
        menu.addAction(delete_action)
        menu.exec(self.list.viewport().mapToGlobal(position))  # type: ignore[arg-type]

    def reveal_selected(self) -> None:
        """Показать папку выбранного проекта в проводнике."""
        path = self.selected_path()
        if path is None:
            return
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", str(path)])  # noqa: S603, S607
            else:  # pragma: no cover - платформа цели — Windows
                subprocess.Popen(["xdg-open", str(path)])  # noqa: S603, S607
        except OSError as exc:  # pragma: no cover
            _log.error("Не удалось открыть проводник: %s", exc)

    def delete_selected(self) -> None:
        """Удалить выбранный проект после подтверждения вводом имени."""
        path = self.selected_path()
        if path is None:
            return
        t = self.translator.tr
        typed, ok = QInputDialog.getText(
            self,
            t("launcher.delete.title"),
            t("launcher.delete.prompt", name=path.name),
        )
        if not ok or typed.strip() != path.name:
            return
        # Страховка: удаляем только каталог, лежащий прямо в projects/.
        if path.parent.resolve() != (self.workspace_root / "projects").resolve():
            return
        try:
            shutil.rmtree(path)
        except OSError as exc:
            QMessageBox.warning(
                self,
                t("launcher.delete.title"),
                t("launcher.delete.failed", error=exc),
            )
        self.reload()


def choose_project(
    workspace_root: Path,
    language: str,
    settings: QSettings,
    *,
    allow_auto_open: bool = True,
) -> LauncherChoice | None:
    """Показать менеджер проектов и вернуть выбор пользователя.

    Если включено автооткрытие и последний проект на месте, окно не показывается.

    Args:
        workspace_root: Рабочий корень с папкой ``projects/``.
        language: Язык интерфейса по умолчанию.
        settings: Хранилище настроек.
        allow_auto_open: Учитывать флаг «открывать последний проект».

    Returns:
        Выбор пользователя или ``None``, если окно закрыли без выбора.
    """
    saved_language = str(settings.value(KEY_LANGUAGE, language, str))
    language = saved_language if saved_language in LANGUAGES else language
    if allow_auto_open and settings.value(KEY_AUTO_OPEN, False, bool):
        last = str(settings.value(KEY_LAST_PROJECT, "", str))
        candidate = workspace_root / "projects" / last
        if last and candidate.is_dir():
            return LauncherChoice(candidate, language)
    dialog = ProjectManagerDialog(workspace_root, language, settings)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.choice


def resolve_launch_project(args: argparse.Namespace, workspace_root: Path) -> bool:
    """Определить проект запуска: из ``--project`` или через менеджер проектов.

    Менеджер не показывается, если проект задан явно, указан ``--no-launcher``
    или снимается скриншот (автоматические проверки).

    Args:
        args: Разобранные аргументы; ``project`` и ``language`` обновляются.
        workspace_root: Рабочий корень с папкой ``projects/``.

    Returns:
        ``False``, если пользователь закрыл менеджер без выбора.
    """
    if args.project is None and (args.no_launcher or args.screenshot):
        args.project = str(workspace_root)
    if args.project is not None:
        return True
    choice = choose_project(
        workspace_root, args.language, QSettings("MyEngine", "Editor")
    )
    if choice is None:
        _log.info("Менеджер проектов закрыт без выбора, выход")
        return False
    args.project = str(choice.project)
    args.language = choice.language
    return True
