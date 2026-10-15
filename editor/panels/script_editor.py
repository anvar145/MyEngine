"""Панель редактора Python-скриптов.

:class:`ScriptEditorPanel` — это «маленькая IDE» внутри редактора движка:

* слева — дерево ``*.py`` файлов проекта (``QFileSystemModel`` + фильтр);
* справа — вкладки открытых файлов (:class:`~editor.panels.code_editor.CodeEditor`)
  и общая консоль вывода снизу;
* сверху — панель инструментов: создать, открыть, сохранить, сохранить всё,
  запустить, остановить, перезагрузить.

Запуск управляется :class:`ScriptRunner`: он держит реестр
:class:`~engine.scripting.runtime.ScriptRuntime`, тикает компоненты сцены,
следит за изменением файлов на диске (горячая перезагрузка) и собирает
сообщения для консоли. Остановка корректно завершает **все** скрипты: каждому
вызывается ``on_destroy``, затем реестр очищается.

Example:
    >>> panel = ScriptEditorPanel(context)
    >>> panel.open_script(Path("scripts/spin.py"))
    >>> panel.run_current()
"""

from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path

from PySide6.QtCore import (
    QDir,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QTextCursor
from PySide6.QtWidgets import (
    QFileDialog,
    QFileSystemModel,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from editor.panels.base import WindowPanel
from editor.panels.code_editor import CodeEditor
from editor.panels.console import QtLogHandler
from editor.panels.find_replace import FindReplaceDialog
from editor.project import scripts_root_for
from editor.theme import icon
from engine.scripting.runtime import ScriptError, ScriptHandle, ScriptRuntime
from engine.scripting.script_component import ScriptComponent

__all__ = [
    "SCRIPTS_FILTER",
    "SCRIPT_SUFFIX",
    "ScriptEditorPanel",
    "ScriptRunner",
    "ScriptSourceFilter",
    "UNSAVED_MARK",
]

_log = logging.getLogger("editor.script_editor")

#: Расширение скриптов.
SCRIPT_SUFFIX = ".py"

#: Фильтр для диалогов открытия и сохранения (запасной, без перевода).
SCRIPTS_FILTER = "Скрипты Python (*.py)"

#: Недопустимые в имени папки символы (Windows).
_BAD_FOLDER_CHARS = frozenset('<>:"/\\|?*')

#: Символ, помечающий несохранённые изменения в заголовке вкладки.
UNSAVED_MARK = "●"

#: Период тика скриптов (миллисекунды): ~60 кадров в секунду.
TICK_INTERVAL_MS = 16

#: Размер окна редактора скриптов по умолчанию (ширина, высота).
DEFAULT_WINDOW_SIZE = (1040, 640)

#: Как часто проверять изменения файлов на диске (миллисекунды).
RELOAD_INTERVAL_MS = 700

#: Шаблон нового скрипта.
NEW_SCRIPT_TEMPLATE = '''"""Скрипт MyEngine."""

from engine.core.math3d import Vector3


def on_start(self):
    """Вызывается один раз при запуске скрипта."""
    self.log("скрипт запущен на", self.game_object.name if self.game_object else "?")


def on_update(self, delta_time):
    """Вызывается каждый кадр: delta_time — секунды с прошлого кадра."""
    if self.transform is None:
        return
    self.transform.rotate(Vector3.up(), 45.0 * delta_time)


def on_destroy(self):
    """Вызывается при остановке скрипта."""
    self.log("скрипт остановлен")
'''


def _public_names(source: str) -> list[str]:
    """Собрать имена, объявленные в скрипте: функции, классы и константы.

    Разбор идёт через :mod:`ast`, поэтому подсветка и автодополнение не зависят
    от того, компилируется ли файл прямо сейчас: даже синтаксически сломанный
    код отдаёт то, что успел разобрать.

    Args:
        source: Исходный текст скрипта.

    Returns:
        Отсортированный список имён верхнего уровня.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return sorted(names)


class ScriptSourceFilter(QSortFilterProxyModel):
    """Фильтр дерева файлов: каталоги и только ``*.py`` файлы.

    Служебные каталоги и файлы (``.venv``, ``.git``, ``__pycache__``) скрыты:
    в большом проекте они превращают дерево скриптов в свалку.

    Args:
        parent: Родительский объект.

    Attributes:
        extensions: Допустимые расширения (с точкой, в нижнем регистре).
        hide_hidden: Скрывать ли имена, начинающиеся с точки и подчёркивания.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.extensions: tuple[str, ...] = (SCRIPT_SUFFIX,)
        self.hide_hidden = True

    def set_extensions(self, extensions: tuple[str, ...]) -> None:
        """Задать допустимые расширения файлов.

        ``invalidateFilter`` в PySide6 помечен deprecated, но именно он
        переспрашивает ``filterAcceptsRow``: без него новые расширения
        применились бы только после следующего изменения модели.

        Args:
            extensions: Например, ``(".py",)``.
        """
        self.extensions = tuple(item.lower() for item in extensions)
        self._invalidate()

    def set_hide_hidden(self, hide: bool) -> None:
        """Скрывать или показывать служебные имена.

        Args:
            hide: ``True`` — не показывать ``.venv``, ``__pycache__`` и подобное.
        """
        self.hide_hidden = bool(hide)
        self._invalidate()

    def _invalidate(self) -> None:
        """Переспросить фильтр у модели, не поднимая предупреждение Qt."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.invalidateFilter()

    def matches_name(self, name: str) -> bool:
        """Подходит ли имя файла под фильтр.

        Вынесено отдельным методом: правило проверяется тестами без файловой
        системы, а модель Qt наполняет каталоги лениво и в тестах пуста.

        Args:
            name: Имя файла (``"spin.py"``).

        Returns:
            ``True``, если файл с таким именем нужно показать.
        """
        text = str(name)
        if self.hide_hidden and (text.startswith(".") or text.startswith("__")):
            return False
        return text.lower().endswith(self.extensions)

    def filterAcceptsRow(  # noqa: N802 - API Qt
        self,
        source_row: int,
        source_parent: QModelIndex,
    ) -> bool:
        """Пропустить каталог или файл с разрешённым расширением.

        Args:
            source_row: Номер строки в исходной модели.
            source_parent: Родительский индекс исходной модели.

        Returns:
            ``True``, если строку нужно показать.
        """
        model = self.sourceModel()
        if model is None:
            return False
        index = model.index(source_row, 0, source_parent)
        if not index.isValid():
            return False
        # Каталоги пропускаем всегда (даже служебные — по ним навигируются),
        # не-каталоги проходит только по расширению и при visible-имени.
        if model.isDir(index):
            return True
        return self.matches_name(model.fileName(index))


class ScriptRunner:
    """Выполнение скриптов для панели: реестр, тик и горячая перезагрузка.

    Args:
        context: Контекст редактора (сцена и корень проекта).
        runtime: Готовый реестр скриптов; ``None`` — создать свой.
        logger: Логгер панели.

    Attributes:
        runtime: Реестр скриптов движка.
    """

    def __init__(
        self,
        context,
        *,
        runtime: ScriptRuntime | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.context = context
        self.log = logger or _log
        # Реестр общий с редактором (см. EditorContext.script_runtime): иначе
        # прикреплённые через инспектор скрипты не получали бы кадры.
        # Сравниваем с None явно: пустой ScriptRuntime ложен (у него есть
        # __len__), поэтому ``shared or ...`` создал бы второй реестр.
        shared = getattr(context, "script_runtime", None)
        if runtime is not None:
            self.runtime = runtime
        elif shared is not None:
            self.runtime = shared
        else:
            self.runtime = ScriptRuntime(
                project_root=getattr(context, "project_root", None)
            )
        self.running = False
        self._elapsed = 0.0
        self._frames = 0

    # ------------------------------------------------------------------ #
    # Запуск и остановка
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Начать выполнение: тикать компоненты и следить за файлами."""
        self.running = True
        self._elapsed = 0.0
        self._frames = 0
        self.log.info("Запуск скриптов: %s", self.runtime.describe())

    def stop(self) -> int:
        """Остановить все скрипты: ``on_destroy`` каждому, затем очистка.

        Returns:
            Сколько скриптов было остановлено.
        """
        count = 0
        for handle in list(self.runtime.scripts.values()):
            component = handle.meta.get("component")
            try:
                if isinstance(component, ScriptComponent):
                    component.unload(destroy=True)
                else:
                    self.runtime.call(handle, "on_destroy")
            except Exception:  # noqa: BLE001 - остановка обязана дойти до конца
                self.log.exception("Ошибка остановки скрипта '%s'", handle.name)
            count += 1
        self.runtime.close()
        self.running = False
        self.log.info("Скрипты остановлены (всего: %d)", count)
        return count

    # ------------------------------------------------------------------ #
    # Кадр
    # ------------------------------------------------------------------ #
    def tick(self, delta: float) -> None:
        """Проверить изменения файлов и обновить компоненты сцены.

        Args:
            delta: Секунды с прошлого тика.
        """
        self.refresh_changed()
        if not self.running:
            return
        self._elapsed += delta
        self._frames += 1
        for component in self.components():
            try:
                component.on_update(delta)
            except Exception:  # noqa: BLE001 - ошибка одного скрипта не рвёт цикл
                self.log.exception("Скрипт '%s' упал в on_update", component.name)

    def components(self) -> list[ScriptComponent]:
        """Все компоненты-скрипты активных объектов сцены.

        Returns:
            Список компонентов в порядке обхода сцены.
        """
        scene = getattr(self.context, "scene", None)
        if scene is None:
            return []
        found: list[ScriptComponent] = []
        try:
            objects = scene.query(ScriptComponent)
        except Exception:  # noqa: BLE001 - сцена может быть заменена на ходу
            return []
        for game_object in objects:
            if not game_object.active_in_hierarchy:
                continue
            for component in game_object.get_components(ScriptComponent):
                if component.is_loaded and component.enabled:
                    found.append(component)
        return found

    def refresh_changed(self) -> list[ScriptHandle]:
        """Перезагрузить скрипты, файлы которых изменились на диске.

        Returns:
            Список перезагруженных записей.
        """
        if not self.runtime.auto_reload:
            return []
        try:
            return self.runtime.refresh(on_reload=self._adopt)
        except Exception:  # noqa: BLE001 - перезагрузка не должна ронять панель
            self.log.exception("Сбой горячей перезагрузки")
            return []

    @staticmethod
    def _adopt(handle: ScriptHandle) -> None:
        """Передать перезагруженную версию компоненту-владельцу.

        Args:
            handle: Новая учётная запись скрипта.
        """
        component = handle.meta.get("component")
        if isinstance(component, ScriptComponent):
            component.adopt(handle)

    # ------------------------------------------------------------------ #
    # Разовый запуск из редактора
    # ------------------------------------------------------------------ #
    def run_once(self, source: str, name: str) -> bool:
        """Выполнить скрипт из редактора без компонента сцены.

        Полезно, чтобы быстро проверить логику: модуль загружается в отдельном
        пространстве имён, получает ``on_start`` и один ``on_update``, а затем
        ``on_destroy``. Исключения попадают в консоль, но не прерывают работу
        редактора.

        Args:
            source: Исходный текст скрипта.
            name: Имя для логов (обычно имя файла).

        Returns:
            ``True``, если скрипт загрузился и отработал без ошибок.
        """
        instance = ScriptComponent(source=source, runtime=self.runtime, name=name)
        instance.load()
        if not instance.is_loaded:
            return False
        try:
            instance.on_start()
            instance.on_update(0.0)
        finally:
            instance.on_destroy()
        self.log.info("Скрипт '%s' выполнен разово", name)
        return True

    def is_attached(self, path: Path | str | None) -> bool:
        """Прикреплён ли скрипт из файла к какому-нибудь объекту сцены.

        Нужно, чтобы не выполнять скрипт дважды: если он уже работает на
        объекте, ``on_start`` и ``on_destroy`` придут ему от сцены.

        Args:
            path: Путь к файлу скрипта.

        Returns:
            ``True``, если в сцене есть загруженный компонент с этим файлом.
        """
        if path is None:
            return False
        target = Path(path)
        for component in self.components():
            if component.path is not None and component.path == target:
                return True
        return False

    def status(self) -> str:
        """Сводка состояния для статусной строки."""
        return self.runtime.describe()


class ScriptEditorPanel(WindowPanel):
    """Отдельное окно со встроенным редактором скриптов.

    Окно не пристыковано к главному интерфейсу, но работает с тем же
    :class:`~editor.context.EditorContext`: сцена, выделение и рантайм скриптов
    общие, поэтому запуск скрипта из окна сразу виден во вьюпорте редактора.

    Args:
        context: Контекст редактора.
        parent: Родительский виджет (окно остаётся самостоятельным).

    Signals:
        script_saved: Скрипт сохранён на диск (путь).
        script_run: Запуск скриптов (путь или пустая строка).
        script_stopped: Скрипты остановлены.
    """

    script_saved = Signal(str)
    script_run = Signal(str)
    script_stopped = Signal()

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.script_editor", context, parent)
        self.runner = ScriptRunner(context)
        self._paths: dict[int, Path | None] = {}

        self._build_actions()
        self._build_toolbar()
        self._build_layout()
        self._connect_signals()
        self._apply_api_names()
        self.set_project_root(self.context.project_root)
        self.context.project_changed.connect(self.set_project_root)
        self.resize(*DEFAULT_WINDOW_SIZE)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(TICK_INTERVAL_MS)
        self._tick_timer.timeout.connect(self._on_tick)
        self.retranslate()

    # ------------------------------------------------------------------ #
    # Построение интерфейса
    # ------------------------------------------------------------------ #
    def _build_actions(self) -> None:
        """Создать действия панели с горячими клавишами."""
        self.action_new = QAction(icon("script_new"), "", self)
        self.action_new.setShortcut(QKeySequence("Ctrl+N"))
        # triggered передаёт checked (bool) — его нельзя отдавать в параметр
        # name, иначе в addTab уедет False вместо имени файла.
        self.action_new.triggered.connect(lambda _checked=False: self.new_script())

        self.action_new_folder = QAction(icon("folder-plus"), "", self)
        self.action_new_folder.triggered.connect(
            lambda _checked=False: self.new_folder()
        )

        self.action_open = QAction(icon("script_open"), "", self)
        self.action_open.setShortcut(QKeySequence("Ctrl+O"))
        self.action_open.triggered.connect(self.open_script_dialog)

        self.action_save = QAction(icon("save"), "", self)
        self.action_save.setShortcut(QKeySequence("Ctrl+S"))
        self.action_save.triggered.connect(self.save_current)

        self.action_save_all = QAction(icon("save_all"), "", self)
        self.action_save_all.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.action_save_all.triggered.connect(self.save_all)

        self.action_run = QAction(icon("play"), "", self)
        self.action_run.setShortcut(QKeySequence("F5"))
        self.action_run.triggered.connect(self.run_current)

        self.action_stop = QAction(icon("stop"), "", self)
        self.action_stop.setShortcut(QKeySequence("Shift+F5"))
        self.action_stop.triggered.connect(self.stop_scripts)

        self.action_reload = QAction(icon("refresh"), "", self)
        self.action_reload.setShortcut(QKeySequence("F6"))
        self.action_reload.triggered.connect(self.reload_current)

        self.action_close_tab = QAction("", self)
        self.action_close_tab.setShortcut(QKeySequence("Ctrl+W"))
        self.action_close_tab.triggered.connect(self.close_current)

        self.action_find = QAction("", self)
        self.action_find.setShortcut(QKeySequence("Ctrl+F"))
        self.action_find.triggered.connect(self.show_find_dialog)

        self.action_find_next = QAction("", self)
        self.action_find_next.setShortcut(QKeySequence("F3"))
        self.action_find_next.triggered.connect(self.find_next)

        self.action_clear_console = QAction("", self)
        self.action_clear_console.triggered.connect(self.clear_console)

        # Окно скриптов дочернее главному: с контекстом по умолчанию его F5/F6
        # неоднозначны с Play Mode и не срабатывают нигде.
        for action in (
            self.action_new,
            self.action_open,
            self.action_save,
            self.action_save_all,
            self.action_run,
            self.action_stop,
            self.action_reload,
            self.action_close_tab,
            self.action_find,
            self.action_find_next,
        ):
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

        self.addActions(
            [
                self.action_new,
                self.action_open,
                self.action_save,
                self.action_save_all,
                self.action_run,
                self.action_stop,
                self.action_reload,
                self.action_close_tab,
                self.action_find,
                self.action_find_next,
                self.action_clear_console,
            ]
        )

    def _build_toolbar(self) -> None:
        """Собрать панель инструментов скриптов."""
        self.toolbar = QToolBar(self)
        self.toolbar.setObjectName("script_toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(self.toolbar.iconSize())
        self.toolbar.addAction(self.action_new)
        self.toolbar.addAction(self.action_open)
        self.toolbar.addAction(self.action_save)
        self.toolbar.addAction(self.action_save_all)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_run)
        self.toolbar.addAction(self.action_stop)
        self.toolbar.addAction(self.action_reload)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.action_find)

    def _build_layout(self) -> None:
        """Собрать дерево файлов, вкладки и консоль в единый сплиттер."""
        container = QWidget(self)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.toolbar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, container)
        self.splitter.setChildrenCollapsible(True)

        # Дерево файлов проекта.
        tree_box = QWidget(self.splitter)
        tree_layout = QVBoxLayout(tree_box)
        tree_layout.setContentsMargins(6, 6, 6, 6)
        tree_layout.setSpacing(4)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.tree_label = QLabel(tree_box)
        self.tree_label.setProperty("role", "section")
        header.addWidget(self.tree_label, 1)
        self.folder_button = QToolButton(tree_box)
        self.folder_button.setObjectName("script_new_folder")
        self.folder_button.setAutoRaise(True)
        self.folder_button.setDefaultAction(self.action_new_folder)
        header.addWidget(self.folder_button)
        tree_layout.addLayout(header)

        self.file_model = QFileSystemModel(self)
        self.file_model.setFilter(
            QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot
        )
        self.file_model.setReadOnly(True)
        self.proxy = ScriptSourceFilter(self)
        self.proxy.setSourceModel(self.file_model)

        self.tree = QTreeView(tree_box)
        self.tree.setObjectName("script_file_tree")
        self.tree.setModel(self.proxy)
        self.tree.setHeaderHidden(True)
        self.tree.setAnimated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setEditTriggers(QTreeView.EditTrigger.NoEditTriggers)
        for column in range(1, self.file_model.columnCount()):
            self.tree.hideColumn(column)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tree_layout.addWidget(self.tree, 1)
        self.splitter.addWidget(tree_box)

        # Вкладки с кодом.
        self.tabs = QTabWidget(self.splitter)
        self.tabs.setObjectName("script_tabs")
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.splitter.addWidget(self.tabs)

        right = QSplitter(Qt.Orientation.Vertical, self.splitter)
        right.addWidget(self.tabs)

        console_box = QWidget(right)
        console_layout = QVBoxLayout(console_box)
        console_layout.setContentsMargins(6, 6, 6, 6)
        console_layout.setSpacing(4)
        header = QWidget(console_box)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.console_label = QLabel(header)
        self.console_label.setProperty("role", "section")
        header_layout.addWidget(self.console_label)
        console_layout.addWidget(header)

        self.console = QPlainTextEdit(console_box)
        self.console.setObjectName("script_console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        self.console.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.console.setFont(CodeEditor().font())
        console_layout.addWidget(self.console, 1)
        right.addWidget(console_box)
        right.setSizes([520, 180])

        self.splitter.addWidget(right)
        self.splitter.setSizes([240, 900])
        outer.addWidget(self.splitter, 1)
        self.setCentralWidget(container)

        self.find_dialog = FindReplaceDialog(self, self, translator=self.tr)
        self.find_dialog.hide()

        self._log_handler = QtLogHandler(logging.INFO)
        self._log_handler.message_logged.connect(self._on_log_message)
        for name in ("engine.scripting", "editor.script_editor"):
            logging.getLogger(name).addHandler(self._log_handler)

    def _connect_signals(self) -> None:
        """Подписать панель на события окна и вкладок."""
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.tree.activated.connect(self._on_tree_activated)
        self.tree.customContextMenuRequested.connect(self._show_tree_menu)
        self.find_dialog.status_message.connect(self.status)
        self.action_run.triggered.connect(self._sync_run_state)

    # ------------------------------------------------------------------ #
    # Локализация
    # ------------------------------------------------------------------ #
    def retranslate(self, *_args: object) -> None:
        """Перевести подписи панели."""
        super().retranslate()
        self.tree_label.setText(self.tr("script.project_files"))
        self.console_label.setText(self.tr("script.console"))
        self.console.setPlaceholderText(self.tr("script.console_hint"))
        self._set_action_text(self.action_new, "script.toolbar.new", "Ctrl+N")
        self.action_new_folder.setText(self.tr("script.folder.new"))
        self.action_new_folder.setToolTip(self.tr("script.folder.new"))
        self._set_action_text(self.action_open, "script.toolbar.open", "Ctrl+O")
        self._set_action_text(self.action_save, "script.toolbar.save", "Ctrl+S")
        self._set_action_text(
            self.action_save_all, "script.toolbar.save_all", "Ctrl+Shift+S"
        )
        self._set_action_text(self.action_run, "script.toolbar.run", "F5")
        self._set_action_text(self.action_stop, "script.toolbar.stop", "Shift+F5")
        self._set_action_text(self.action_reload, "script.toolbar.reload", "F6")
        self._set_action_text(self.action_find, "script.toolbar.find", "Ctrl+F")
        self.action_close_tab.setText(self.tr("script.close_tab"))
        self.action_find_next.setText(self.tr("script.find.next"))
        self.action_clear_console.setText(self.tr("console.clear"))
        self.toolbar.setWindowTitle(self.tr("script.toolbar.title"))
        self.find_dialog.set_translator(self.tr)
        for index in range(self.tabs.count()):
            self._refresh_tab_title(index)

    def _set_action_text(self, action: QAction, key: str, shortcut: str) -> None:
        """Задать текст, подсказку и статусную подсказку действия.

        Args:
            action: Действие панели.
            key: Ключ перевода.
            shortcut: Читаемое обозначение горячей клавиши.
        """
        text = self.tr(key)
        action.setText(text)
        action.setToolTip(f"{text} ({shortcut})")
        action.setStatusTip(text)

    # ------------------------------------------------------------------ #
    # Проект и дерево файлов
    # ------------------------------------------------------------------ #
    def set_project_root(self, root: Path | str | None) -> None:
        """Направить дерево на каталог проекта.

        Args:
            root: Корень проекта; ``None`` — оставить текущий.
        """
        if root is None:
            return
        path = Path(root)
        if not path.is_dir():
            self.status(self.tr("script.status.no_project", path=str(path)))
            return
        self._root = path
        scripts = scripts_root_for(path)
        index = self.file_model.setRootPath(str(scripts))
        self.tree.setRootIndex(self.proxy.mapFromSource(index))
        for column in range(1, self.file_model.columnCount()):
            self.tree.hideColumn(column)

    def project_root(self) -> Path | None:
        """Корень текущего проекта."""
        return getattr(self, "_root", None)

    def scripts_root(self) -> Path:
        """Корень дерева скриптов (``<проект>/Assets/Scripts``)."""
        root = self.project_root() or Path(self.context.project_root)
        return scripts_root_for(root)

    # ------------------------------------------------------------------ #
    # Работа с файлами
    # ------------------------------------------------------------------ #
    def open_script(self, path: Path | str, *, line: int = 0) -> CodeEditor:
        """Открыть скрипт во вкладке (или переключиться на уже открытую).

        Args:
            path: Путь к файлу.
            line: Строка, на которую поставить курсор (нумерация с 1).

        Returns:
            Редактор открытого файла.
        """
        target = Path(path)
        existing = self._editor_for_path(target)
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            self._go_to_line(existing, line)
            return existing

        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            self.status(self.tr("script.status.open_failed", message=str(exc)))
            _log.error("Не удалось открыть скрипт '%s': %s", target, exc)
            return self.new_script(name=target.name)

        editor = self._add_editor(text, target, target.name)
        self._go_to_line(editor, line)
        self.status(self.tr("script.status.opened", name=target.name))
        return editor

    def open_script_dialog(self) -> CodeEditor | None:
        """Спросить файл и открыть его.

        Returns:
            Редактор открытого файла или ``None``, если выбор отменён.
        """
        start = str(self.scripts_root())
        path, _filter = QFileDialog.getOpenFileName(
            self,
            self.tr("script.dialog.open"),
            start,
            f"{self.tr('script.filter')} (*{SCRIPT_SUFFIX})",
        )
        if not path:
            return None
        return self.open_script(path)

    def new_script(self, name: str | None = None) -> CodeEditor:
        """Создать скрипт во вкладке, не записывая его на диск.

        Файл появится только при сохранении (Ctrl+S): пока пользователь не
        решил, что скрипт нужен, папка проекта не засоряется. Папка для
        сохранения предлагается по выделению в дереве.

        Args:
            name: Имя будущего файла или ``None`` — подобрать свободное.

        Returns:
            Редактор нового скрипта.
        """
        if name is None:
            name = self.unique_script_name(self.selected_folder())
        editor = self._add_editor(NEW_SCRIPT_TEMPLATE, None, name)
        editor.document().setModified(True)
        self._refresh_tab_title(self.tabs.indexOf(editor))
        self.status(self.tr("script.status.created", name=name))
        return editor

    @staticmethod
    def validate_folder_name(name: str) -> str | None:
        """Проверить имя новой папки.

        Args:
            name: Введённое имя.

        Returns:
            Ключ перевода ошибки или ``None``, если имя допустимо.
        """
        text = name.strip()
        if not text:
            return "script.folder.error_empty"
        if text in {".", ".."} or text.endswith("."):
            return "script.folder.error_invalid"
        if any(ch in _BAD_FOLDER_CHARS or ord(ch) < 32 for ch in text):
            return "script.folder.error_invalid"
        return None

    def new_folder(self, name: str | None = None) -> Path | None:
        """Создать подпапку в выбранной папке дерева скриптов.

        Args:
            name: Имя папки; ``None`` — спросить в диалоге.

        Returns:
            Путь созданной папки или ``None`` (отмена, ошибка, имя занято).
        """
        parent = self.selected_folder()
        if name is None:
            name, accepted = QInputDialog.getText(
                self,
                self.tr("script.folder.new"),
                self.tr("script.folder.prompt"),
            )
            if not accepted:
                return None
        error = self.validate_folder_name(name)
        if error is not None:
            self.status(self.tr(error))
            return None
        target = parent / name.strip()
        if target.exists():
            self.status(self.tr("script.folder.exists", name=target.name))
            return None
        try:
            target.mkdir(parents=True)
        except OSError as exc:
            self.status(self.tr("script.folder.failed", message=str(exc)))
            _log.error("Не удалось создать папку '%s': %s", target, exc)
            return None
        self.status(self.tr("script.folder.created", name=target.name))
        return target

    def _show_tree_menu(self, position) -> None:
        """Контекстное меню дерева: новый скрипт и новая папка."""
        index = self.tree.indexAt(position)
        if index.isValid():
            self.tree.setCurrentIndex(index)
        menu = QMenu(self)
        menu.addAction(self.action_new)
        menu.addAction(self.action_new_folder)
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def selected_folder(self) -> Path:
        """Папка для нового скрипта: выделенная в дереве или ``Assets/Scripts``."""
        index = self.tree.currentIndex()
        if index.isValid():
            source = self.proxy.mapToSource(index)
            path = Path(self.file_model.filePath(source))
            return path if self.file_model.isDir(source) else path.parent
        return self.scripts_root()

    def unique_script_name(
        self, folder: Path | None = None, base: str = "new_script"
    ) -> str:
        """Свободное имя файла с учётом диска и открытых вкладок.

        Args:
            folder: Папка назначения; ``None`` — корень скриптов проекта.
            base: Имя без расширения.

        Returns:
            ``new_script.py``, ``new_script_2.py`` и так далее.
        """
        target = Path(folder) if folder is not None else self.scripts_root()
        taken = {path.resolve() for path in self._paths.values() if path is not None}
        untitled = {
            self.tabs.tabText(i).lstrip(UNSAVED_MARK + " ")
            for i in range(self.tabs.count())
            if self._paths.get(id(self.tabs.widget(i))) is None
        }
        number = 1
        while True:
            suffix = "" if number == 1 else f"_{number}"
            name = f"{base}{suffix}{SCRIPT_SUFFIX}"
            candidate = target / name
            if (
                not candidate.exists()
                and candidate.resolve() not in taken
                and name not in untitled
            ):
                return name
            number += 1

    def save_current(self) -> bool:
        """Сохранить скрипт в текущей вкладке.

        Returns:
            ``True``, если файл записан.
        """
        editor = self.current_editor()
        if editor is None:
            return False
        return self.save_editor(editor)

    def save_editor(self, editor: CodeEditor) -> bool:
        """Сохранить указанный редактор (при необходимости спросив путь).

        Args:
            editor: Редактор скрипта.

        Returns:
            ``True``, если файл записан.
        """
        path = self._paths.get(id(editor))
        if path is None:
            path = self._ask_new_path(self._tab_name(editor))
            if path is None:
                return False
            self._paths[id(editor)] = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(editor.text(), encoding="utf-8")
        except OSError as exc:
            self.status(self.tr("script.status.save_failed", message=str(exc)))
            _log.error("Не удалось сохранить '%s': %s", path, exc)
            return False

        editor.mark_saved()
        index = self.tabs.indexOf(editor)
        self.tabs.setTabText(index, path.name)
        self.tabs.setTabToolTip(index, str(path))
        self.status(self.tr("script.status.saved", name=path.name))
        self.script_saved.emit(str(path))
        # Файл изменился на диске — реестр подхватит его при следующем тике.
        self.runner.refresh_changed()
        return True

    def save_all(self) -> int:
        """Сохранить все вкладки с изменениями.

        Returns:
            Число сохранённых файлов.
        """
        saved = 0
        for number in range(self.tabs.count()):
            editor = self.tabs.widget(number)
            if (
                isinstance(editor, CodeEditor)
                and editor.is_modified()
                and self.save_editor(editor)
            ):
                saved += 1
        if saved:
            self.status(self.tr("script.status.saved_all", count=saved))
        return saved

    def close_tab(self, index: int) -> bool:
        """Закрыть вкладку, спросив подтверждение при несохранённых правках.

        Args:
            index: Номер вкладки.

        Returns:
            ``True``, если вкладка закрыта.
        """
        editor = self.tabs.widget(index)
        if not isinstance(editor, CodeEditor):
            self.tabs.removeTab(index)
            return True
        if editor.is_modified():
            answer = QMessageBox.question(
                self,
                self.tr("script.dialog.unsaved_title"),
                self.tr("script.dialog.unsaved", name=self._tab_name(editor)),
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save and not self.save_editor(
                editor
            ):
                return False
        self._detach_editor(editor)
        self.tabs.removeTab(index)
        editor.deleteLater()
        return True

    def close_current(self) -> bool:
        """Закрыть текущую вкладку.

        Returns:
            ``True``, если вкладка закрыта.
        """
        index = self.tabs.currentIndex()
        return self.close_tab(index) if index >= 0 else False

    # ------------------------------------------------------------------ #
    # Запуск и остановка
    # ------------------------------------------------------------------ #
    def run_current(self) -> bool:
        """Запустить скрипты: тик компонентов сцены и разовый прогон вкладки.

        Returns:
            ``True``, если окружение запущено.
        """
        editor = self.current_editor()
        path = self._paths.get(id(editor)) if editor is not None else None
        self.runner.start()
        self.action_run.setEnabled(False)
        self.action_stop.setEnabled(True)
        self._tick_timer.start()
        if editor is not None:
            name = path.name if path is not None else self._tab_name(editor)
            if self.runner.is_attached(path):
                # Скрипт уже работает на объекте: лишний разовый прогон только
                # продублировал бы его вывод и побочные эффекты.
                self.status(self.tr("script.status.already_attached", name=name))
            elif not self.runner.run_once(editor.text(), name):
                self.status(self.tr("script.status.run_failed", name=name))
                self.script_run.emit(str(path or ""))
                return False
        self.status(self.tr("script.status.running", info=self.runner.status()))
        self.script_run.emit(str(path or ""))
        return True

    def stop_scripts(self) -> int:
        """Остановить все скрипты и снять блокировку кнопки «Стоп».

        Returns:
            Число остановленных скриптов.
        """
        stopped = self.runner.stop()
        self._tick_timer.stop()
        self.action_run.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.status(self.tr("script.status.stopped", count=stopped))
        self.script_stopped.emit()
        return stopped

    def is_running(self) -> bool:
        """Выполняются ли скрипты прямо сейчас."""
        return self.runner.running

    def reload_current(self) -> bool:
        """Перезагрузить модуль текущей вкладки (горячая перезагрузка).

        Returns:
            ``True``, если скрипт перезагружен.
        """
        handle = self._handle_for_current()
        if handle is None:
            self.status(self.tr("script.status.reload_manual"))
            self.runner.refresh_changed()
            return False
        try:
            self.runner.runtime.reload(handle.token)
        except (ScriptError, KeyError) as exc:
            self.status(self.tr("script.status.reload_failed", message=str(exc)))
            return False
        self.status(self.tr("script.status.reloaded", name=handle.name))
        return True

    def reload_all(self) -> int:
        """Перезагрузить все изменившиеся скрипты немедленно.

        Returns:
            Число перезагруженных скриптов.
        """
        reloaded = self.runner.refresh_changed()
        if reloaded:
            self.status(
                self.tr("script.status.reloaded_all", count=len(reloaded))
            )
        return len(reloaded)

    # ------------------------------------------------------------------ #
    # Поиск
    # ------------------------------------------------------------------ #
    def show_find_dialog(self) -> FindReplaceDialog:
        """Показать диалог поиска и замены по активной вкладке.

        Returns:
            Диалог поиска.
        """
        if self.current_editor() is None:
            self.status(self.tr("script.status.need_file"))
            return self.find_dialog
        self.find_dialog.show_for()
        return self.find_dialog

    def find_next(self) -> bool:
        """Повторить последний поиск без открытия диалога.

        Returns:
            ``True``, если вхождение найдено.
        """
        editor = self.current_editor()
        if editor is None:
            return False
        needle = self.find_dialog.find_edit.text()
        if not needle:
            self.show_find_dialog()
            return False
        flags = self.find_dialog.find_flags()
        found = editor.find(needle, flags)
        if not found:
            cursor = editor.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            editor.setTextCursor(cursor)
            found = editor.find(needle, flags)
        self.status(
            self.tr(
                "script.find.found" if found else "script.find.not_found",
                text=needle,
            )
        )
        return bool(found)

    # ------------------------------------------------------------------ #
    # Консоль
    # ------------------------------------------------------------------ #
    def append_console(self, text: str, level: int = logging.INFO) -> None:
        """Добавить строку в консоль панели с окраской по уровню.

        Args:
            text: Текст сообщения.
            level: Уровень логирования.
        """
        from editor.theme import PALETTE

        colors = {
            logging.DEBUG: PALETTE["text_muted"],
            logging.INFO: PALETTE["text"],
            logging.WARNING: PALETTE["warning"],
            logging.ERROR: PALETTE["error"],
            logging.CRITICAL: PALETTE["error"],
        }
        color = colors.get(level, PALETTE["text"])
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.console.appendHtml(f'<span style="color:{color}">{escaped}</span>')
        self.console.moveCursor(self.console.textCursor().MoveOperation.End)

    def clear_console(self) -> None:
        """Очистить консоль панели."""
        self.console.clear()

    # ------------------------------------------------------------------ #
    # Текущая вкладка и внутреннее
    # ------------------------------------------------------------------ #
    def current_editor(self) -> CodeEditor | None:
        """Редактор активной вкладки (``None``, если вкладок нет)."""
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, CodeEditor) else None

    def current_script_path(self) -> Path | None:
        """Путь к файлу в активной вкладке (``None`` для нового скрипта)."""
        editor = self.current_editor()
        return self._paths.get(id(editor)) if editor is not None else None

    def editors(self) -> list[CodeEditor]:
        """Все открытые редакторы в порядке вкладок."""
        return [
            widget
            for index in range(self.tabs.count())
            if isinstance((widget := self.tabs.widget(index)), CodeEditor)
        ]

    def open_paths(self) -> list[Path]:
        """Пути всех сохранённых вкладок."""
        return [
            path
            for editor in self.editors()
            if (path := self._paths.get(id(editor))) is not None
        ]

    def _add_editor(self, text: str, path: Path | None, name: str) -> CodeEditor:
        """Создать вкладку с редактором и подписать его на команды панели.

        Args:
            text: Исходный текст.
            path: Путь к файлу (``None`` для нового).
            name: Имя вкладки.

        Returns:
            Созданный редактор.
        """
        editor = CodeEditor(self.tabs)
        editor.set_text(text)
        self._apply_api_names(editor)
        index = self.tabs.addTab(editor, name)
        self.tabs.setTabToolTip(index, str(path) if path is not None else name)
        self._paths[id(editor)] = path

        editor.modificationChanged.connect(
            lambda _changed, editor=editor: self._refresh_tab_title(
                self.tabs.indexOf(editor)
            )
        )
        editor.save_requested.connect(lambda editor=editor: self.save_editor(editor))
        editor.run_requested.connect(self.run_current)
        editor.stop_requested.connect(self.stop_scripts)
        editor.reload_requested.connect(self.reload_current)
        editor.cursor_position_changed.connect(self._on_cursor_moved)
        editor.textChanged.connect(
            lambda editor=editor: self._refresh_completion(editor)
        )
        self.tabs.setCurrentIndex(index)
        self._refresh_tab_title(index)
        return editor

    def _detach_editor(self, editor: CodeEditor) -> None:
        """Убрать служебные записи редактора при закрытии вкладки.

        Args:
            editor: Закрываемый редактор.
        """
        self._paths.pop(id(editor), None)

    def _refresh_tab_title(self, index: int) -> None:
        """Обновить заголовок вкладки: имя, пометка правок и подсказка.

        Args:
            index: Номер вкладки.
        """
        if not 0 <= index < self.tabs.count():
            return
        editor = self.tabs.widget(index)
        if not isinstance(editor, CodeEditor):
            return
        path = self._paths.get(id(editor))
        current = self.tabs.tabText(index).lstrip(UNSAVED_MARK + " ")
        name = path.name if path is not None else current
        mark = f"{UNSAVED_MARK} " if editor.is_modified() else ""
        self.tabs.setTabText(index, f"{mark}{name}")
        self.tabs.setTabToolTip(index, str(path) if path is not None else name)

    def _refresh_completion(self, editor: CodeEditor) -> None:
        """Дополнить словарь подсказок именами из текущего файла.

        Args:
            editor: Редактор, содержимое которого изменилось.
        """
        words = _public_names(editor.text())
        if words == getattr(editor, "_script_names", None):
            return
        editor._script_names = words  # noqa: SLF001 - кэш для сравнения
        from engine.scripting.syntax import completion_words

        editor._words = completion_words(  # noqa: SLF001 - словарь подсказок
            editor.api_names, words
        )
        editor._model.setStringList(editor._words)  # noqa: SLF001

    def _apply_api_names(self, editor: CodeEditor | None = None) -> None:
        """Собрать имена API движка и передать их подсветке.

        Args:
            editor: Конкретный редактор; ``None`` — все открытые.
        """
        names = sorted(ScriptComponent.API_NAMES)
        targets: list[CodeEditor | None] = [editor] if editor is not None else list(
            self.editors()
        )
        for target in targets:
            if target is not None:
                target.highlighter.set_api_names(names)

    def _tab_name(self, editor: CodeEditor) -> str:
        """Имя вкладки без пометки несохранённых правок."""
        return self.tabs.tabText(self.tabs.indexOf(editor)).lstrip(UNSAVED_MARK + " ")

    def _editor_for_path(self, path: Path) -> CodeEditor | None:
        """Найти открытую вкладку по пути файла."""
        target = Path(path)
        for editor in self.editors():
            stored = self._paths.get(id(editor))
            if stored is not None and stored == target:
                return editor
        return None

    def _ask_new_path(self, suggested: str) -> Path | None:
        """Спросить путь для сохранения нового скрипта.

        Args:
            suggested: Имя файла по умолчанию.

        Returns:
            Выбранный путь или ``None``.
        """
        start = self.selected_folder()
        if not suggested.endswith(SCRIPT_SUFFIX):
            suggested += SCRIPT_SUFFIX
        path, _filter = QFileDialog.getSaveFileName(
            self,
            self.tr("script.dialog.save"),
            str(Path(start) / suggested),
            f"{self.tr('script.filter')} (*{SCRIPT_SUFFIX})",
        )
        return Path(path) if path else None

    def _go_to_line(self, editor: CodeEditor, line: int) -> None:
        """Поставить курсор на строку.

        Args:
            editor: Редактор скрипта.
            line: Номер строки (1 — первая); ``0`` — не перемещать.
        """
        if line <= 0:
            editor.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        from PySide6.QtGui import QTextCursor

        cursor = QTextCursor(editor.document().findBlockByNumber(line - 1))
        editor.setTextCursor(cursor)
        editor.centerCursor()
        editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_tree_activated(self, index: QModelIndex) -> None:
        """Открыть файл по двойному клику в дереве.

        Args:
            index: Индекс прокси-модели.
        """
        source = self.proxy.mapToSource(index)
        if self.file_model.isDir(source):
            self.tree.setExpanded(index, not self.tree.isExpanded(index))
            return
        self.open_script(self.file_model.filePath(source))

    def _on_tab_changed(self, index: int) -> None:
        """Обновить состояние подсказок при смене вкладки.

        Args:
            index: Номер новой активной вкладки.
        """
        editor = self.tabs.widget(index)
        if isinstance(editor, CodeEditor):
            editor.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_cursor_moved(self, line: int, column: int) -> None:
        """Показать позицию курсора в статусной строке.

        Args:
            line: Номер строки.
            column: Номер столбца.
        """
        self.status(self.tr("script.status.cursor", line=line, column=column))

    def _on_log_message(self, text: str, level: int) -> None:
        """Показать запись лога в консоли панели.

        Args:
            text: Текст записи.
            level: Уровень логирования.
        """
        self.append_console(text, level)

    def _on_tick(self) -> None:
        """Кадр скриптов: обновить компоненты и подсветить ошибки."""
        delta = self._tick_timer.interval() / 1000.0
        self.runner.tick(delta)

    def _handle_for_current(self) -> ScriptHandle | None:
        """Учётная запись скрипта, открытого в текущей вкладке."""
        path = self._paths.get(id(self.current_editor()))
        if path is None:
            return None
        return self.runner.runtime.by_path(path)

    def _sync_run_state(self, *_args: object) -> None:
        """Согласовать доступность кнопок «Запустить» и «Стоп»."""
        running = self.runner.running
        self.action_run.setEnabled(not running)
        self.action_stop.setEnabled(running)

    def confirm_close(self) -> bool:
        """Спросить про каждый несохранённый скрипт (Сохранить/Не сохранять/Отмена).

        Returns:
            ``False``, если пользователь отменил закрытие.
        """
        for editor in self.editors():
            if not editor.is_modified():
                continue
            self.tabs.setCurrentWidget(editor)
            answer = QMessageBox.question(
                self,
                self.tr("script.dialog.unsaved_title"),
                self.tr("script.dialog.unsaved", name=self._tab_name(editor)),
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save:
                if not self.save_editor(editor):
                    return False
            else:
                editor.document().setModified(False)
        return True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - API Qt
        """Спросить про несохранённое, остановить скрипты и снять логи.

        Args:
            event: Событие закрытия.
        """
        if not self.confirm_close():
            event.ignore()
            return
        self._tick_timer.stop()
        if self.runner.running:
            self.runner.stop()
        # Отпускаем слежение за каталогом проекта: иначе модель продолжает
        # держать наблюдателя за уже удалённым каталогом.
        self.file_model.setRootPath(QDir.rootPath())
        for name in ("engine.scripting", "editor.script_editor"):
            logging.getLogger(name).removeHandler(self._log_handler)
        self.find_dialog.close()
        super().closeEvent(event)

    def __repr__(self) -> str:
        return (
            f"<ScriptEditorPanel вкладок={self.tabs.count()} "
            f"запущено={self.runner.running} {self.runner.status()}>"
        )
