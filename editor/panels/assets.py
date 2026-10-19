"""Панель ассетов: файлы проекта с фильтром по типам и добавлением в сцену."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import (
    QDir,
    QMimeData,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileIconProvider,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from editor.mime import PREFAB_MIME
from editor.panels.base import DockPanel
from editor.project import assets_root_for as _assets_root_for
from editor.theme import PALETTE, icon

__all__ = [
    "AssetsPanel",
    "MODEL_EXTENSIONS",
    "SCENE_EXTENSIONS",
    "SCRIPT_EXTENSIONS",
    "TEXTURE_EXTENSIONS",
]
#: Расширения, которые редактор умеет добавлять в сцену как объект.
MODEL_EXTENSIONS = frozenset({".obj", ".glb", ".gltf", ".fbx"})
TEXTURE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".dds", ".gif"})
SCRIPT_EXTENSIONS = frozenset({".py", ".md"})
#: Файлы сцен; сравниваются по окончанию имени (двойное расширение).
SCENE_EXTENSIONS = frozenset({".scene.json"})

#: Категории фильтра: ключ перевода → набор расширений (пустой — все файлы).
FILTER_CATEGORIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("assets.filter.all", frozenset()),
    ("assets.filter.models", MODEL_EXTENSIONS),
    ("assets.filter.textures", TEXTURE_EXTENSIONS),
    ("assets.filter.scripts", SCRIPT_EXTENSIONS),
    ("assets.filter.scenes", SCENE_EXTENSIONS),
)


def _unique_path(path: Path) -> Path:
    """Свободное имя файла: ``name.ext`` → ``name (1).ext`` и т. д."""
    candidate = path
    number = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
        number += 1
    return candidate


class _AssetFilter(QSortFilterProxyModel):
    """Фильтр файлов по расширению (папки показываются всегда)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._extensions: frozenset[str] = frozenset()

    def set_extensions(self, extensions: frozenset[str]) -> None:
        """Ограничить список расширениями (пустой набор — без фильтра)."""
        # Новый API Qt (6.10+): beginFilterChange/endFilterChange. Старые
        # invalidateFilter/invalidateRowsFilter помечены устаревшими.
        self.beginFilterChange()
        self._extensions = extensions
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(  # noqa: N802 - API Qt
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        """Пропустить папки и подходящие по расширению файлы."""
        model = self.sourceModel()
        if model is None:
            return True
        index = model.index(source_row, 0, source_parent)
        if model.isDir(index):
            return True
        name = str(model.fileName(index)).lower()
        if name.endswith(".meta"):
            # Служебные файлы импорта лежат рядом с ассетами, но не нужны глазу.
            return False
        if not self._extensions:
            return True
        return name.endswith(tuple(self._extensions))


_FILE_ICONS = {
    ".py": "file-code",
    ".json": "file-code",
    ".glsl": "file-code",
    ".vert": "file-code",
    ".frag": "file-code",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".gif": "image",
    ".dds": "image",
    ".obj": "box",
    ".gltf": "box",
    ".glb": "box",
    ".fbx": "box",
}


class _LucideIconProvider(QFileIconProvider):
    """Иконки файлов в стиле темы вместо системных (жёлтые папки Windows)."""

    def icon(self, arg):  # noqa: A003 - API Qt
        if isinstance(arg, QFileIconProvider.IconType):
            is_folder = arg == QFileIconProvider.IconType.Folder
            return icon("folder" if is_folder else "file")
        if arg.isDir():
            return icon("folder")
        suffix = "." + arg.suffix().lower()
        return icon(_FILE_ICONS.get(suffix, "file"))

#: Суффикс файла префаба.
PREFAB_SUFFIX = ".prefab.json"


class _ImportAwareModel(QFileSystemModel):
    """Файловая модель, подсвечивающая состояние импорта файлов."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._context = context
        self._icons = _LucideIconProvider()
        self.setIconProvider(self._icons)

    def mimeData(self, indexes: list[QModelIndex]) -> QMimeData:  # noqa: N802 - API Qt
        """Данные перетаскивания; для ``*.prefab.json`` добавляется префаб-тип."""
        data = super().mimeData(indexes)
        for index in indexes:
            if index.column() == 0 and not self.isDir(index):
                path = self.filePath(index)
                if path.lower().endswith(PREFAB_SUFFIX):
                    data.setData(PREFAB_MIME, path.encode("utf-8"))
                    break
        return data

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        """Отметить файлы в очереди (``…``) и упавшие импорты (красный цвет)."""
        value = super().data(index, role)
        manager = self._context.imports
        if manager is None or index.column() != 0 or self.isDir(index):
            return value
        if role not in (
            Qt.ItemDataRole.DisplayRole,
            Qt.ItemDataRole.ForegroundRole,
            Qt.ItemDataRole.ToolTipRole,
        ):
            return value
        guid = manager.db.guid_of(Path(self.filePath(index)))
        if guid is None:
            return value
        state = manager.state(guid).value
        if role == Qt.ItemDataRole.DisplayRole and state in ("queued", "running"):
            return f"{value}  …"
        if state == "failed":
            if role == Qt.ItemDataRole.ForegroundRole:
                return QBrush(QColor(PALETTE["error"]))
            if role == Qt.ItemDataRole.ToolTipRole:
                return manager.failures().get(guid, "")
        return value


class _AssetTree(QTreeView):
    """Дерево ассетов, отличающее клик от начала перетаскивания.

    Выбор ассета по нажатию кнопки очищал бы выделение объекта до того, как
    файл дотащат до поля ссылки в инспекторе. Поэтому при выборе мышью выбор
    откладывается до отпускания кнопки, а начало drag его отменяет.
    """

    released = Signal()
    drag_started = Signal()

    def startDrag(self, supported_actions) -> None:  # noqa: N802 - API Qt
        """Сообщить о начале перетаскивания и запустить его."""
        self.drag_started.emit()
        super().startDrag(supported_actions)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Сообщить об отпускании кнопки после стандартной обработки."""
        super().mouseReleaseEvent(event)
        self.released.emit()


class AssetsPanel(DockPanel):
    """Обозреватель файлов проекта.

    Signals:
        model_activated: Двойной клик по файлу модели (аргумент — путь).
        scene_activated: Двойной клик по файлу сцены (аргумент — путь).
        script_activated: Двойной клик по скрипту ``.py`` (аргумент — путь).
        asset_selected: Выбран файл (путь и расширение) — для статусной строки.
    """

    model_activated = Signal(str)
    scene_activated = Signal(str)
    asset_selected = Signal(str)
    script_activated = Signal(str)

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.assets", context, parent)
        self.root = _assets_root_for(context.project_root)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self.filter_label = QLabel(container)
        self.filter_label.setProperty("role", "hint")
        self.filter_combo = QComboBox(container)
        for title_key, _extensions in FILTER_CATEGORIES:
            self.filter_combo.addItem("", title_key)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        controls.addWidget(self.filter_label)
        controls.addWidget(self.filter_combo, 1)
        layout.addLayout(controls)

        self.fs_model = _ImportAwareModel(context, self)
        self.fs_model.setRootPath(QDir.rootPath())
        self.fs_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        self.proxy = _AssetFilter(self)
        self.proxy.setSourceModel(self.fs_model)

        self._pending_path: str | None = None
        self.tree = _AssetTree(container)
        self.tree.released.connect(self._flush_pending)
        self.tree.drag_started.connect(self._cancel_pending)
        self.tree.setModel(self.proxy)
        self.tree.setRootIndex(self.proxy.mapFromSource(self.fs_model.index(str(self.root))))
        self.tree.setHeaderHidden(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        # файлы можно тащить в поля скриптов (uri-list от QFileSystemModel)
        self.tree.setDragEnabled(True)
        self.tree.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        # QFileSystemModel отдаёт 4 колонки (имя, размер, тип, дата) — в панели
        # ассетов нужна только первая, иначе имена обрезаются.
        for column in range(1, self.fs_model.columnCount()):
            self.tree.setColumnHidden(column, True)
        self.tree.doubleClicked.connect(self._on_double_click)
        self.tree.selectionModel().currentChanged.connect(self._on_current_changed)
        layout.addWidget(self.tree, 1)

        self.hint = QLabel(self.tr("assets.hint"), container)
        self.hint.setProperty("role", "hint")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.setWidget(container)

        self._apply_filter(0)
        self.setAcceptDrops(True)
        self.context.imports_changed.connect(self.tree.viewport().update)
        self.context.project_changed.connect(self.set_project_root)
        self.retranslate()

    # ------------------------------------------------------------------ #
    # Проект
    # ------------------------------------------------------------------ #
    def set_project_root(self, root: object) -> None:
        """Показать файлы другого проекта.

        Args:
            root: Новый корень проекта (путь или строка).
        """
        path = Path(str(root))
        self.root = _assets_root_for(path)
        index = self.fs_model.index(str(self.root))
        self.tree.setRootIndex(self.proxy.mapFromSource(index))
        self.status(self.tr("assets.roots", path=str(self.root)))

    # ------------------------------------------------------------------ #
    # Локализация
    # ------------------------------------------------------------------ #
    def retranslate(self, *_args: object) -> None:
        """Перевести подписи панели."""
        super().retranslate()
        self.filter_label.setText(self.tr("assets.filters"))
        for index, (title_key, _extensions) in enumerate(FILTER_CATEGORIES):
            self.filter_combo.setItemText(index, self.tr(title_key))
        self.hint.setText(self.tr("assets.hint"))

    # ------------------------------------------------------------------ #
    # Фильтр и выбор
    # ------------------------------------------------------------------ #
    def _on_filter_changed(self, index: int) -> None:
        """Применить выбранную категорию фильтра."""
        self._apply_filter(index)

    def _apply_filter(self, index: int) -> None:
        """Передать набор расширений в прокси-модель."""
        if 0 <= index < len(FILTER_CATEGORIES):
            self.proxy.set_extensions(FILTER_CATEGORIES[index][1])

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        """Сообщить о выборе файла."""
        path = self.path_for_index(current)
        if path is None or not path.is_file():
            return
        if QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            self._pending_path = str(path)  # выбор мышью - после отпускания
        else:
            self._pending_path = None
            self.asset_selected.emit(str(path))

    def _flush_pending(self) -> None:
        """Сообщить об отложенном выборе: кнопку отпустили без drag."""
        path, self._pending_path = self._pending_path, None
        if path is not None:
            self.asset_selected.emit(path)

    def _cancel_pending(self) -> None:
        """Отменить отложенный выбор: файл потащили."""
        self._pending_path = None

    def _on_double_click(self, index: QModelIndex) -> None:
        """Добавить модель в сцену по двойному клику."""
        path = self.path_for_index(index)
        if path is None or not path.is_file():
            return
        if path.name.lower().endswith(".scene.json"):
            self.scene_activated.emit(str(path))
        elif path.suffix.lower() == ".py":
            self.script_activated.emit(str(path))
        elif path.suffix.lower() in MODEL_EXTENSIONS:
            self.model_activated.emit(str(path))
        else:
            self.status(self.tr("status.unsupported_asset", name=path.name))

    def path_for_index(self, index: QModelIndex) -> Path | None:
        """Путь к файлу по индексу прокси-модели.

        Args:
            index: Индекс в дереве.

        Returns:
            Путь или ``None``.
        """
        if not index.isValid():
            return None
        source_index = self.proxy.mapToSource(index)
        return Path(self.fs_model.filePath(source_index))

    # ------------------------------------------------------------------ #
    # Перетаскивание файлов из проводника
    # ------------------------------------------------------------------ #
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Принять перетаскивание локальных файлов."""
        if (
            event.source() not in (self.tree, self.tree.viewport())
            and event.mimeData().hasUrls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - API Qt
        """Скопировать брошенные файлы в проект."""
        if event.source() in (self.tree, self.tree.viewport()):
            return  # перетаскивание из самой панели — не импорт
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        copied = self.import_files(paths)
        if copied:
            event.acceptProposedAction()

    def import_files(self, paths: list[Path], target: Path | None = None) -> list[Path]:
        """Скопировать файлы в каталог ассетов и сразу зарегистрировать их.

        Args:
            paths: Исходные файлы (каталоги пропускаются).
            target: Каталог назначения; по умолчанию — выбранная папка
                или корень ассетов.

        Returns:
            Пути скопированных файлов (существующие не перезаписываются).
        """
        folder = target or self._current_folder()
        folder.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for source in paths:
            if not source.is_file():
                continue
            destination = _unique_path(folder / source.name)
            try:
                shutil.copy2(source, destination)
            except OSError as exc:
                self.status(
                    self.tr("assets.copy_failed", name=source.name, error=str(exc))
                )
                continue
            copied.append(destination)
        if copied:
            self.status(self.tr("assets.copied", count=len(copied)))
            if self.context.imports is not None:
                self.context.imports.poll(force=True)
        return copied

    def _current_folder(self) -> Path:
        """Папка, выбранная в дереве (или файл → его папка), иначе корень."""
        path = self.path_for_index(self.tree.currentIndex())
        if path is None:
            return self.root
        folder = path if path.is_dir() else path.parent
        try:
            folder.relative_to(self.root)
        except ValueError:
            return self.root
        return folder

    def select_path(self, path: Path) -> None:
        """Раскрыть дерево и выделить файл (используется при старте)."""
        index = self.proxy.mapFromSource(self.fs_model.index(str(path)))
        if index.isValid():
            self.tree.setCurrentIndex(index)
            self.tree.scrollTo(index)
