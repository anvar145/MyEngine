"""Встроенный выбор скрипта: дерево папки ``scripts`` проекта с подпапками."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from editor.panels.script_editor import SCRIPT_SUFFIX, ScriptSourceFilter

#: Роль элемента дерева, в которой лежит абсолютный путь файла.
PATH_ROLE = Qt.ItemDataRole.UserRole


def list_scripts(root: Path) -> list[Path]:
    """Все скрипты ``*.py`` в каталоге и его подпапках (без служебных имён).

    Args:
        root: Каталог ``scripts`` проекта.

    Returns:
        Отсортированный список путей; пустой, если каталога нет.
    """
    if not root.is_dir():
        return []
    rule = ScriptSourceFilter()
    return sorted(
        path
        for path in root.rglob(f"*{SCRIPT_SUFFIX}")
        if path.is_file()
        and rule.matches_name(path.name)
        and not any(
            part.startswith((".", "__")) for part in path.relative_to(root).parts[:-1]
        )
    )


class ScriptPicker(QDialog):
    """Диалог выбора скрипта; показывает только содержимое ``scripts``.

    Args:
        scripts_dir: Каталог со скриптами проекта.
        title: Заголовок окна.
        parent: Родительский виджет.

    Attributes:
        tree: Дерево папок и файлов.
    """

    def __init__(
        self,
        scripts_dir: Path,
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.scripts_dir = scripts_dir
        self.setWindowTitle(title)
        self.resize(420, 480)
        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.tree.itemDoubleClicked.connect(lambda *_: self._accept_if_file())
        self.tree.itemSelectionChanged.connect(self._update_ok)
        self._fill()
        self._update_ok()

    def _fill(self) -> None:
        """Построить дерево по списку скриптов."""
        folders: dict[Path, QTreeWidgetItem] = {}
        for path in list_scripts(self.scripts_dir):
            parent: QTreeWidgetItem | None = None
            relative = path.relative_to(self.scripts_dir)
            current = Path()
            for part in relative.parts[:-1]:
                current = current / part
                node = folders.get(current)
                if node is None:
                    node = QTreeWidgetItem([part])
                    if parent is None:
                        self.tree.addTopLevelItem(node)
                    else:
                        parent.addChild(node)
                    folders[current] = node
                parent = node
            leaf = QTreeWidgetItem([path.name])
            leaf.setData(0, PATH_ROLE, str(path))
            if parent is None:
                self.tree.addTopLevelItem(leaf)
            else:
                parent.addChild(leaf)
        self.tree.expandAll()

    def selected_path(self) -> Path | None:
        """Путь выбранного файла или ``None`` (выбрана папка/ничего)."""
        items = self.tree.selectedItems()
        if not items:
            return None
        value = items[0].data(0, PATH_ROLE)
        return Path(value) if value else None

    def _update_ok(self) -> None:
        """Кнопка OK доступна только при выбранном файле."""
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self.selected_path() is not None
        )

    def _accept_if_file(self) -> None:
        """Двойной клик по файлу подтверждает выбор."""
        if self.selected_path() is not None:
            self.accept()
