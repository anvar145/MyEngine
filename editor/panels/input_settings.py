"""Панель «Настройки ввода»: ремаппинг действий проекта (``Settings/input.json``).

Показывает действия ``InputMap`` деревом: у действия - список привязок, у каждой
кнопки «Изменить» и «Удалить». «Сохранить» пишет в проект только отличия от
дефолтов; плеер читает файл при старте.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from editor.panels.base import DockPanel
from editor.play_mode_qt_input import qt_key_name
from editor.project import SETTINGS_DIRNAME
from engine.core.input import Binding, InputMap, InputState, default_input_map

__all__ = ["BindingCaptureDialog", "InputSettingsPanel", "binding_text"]

#: Имя файла привязок внутри ``Settings``.
SETTINGS_FILENAME = "input.json"

_MOUSE_NAMES = {
    Qt.LeftButton: "left",
    Qt.RightButton: "right",
    Qt.MiddleButton: "middle",
}


def binding_text(binding: Binding) -> str:
    """Короткое описание привязки: ``keyboard: w`` (``x-1``, ``Y`` у осей)."""
    text = f"{binding.device}: {binding.code}"
    if binding.scale != 1.0:
        text += f"  x{binding.scale:g}"
    if binding.axis:
        text += "  (Y)"
    return text


class BindingCaptureDialog(QDialog):
    """Модальный диалог «Нажмите клавишу…»: ждёт клавишу или кнопку мыши.

    ``Esc`` или «Отмена» закрывают диалог без результата.

    Attributes:
        result_binding: ``(device, code)`` выбранной привязки или ``None``.
    """

    def __init__(
        self, title: str, hint: str, cancel: str, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.result_binding: tuple[str, str] | None = None
        layout = QVBoxLayout(self)
        self.label = QLabel(hint, self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self.cancel_button = QPushButton(cancel, self)
        # Space/Enter не должны «нажимать» кнопку: они - кандидаты в привязки.
        self.cancel_button.setFocusPolicy(Qt.NoFocus)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.setMinimumWidth(280)

    def capture_key(self, key: int) -> None:
        """Принять клавишу Qt (``Esc`` - отмена; неизвестные игнорируются)."""
        if key == Qt.Key_Escape:
            self.reject()
            return
        name = qt_key_name(key)
        if name is not None:
            self.result_binding = ("keyboard", name)
            self.accept()

    def capture_mouse(self, name: str) -> None:
        """Принять кнопку мыши (``left``/``right``/``middle``)."""
        self.result_binding = ("mouse", name)
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if not event.isAutoRepeat():
            self.capture_key(event.key())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        name = _MOUSE_NAMES.get(event.button())
        if name is not None:
            self.capture_mouse(name)


class InputSettingsPanel(DockPanel):
    """Редактор привязок действий ввода проекта."""

    def __init__(self, context, parent: QWidget | None = None) -> None:
        super().__init__("panel.input", context, parent)
        self.input_map: InputMap = default_input_map()
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.tree = QTreeWidget(container)
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(True)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        layout.addWidget(self.tree, 1)

        row = QHBoxLayout()
        self.reset_all_button = QPushButton(container)
        self.reset_all_button.clicked.connect(self.reset_all)
        self.save_button = QPushButton(container)
        self.save_button.clicked.connect(self.save)
        self.load_button = QPushButton(container)
        self.load_button.clicked.connect(self.load)
        for button in (self.reset_all_button, self.load_button, self.save_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.status_label = QLabel(container)
        layout.addWidget(self.status_label)
        self.setWidget(container)

        context.project_changed.connect(lambda *_: self.load(silent=True))
        self.load(silent=True)
        self.retranslate()

    # ------------------------------------------------------------------ #
    @property
    def settings_file(self) -> Path:
        """Файл привязок активного проекта."""
        return Path(self.context.project_root) / SETTINGS_DIRNAME / SETTINGS_FILENAME

    def retranslate(self, *_args: object) -> None:
        """Перевести подписи панели."""
        super().retranslate()
        self.reset_all_button.setText(self.tr("input.reset_all"))
        self.save_button.setText(self.tr("input.save"))
        self.load_button.setText(self.tr("input.load"))
        self.refresh()

    def refresh(self) -> None:
        """Перестроить дерево по текущей карте."""
        self.tree.clear()
        for name in self.input_map.names():
            action = self.input_map.actions[name]
            title = f"{name}  [{action.kind}]"
            if not self.input_map.is_default(name):
                title += "  *"
            top = QTreeWidgetItem([title, ""])
            self.tree.addTopLevelItem(top)
            top.setData(0, Qt.UserRole, name)
            self.tree.setItemWidget(
                top,
                1,
                self._buttons(
                    (self.tr("input.add"), partial(self.add_binding, name)),
                    (self.tr("input.reset"), partial(self.reset_action, name)),
                ),
            )
            for index, binding in enumerate(action.bindings):
                child = QTreeWidgetItem([binding_text(binding), ""])
                top.addChild(child)
                self.tree.setItemWidget(
                    child,
                    1,
                    self._buttons(
                        (
                            self.tr("input.edit"),
                            partial(self.edit_binding, name, index),
                        ),
                        (
                            self.tr("input.remove"),
                            partial(self.remove_binding, name, index),
                        ),
                    ),
                )
            top.setExpanded(True)

    def _buttons(self, *specs) -> QWidget:
        box = QWidget(self.tree)
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        for text, handler in specs:
            button = QPushButton(text, box)
            # В строке дерева кнопка не должна раздувать высоту строки.
            button.setFixedHeight(22)
            button.setStyleSheet("padding: 0 8px;")
            button.clicked.connect(handler)
            lay.addWidget(button)
        return box

    # ------------------------------------------------------------------ #
    # Операции
    # ------------------------------------------------------------------ #
    def ask_binding(self) -> tuple[str, str] | None:
        """Показать диалог «Нажмите клавишу…»; ``None`` при отмене."""
        dialog = BindingCaptureDialog(
            self.tr("input.capture.title"),
            self.tr("input.capture.hint"),
            self.tr("input.capture.cancel"),
            self,
        )
        dialog.exec()
        return dialog.result_binding

    def apply_capture(self, action: str, index: int, picked: tuple[str, str]) -> None:
        """Записать ``picked`` в привязку (ось и знак старой сохраняются)."""
        state = InputState(input_map=self.input_map)
        state.begin_rebind(action, index)
        device, code = picked
        if device == "mouse":
            state.mouse_button(code, True)
        else:
            state.key_down(code)
        self.refresh()

    def edit_binding(self, action: str, index: int) -> None:
        """Заменить привязку на нажатую пользователем."""
        picked = self.ask_binding()
        if picked is not None:
            self.apply_capture(action, index, picked)

    def add_binding(self, action: str) -> None:
        """Добавить привязку, нажатую пользователем."""
        picked = self.ask_binding()
        if picked is not None:
            index = len(self.input_map.actions[action].bindings)
            self.apply_capture(action, index, picked)

    def remove_binding(self, action: str, index: int) -> None:
        """Удалить привязку."""
        self.input_map.remove_binding(action, index)
        self.refresh()

    def reset_action(self, action: str) -> None:
        """Вернуть действие к дефолту (у пользовательских дефолта нет)."""
        try:
            self.input_map.reset_to_default(action)
        except KeyError:
            return
        self.refresh()

    def reset_all(self) -> None:
        """Вернуть все действия к дефолту."""
        self.input_map.reset_all()
        self.refresh()

    def save(self) -> None:
        """Записать в ``Settings/input.json`` только отличия от дефолтов."""
        self.input_map.save(self.settings_file, only_changes=True)
        self._say(self.tr("input.saved", path=str(self.settings_file)))

    def load(self, silent: bool = False) -> None:
        """Загрузить привязки проекта поверх дефолтов."""
        path = self.settings_file
        if path.is_file():
            try:
                self.input_map = InputMap.load(path)
            except (OSError, ValueError) as error:
                self._say(self.tr("input.load_failed", error=str(error)))
                return
            if not silent:
                self._say(self.tr("input.loaded", path=str(path)))
        else:
            self.input_map = default_input_map()
            if not silent:
                self._say(self.tr("input.no_file"))
        self.refresh()

    def _say(self, message: str) -> None:
        self.status_label.setText(message)
        self.status(message)
