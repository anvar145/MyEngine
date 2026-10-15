"""Базовые классы панелей редактора.

Большинство панелей — это ``QDockWidget``: их можно перетаскивать, отстыковывать
в отдельные окна, сворачивать и скрывать через меню «Вид». Общий предок решает
три задачи: единый доступ к :class:`~editor.context.EditorContext`, автоматический
перевод заголовка и повторный перевод содержимого при смене языка.

Для тех панелей, которым тесно в доке, есть :class:`WindowPanel` — то же самое,
но самостоятельным окном: оно не пристыковывается к главному, зато живёт рядом с
ним и работает с тем же контекстом.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from editor.context import EditorContext

__all__ = ["DockPanel", "WindowPanel"]


class _PanelMixin:
    """Общая часть панелей: контекст, перевод строк и статусные сообщения.

    Сигнал ``status_message`` объявляют конкретные классы: PySide6 умеет
    регистрировать ``Signal`` только в классах, унаследованных от ``QObject``.
    """

    context: EditorContext
    title_key: str

    def _init_panel(
        self, title_key: str, context: EditorContext, object_name: str
    ) -> None:
        """Запомнить контекст и подписаться на смену языка.

        Args:
            title_key: Ключ перевода заголовка.
            context: Контекст редактора.
            object_name: Имя объекта Qt (для сохранения состояния и тестов).
        """
        self.context = context
        self.title_key = title_key
        self.setObjectName(object_name)
        context.translator.language_changed.connect(self.retranslate)
        self.setWindowTitle(self.context.tr(self.title_key))

    def retranslate(self, *_args: object) -> None:
        """Перевести заголовок и содержимое панели.

        Наследники переопределяют метод и вызывают ``super().retranslate()``.
        """
        self.setWindowTitle(self.context.tr(self.title_key))
        self.setToolTip(self.context.tr(self.title_key))

    def tr(self, key: str, **kwargs: object) -> str:  # noqa: A003 - привычное имя Qt
        """Перевести строку через переводчик редактора."""
        return self.context.tr(key, **kwargs)

    def status(self, message: str) -> None:
        """Отправить сообщение в статусную строку."""
        self.status_message.emit(message)


class DockPanel(_PanelMixin, QDockWidget):
    """Док-панель с поддержкой локализации.

    Args:
        title_key: Ключ перевода заголовка (например, ``"panel.hierarchy"``).
        context: Контекст редактора.
        parent: Родительский виджет.

    Signals:
        status_message: Сообщение для статусной строки редактора.
    """

    status_message = Signal(str)

    def __init__(
        self,
        title_key: str,
        context: EditorContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_panel(title_key, context, title_key.replace(".", "_"))
        self.setAllowedAreas(
            Qt.LeftDockWidgetArea
            | Qt.RightDockWidgetArea
            | Qt.BottomDockWidgetArea
            | Qt.TopDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        self.setTitleBarWidget(None)


class WindowPanel(_PanelMixin, QMainWindow):
    """Самостоятельное окно редактора с поддержкой локализации.

    Окно не пристыковывается к главному, но делит с ним контекст: сцена,
    выделение, переводчик и рантайм скриптов общие. Родитель передаётся только
    для владения временем жизни — окно остаётся отдельным.

    Args:
        title_key: Ключ перевода заголовка.
        context: Контекст редактора.
        parent: Родительский виджет (окно всё равно самостоятельное).

    Signals:
        status_message: Сообщение для статусной строки редактора.
    """

    status_message = Signal(str)

    def __init__(
        self,
        title_key: str,
        context: EditorContext,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._init_panel(title_key, context, f"{title_key.replace('.', '_')}_window")

    def place_next_to(self, anchor: QWidget | None) -> None:
        """Поставить окно со сдвигом от главного, чтобы не закрывать его.

        Args:
            anchor: Главное окно; ``None`` или невидимое окно ничего не меняет.
        """
        if anchor is None or not anchor.isVisible():
            return
        origin = anchor.frameGeometry().topLeft()
        self.move(origin.x() + 48, origin.y() + 48)
