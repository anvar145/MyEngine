"""Переиспользуемые виджеты редактора.

Здесь собраны небольшие элементы, которые нужны нескольким панелям: редактор
трёхкомпонентного вектора, кнопка выбора цвета и заголовок секции. Все они
умеют переводить свои подписи, поэтому интерфейс остаётся локализованным.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from editor.theme import icon
from engine.core.math3d import Vector3

__all__ = [
    "ColorButton",
    "FocusComboBox",
    "FocusDoubleSpinBox",
    "FocusSpinBox",
    "FormSection",
    "SectionBox",
    "SideRail",
    "VectorEditor",
    "fit_widths",
]


_AXIS_LABEL_WIDTH = 14


def _wheel_only_with_focus(widget: QWidget, event, base_wheel) -> None:
    """Принять колесо только у поля с фокусом, иначе отдать родителю (скролл)."""
    if widget.hasFocus():
        base_wheel(event)
    else:
        event.ignore()


_SELECT_ALL_REASONS = (
    Qt.FocusReason.MouseFocusReason,
    Qt.FocusReason.TabFocusReason,
    Qt.FocusReason.BacktabFocusReason,
    Qt.FocusReason.ShortcutFocusReason,
)


def _select_all_later(spin: QAbstractSpinBox, event) -> None:
    """Выделить всё содержимое поля после получения фокуса.

    Выделение откладывается: иначе его затирает обработка нажатия мыши,
    которая ставит курсор уже после ``focusInEvent``.
    """
    if event.reason() in _SELECT_ALL_REASONS:
        QTimer.singleShot(0, spin.selectAll)


class FocusDoubleSpinBox(QDoubleSpinBox):
    """``QDoubleSpinBox``: колесо — только с фокусом, клик выделяет всё поле."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def focusInEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        super().focusInEvent(event)
        _select_all_later(self, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        had_focus = self.hasFocus()
        super().mousePressEvent(event)
        if not had_focus:
            QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        _wheel_only_with_focus(self, event, super().wheelEvent)


class FocusSpinBox(QSpinBox):
    """``QSpinBox``: колесо — только с фокусом, клик выделяет всё поле."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def focusInEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        super().focusInEvent(event)
        _select_all_later(self, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        had_focus = self.hasFocus()
        super().mousePressEvent(event)
        if not had_focus:
            QTimer.singleShot(0, self.selectAll)

    def wheelEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        _wheel_only_with_focus(self, event, super().wheelEvent)


class FocusComboBox(QComboBox):
    """``QComboBox``: колесо переключает пункт только у поля с фокусом."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Ширина не по самому длинному пункту (иначе длинный путь ассета
        # распирает панель).
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setMinimumContentsLength(8)

    def wheelEvent(self, event) -> None:  # noqa: N802 - имя из Qt
        _wheel_only_with_focus(self, event, super().wheelEvent)


class VectorEditor(QWidget):
    """Редактор вектора из трёх числовых полей (X, Y, Z).

    Args:
        minimum: Минимальное значение полей.
        maximum: Максимальное значение полей.
        step: Шаг изменения.
        decimals: Число знаков после запятой.
        labels: Подписи осей.
        parent: Родительский виджет.

    Signals:
        value_changed: Значение изменено пользователем (аргумент — ``Vector3``).
    """

    value_changed = Signal(object)

    def __init__(
        self,
        *,
        minimum: float = -10_000.0,
        maximum: float = 10_000.0,
        step: float = 0.1,
        decimals: int = 3,
        labels: tuple[str, str, str] = ("X", "Y", "Z"),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._spins: list[QDoubleSpinBox] = []
        for label in labels:
            caption = QLabel(label, self)
            caption.setProperty("role", "axis")
            caption.setProperty("axis", label.lower())
            caption.setFixedWidth(_AXIS_LABEL_WIDTH)
            layout.addWidget(caption)
            box = FocusDoubleSpinBox(self)
            box.setRange(minimum, maximum)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setKeyboardTracking(False)
            box.setMinimumWidth(40)
            # Равный stretch и политика Expanding: ширина не зависит от текста.
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            box.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            box.valueChanged.connect(self._on_changed)
            layout.addWidget(box, 1)
            self._spins.append(box)

    # ------------------------------------------------------------------ #
    # Значение
    # ------------------------------------------------------------------ #
    def value(self) -> Vector3:
        """Текущее значение вектора."""
        return Vector3(
            self._spins[0].value(), self._spins[1].value(), self._spins[2].value()
        )

    def set_value(self, value: Vector3, *, emit: bool = False) -> None:
        """Установить значение без лишних сигналов.

        Args:
            value: Новое значение.
            emit: Испускать ли ``value_changed``.
        """
        for spin, component in zip(
            self._spins, (value.x, value.y, value.z), strict=True
        ):
            spin.blockSignals(not emit)
            spin.setValue(component)
            spin.blockSignals(False)

    def set_editable(self, editable: bool) -> None:
        """Включить или выключить все поля."""
        for spin in self._spins:
            spin.setEnabled(editable)

    def _on_changed(self, *_args: object) -> None:
        """Передать новое значение подписчикам."""
        self.value_changed.emit(self.value())


class ColorButton(QPushButton):
    """Кнопка-образец цвета с открытием системного диалога выбора цвета.

    Args:
        color: Начальный цвет.
        parent: Родительский виджет.

    Signals:
        color_changed: Цвет изменён пользователем (аргумент — ``Vector3``).
    """

    color_changed = Signal(object)

    def __init__(
        self, color: Vector3 | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._vector = color or Vector3(1.0, 1.0, 1.0)
        self.setMinimumHeight(26)
        self.clicked.connect(self._pick)
        self._update_swatch()

    # ------------------------------------------------------------------ #
    # Значение
    # ------------------------------------------------------------------ #
    def vector(self) -> Vector3:
        """Текущий цвет как вектор ``0..1``."""
        return self._vector

    def set_vector(self, value: Vector3, *, emit: bool = False) -> None:
        """Задать цвет.

        Args:
            value: Новый цвет.
            emit: Испускать ли ``color_changed``.
        """
        self._vector = value
        self._update_swatch()
        if emit:
            self.color_changed.emit(value)

    def set_label(self, text: str) -> None:
        """Обновить текст кнопки (используется при смене языка)."""
        self._label = text
        self._update_swatch()

    def _update_swatch(self) -> None:
        """Показать текущий цвет фоном кнопки."""
        color = QColor.fromRgbF(
            _clamp(self._vector.x), _clamp(self._vector.y), _clamp(self._vector.z)
        )
        text_color = "#141312" if color.lightness() > 140 else "#e8e4de"
        self.setStyleSheet(
            f"QPushButton {{ background-color: {color.name()}; color: {text_color}; "
            f"border: 1px solid #3a3835; border-radius: 7px; padding: 4px 10px; }}"
        )
        self.setText(f"{getattr(self, '_label', '')}  {color.name()}".strip())

    def _pick(self) -> None:
        """Открыть диалог выбора цвета."""
        current = QColor.fromRgbF(
            _clamp(self._vector.x), _clamp(self._vector.y), _clamp(self._vector.z)
        )
        chosen = QColorDialog.getColor(current, self, self.tr("Select colour"))
        if chosen.isValid():
            self.set_vector(
                Vector3(chosen.redF(), chosen.greenF(), chosen.blueF()), emit=True
            )


class SectionBox(QWidget):
    """Секция инспектора: заголовок и контейнер для строк формы.

    Строка заголовка по умолчанию содержит только название. Секция компонента,
    который можно выключить или удалить, дополняет её
    :meth:`add_header_controls`: галочкой «включён» и кнопкой «удалить». Серая
    полоса слева у выключенного компонента - динамическое свойство ``frozen``
    строки заголовка (:meth:`set_frozen`, стиль в ``dark.qss.tpl``).

    Args:
        title: Заголовок секции.
        parent: Родительский виджет.

    Signals:
        enabled_toggled: Пользователь переключил галочку «включён» (``bool``).
        remove_requested: Пользователь нажал кнопку «удалить».
    """

    enabled_toggled = Signal(bool)
    remove_requested = Signal()

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 6)
        self._layout.setSpacing(4)
        self.header = QFrame(self)
        self.header.setProperty("role", "sectionHeader")
        self.header.setProperty("frozen", False)
        self._header_layout = QHBoxLayout(self.header)
        self._header_layout.setContentsMargins(6, 0, 0, 0)
        self._header_layout.setSpacing(6)
        self.title = QLabel(title, self.header)
        self.title.setProperty("role", "section")
        self.title.setProperty("frozen", False)
        self.title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._header_layout.addWidget(self.title, 1)
        self._layout.addWidget(self.header)
        self.enabled_check: QCheckBox | None = None
        self.remove_button: QPushButton | None = None

    def set_title(self, title: str) -> None:
        """Заменить заголовок секции."""
        self.title.setText(title)

    def add_header_controls(
        self,
        *,
        enabled: bool,
        enabled_text: str,
        remove_text: str,
        enabled_tooltip: str = "",
        remove_tooltip: str = "",
    ) -> None:
        """Добавить в строку заголовка галочку «включён» и кнопку «удалить».

        Сама секция ничего не включает и не удаляет: она шлёт сигналы
        :attr:`enabled_toggled` и :attr:`remove_requested`, а владелец решает,
        что делать (и спрашивает подтверждение).

        Args:
            enabled: Начальное состояние компонента.
            enabled_text: Подпись галочки.
            remove_text: Подпись кнопки.
            enabled_tooltip: Подсказка галочки.
            remove_tooltip: Подсказка кнопки.
        """
        if self.enabled_check is not None:
            return
        self.enabled_check = QCheckBox(enabled_text, self.header)
        self.enabled_check.setToolTip(enabled_tooltip)
        self.enabled_check.clicked.connect(self.enabled_toggled)
        self.remove_button = QPushButton(remove_text, self.header)
        self.remove_button.setToolTip(remove_tooltip)
        self.remove_button.clicked.connect(
            lambda _checked=False: self.remove_requested.emit()
        )
        self._header_layout.addWidget(self.enabled_check)
        self._header_layout.addWidget(self.remove_button)
        self.set_header_enabled(enabled)

    def set_header_enabled(self, enabled: bool) -> None:
        """Показать состояние компонента: галочка и серая полоса (без сигналов)."""
        if self.enabled_check is not None:
            self.enabled_check.setChecked(enabled)
        self.set_frozen(not enabled)

    def set_frozen(self, frozen: bool) -> None:
        """Включить или убрать серую полосу и приглушённый цвет заголовка."""
        for widget in (self.header, self.title):
            widget.setProperty("frozen", bool(frozen))
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)

    def add_widget(self, widget: QWidget) -> None:
        """Добавить виджет в секцию."""
        self._layout.addWidget(widget)

    def add_layout(self, layout: object) -> None:
        """Добавить вложенную раскладку."""
        self._layout.addLayout(layout)  # type: ignore[arg-type]


class FormSection(SectionBox):
    """Секция с формой «подпись — поле» и короткими фабриками полей.

    Args:
        title: Заголовок секции.
        parent: Родительский виджет.
    """

    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.form = QFormLayout()
        self.form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.form.setHorizontalSpacing(12)
        self.form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.add_layout(self.form)

    def add_spin(
        self,
        label: str,
        value: float,
        slot,
        *,
        minimum: float = 0.0,
        maximum: float = 100.0,
        step: float = 0.1,
        decimals: int = 2,
        suffix: str = "",
        tooltip: str = "",
    ) -> FocusDoubleSpinBox:
        """Добавить числовое поле (колесо работает только при фокусе)."""
        spin = FocusDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(suffix)
        spin.setValue(value)
        if tooltip:
            spin.setToolTip(tooltip)
        spin.valueChanged.connect(slot)
        self.form.addRow(label, spin)
        return spin

    def add_int(
        self,
        label: str,
        value: int,
        slot,
        *,
        minimum: int = 0,
        maximum: int = 100,
        tooltip: str = "",
    ) -> FocusSpinBox:
        """Добавить целочисленное поле."""
        spin = FocusSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        if tooltip:
            spin.setToolTip(tooltip)
        spin.valueChanged.connect(slot)
        self.form.addRow(label, spin)
        return spin

    def add_check(
        self, text: str, checked: bool, slot, *, tooltip: str = "", label: str = ""
    ) -> QCheckBox:
        """Добавить флажок."""
        box = QCheckBox(text, self)
        box.setChecked(checked)
        if tooltip:
            box.setToolTip(tooltip)
        box.toggled.connect(slot)
        self.form.addRow(label, box)
        return box

    def add_combo(
        self, label: str, items: list[tuple[str, object]], current: object, slot,
        *, tooltip: str = "",
    ) -> FocusComboBox:
        """Добавить выпадающий список из пар ``(подпись, данные)``."""
        combo = FocusComboBox(self)
        for text, data in items:
            combo.addItem(text, data)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        if tooltip:
            combo.setToolTip(tooltip)
        combo.currentIndexChanged.connect(
            lambda _index, box=combo: slot(box.currentData())
        )
        self.form.addRow(label, combo)
        return combo


#: Минимальная ширина полей ввода в формах: ниже панель не сжимается.
_FIELD_MIN_WIDTH = 48


def fit_widths(root: QWidget) -> None:
    """Не дать полям ввода раздувать минимальную ширину панели.

    ``minimumSizeHint`` поля зависит от текста (длинный путь, placeholder,
    подпись кнопки), и форма не может стать уже суммы таких минимумов. Задаём
    полям небольшой явный минимум, строкам форм — перенос длинных строк, а
    подписям форм — перенос слов.

    Args:
        root: Корень поддерева (содержимое инспектора, секция).
    """
    for form in root.findChildren(QFormLayout):
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            label = item.widget() if item is not None else None
            if isinstance(label, QLabel):
                label.setWordWrap(True)
    for field in (
        *root.findChildren(QLineEdit),
        *root.findChildren(QAbstractSpinBox),
        *root.findChildren(QComboBox),
    ):
        if field.minimumWidth() == 0:
            field.setMinimumWidth(_FIELD_MIN_WIDTH)
        policy = field.sizePolicy()
        if policy.horizontalPolicy() == QSizePolicy.Policy.Preferred:
            policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
            field.setSizePolicy(policy)
    for button in root.findChildren(QPushButton):
        # Длинная подпись кнопки не должна задавать минимум панели: текст
        # обрезается, полный текст — в подсказке.
        if button.minimumWidth() == 0 and button.minimumSizeHint().width() > 120:
            if not button.toolTip():
                button.setToolTip(button.text())
            button.setMinimumWidth(_FIELD_MIN_WIDTH)
            vertical = button.sizePolicy().verticalPolicy()
            button.setSizePolicy(QSizePolicy.Policy.Ignored, vertical)
    for box in root.findChildren(QCheckBox):
        if box.minimumWidth() == 0 and box.minimumSizeHint().width() > 120:
            if not box.toolTip():
                box.setToolTip(box.text())
            box.setMinimumWidth(_FIELD_MIN_WIDTH)
    for label in root.findChildren(QLabel):
        if not label.wordWrap() and label.minimumSizeHint().width() > 120:
            label.setWordWrap(True)


def _clamp(value: float) -> float:
    """Ограничить компоненту цвета диапазоном ``0..1``."""
    return max(0.0, min(1.0, float(value)))


class SideRail(QToolBar):
    """Левая иконочная рейка: кнопки показа и скрытия док-панелей.

    Кнопка подсвечена, пока панель видна. Клик по кнопке скрытой или
    спрятанной за вкладкой панели показывает её и выводит наверх, клик по
    видимой — скрывает. Рейка не владеет раскладкой: состояние берётся из
    самих доков, поэтому сохранение и сброс раскладки её не затрагивают.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sideRail")
        self.setMovable(False)
        self.setFloatable(False)
        self.setOrientation(Qt.Orientation.Vertical)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.buttons: dict[str, QAction] = {}
        self._docks: dict[str, QDockWidget] = {}

    def add_dock(self, key: str, dock: QDockWidget, icon_name: str) -> QAction:
        """Добавить кнопку для док-панели.

        Args:
            key: Ключ перевода названия панели (``"panel.hierarchy"``).
            dock: Управляемая док-панель.
            icon_name: Имя иконки из ``assets/icons``.

        Returns:
            Действие кнопки (checkable).
        """
        action = QAction(icon(icon_name), "", self)
        action.setCheckable(True)
        action.setChecked(dock.isVisible())
        action.triggered.connect(lambda _checked=False, k=key: self._toggle(k))
        dock.visibilityChanged.connect(lambda _v=False, k=key: self.sync(k))
        self.addAction(action)
        self.buttons[key] = action
        self._docks[key] = dock
        return action

    def add_action(self, key: str, action: QAction) -> None:
        """Добавить готовое действие (например, окно скриптов) в конец рейки."""
        self.addAction(action)
        self.buttons[key] = action

    def sync(self, key: str) -> None:
        """Обновить подсветку кнопки панели по её видимости."""
        dock = self._docks.get(key)
        action = self.buttons.get(key)
        if dock is not None and action is not None:
            action.setChecked(dock.isVisible())

    def _toggle(self, key: str) -> None:
        """Показать панель наверх или скрыть, если она уже на виду."""
        dock = self._docks[key]
        if dock.isVisible():
            dock.hide()
        else:
            dock.show()
            dock.raise_()
        self.sync(key)
