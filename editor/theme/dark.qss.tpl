/* MyEngine Editor — тёмная тёплая тема (шаблон QSS).
 *
 * Цвета задаются токенами «@имя» и подставляются из PALETTE
 * (editor/theme/__init__.py) один раз при запуске: единственный источник
 * правды. Без синих оттенков, терракотовый акцент (@accent), скругления 8 px.
 * QSS не умеет transitions и теней, поэтому всё статично — это дёшево;
 * плавность даёт editor/anim.py (геометрия и прозрачность).
 */

QWidget {
    background-color: @window;
    color: @text;
    font-family: "Inter", "Segoe UI", "Noto Sans", sans-serif;
    font-size: 13px;
}

QMainWindow::separator {
    background-color: @window;
    width: 3px;
    height: 3px;
}

QMainWindow::separator:hover {
    background-color: @accent;
}

/* ------------------------------ меню и тулбар ---------------------------- */

QMenuBar {
    background-color: @panel;
    border-bottom: 1px solid @border;
    padding: 2px 6px;
}

QMenuBar::item {
    padding: 5px 10px;
    border-radius: 6px;
    background: transparent;
}

QMenuBar::item:selected {
    background-color: @surface_alt;
}

QMenu {
    background-color: @surface;
    border: 1px solid @border_light;
    border-radius: 10px;
    padding: 5px;
}

QMenu::item {
    padding: 6px 26px 6px 12px;
    border-radius: 6px;
}

QMenu::item:selected {
    background-color: @accent_soft;
    color: @text;
}

QMenu::item:disabled {
    color: @text_dim;
}

QMenu::separator {
    height: 1px;
    background-color: @border;
    margin: 5px 8px;
}

QToolBar {
    background-color: @panel;
    border: none;
    border-bottom: 1px solid @border;
    spacing: 4px;
    padding: 4px 8px;
}

QToolBar::separator {
    background-color: @border;
    width: 1px;
    margin: 6px 6px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 4px 6px;
}

QToolButton:hover {
    background-color: @surface_alt;
}

QToolButton:pressed {
    background-color: @border;
}

QToolButton:checked {
    background-color: @accent_soft;
    border-color: transparent;
}

/* Левая иконочная рейка (editor/widgets.py::SideRail). */
QToolBar#sideRail {
    background-color: @window;
    border: none;
    border-right: 1px solid @border;
    padding: 8px 5px;
    spacing: 6px;
}

QToolBar#sideRail QToolButton {
    padding: 6px;
    border-radius: 9px;
}

QToolBar#sideRail QToolButton:checked {
    background-color: @surface_alt;
}

/* ------------------------------- док-панели ------------------------------ */

QDockWidget {
    font-weight: 600;
}

QDockWidget::title {
    background-color: @panel;
    border-bottom: 1px solid @border;
    padding: 8px 12px;
    text-align: left;
}

QDockWidget {
    titlebar-close-icon: url(@icons/ui/close.svg);
    titlebar-normal-icon: url(@icons/ui/float.svg);
}

QDockWidget::close-button,
QDockWidget::float-button {
    background-color: transparent;
    border: none;
    padding: 0;
    width: 18px;
    height: 18px;
    icon-size: 14px;
}

QDockWidget::close-button:hover,
QDockWidget::float-button:hover {
    background-color: @surface_alt;
    border-radius: 5px;
}

/* ---------------------------------- вкладки ------------------------------ */

QTabWidget::pane {
    border: 1px solid @border;
    border-radius: 8px;
    top: -1px;
}

QTabBar {
    background: transparent;
}

QTabBar::tab {
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 14px;
    margin-right: 2px;
    color: @text_muted;
}

QTabBar::tab:selected {
    color: @text;
    border-bottom-color: @accent;
}

QTabBar::close-button {
    image: url(@icons/ui/close.svg);
    subcontrol-position: right;
    border-radius: 4px;
    padding: 2px;
}

QTabBar::close-button:hover {
    image: url(@icons/ui/close-hover.svg);
    background-color: @surface_alt;
}

QTabBar::tab:hover:!selected {
    color: @text;
    background-color: @hover;
}

/* ---------------------------------- деревья ----------------------------- */

QTreeView,
QListView,
QTableView {
    background-color: @surface;
    alternate-background-color: @surface;
    border: 1px solid @border;
    border-radius: 8px;
    outline: none;
    selection-background-color: @accent_soft;
    selection-color: @text;
}

QTreeView::item,
QListView::item {
    padding: 5px 6px;
    border-radius: 6px;
}

QTreeView::item:hover,
QListView::item:hover {
    background-color: @hover;
}

QTreeView::item:selected,
QListView::item:selected {
    background-color: @accent_soft;
    color: @text;
}

QTreeView::branch {
    background-color: transparent;
}

QHeaderView::section {
    background-color: @panel;
    color: @text_muted;
    border: none;
    border-right: 1px solid @border;
    border-bottom: 1px solid @border;
    padding: 5px 8px;
}

/* ---------------------------------- вводы ------------------------------- */

QLineEdit,
QPlainTextEdit,
QTextEdit,
QSpinBox,
QDoubleSpinBox,
QComboBox {
    background-color: @surface;
    border: 1px solid @border;
    border-radius: 7px;
    padding: 5px 8px;
    selection-background-color: @accent_soft;
    selection-color: @text;
}

QLineEdit:hover,
QSpinBox:hover,
QDoubleSpinBox:hover,
QComboBox:hover {
    border-color: @border_light;
}

QLineEdit:focus,
QPlainTextEdit:focus,
QTextEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus,
QComboBox:focus {
    border-color: @accent;
}

QLineEdit:disabled,
QSpinBox:disabled,
QDoubleSpinBox:disabled,
QComboBox:disabled {
    color: @text_dim;
    background-color: @panel;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: @surface;
    border: 1px solid @border_light;
    border-radius: 8px;
    selection-background-color: @accent_soft;
    padding: 4px;
}

QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button,
QSpinBox::up-button,
QSpinBox::down-button {
    background-color: transparent;
    border: none;
    width: 14px;
}

QDoubleSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover,
QSpinBox::down-button:hover {
    background-color: @surface_alt;
}

/* --------------------------------- кнопки ------------------------------- */

QPushButton {
    background-color: @surface_alt;
    border: 1px solid @border_light;
    border-radius: 8px;
    padding: 6px 14px;
}

QPushButton:hover {
    background-color: @border;
}

QPushButton:pressed {
    background-color: @hover;
}

QPushButton:disabled {
    color: @text_dim;
    background-color: @panel;
    border-color: @border;
}

QPushButton[role="primary"],
QPushButton[accent="true"] {
    background-color: @accent;
    border-color: @accent;
    color: @on_accent;
    font-weight: 600;
}

QPushButton[role="primary"]:hover,
QPushButton[accent="true"]:hover {
    background-color: @accent_hover;
    border-color: @accent_hover;
}

QPushButton[role="primary"]:disabled {
    background-color: @accent_soft;
    border-color: @accent_soft;
    color: @text_dim;
}

QPushButton[role="ghost"] {
    background-color: transparent;
    border-color: transparent;
    color: @text_muted;
}

QPushButton[role="ghost"]:hover {
    background-color: @surface_alt;
    color: @text;
}

/* --------------------------------- чекбоксы ----------------------------- */

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid @border_light;
    border-radius: 5px;
    background-color: @surface;
}

QCheckBox::indicator:checked {
    background-color: @accent;
    border-color: @accent;
}

QCheckBox::indicator:hover {
    border-color: @accent;
}

/* ---------------------------------- группы ------------------------------ */

QGroupBox {
    border: 1px solid @border;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 8px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: @text_muted;
}

/* ---------------------------------- прочее ------------------------------ */

QStatusBar {
    background-color: @panel;
    border-top: 1px solid @border;
    color: @text_muted;
}

QStatusBar::item {
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background-color: @border_light;
    border-radius: 3px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background-color: @text_dim;
}

QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background-color: @border_light;
    border-radius: 3px;
    min-width: 28px;
}

QScrollBar::handle:horizontal:hover {
    background-color: @text_dim;
}

QScrollBar::add-line,
QScrollBar::sub-line,
QScrollBar::add-page,
QScrollBar::sub-page {
    background: none;
    border: none;
    height: 0;
    width: 0;
}

QSplitter::handle {
    background-color: @window;
}

QSplitter::handle:hover {
    background-color: @accent;
}

QLabel[role="section"] {
    color: @text_muted;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* строка заголовка секции компонента; серая полоса - компонент выключен */
QFrame[role="sectionHeader"] {
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
}

QFrame[role="sectionHeader"][frozen="true"] {
    border-left: 3px solid @text_dim;
}

QLabel[role="section"][frozen="true"] {
    color: @text_dim;
}

QLabel[role="hint"] {
    color: @text_muted;
}

QLabel[role="error"] {
    color: @error;
}

QToolTip {
    background-color: @surface;
    color: @text;
    border: 1px solid @border_light;
    border-radius: 6px;
    padding: 5px 8px;
}

QProgressBar {
    background-color: @surface;
    border: 1px solid @border;
    border-radius: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: @accent;
    border-radius: 5px;
}

QSlider::groove:horizontal {
    height: 4px;
    background: @border;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: @accent;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: @text;
}

/* Оси X/Y/Z — условные цвета (красный/зелёный/синий), как в гизмо. */
QLabel[role="axis"] {
    font-weight: bold;
    padding: 0 1px;
    color: @text_muted;
}

QLabel[role="axis"][axis="x"] {
    color: #e06c75;
}

QLabel[role="axis"][axis="y"] {
    color: #8fbf6b;
}

QLabel[role="axis"][axis="z"] {
    color: #6aa3d9;
}

/* ---------------------------- менеджер проектов -------------------------- */

QFrame#launcherSidebar {
    background-color: @panel;
    border: none;
    border-right: 1px solid @border;
}

QFrame#launcherSidebar QLabel,
QFrame#launcherSidebar QCheckBox {
    background: transparent;
}

QPushButton[role="nav"] {
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    text-align: left;
    color: @text_muted;
}

QPushButton[role="nav"]:hover {
    background-color: @surface_alt;
    color: @text;
}

QPushButton[role="nav"]:checked {
    background-color: @accent_soft;
    color: @text;
}

QLabel#launcherTitle {
    font-size: 20px;
    font-weight: 600;
    background: transparent;
}

QLabel#launcherHeading {
    font-size: 18px;
    font-weight: 600;
    background: transparent;
}

QListWidget#projectList {
    background-color: transparent;
    border: none;
}

QListWidget#projectList::item {
    background-color: @surface;
    border: 1px solid @border;
    border-radius: 12px;
    padding: 10px;
    margin: 5px;
}

QListWidget#projectList::item:hover {
    background-color: @hover;
    border-color: @border_light;
}

QListWidget#projectList::item:selected {
    background-color: @accent_soft;
    border-color: @accent;
}

QFrame#launcherCard {
    background-color: @surface;
    border: 1px solid @border;
    border-radius: 12px;
}

QFrame#launcherSidebar QLabel#launcherLogo {
    background-color: @accent;
    border-radius: 9px;
}

QPushButton[role="template"] {
    background-color: @surface;
    border: 1px solid @border;
    border-radius: 12px;
    padding: 12px;
    text-align: left;
}

QPushButton[role="template"]:hover {
    background-color: @hover;
    border-color: @border_light;
}

QPushButton[role="template"]:checked {
    background-color: @accent_soft;
    border-color: @accent;
}
