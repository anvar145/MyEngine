"""Общие фикстуры тестов редактора.

Тесты не показывают окно: ``QOpenGLWidget`` создаётся, но GL-контекст не
инициализируется до первого показа, поэтому проверки интерфейса работают и в
среде без графики (достаточно платформенного плагина Qt).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Платформу выбираем в фикстуре, а не при импорте: иначе она зафиксируется для
# всего процесса pytest и GL-тесты движка не смогут создать контекст.


@pytest.fixture(scope="session")
def qapp():
    """Единственный на весь набор ``QApplication`` с применённой темой.

    Returns:
        Экземпляр ``QApplication``.
    """
    pytest.importorskip("PySide6", reason="для тестов редактора нужен PySide6")
    # На Windows доступна оконная платформа (нужна GL-тестам движка); в других
    # системах достаточно offscreen: окно редактора в тестах не показывается.
    os.environ.setdefault(
        "QT_QPA_PLATFORM", "windows" if os.name == "nt" else "offscreen"
    )
    from PySide6.QtWidgets import QApplication

    from editor.theme import apply_theme

    application = QApplication.instance() or QApplication([])
    apply_theme(application)
    return application


@pytest.fixture(scope="session", autouse=True)
def _isolated_qsettings(tmp_path_factory):
    """Увести ``QSettings`` в ini во временной папке.

    ``QSettings(org, app)`` на Windows пишет в реестр: тесты, закрывающие окно,
    иначе портят реальную сохранённую геометрию и раскладку редактора.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings

    folder = tmp_path_factory.mktemp("qsettings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(folder))
    yield


@pytest.fixture(autouse=True)
def _flush_qt_objects(qapp):
    """Собрать мусор из Qt-объектов теста, пока приложение ещё живо.

    Виджеты без родителя (редакторы кода, всплывающие подсказки, файловые
    модели) PySide6 уничтожает неохотно: если оставить их сборщику мусора,
    pytest соберёт их на выходе уже после разбора фикстур — и процесс падает
    с ``Fatal Python error: Aborted`` прямо в ``gc_collect_harder``. Поэтому
    собираем мусор сразу после каждого теста, когда ``QApplication`` и цикл
    событий в порядке.
    """
    yield
    import gc

    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    if application is not None:
        application.processEvents()
    gc.collect()


@pytest.fixture(autouse=True)
def _no_modal_questions(qapp, monkeypatch):
    """Не давать ``QMessageBox.question`` блокировать тест при закрытии окон.

    По умолчанию отвечаем «Не сохранять»; тесты диалогов подменяют ответ сами.
    """
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Discard),
    )


@pytest.fixture()
def context(qapp):
    """Контекст редактора с пустой сценой.

    Returns:
        :class:`editor.context.EditorContext`.
    """
    from editor.context import EditorContext

    return EditorContext(project_root=ROOT, language="ru")


@pytest.fixture()
def window(qapp, context):
    """Главное окно редактора (без показа).

    Yields:
        :class:`editor.main_window.MainWindow`.
    """
    from editor.main_window import MainWindow

    main_window = MainWindow(context)
    yield main_window
    main_window.close()
    main_window.deleteLater()
