"""Тесты редактора скриптов: код, поиск, дерево файлов и запуск.

Панель проверяется без показа окна (см. ``editor/tests/conftest.py``), поэтому
тесты работают и в окружении без графики. QFileSystemModel наполняет каталоги
лениво, по запросу вида, поэтому фильтр дерева проверяется напрямую через
``filterAcceptsRow``, а не через количество строк в представлении.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from editor.panels.code_editor import (
    TAB_WIDTH,
    CodeEditor,
    code_font,
    matching_bracket,
)
from editor.panels.find_replace import FindReplaceDialog, resolve_text_target
from editor.panels.script_editor import (
    SCRIPT_SUFFIX,
    UNSAVED_MARK,
    ScriptEditorPanel,
    ScriptRunner,
    ScriptSourceFilter,
)
from engine.scripting import ScriptComponent

#: Исходник скрипта для тестов панели.
SPIN_SOURCE = '''"""Вращение."""

from engine.core.math3d import Vector3

SPEED = 120.0


def on_start(self):
    self.log("старт")
    print("печать из скрипта")


def on_update(self, delta_time):
    self.transform.rotate(Vector3.up(), SPEED * delta_time)
'''


def make_project(root: Path) -> Path:
    """Создать проект с каталогом скриптов и лишним файлом.

    Args:
        root: Каталог проекта.

    Returns:
        Путь к каталогу ``Assets/Scripts``.
    """
    scripts = root / "Assets" / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "spin.py").write_text(SPIN_SOURCE, encoding="utf-8")
    (root / "readme.md").write_text("# проект\n", encoding="utf-8")
    return scripts


def ru_translator():
    """Переводчик редактора для русского языка (реальные подписи диалогов).

    Returns:
        Функция перевода ``tr(key, **kwargs)``.
    """
    from editor.i18n import Translator

    return Translator("ru").tr


@pytest.fixture()
def panel(qapp, tmp_path: Path):
    """Панель редактора скриптов с проектом во временном каталоге.

    Yields:
        :class:`~editor.panels.script_editor.ScriptEditorPanel`.
    """
    from editor.context import EditorContext

    make_project(tmp_path)
    context = EditorContext(project_root=tmp_path, language="ru")
    widget = ScriptEditorPanel(context)
    yield widget
    widget.close()
    widget.deleteLater()


class TestCodeEditor:
    def test_text_round_trip(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("value = 1\n")
        assert editor.text() == "value = 1\n"
        assert editor.is_modified() is False

    def test_modification_flag(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("a = 1\n")
        editor.insertPlainText("b = 2\n")
        assert editor.is_modified() is True
        editor.mark_saved()
        assert editor.is_modified() is False

    def test_monospace_font_is_used(self, qapp) -> None:
        font = code_font()
        assert font.fixedPitch() or font.styleHint() == font.StyleHint.Monospace

    def test_highlighter_is_attached(self, qapp) -> None:
        editor = CodeEditor()
        assert editor.highlighter is not None
        assert "game_object" in editor.highlighter.api_names

    def test_completion_words_include_engine_api(self, qapp) -> None:
        editor = CodeEditor()
        words = editor.completion_words()
        assert "on_update" in words
        assert "transform" in words
        assert "print" in words

    def test_completion_prefix_from_line(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("self.game_obj")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert editor.completion_prefix() == "game_obj"

    def test_show_completions_does_not_crash(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("self.trans")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        editor.show_completions()
        editor.hide_completions()
        assert editor.completion_prefix() == "trans"

    def test_insert_completion_replaces_prefix(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("self.trans")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        editor._completer_visible = True
        editor._completer.setCompletionPrefix("trans")
        editor.insert_completion("self.transform")
        assert editor.text() == "self.transform"


class TestIndentation:
    def test_indent_after_colon(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("def on_start(self):")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert editor.indent_for_new_line() == " " * TAB_WIDTH

    def test_indent_is_inherited(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("if value:")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        editor._insert_new_line()
        cursor = editor.textCursor()
        cursor.insertText("total = 1")
        editor.setTextCursor(cursor)
        assert editor.indent_for_new_line() == " " * TAB_WIDTH

    def test_dedent_after_return(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("    return")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert editor.indent_for_new_line() == ""

    def test_new_line_keeps_indent(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("def f():")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert editor._insert_new_line() is True
        assert editor.text().endswith("\n" + " " * TAB_WIDTH)

    def test_indent_selection_shifts_all_lines(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("a = 1\nb = 2\n")
        cursor = editor.textCursor()
        cursor.select(cursor.SelectionType.Document)
        editor.setTextCursor(cursor)
        editor.indent_selection()
        assert editor.text() == f"{' ' * TAB_WIDTH}a = 1\n{' ' * TAB_WIDTH}b = 2\n"

    def test_unindent_selection_restores(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text(f"{' ' * TAB_WIDTH}a = 1\n{' ' * TAB_WIDTH}b = 2\n")
        cursor = editor.textCursor()
        cursor.select(cursor.SelectionType.Document)
        editor.setTextCursor(cursor)
        editor.unindent_selection()
        assert editor.text() == "a = 1\nb = 2\n"

    def test_indent_without_selection_inserts_spaces(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("")
        editor.indent_selection()
        assert editor.text() == " " * TAB_WIDTH

    def test_smart_backspace_removes_full_step(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text(" " * TAB_WIDTH)
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert editor._smart_backspace() is True
        assert editor.text() == ""

    def test_smart_backspace_ignores_partial_indent(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text(" ")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert editor._smart_backspace() is False


class TestBrackets:
    def test_autoclose_parenthesis(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("")
        assert editor._autoclose_bracket("(") is True
        assert editor.text() == "()"
        assert editor.textCursor().position() == 1

    def test_skip_closing_character(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("()")
        cursor = editor.textCursor()
        cursor.setPosition(1)
        editor.setTextCursor(cursor)
        assert editor._skip_closing_char(")") is True
        assert editor.textCursor().position() == 2

    def test_wrap_selection(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("value")
        cursor = editor.textCursor()
        cursor.select(cursor.SelectionType.WordUnderCursor)
        editor.setTextCursor(cursor)
        assert editor._autoclose_bracket("[") is True
        assert editor.text() == "[value]"

    def test_quote_inside_word_is_plain(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("ab")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert editor._autoclose_bracket("'") is False

    def test_matching_bracket_helper(self) -> None:
        assert matching_bracket("f(a[b])", 2) == (1, 6, True)
        assert matching_bracket("f(a", 1) == (1, 1, False)
        assert matching_bracket("abc", 1) is None


class TestCommentToggle:
    def test_comment_and_uncomment_selection(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("a = 1\nb = 2\n")
        cursor = editor.textCursor()
        cursor.select(cursor.SelectionType.Document)
        editor.setTextCursor(cursor)
        editor.toggle_comment()
        assert editor.text() == "# a = 1\n# b = 2\n"
        cursor = editor.textCursor()
        cursor.select(cursor.SelectionType.Document)
        editor.setTextCursor(cursor)
        editor.toggle_comment()
        assert editor.text() == "a = 1\nb = 2\n"

    def test_comment_keeps_empty_lines(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("a = 1\n\nb = 2\n")
        cursor = editor.textCursor()
        cursor.select(cursor.SelectionType.Document)
        editor.setTextCursor(cursor)
        editor.toggle_comment()
        assert editor.text() == "# a = 1\n\n# b = 2\n"


class TestFindReplaceDialog:
    def test_resolve_target_from_editor(self, qapp) -> None:
        editor = CodeEditor()
        assert resolve_text_target(editor) is editor

    def test_resolve_target_rejects_unknown(self, qapp) -> None:
        with pytest.raises(TypeError):
            resolve_text_target(object())

    def test_find_next_and_status(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("alpha beta alpha\n")
        dialog = FindReplaceDialog(editor, translator=ru_translator())
        dialog.set_find_text("alpha")
        assert dialog.find_next() is True
        assert "Найдено" in dialog.result_text()
        assert dialog.find_next() is True
        assert dialog.find_next() is True  # поиск зацикливается на начало

    def test_find_reports_missing(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("alpha\n")
        dialog = FindReplaceDialog(editor, translator=ru_translator())
        dialog.set_find_text("омега")
        assert dialog.find_next() is False
        assert "Не найдено" in dialog.result_text()

    def test_find_empty_query(self, qapp) -> None:
        editor = CodeEditor()
        dialog = FindReplaceDialog(editor)
        assert dialog.find_next() is False

    def test_case_sensitivity_flag(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("Value value\n")
        dialog = FindReplaceDialog(editor)
        dialog.set_find_text("value")
        dialog.case_box.setChecked(True)
        assert dialog.find_next() is True
        assert editor.textCursor().selectedText() == "value"

    def test_replace_all_counts(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("x = 1\nx = 2\nx = 3\n")
        dialog = FindReplaceDialog(editor)
        dialog.set_find_text("x")
        dialog.replace_edit.setText("y")
        assert dialog.replace_all() == 3
        assert editor.text() == "y = 1\ny = 2\ny = 3\n"

    def test_replace_current(self, qapp) -> None:
        editor = CodeEditor()
        editor.set_text("value = 1\n")
        dialog = FindReplaceDialog(editor)
        dialog.set_find_text("value")
        dialog.replace_edit.setText("total")
        assert dialog.find_next() is True
        assert dialog.replace_current() is True
        assert "total" in editor.text()

    def test_translator_is_used(self, qapp) -> None:
        editor = CodeEditor()
        dialog = FindReplaceDialog(editor, translator=lambda key, **kw: f"<{key}>")
        dialog.retranslate()
        assert dialog.find_next_button.text() == "<script.find.next>"


class TestScriptSourceFilter:
    def _populated_model(self, qapp, root: Path) -> tuple:
        """Собрать модель файловой системы, реально наполнившую каталог.

        ``QFileSystemModel`` загружает содержимое лениво и только когда его
        кто-то запрашивает, поэтому в тесте создаётся настоящее представление и
        прокручивается цикл событий: иначе количество строк всегда равно нулю.

        Args:
            qapp: Приложение Qt (нужно для прокрутки событий).
            root: Каталог проекта.

        Returns:
            Пару «модель, представление» (представление нужно держать живым).
        """
        import time as _time

        from PySide6.QtCore import QDir
        from PySide6.QtWidgets import QFileSystemModel, QTreeView

        model = QFileSystemModel()
        model.setFilter(
            QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot
        )
        view = QTreeView()
        view.setModel(model)
        view.resize(400, 300)
        view.show()
        view.setRootIndex(model.setRootPath(str(root)))
        for _ in range(30):
            qapp.processEvents()
            _time.sleep(0.005)
        return model, view

    def test_only_python_files_pass(self, qapp, tmp_path: Path) -> None:
        make_project(tmp_path)
        model, view = self._populated_model(qapp, tmp_path)
        proxy = ScriptSourceFilter()
        proxy.setSourceModel(model)
        root_index = model.index(str(tmp_path))
        accepted = {
            model.fileName(model.index(row, 0, root_index)): proxy.filterAcceptsRow(
                row, root_index
            )
            for row in range(model.rowCount(root_index))
        }
        view.close()
        assert accepted.get("Assets") is True
        assert accepted.get("readme.md") is False

    def test_extension_matching_is_case_insensitive(self, qapp) -> None:
        proxy = ScriptSourceFilter()
        assert proxy.matches_name("spin.py") is True
        assert proxy.matches_name("SPIN.PY") is True
        assert proxy.matches_name("readme.md") is False
        assert proxy.matches_name("script") is False

    def test_service_names_are_hidden(self, qapp) -> None:
        proxy = ScriptSourceFilter()
        assert proxy.matches_name(".sleep.py") is False
        assert proxy.matches_name("__init__.py") is False
        proxy.set_hide_hidden(False)
        assert proxy.matches_name("__init__.py") is True

    def test_extensions_are_configurable(self, qapp) -> None:
        proxy = ScriptSourceFilter()
        assert proxy.extensions == (SCRIPT_SUFFIX,)
        proxy.set_extensions((".py", ".pyi"))
        assert proxy.extensions == (".py", ".pyi")
        assert proxy.matches_name("stub.pyi") is True
        proxy.set_extensions((".PY",))
        assert proxy.extensions == (".py",)

    def test_missing_source_model_is_safe(self, qapp) -> None:
        proxy = ScriptSourceFilter()
        assert proxy.filterAcceptsRow(0, proxy.index(-1, -1)) is False


class TestScriptRunner:
    def test_run_once_executes_hooks(self, qapp, tmp_path: Path, caplog) -> None:
        from editor.context import EditorContext

        make_project(tmp_path)
        context = EditorContext(project_root=tmp_path, language="ru")
        runner = ScriptRunner(context)
        with caplog.at_level(logging.INFO, logger="engine.scripting"):
            assert runner.run_once(SPIN_SOURCE, "spin.py") is True
        assert len(runner.runtime) == 0  # разовый прогон не оставляет записей

    def test_run_once_reports_broken_source(self, qapp, tmp_path: Path) -> None:
        from editor.context import EditorContext

        context = EditorContext(project_root=tmp_path, language="ru")
        runner = ScriptRunner(context)
        assert runner.run_once("def broken(:\n", "bad.py") is False

    def test_tick_updates_scene_components(self, qapp, tmp_path: Path) -> None:
        from editor.context import EditorContext

        scripts = make_project(tmp_path)
        context = EditorContext(project_root=tmp_path, language="ru")
        runner = ScriptRunner(context)
        # Объект с мешем нужен, чтобы трансформ участвовал в кадре сцены.
        cube = context.create_object("Cube", mesh="cube")
        cube.add_component(
            ScriptComponent(scripts / "spin.py", runtime=runner.runtime)
        )
        runner.start()
        runner.tick(0.1)
        assert cube.transform.rotation.w < 1.0
        assert len(runner.components()) == 1

    def test_components_skips_inactive(self, qapp, tmp_path: Path) -> None:
        from editor.context import EditorContext

        scripts = make_project(tmp_path)
        context = EditorContext(project_root=tmp_path, language="ru")
        runner = ScriptRunner(context)
        game_object = context.create_object("Empty")
        game_object.add_component(
            ScriptComponent(scripts / "spin.py", runtime=runner.runtime)
        )
        assert len(runner.components()) == 1
        game_object.active_self = False
        assert runner.components() == []

    def test_stop_unloads_everything(self, qapp, tmp_path: Path) -> None:
        from editor.context import EditorContext

        scripts = make_project(tmp_path)
        context = EditorContext(project_root=tmp_path, language="ru")
        runner = ScriptRunner(context)
        first = context.create_object("A")
        second = context.create_object("B")
        first.add_component(ScriptComponent(scripts / "spin.py", runtime=runner.runtime))
        second.add_component(ScriptComponent(scripts / "spin.py", runtime=runner.runtime))
        runner.start()
        assert len(runner.runtime) == 2
        assert runner.stop() == 2
        assert len(runner.runtime) == 0
        assert runner.running is False

    def test_auto_reload_applies_to_scene_component(self, qapp, tmp_path: Path) -> None:
        from editor.context import EditorContext

        scripts = make_project(tmp_path)
        path = scripts / "spin.py"
        context = EditorContext(project_root=tmp_path, language="ru")
        runner = ScriptRunner(context)
        game_object = context.create_object("Cube")
        component = game_object.add_component(
            ScriptComponent(path, runtime=runner.runtime)
        )
        path.write_text(
            "def on_update(self, delta_time):\n"
            "    self.transform.position = self.transform.position + "
            "Vector3(0, delta_time, 0)\n",
            encoding="utf-8",
        )
        future = time.time() + 5.0
        os.utime(path, (future, future))
        reloaded = runner.refresh_changed()
        assert [handle.name for handle in reloaded] == ["spin.py"]
        assert component.handle.revision == 2
        runner.start()
        before = game_object.transform.position
        runner.tick(0.5)
        assert game_object.transform.position.y - before.y == pytest.approx(0.5)


class TestScriptEditorPanel:
    def test_panel_is_a_separate_window(self, panel: ScriptEditorPanel) -> None:
        """Редактор скриптов — самостоятельное окно, а не док-панель."""
        from PySide6.QtWidgets import QDockWidget, QMainWindow

        assert isinstance(panel, QMainWindow)
        assert not isinstance(panel, QDockWidget)
        # Окно верхнего уровня: у него нет родителя-контейнера.
        assert panel.isWindow() is True
        assert panel.windowTitle() == "Редактор скриптов"
        assert panel.centralWidget() is not None

    def test_window_is_not_docked_in_main_window(self, window) -> None:
        """Окно не добавляется в доки главного окна."""
        from PySide6.QtWidgets import QDockWidget

        docks = window.findChildren(QDockWidget)
        assert window.script_editor not in docks
        assert all(
            dock is not window.script_editor for dock in window.children()
            if isinstance(dock, QDockWidget)
        )

    def test_window_shares_context_with_main_window(self, window) -> None:
        """Взаимодействие с главным окном идёт через общий контекст."""
        assert window.script_editor.context is window.context

    def test_show_and_hide_window(self, window) -> None:
        window.show()
        try:
            window.set_script_panel_visible(True)
            assert window.script_editor.isVisible() is True
            assert window.action_script_panel.isChecked() is True
            window.set_script_panel_visible(False)
            assert window.script_editor.isVisible() is False
        finally:
            window.hide()

    def test_default_window_size_is_usable(self, panel: ScriptEditorPanel) -> None:
        from editor.panels.script_editor import DEFAULT_WINDOW_SIZE

        assert (panel.width(), panel.height()) == DEFAULT_WINDOW_SIZE

    def test_window_is_placed_next_to_main(self, window) -> None:
        """Окно открывается со сдвигом от главного, а не поверх него."""
        window.show()  # при первом показе окно вписывается в экран
        window.resize(900, 700)
        window.move(120, 120)
        window.script_editor.move(0, 0)
        try:
            window.set_script_panel_visible(True)
            position = window.script_editor.pos()
            assert position.x() >= 120
            assert position.y() >= 120
        finally:
            window.set_script_panel_visible(False)
            window.hide()

    def test_has_toolbar_actions(self, panel: ScriptEditorPanel) -> None:
        names = [action.text() for action in panel.toolbar.actions()]
        assert "Новый скрипт" in names
        assert "Сохранить всё" in names
        assert "Запустить скрипты" in names
        assert "Остановить скрипты" in names

    def test_shortcuts_are_registered(self, panel: ScriptEditorPanel) -> None:
        shortcuts = {
            panel.action_save.shortcut().toString(),
            panel.action_new.shortcut().toString(),
            panel.action_run.shortcut().toString(),
            panel.action_stop.shortcut().toString(),
        }
        assert {"Ctrl+S", "Ctrl+N", "F5", "Shift+F5"} <= shortcuts

    def test_open_script_creates_tab(self, panel: ScriptEditorPanel, tmp_path: Path) -> None:
        editor = panel.open_script(tmp_path / "Assets" / "Scripts" / "spin.py")
        assert panel.tabs.count() == 1
        assert panel.tabs.tabText(0) == "spin.py"
        assert editor.text().startswith('"""Вращение."""')

    def test_open_same_script_twice_reuses_tab(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        path = tmp_path / "Assets" / "Scripts" / "spin.py"
        first = panel.open_script(path)
        second = panel.open_script(path)
        assert first is second
        assert panel.tabs.count() == 1

    def test_open_missing_script_falls_back_to_new(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        editor = panel.open_script(tmp_path / "нет.py")
        assert editor is not None
        assert panel.tabs.count() == 1

    def test_new_script_is_marked_modified(self, panel: ScriptEditorPanel) -> None:
        panel.new_script("fresh.py")
        assert panel.tabs.tabText(0).startswith(UNSAVED_MARK)
        editor = panel.current_editor()
        assert editor is not None
        assert "def on_update" in editor.text()

    def test_new_script_action_does_not_touch_disk(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        """Новый скрипт живёт во вкладке и не пишется на диск до Ctrl+S."""
        folder = tmp_path / "Assets" / "Scripts"
        before = sorted(p.name for p in folder.iterdir())
        panel.action_new.trigger()
        assert panel.tabs.count() == 1
        editor = panel.current_editor()
        assert editor is not None
        assert editor.is_modified() is True
        assert panel.open_paths() == []
        assert panel.tabs.tabText(0).endswith("new_script.py")
        assert sorted(p.name for p in folder.iterdir()) == before

    def test_new_script_names_do_not_collide(self, panel: ScriptEditorPanel) -> None:
        panel.new_script()
        panel.new_script()
        names = [panel.tabs.tabText(i).lstrip(UNSAVED_MARK + " ") for i in range(2)]
        assert names == ["new_script.py", "new_script_2.py"]

    def test_new_script_skips_names_on_disk(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        folder = tmp_path / "Assets" / "Scripts"
        (folder / "new_script.py").write_text("x = 1\n", encoding="utf-8")
        panel.new_script()
        assert panel.tabs.tabText(0).lstrip(UNSAVED_MARK + " ") == "new_script_2.py"
        assert (folder / "new_script.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_unique_script_name_accepts_none(self, panel: ScriptEditorPanel) -> None:
        assert panel.unique_script_name(None) == "new_script.py"

    def test_save_new_script_writes_to_chosen_path(
        self, panel: ScriptEditorPanel, tmp_path: Path, monkeypatch
    ) -> None:
        sub = tmp_path / "Assets" / "Scripts" / "ai"
        sub.mkdir()
        target = sub / "brain.py"
        monkeypatch.setattr(panel, "_ask_new_path", lambda _suggested: target)
        panel.new_script()
        assert panel.save_current() is True
        assert target.is_file()
        assert panel.open_paths() == [target]

    def test_discarding_untitled_script_leaves_disk_empty(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        folder = tmp_path / "Assets" / "Scripts"
        before = sorted(p.name for p in folder.iterdir())
        panel.new_script()
        # Фикстура отвечает на вопрос «Не сохранять».
        assert panel.confirm_close() is True
        assert sorted(p.name for p in folder.iterdir()) == before

    def test_tree_root_is_scripts_folder(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        assert panel.scripts_root() == tmp_path / "Assets" / "Scripts"
        root = panel.proxy.mapToSource(panel.tree.rootIndex())
        assert Path(panel.file_model.filePath(root)) == panel.scripts_root()

    def test_new_folder_creates_nested_directory(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        created = panel.new_folder("enemies")
        assert created == tmp_path / "Assets" / "Scripts" / "enemies"
        assert created.is_dir()
        assert panel.new_folder("enemies") is None

    def test_new_folder_is_created_in_selected_folder(
        self, panel: ScriptEditorPanel, tmp_path: Path, monkeypatch
    ) -> None:
        parent = tmp_path / "Assets" / "Scripts" / "ai"
        parent.mkdir()
        monkeypatch.setattr(panel, "selected_folder", lambda: parent)
        created = panel.new_folder("deep")
        assert created == parent / "deep"

    @pytest.mark.parametrize("name", ["", "  ", "..", "a/b", "a\\b", "x:y", "bad?", "dot."])
    def test_new_folder_rejects_bad_names(
        self, panel: ScriptEditorPanel, tmp_path: Path, name: str
    ) -> None:
        folder = tmp_path / "Assets" / "Scripts"
        before = sorted(p.name for p in folder.iterdir())
        assert panel.new_folder(name) is None
        assert sorted(p.name for p in folder.iterdir()) == before

    def test_close_asks_about_unsaved(
        self, panel: ScriptEditorPanel, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        editor = panel.new_script("draft.py")
        assert editor.is_modified()
        monkeypatch.setattr(
            QMessageBox,
            "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
        )
        assert panel.confirm_close() is False
        monkeypatch.setattr(
            QMessageBox,
            "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Discard),
        )
        assert panel.confirm_close() is True

    def test_save_writes_file(self, panel: ScriptEditorPanel, tmp_path: Path) -> None:
        editor = panel.open_script(tmp_path / "Assets" / "Scripts" / "spin.py")
        editor.set_text("VALUE = 42\n")
        assert panel.save_editor(editor) is True
        assert (tmp_path / "Assets" / "Scripts" / "spin.py").read_text(encoding="utf-8") == (
            "VALUE = 42\n"
        )
        assert UNSAVED_MARK not in panel.tabs.tabText(0)

    def test_save_all_counts_changed_tabs(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        first = panel.open_script(tmp_path / "Assets" / "Scripts" / "spin.py")
        second = panel.new_script("second.py")
        panel._paths[id(second)] = tmp_path / "Assets" / "Scripts" / "second.py"
        # Правки вносим как пользователь: set_text подставляет текст
        # программно и намеренно сохраняет прежний флаг изменения.
        first.insertPlainText("# правка\n")
        second.insertPlainText("# ещё правка\n")
        assert first.is_modified() is True
        assert panel.save_all() == 2
        assert (tmp_path / "Assets" / "Scripts" / "second.py").is_file()

    def test_set_text_keeps_modified_state(self, panel: ScriptEditorPanel) -> None:
        editor = panel.new_script("fresh.py")
        assert editor.is_modified() is True
        editor.set_text("VALUE = 1\n")
        assert editor.is_modified() is True
        editor.mark_saved()
        editor.set_text("VALUE = 2\n")
        assert editor.is_modified() is False

    def test_run_current_reports_success(self, panel: ScriptEditorPanel) -> None:
        panel.new_script("run.py")
        assert panel.run_current() is True
        assert panel.is_running() is True
        assert panel.action_run.isEnabled() is False
        assert panel.action_stop.isEnabled() is True

    def test_attached_script_is_not_run_twice(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        path = tmp_path / "Assets" / "Scripts" / "spin.py"
        panel.open_script(path)
        panel.context.create_object("Cube").add_component(
            ScriptComponent(path, runtime=panel.runner.runtime)
        )
        assert panel.runner.is_attached(path) is True
        assert panel.runner.is_attached(tmp_path / "Assets" / "Scripts" / "other.py") is False
        assert panel.runner.is_attached(None) is False

        messages: list[str] = []
        panel.status_message.connect(messages.append)
        assert panel.run_current() is True
        assert any("уже работает" in message for message in messages)

    def test_run_once_detaches_inline_script(self, panel: ScriptEditorPanel) -> None:
        panel.new_script("once.py")
        assert panel.run_current() is True
        # Разовый прогон не оставляет записей в реестре.
        assert len(panel.runner.runtime) == 0

    def test_run_console_receives_output(self, panel: ScriptEditorPanel) -> None:
        editor = panel.new_script("run.py")
        editor.set_text(
            'def on_start(self):\n    print("строка из скрипта")\n'
        )
        assert panel.run_current() is True
        assert "строка из скрипта" in panel.console.toPlainText()

    def test_run_current_reports_syntax_error(self, panel: ScriptEditorPanel) -> None:
        editor = panel.new_script("bad.py")
        editor.set_text("def broken(:\n")
        assert panel.run_current() is False
        assert "синтаксическая" in panel.console.toPlainText()

    def test_stop_resets_actions(self, panel: ScriptEditorPanel) -> None:
        panel.new_script("run.py")
        panel.run_current()
        assert panel.stop_scripts() >= 0
        assert panel.is_running() is False
        assert panel.action_run.isEnabled() is True
        assert panel.action_stop.isEnabled() is False

    def test_clear_console(self, panel: ScriptEditorPanel) -> None:
        panel.append_console("сообщение")
        assert "сообщение" in panel.console.toPlainText()
        panel.clear_console()
        assert panel.console.toPlainText() == ""

    def test_close_tab_without_changes(self, panel: ScriptEditorPanel, tmp_path: Path) -> None:
        panel.open_script(tmp_path / "Assets" / "Scripts" / "spin.py")
        assert panel.close_current() is True
        assert panel.tabs.count() == 0

    def test_close_last_tab_is_safe(self, panel: ScriptEditorPanel) -> None:
        assert panel.close_current() is False

    def test_reload_without_running_script(self, panel: ScriptEditorPanel) -> None:
        panel.new_script("fresh.py")
        assert panel.reload_current() is False

    def test_reload_all(self, panel: ScriptEditorPanel) -> None:
        assert panel.reload_all() == 0

    def test_editors_and_paths(self, panel: ScriptEditorPanel, tmp_path: Path) -> None:
        path = tmp_path / "Assets" / "Scripts" / "spin.py"
        panel.open_script(path)
        panel.new_script("fresh.py")
        assert len(panel.editors()) == 2
        assert panel.open_paths() == [path]
        assert panel.current_script_path() is None  # активна новая вкладка

    def test_find_dialog_targets_current_tab(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        editor = panel.open_script(tmp_path / "Assets" / "Scripts" / "spin.py")
        dialog = panel.show_find_dialog()
        assert dialog.editor is editor

    def test_find_next_without_query_opens_dialog(
        self, panel: ScriptEditorPanel, tmp_path: Path
    ) -> None:
        panel.open_script(tmp_path / "Assets" / "Scripts" / "spin.py")
        assert panel.find_next() is False

    def test_find_next_finds_text(self, panel: ScriptEditorPanel, tmp_path: Path) -> None:
        panel.open_script(tmp_path / "Assets" / "Scripts" / "spin.py")
        panel.find_dialog.set_find_text("on_update")
        assert panel.find_next() is True

    def test_cursor_status_message(self, panel: ScriptEditorPanel) -> None:
        messages: list[str] = []
        panel.status_message.connect(messages.append)
        editor = panel.new_script("pos.py")
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        editor.setTextCursor(cursor)
        assert any("Строка" in message for message in messages)

    def test_retranslate_to_english(self, panel: ScriptEditorPanel) -> None:
        panel.context.set_language("en")
        assert panel.windowTitle() == "Script editor"
        assert panel.action_run.text() == "Run scripts"
        assert panel.tree_label.text().startswith("Project files")
        panel.context.set_language("ru")
        assert panel.action_run.text() == "Запустить скрипты"

    def test_project_root_missing_is_reported(
        self, qapp, tmp_path: Path
    ) -> None:
        from editor.context import EditorContext

        context = EditorContext(project_root=tmp_path, language="ru")
        widget = ScriptEditorPanel(context)
        messages: list[str] = []
        widget.status_message.connect(messages.append)
        widget.set_project_root(tmp_path / "нет-такого")
        assert any("не найден" in message for message in messages)
        widget.close()

    def test_repr_is_informative(self, panel: ScriptEditorPanel) -> None:
        panel.new_script("x.py")
        assert "вкладок=1" in repr(panel)


class TestMainWindowIntegration:
    def test_script_panel_is_created(self, window) -> None:
        assert window.script_editor is not None
        assert isinstance(window.script_editor, ScriptEditorPanel)

    def test_script_menu_present(self, window) -> None:
        titles = [menu.title() for menu in window.menuBar().findChildren(type(window.menu_script))]
        assert any("Скрипты" in title for title in titles)

    def test_toggle_action_shows_panel(self, window) -> None:
        window.show()
        try:
            window.set_script_panel_visible(True)
            assert window.script_editor.isVisible() is True
            assert window.action_script_panel.isChecked() is True
            window.set_script_panel_visible(False)
            assert window.script_editor.isVisible() is False
        finally:
            window.hide()

    def test_attach_action_is_replaced_by_inspector_section(self, window) -> None:
        """Команды «прикрепить скрипт» больше нет: скрипты живут в инспекторе."""
        assert not hasattr(window, "action_attach_script")
        assert not hasattr(window, "attach_script_to_selected")
        texts = [action.text() for action in window.menu_script.actions()]
        assert not any("рикрепить" in text for text in texts)

    def test_script_runtime_is_shared_with_window(self, window) -> None:
        """Редактор скриптов тикает тот же реестр, куда пишет инспектор."""
        assert window.script_editor.runner.runtime is window.context.script_runtime

    def test_panel_logs_reach_editor_console(self, window, tmp_path: Path) -> None:
        scripts = make_project(tmp_path)
        window.script_editor.open_script(scripts / "spin.py")
        window.script_editor.run_current()
        assert any(
            "печать из скрипта" in text for text, _level in window.console.history
        )
        window.script_editor.stop_scripts()

    def test_reset_layout_hides_script_panel(self, window) -> None:
        window.set_script_panel_visible(True)
        window.reset_layout()
        assert window.script_editor.isVisible() is False
        assert window.action_script_panel.isChecked() is False


class TestScriptsInInspector:
    """Прикрепление скриптов к объекту через раздел «Скрипты» инспектора."""

    def _attach(self, window, tmp_path: Path, monkeypatch, name: str = "spin.py"):
        """Прикрепить скрипт к выделенному объекту, минуя диалог выбора файла."""
        scripts = make_project(tmp_path)
        window.context.set_project_root(tmp_path)
        monkeypatch.setattr(
            window.inspector, "_pick_script_path", lambda: scripts / name
        )
        window.inspector._add_script()
        return window.context.selection.primary

    def test_add_script_attaches_component(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        cube = self._attach(window, tmp_path, monkeypatch)
        components = cube.get_components(ScriptComponent)
        assert len(components) == 1
        component = components[0]
        assert component.name == "spin.py"
        assert component.path == tmp_path / "Assets" / "Scripts" / "spin.py"
        # Скрипт получает объект как базовый и запускается на старте.
        assert component.game_object is cube
        assert component.auto_start is True
        assert component.is_loaded is True

    def test_attached_script_uses_shared_runtime(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        cube = self._attach(window, tmp_path, monkeypatch)
        component = cube.get_component(ScriptComponent)
        assert component.runtime is window.context.script_runtime
        assert component.handle is not None

    def test_add_script_without_selection_is_safe(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        window.context.selection.clear()
        monkeypatch.setattr(window.inspector, "_pick_script_path", lambda: None)
        window.inspector._add_script()  # не должно бросить исключение
        assert window.context.selection.is_empty

    def test_cancelled_dialog_adds_nothing(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        cube = window.context.selection.primary
        monkeypatch.setattr(window.inspector, "_pick_script_path", lambda: None)
        window.inspector._add_script()
        assert cube.get_components(ScriptComponent) == []

    def test_auto_start_checkbox_toggles_flag(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        cube = self._attach(window, tmp_path, monkeypatch)
        component = cube.get_component(ScriptComponent)
        window.inspector._apply_auto_start(component, False)
        assert component.auto_start is False
        window.inspector._apply_auto_start(component, True)
        assert component.auto_start is True

    def test_remove_script_detaches_and_unloads(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        cube = self._attach(window, tmp_path, monkeypatch)
        component = cube.get_component(ScriptComponent)
        handle = component.handle
        assert handle is not None

        window.inspector._remove_script(component)

        assert cube.get_components(ScriptComponent) == []
        assert component.handle is None
        assert handle.token not in window.context.script_runtime.scripts

    def test_script_section_is_rebuilt_on_selection(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QLabel, QPushButton

        cube = self._attach(window, tmp_path, monkeypatch)
        labels = [label.text() for label in window.inspector.findChildren(QLabel)]
        assert "Скрипты" in labels
        assert "spin.py" in labels
        buttons = [button.text() for button in window.inspector.findChildren(QPushButton)]
        assert "Добавить скрипт…" in buttons
        assert "Убрать" in buttons
        assert window.inspector._scripts()[0].game_object is cube

    def test_scripts_section_hint_without_scripts(self, window) -> None:
        from PySide6.QtWidgets import QLabel

        labels = [label.text() for label in window.inspector.findChildren(QLabel)]
        assert "К объекту не прикреплено ни одного скрипта" in labels

    def test_scripts_are_saved_with_scene(
        self, window, tmp_path: Path, monkeypatch
    ) -> None:
        from editor import scene_io

        cube = self._attach(window, tmp_path, monkeypatch)
        window.inspector._apply_auto_start(cube.get_component(ScriptComponent), False)

        data = scene_io.scene_to_dict(window.context.scene)
        entry = next(item for item in data["objects"] if item["name"] == cube.name)
        payload = next(
            item
            for item in entry["components"]
            if item["type"] == "ScriptComponent"
        )
        assert payload["path"].endswith("spin.py")
        assert payload["auto_start"] is False

    def test_scripts_are_restored_from_scene(self, window, tmp_path: Path) -> None:
        from editor import scene_io

        scripts = make_project(tmp_path)
        cube = window.context.selection.primary
        cube.add_component(
            ScriptComponent(scripts / "spin.py", auto_start=False)
        )
        data = scene_io.scene_to_dict(window.context.scene)
        restored = scene_io.scene_from_dict(data)

        restored_cube = next(
            item for item in restored.objects if item.name == cube.name
        )
        component = restored_cube.get_component(ScriptComponent)
        assert component is not None
        assert component.path == scripts / "spin.py"
        assert component.auto_start is False
