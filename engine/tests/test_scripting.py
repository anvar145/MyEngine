"""Тесты слоя скриптинга: изоляция пространств имён, песочница, перезагрузка.

Тесты не требуют Qt для основной части: ядро скриптинга headless-совместимо,
поэтому проверяются и загрузка модулей, и перехват исключений, и жизненный цикл
компонента. Подсветка синтаксиса, которой нужен PySide6, помечена отдельно.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from engine.core.components import MeshRenderer
from engine.core.math3d import Vector3
from engine.core.scene import Scene
from engine.scripting import (
    ALLOWED_MODULES,
    ALLOWED_ROOTS,
    API_NAMES,
    ScriptComponent,
    ScriptError,
    ScriptRuntime,
    create_builtins,
    format_traceback,
    load_module,
    safe_import,
)

#: Исходник скрипта-образца: проверяет все три хука и вывод в лог.
SAMPLE = '''"""Образец."""

from engine.core.math3d import Vector3

SPEED = 90.0
CALLS = 0


def on_start(self):
    self.log("старт")
    print("печать из скрипта")


def on_update(self, delta_time):
    global CALLS
    CALLS += 1
    self.transform.rotate(Vector3.up(), SPEED * delta_time)


def on_destroy(self):
    self.log("финиш")
'''


def write_script(directory: Path, name: str, source: str) -> Path:
    """Записать скрипт в каталог и вернуть путь.

    Args:
        directory: Каталог для файла.
        name: Имя файла.
        source: Исходный текст.

    Returns:
        Путь к записанному файлу.
    """
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


def touch_future(path: Path) -> None:
    """Сделать файл «новее», чем он был при загрузке.

    Метка времени файловой системы на Windows имеет разрешение порядка
    10 мс, поэтому без сдвига тест горячей перезагрузки был бы неустойчив.

    Args:
        path: Путь к файлу.
    """
    future = time.time() + 5.0
    os.utime(path, (future, future))


class TestModuleLoading:
    def test_load_module_returns_isolated_module(self) -> None:
        module = load_module("demo", "VALUE = 41 + 1\n")
        assert module.VALUE == 42
        assert module.__name__ == "demo"

    def test_namespaces_are_independent(self) -> None:
        first = load_module("iso_a", "VALUE = 1\n")
        second = load_module("iso_b", "VALUE = 2\n")
        assert (first.VALUE, second.VALUE) == (1, 2)
        assert first.__dict__ is not second.__dict__

    def test_module_functions_see_module_globals(self) -> None:
        module = load_module("counter", "COUNT = 0\n\n\ndef bump():\n    global COUNT\n    COUNT += 1\n")
        module.bump()
        module.bump()
        assert module.COUNT == 2

    def test_previous_state_survives_reload(self) -> None:
        before = load_module("stateful", "COUNTER = 7\n").__dict__
        after = load_module("stateful", "OTHER = 1\n", previous=before)
        assert after.COUNTER == 7
        assert after.OTHER == 1

    def test_previous_state_is_replaced_by_new_code(self) -> None:
        before = load_module("stateful", "COUNTER = 7\n").__dict__
        after = load_module("stateful", "COUNTER = 9\n", previous=before)
        assert after.COUNTER == 9

    def test_syntax_error_is_wrapped(self) -> None:
        with pytest.raises(ScriptError) as info:
            load_module("broken", "def f(self)\n    pass\n", "broken.py")
        assert info.value.syntax is True
        assert "синтаксическая ошибка" in str(info.value)
        assert "broken.py" in info.value.text

    def test_top_level_exception_is_wrapped(self) -> None:
        with pytest.raises(ScriptError) as info:
            load_module("boom", "raise ValueError('нарочно')\n", "boom.py")
        assert info.value.syntax is False
        assert "ValueError" in str(info.value)

    def test_traceback_hides_loader_frames(self) -> None:
        try:
            load_module("boom", "raise ValueError('нарочно')\n", "boom.py")
        except ScriptError as exc:
            assert "runtime.py" not in exc.text
            assert "boom.py" in exc.text
            assert "ValueError: нарочно" in exc.text


class TestSandbox:
    def test_standard_modules_are_allowed(self) -> None:
        module = load_module("good", "import math\nimport random\nPI = math.pi\n")
        assert module.PI == pytest.approx(3.14159, rel=1e-4)  # noqa: SIM300 - явно видно допуск
        assert module.random is not None

    def test_preloaded_modules_are_available_by_name(self) -> None:
        module = load_module("pre", "VALUE = math.floor(2.7)\n")
        assert module.VALUE == 2

    def test_engine_imports_are_allowed(self) -> None:
        module = load_module(
            "engine_user", "from engine.core.math3d import Vector3\nUP = Vector3.up()\n"
        )
        assert module.UP.almost_equals(Vector3(0, 1, 0), 1e-6)

    def test_future_import_is_allowed(self) -> None:
        """Привычная шапка ``from __future__ import annotations`` не запрещена."""
        module = load_module(
            "future_user",
            "from __future__ import annotations\n\n\ndef f(x: int) -> int:\n    return x\n",
        )
        assert module.f(3) == 3

    def test_demo_script_loads(self) -> None:
        """Демонстрационный скрипт проекта должен загружаться без ошибок."""
        demo = Path(__file__).resolve().parents[2] / "scripts" / "spin_demo.py"
        module = load_module("spin_demo", demo.read_text(encoding="utf-8"), str(demo))
        assert callable(module.on_start)
        assert callable(module.on_update)
        assert callable(module.on_destroy)

    def test_forbidden_module_raises(self) -> None:
        with pytest.raises(ScriptError) as info:
            load_module("evil", "import os\n")
        assert "os" in str(info.value)

    def test_forbidden_submodule_raises(self) -> None:
        with pytest.raises(ScriptError):
            load_module("evil2", "import subprocess\n")

    def test_dangerous_builtins_are_removed(self) -> None:
        namespace = create_builtins()
        for name in ("eval", "exec", "open", "compile", "__import__"):
            assert name not in namespace or name == "__import__"
        assert "__import__" in namespace

    def test_print_is_redirected_to_log(self, caplog) -> None:
        with caplog.at_level("INFO", logger="engine.scripting.print"):
            module = load_module("printer", "print('привет', 'мир')\n")
        assert module is not None
        assert any("привет мир" in record.message for record in caplog.records)

    def test_safe_import_rejects_relative(self) -> None:
        with pytest.raises(ImportError):
            safe_import("math", level=1)

    def test_allowed_sets_are_consistent(self) -> None:
        assert "engine" in ALLOWED_ROOTS
        assert "os" not in ALLOWED_MODULES
        assert "math" in ALLOWED_MODULES


class TestFormatTraceback:
    def test_truncates_long_tracebacks(self) -> None:
        def deep(level: int) -> None:
            if level:
                deep(level - 1)
            else:
                raise RuntimeError("дно")

        try:
            deep(30)
        except RuntimeError as exc:
            text = format_traceback(exc, limit=5)
        assert "скрыто" in text
        assert len(text.splitlines()) <= 6


class TestScriptRuntime:
    def test_load_and_call_lifecycle(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        assert handle.name == "spin.py"
        assert len(runtime) == 1
        assert runtime.by_path(path) is handle

        # Хуки вызываются с реальным компонентом: скрипт обращается к
        # self.transform, поэтому объект должен быть привязан к сцене.
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        cube.add_component(MeshRenderer, mesh="cube")
        instance = cube.add_component(ScriptComponent(path, runtime=runtime))
        runtime.call(handle, "on_start", instance)
        for _ in range(3):
            runtime.call(handle, "on_update", instance, 0.1)
        assert handle.module.CALLS == 3
        assert handle.errors == 0

    def test_call_reports_missing_hook_as_none(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "bare.py", "VALUE = 1\n")
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        assert runtime.call(handle, "on_update", None, 0.1) is None
        assert handle.errors == 0

    def test_call_captures_exception_and_raises_count(self, tmp_path: Path, caplog) -> None:
        path = write_script(
            tmp_path, "boom.py", "def on_update(self, dt):\n    raise ValueError('крах')\n"
        )
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        with caplog.at_level("ERROR", logger="engine.scripting"):
            assert runtime.call(handle, "on_update", None, 0.1) is None
        assert handle.errors == 1
        assert any("крах" in record.getMessage() for record in caplog.records)
        assert any("[script]" in record.getMessage() for record in caplog.records)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        runtime = ScriptRuntime(project_root=tmp_path)
        with pytest.raises(ScriptError):
            runtime.load(tmp_path / "нет.py")

    def test_unregister_and_close(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        assert runtime.unregister(handle.token) is True
        assert runtime.unregister("нет-такого") is False
        runtime.load(path)
        runtime.close()
        assert len(runtime) == 0
        assert runtime.active_scripts == ()

    def test_needs_reload_detects_change(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        assert runtime.needs_reload(handle) is False
        touch_future(path)
        assert runtime.needs_reload(handle) is True

    def test_reload_keeps_token_and_bumps_revision(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        write_script(tmp_path, "spin.py", "MARK = 'v2'\n")
        touch_future(path)
        updated = runtime.reload(handle.token)
        assert updated.token == handle.token
        assert updated.revision == 2
        assert updated.module.MARK == "v2"
        assert runtime.by_path(path) is updated

    def test_refresh_reloads_only_changed(self, tmp_path: Path) -> None:
        first = write_script(tmp_path, "a.py", "MARK = 1\n")
        write_script(tmp_path, "b.py", "MARK = 2\n")
        runtime = ScriptRuntime(project_root=tmp_path)
        runtime.load(first)
        runtime.load(tmp_path / "b.py")
        assert runtime.pending_reloads() == []
        write_script(tmp_path, "a.py", "MARK = 11\n")
        touch_future(first)
        pending = runtime.pending_reloads()
        assert [handle.name for handle in pending] == ["a.py"]
        reloaded = runtime.refresh()
        assert [handle.name for handle in reloaded] == ["a.py"]
        assert reloaded[0].module.MARK == 11

    def test_refresh_calls_back_into_owner(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        runtime.load(path)
        seen: list[str] = []
        write_script(tmp_path, "spin.py", "MARK = 3\n")
        touch_future(path)
        runtime.refresh(
            on_destroy=lambda old: seen.append(f"destroy:{old.revision}"),
            on_reload=lambda new: seen.append(f"reload:{new.revision}"),
        )
        assert seen == ["destroy:1", "reload:2"]

    def test_reload_warns_about_leaks(self, tmp_path: Path, caplog) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        write_script(tmp_path, "spin.py", "MARK = 3\n")
        touch_future(path)
        with caplog.at_level("WARNING", logger="engine.scripting"):
            runtime.reload(handle.token)
        assert any("остаться в памяти" in record.getMessage() for record in caplog.records)

    def test_reload_reports_broken_file(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.load(path)
        write_script(tmp_path, "spin.py", "def broken(:\n")
        touch_future(path)
        with pytest.raises(ScriptError):
            runtime.reload(handle.token)

    def test_reload_unknown_token(self, tmp_path: Path) -> None:
        runtime = ScriptRuntime(project_root=tmp_path)
        with pytest.raises(KeyError):
            runtime.reload("нет-такого")

    def test_reload_inline_script_is_rejected(self, tmp_path: Path) -> None:
        runtime = ScriptRuntime(project_root=tmp_path)
        handle = runtime.register("inline", load_module("inline", "VALUE = 1\n"))
        with pytest.raises(ScriptError):
            runtime.reload(handle.token)

    def test_auto_reload_can_be_disabled(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path, auto_reload=False)
        runtime.load(path)
        touch_future(path)
        assert runtime.pending_reloads() == []
        assert runtime.refresh() == []

    def test_call_all_counts_active(self, tmp_path: Path) -> None:
        first = write_script(tmp_path, "a.py", SAMPLE)
        second = write_script(tmp_path, "b.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        one = runtime.load(first)
        two = runtime.load(second)
        assert runtime.call_all("on_start", None) == 2
        two.active = False
        assert runtime.call_all("on_start", None) == 1
        one.active = False
        assert runtime.call_all("on_start", None) == 0

    def test_describe_and_repr(self, tmp_path: Path) -> None:
        runtime = ScriptRuntime(project_root=tmp_path)
        runtime.load(write_script(tmp_path, "spin.py", SAMPLE))
        assert "скриптов=1" in runtime.describe()
        assert "ScriptRuntime" in repr(runtime)


class TestScriptComponent:
    def test_requires_source_or_path(self) -> None:
        with pytest.raises(ValueError):
            ScriptComponent()

    def test_loads_on_attach_and_reports_state(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        assert component.is_loaded
        assert component.name == "spin.py"
        assert component.game_object is cube

    def test_lifecycle_hooks_reach_script(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        cube.add_component(MeshRenderer, mesh="cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        component.on_start()
        for _ in range(2):
            component.on_update(0.5)
        assert component.module.CALLS == 2
        assert component.delta_time == 0.5
        assert component.time.frame == 2
        assert component.time.elapsed == pytest.approx(1.0)

    def test_scene_update_drives_component(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        cube.add_component(MeshRenderer, mesh="cube")
        cube.add_component(ScriptComponent(path, runtime=runtime))
        scene.update(0.1)
        scene.update(0.1)
        assert cube.transform.rotation.w < 1.0  # куб повернулся

    def test_script_sees_engine_api(self, tmp_path: Path) -> None:
        source = (
            "def on_start(self):\n"
            "    self.log('имя', self.game_object.name)\n"
            "    self.log('сцена', self.scene.name)\n"
            "    self.log('своя', self.transform.position.y)\n"
            "    self.log('ввод', self.input.is_key_down('space'))\n"
            "    self.log('время', self.time.frame)\n"
        )
        path = write_script(tmp_path, "api.py", source)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("ApiScene")
        cube = scene.create_game_object("ApiCube", position=Vector3(0, 2, 0))
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        component.on_start()
        assert component.handle.errors == 0

    def test_input_state(self) -> None:
        component = ScriptComponent(source="VALUE = 1\n")
        assert component.input.is_key_down("space") is False
        component.input.press("Space")
        assert component.input.is_key_down("space") is True
        assert "space" in component.input
        component.input.release("SPACE")
        assert component.input.is_key_down("space") is False
        component.input.press("w")
        component.input.clear()
        assert component.input.pressed_keys == set()

    def test_unattached_component_has_no_transform(self) -> None:
        """Разовый запуск без объекта: ``self.transform`` равен ``None``."""
        component = ScriptComponent(source="VALUE = 1\n")
        assert component.game_object is None
        assert component.transform is None
        assert component.scene is None

    def test_standalone_script_without_object_runs(self, tmp_path: Path) -> None:
        """Скрипт, написанный с проверкой ``self.transform is None``, не падает."""
        path = write_script(
            tmp_path,
            "standalone.py",
            "def on_start(self):\n"
            "    self.log('нет объекта' if self.game_object is None else 'есть')\n"
            "\n"
            "def on_update(self, delta_time):\n"
            "    if self.transform is None:\n"
            "        return\n"
            "    raise AssertionError('сюда попадать не должны')\n",
        )
        runtime = ScriptRuntime(project_root=tmp_path)
        component = ScriptComponent(path, runtime=runtime)
        component.load()
        component.on_start()
        component.on_update(0.0)
        assert component.handle.errors == 0
        component.on_destroy()

    def test_demo_script_runs_standalone(self) -> None:
        """Демонстрационный скрипт должен переживать разовый запуск из редактора."""
        demo = Path(__file__).resolve().parents[2] / "scripts" / "spin_demo.py"
        runtime = ScriptRuntime()
        component = ScriptComponent(
            source=demo.read_text(encoding="utf-8"), name="spin_demo.py", runtime=runtime
        )
        component.load()
        assert component.is_loaded
        component.on_start()
        component.on_update(0.0)
        assert component.handle.errors == 0
        component.on_destroy()

    def test_namespace_publishing(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "ns.py", "def on_start(self):\n    pass\n")
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        namespace = component.module.__dict__
        assert namespace["Vector3"] is Vector3
        assert namespace["scene"] is scene
        assert namespace["game_object"] is cube
        assert namespace["input"] is component.input
        assert namespace["time"] is component.time

    def test_two_instances_get_separate_namespaces(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "shared.py", "STATE = []\n")
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        first = scene.create_game_object("A").add_component(
            ScriptComponent(path, runtime=runtime)
        )
        second = scene.create_game_object("B").add_component(
            ScriptComponent(path, runtime=runtime)
        )
        first.module.STATE.append("a")
        assert second.module.STATE == []
        assert len(runtime) == 2

    def test_reload_restarts_script(self, tmp_path: Path, caplog) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        cube.add_component(MeshRenderer, mesh="cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        component.on_start()

        write_script(
            tmp_path,
            "spin.py",
            "def on_update(self, delta_time):\n"
            "    self.transform.position = self.transform.position + "
            "Vector3(0, delta_time, 0)\n",
        )
        touch_future(path)
        with caplog.at_level("WARNING", logger="engine.scripting"):
            reloaded = runtime.refresh(on_reload=lambda handle: component.adopt(handle))
        assert [handle.name for handle in reloaded] == ["spin.py"]
        assert component.handle.revision == 2
        before = cube.transform.position
        component.on_update(0.5)
        assert cube.transform.position.y - before.y == pytest.approx(0.5)

    def test_reset_restarts_from_scratch(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        component.on_start()
        component.reset()
        assert component.is_loaded

    def test_failed_load_is_reported_not_raised(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "bad.py", "def broken(:\n")
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        component = scene.create_game_object("Cube").add_component(
            ScriptComponent(path, runtime=runtime)
        )
        assert component.is_loaded is False
        assert "синтаксическая" in component.last_error

    def test_strict_mode_raises(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "bad.py", "def broken(:\n")
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        with pytest.raises(ScriptError):
            scene.create_game_object("Cube").add_component(
                ScriptComponent(path, runtime=runtime, strict=True)
            )

    def test_destroy_calls_on_destroy_and_unregisters(self, tmp_path: Path, caplog) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        cube.add_component(ScriptComponent(path, runtime=runtime))
        assert len(runtime) == 1
        with caplog.at_level("INFO", logger="engine.scripting.component"):
            cube.destroy(immediate=True)
        assert len(runtime) == 0
        assert any("финиш" in record.getMessage() for record in caplog.records)

    def test_remove_component_unloads(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        cube = Scene("Test").create_game_object("Cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        assert cube.remove_component(component) is True
        assert len(runtime) == 0

    def test_error_in_update_does_not_break_scene(self, tmp_path: Path) -> None:
        path = write_script(
            tmp_path, "boom.py", "def on_update(self, dt):\n    raise RuntimeError('бум')\n"
        )
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        scene.update(0.1)
        assert component.handle.errors == 1

    def test_api_names_exposed(self) -> None:
        assert "game_object" in API_NAMES
        assert "scene" in API_NAMES
        assert ScriptComponent.API_NAMES == API_NAMES

    def test_auto_start_off_skips_hooks(self, tmp_path: Path) -> None:
        """Скрипт, выключенный на старте, не получает on_start и кадров."""
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        cube.add_component(MeshRenderer, mesh="cube")
        component = cube.add_component(
            ScriptComponent(path, runtime=runtime, auto_start=False)
        )
        assert component.is_loaded, "скрипт должен остаться загруженным"
        component.on_start()
        component.on_update(0.5)
        assert component.is_running is False
        assert component.module.CALLS == 0
        assert cube.transform.rotation.w == pytest.approx(1.0)

    def test_auto_start_can_be_enabled_later(self, tmp_path: Path) -> None:
        """Выключенный скрипт запускается, когда флаг включают."""
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        scene = Scene("Test")
        cube = scene.create_game_object("Cube")
        cube.add_component(MeshRenderer, mesh="cube")
        component = cube.add_component(
            ScriptComponent(path, runtime=runtime, auto_start=False)
        )
        assert component.auto_start is False

        component.auto_start = True
        component.on_start()
        component.on_update(0.5)

        assert component.is_running is True
        assert component.module.CALLS == 1

    def test_auto_start_flag_survives_serialization(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        component = ScriptComponent(path, auto_start=False)
        payload = component.to_dict()
        assert payload["auto_start"] is False
        restored = ScriptComponent.from_dict(payload)
        assert restored.auto_start is False
        assert restored.path == path

    def test_serialization_round_trip(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        component = ScriptComponent(path)
        payload = component.to_dict()
        assert payload["path"] == str(path)
        restored = ScriptComponent.from_dict(payload)
        assert restored.path == path

    def test_inline_serialization(self) -> None:
        component = ScriptComponent(source="VALUE = 1\n")
        payload = component.to_dict()
        assert "source" in payload and "path" not in payload
        assert ScriptComponent.from_dict(payload).source == "VALUE = 1\n"

    def test_repr_mentions_owner(self, tmp_path: Path) -> None:
        path = write_script(tmp_path, "spin.py", SAMPLE)
        runtime = ScriptRuntime(project_root=tmp_path)
        cube = Scene("Test").create_game_object("Cube")
        component = cube.add_component(ScriptComponent(path, runtime=runtime))
        assert "Cube" in repr(component)
        assert "spin.py" in repr(component)

    def test_default_runtime_is_shared(self, tmp_path: Path) -> None:
        from engine.scripting.script_component import (
            default_runtime,
            set_default_runtime,
        )

        custom = ScriptRuntime(project_root=tmp_path)
        set_default_runtime(custom)
        try:
            component = ScriptComponent(source="VALUE = 1\n")
            component.load()
            assert len(custom) == 1
            assert component.runtime is custom
        finally:
            set_default_runtime(None)
        assert default_runtime() is not None


class TestHighlighterData:
    def test_keywords_and_builtins_are_present(self) -> None:
        from engine.scripting.syntax import (
            BUILTIN_NAMES,
            ENGINE_API,
            KEYWORDS,
            LIFECYCLE_HOOKS,
            TOKEN_COLORS,
            completion_words,
        )

        assert "def" in KEYWORDS and "class" in KEYWORDS
        assert "print" in BUILTIN_NAMES
        assert "self.game_object" in ENGINE_API
        assert "on_update" in LIFECYCLE_HOOKS
        words = completion_words(["моя_функция"])
        assert "моя_функция" in words
        assert "on_update" in words
        for token in ("keyword", "string", "comment"):
            assert token in TOKEN_COLORS


def test_highlighter_requires_pyside() -> None:
    """Заглушка и настоящая подсветка взаимоисключающи, но импорт не падает."""
    from engine.scripting import python_highlighter

    if python_highlighter.QT_AVAILABLE:
        assert python_highlighter.PythonHighlighter is not None
    else:  # pragma: no cover - только в окружении без PySide6
        with pytest.raises(ImportError):
            python_highlighter.PythonHighlighter(None)
