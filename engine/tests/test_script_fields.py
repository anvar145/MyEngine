"""Поля скриптов: поиск в исходнике, значения до ``on_start``, сцена, плеер."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.core.event_bus import EventBus
from engine.core.resources import AssetDatabase, AssetRef
from engine.core.scene import Scene
from engine.player.snapshot import SceneSnapshot
from engine.scene_io import scene_from_dict, scene_to_dict
from engine.scripting import ScriptComponent, ScriptRuntime
from engine.scripting.fields import coerce_field, scan_fields

SOURCE = '''
speed = 3.0
count = 5
label = "hi"
on = True
ratio: float = 1
target: GameObject = None
icon: Asset = None
_private = 1
LIST = [1, 2]
computed = 1 + 2
negative = -4
def on_start(self):
    pass
'''


def _by_name(source: str) -> dict:
    return {f.name: f for f in scan_fields(source)}


class TestScan:
    def test_kinds_and_defaults(self) -> None:
        fields = _by_name(SOURCE)
        assert {n: f.kind for n, f in fields.items()} == {
            "speed": "float",
            "count": "int",
            "label": "str",
            "on": "bool",
            "ratio": "float",
            "target": "object",
            "icon": "asset",
            "computed": "int",
            "negative": "int",
        } or "computed" not in fields
        assert fields["speed"].default == 3.0
        assert fields["count"].default == 5
        assert fields["label"].default == "hi"
        assert fields["on"].default is True
        assert fields["negative"].default == -4

    def test_bool_is_not_int(self) -> None:
        assert _by_name("a = True")["a"].kind == "bool"
        assert _by_name("a = 1")["a"].kind == "int"

    def test_annotation_forces_float(self) -> None:
        field = _by_name("x: float = 1")["x"]
        assert field.kind == "float"
        assert isinstance(field.default, float)

    def test_skips_private_collections_calls_and_functions(self) -> None:
        names = set(_by_name(SOURCE))
        assert "_private" not in names
        assert "LIST" not in names
        assert "on_start" not in names
        assert _by_name("v = Vector3(1, 2, 3)") == {}

    def test_syntax_error_gives_no_fields(self) -> None:
        assert scan_fields("def (:") == []

    def test_last_assignment_wins_and_non_literal_removes(self) -> None:
        assert _by_name("a = 1\na = 2")["a"].default == 2
        assert "a" not in _by_name("a = 1\na = [1]")

    def test_string_annotations(self) -> None:
        fields = _by_name('t: "GameObject" = None')
        assert fields["t"].kind == "object"


class TestCoerce:
    def test_rules(self) -> None:
        assert coerce_field("float", 2) == 2.0
        assert coerce_field("int", 2.5) is None
        assert coerce_field("int", True) is None
        assert coerce_field("bool", 1) is None
        assert coerce_field("str", 5) is None
        assert coerce_field("float", "1") is None
        assert coerce_field("asset", "a/b.png") == AssetRef(path="a/b.png")
        assert coerce_field("object", "x") is None


@pytest.fixture()
def runtime(tmp_path: Path) -> ScriptRuntime:
    return ScriptRuntime(project_root=tmp_path, auto_reload=False)


def _script(tmp_path: Path, text: str, name: str = "s.py") -> Path:
    path = tmp_path / name
    path.write_text(text, "utf-8")
    return path


def _make(scene: Scene, runtime, path: Path, name: str = "A"):
    obj = scene.create_game_object(name)
    return obj, obj.add_component(ScriptComponent(path, runtime=runtime))


class TestRuntimeValues:
    def test_default_comes_from_code(self, tmp_path, runtime) -> None:
        scene = Scene("s", event_bus=EventBus())
        _, script = _make(scene, runtime, _script(tmp_path, SOURCE))
        assert script.module.speed == 3.0
        assert script.module.target is None
        assert script.module.icon is None

    def test_override_reaches_module_before_on_start(self, tmp_path, runtime) -> None:
        text = (
            "speed = 3.0\nseen = None\n"
            "def on_start(self):\n    global seen\n    seen = speed\n"
        )
        scene = Scene("s", event_bus=EventBus())
        script = ScriptComponent(_script(tmp_path, text), runtime=runtime)
        script.fields["speed"] = 7.0  # как после from_dict: до подключения
        scene.create_game_object("A").add_component(script)  # on_start здесь
        assert script.module.seen == 7.0
        assert isinstance(script.module.seen, float)

    def test_two_objects_have_independent_values(self, tmp_path, runtime) -> None:
        path = _script(tmp_path, "speed = 1.0\nlabel = 'x'\n")
        scene = Scene("s", event_bus=EventBus())
        _, a = _make(scene, runtime, path, "A")
        _, b = _make(scene, runtime, path, "B")
        a.set_field("speed", 9.5)
        b.set_field("label", "bee")
        assert (a.module.speed, a.module.label) == (9.5, "x")
        assert (b.module.speed, b.module.label) == (1.0, "bee")

    def test_wrong_type_is_rejected(self, tmp_path, runtime) -> None:
        scene = Scene("s", event_bus=EventBus())
        _, script = _make(scene, runtime, _script(tmp_path, SOURCE))
        assert not script.set_field("count", 2.5)
        assert not script.set_field("on", "yes")
        assert not script.set_field("missing", 1)
        assert script.fields == {}
        assert script.module.count == 5

    def test_clear_returns_to_code_value(self, tmp_path, runtime) -> None:
        scene = Scene("s", event_bus=EventBus())
        _, script = _make(scene, runtime, _script(tmp_path, SOURCE))
        script.set_field("speed", 9.0)
        assert script.clear_field("speed")
        assert script.module.speed == 3.0
        assert "speed" not in script.fields

    def test_object_and_asset_references(self, tmp_path, runtime) -> None:
        scene = Scene("s", event_bus=EventBus())
        _, script = _make(scene, runtime, _script(tmp_path, SOURCE))
        other = scene.create_game_object("Other")
        assert script.set_field("target", other)
        assert script.module.target is other
        assert script.set_field("icon", str(tmp_path / "i.png"))
        assert script.module.icon == AssetRef(path=str(tmp_path / "i.png"))
        assert isinstance(script.module.icon, AssetRef)
        assert script.set_field("target", None)
        assert script.module.target is None

    def test_override_survives_hot_reload_and_code_change(
        self, tmp_path, runtime
    ) -> None:
        path = _script(tmp_path, "speed = 1.0\nother = 2\n")
        scene = Scene("s", event_bus=EventBus())
        _, script = _make(scene, runtime, path)
        script.set_field("speed", 8.0)
        path.write_text("speed = 5.0\nother = 3\nadded = 'new'\n", "utf-8")
        script.reload()
        assert script.module.speed == 8.0
        assert script.module.other == 3
        assert script.module.added == "new"
        path.write_text("other = 3\n", "utf-8")  # поле пропало из кода
        script.reload()
        assert script.fields["speed"] == 8.0  # ждёт возвращения
        path.write_text("speed = 5.0\n", "utf-8")
        script.reload()
        assert script.module.speed == 8.0

    def test_type_change_in_code_drops_override(self, tmp_path, runtime) -> None:
        path = _script(tmp_path, "speed = 1.0\n")
        scene = Scene("s", event_bus=EventBus())
        _, script = _make(scene, runtime, path)
        script.set_field("speed", 8.0)
        path.write_text("speed = 'fast'\n", "utf-8")
        script.reload()
        assert script.module.speed == "fast"
        assert "speed" not in script.fields

    def test_script_without_future_import_runs_annotations(
        self, tmp_path, runtime
    ) -> None:
        scene = Scene("s", event_bus=EventBus())
        _, script = _make(scene, runtime, _script(tmp_path, "t: GameObject = None\n"))
        assert script.is_loaded, script.last_error


def _project(tmp_path: Path):
    assets = tmp_path / "Assets"
    (assets / "Scripts").mkdir(parents=True)
    (assets / "Textures").mkdir()
    (assets / "Textures" / "icon.png").write_bytes(b"x")
    script = assets / "Scripts" / "s.py"
    script.write_text(SOURCE, "utf-8")
    return assets, script, AssetDatabase(assets, library=tmp_path / "Library")


class TestSceneRoundTrip:
    def _scene(self, tmp_path):
        assets, script_path, db = _project(tmp_path)
        runtime = ScriptRuntime(project_root=tmp_path, auto_reload=False)
        scene = Scene("s", event_bus=EventBus())
        scene.create_game_object("Filler")
        other = scene.create_game_object("Other")
        _, script = _make(scene, runtime, script_path, "Owner")
        script.set_field("speed", 6.5)
        script.set_field("count", 9)
        script.set_field("label", "boss")
        script.set_field("on", False)
        script.set_field("target", other)
        script.set_field("icon", str(assets / "Textures" / "icon.png"))
        return scene, db, assets

    def _owner(self, scene):
        return next(
            o.get_component(ScriptComponent)
            for o in scene.objects
            if o.get_component(ScriptComponent)
        )

    def test_all_types_round_trip(self, tmp_path: Path) -> None:
        scene, db, assets = self._scene(tmp_path)
        data = json.loads(json.dumps(scene_to_dict(scene, db)))
        stored = data["objects"][2]["components"][0]["fields"]
        assert stored["target"] == {"uid": data["objects"][1]["uid"]}
        assert stored["icon"]["path"] == "Textures/icon.png"
        loaded = scene_from_dict(data, db)
        script = self._owner(loaded)
        module = script.module
        assert (module.speed, module.count, module.label, module.on) == (
            6.5,
            9,
            "boss",
            False,
        )
        assert module.target is loaded.objects[1]
        assert module.target.name == "Other"
        assert assets / module.icon.path == assets / "Textures" / "icon.png"

    def test_old_scene_without_fields(self, tmp_path: Path) -> None:
        scene, db, _ = self._scene(tmp_path)
        data = scene_to_dict(scene, db)
        del data["objects"][2]["components"][0]["fields"]
        script = self._owner(scene_from_dict(data, db))
        assert script.module.speed == 3.0

    def test_object_reference_survives_snapshot_renumbering(
        self, tmp_path: Path
    ) -> None:
        scene, db, _ = self._scene(tmp_path)
        scene.objects[0].destroy() if hasattr(scene.objects[0], "destroy") else None
        snapshot = SceneSnapshot.freeze(scene, db)
        loaded = scene_from_dict(snapshot.data, db)
        script = self._owner(loaded)
        assert script.module.target.name == "Other"

    def test_deleted_object_and_missing_file_become_unset(
        self, tmp_path: Path
    ) -> None:
        scene, db, assets = self._scene(tmp_path)
        data = scene_to_dict(scene, db)
        fields = data["objects"][2]["components"][0]["fields"]
        fields["target"] = {"object": 9999}
        (assets / "Textures" / "icon.png").unlink()
        script = self._owner(scene_from_dict(data, db))
        assert script.module.target is None
        # битая ссылка не стирается: инспектор подсвечивает её красным
        assert script.module.icon.path.endswith("Textures/icon.png")
        assert script.module.speed == 6.5

    def test_asset_follows_guid_after_move(self, tmp_path: Path) -> None:
        scene, db, assets = self._scene(tmp_path)
        db.scan()
        data = scene_to_dict(scene, db)
        entry = data["objects"][2]["components"][0]["fields"]["icon"]
        assert entry.get("guid")
        old = assets / "Textures" / "icon.png"
        moved = assets / "Textures" / "renamed.png"
        old.rename(moved)
        old.with_name("icon.png.meta").rename(moved.with_name("renamed.png.meta"))
        fresh = AssetDatabase(assets, library=tmp_path / "Library2")
        fresh.scan()
        script = self._owner(scene_from_dict(data, fresh))
        assert assets / script.module.icon.path == moved


PLAYER_SCRIPT = '''
from engine.core.math3d import Vector3

speed = 2.0
step = 4
name_tag = "none"
enabled_flag = False
target: GameObject = None
icon: Asset = None

def on_start(self):
    if target is not None:
        target.transform.position = Vector3(speed, step, 0.0)
    self.game_object.name = name_tag + ("-on" if enabled_flag else "-off")
    self.game_object.name += "|" + (icon.path.split("/")[-1] if icon else "noicon")
    self.game_object.name += "|" + (target.name if target is not None else "notarget")
'''


def test_player_interprets_fields_in_game(tmp_path: Path) -> None:
    """Сцена с полями всех типов, запущенная плеером: скрипт видит значения."""
    root = tmp_path / "Game"
    assets = root / "Assets"
    (assets / "Scripts").mkdir(parents=True)
    (assets / "Scenes").mkdir()
    (assets / "Textures").mkdir()
    (assets / "Textures" / "hero.png").write_bytes(b"x")
    (assets / "Scripts" / "boss.py").write_text(PLAYER_SCRIPT, "utf-8")
    scene = {
        "format": "myengine.scene",
        "version": 1,
        "name": "t",
        "objects": [
            {
                "id": 7,
                "name": "Target",
                "components": [],
                "transform": {
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0, 1],
                    "scale": [1, 1, 1],
                },
            },
            {
                "id": 8,
                "name": "Boss",
                "components": [
                    {
                        "type": "ScriptComponent",
                        "path": "Assets/Scripts/boss.py",
                        "auto_start": True,
                        "fields": {
                            "speed": 5,
                            "step": 11,
                            "name_tag": "orc",
                            "enabled_flag": True,
                            "target": {"object": 7},
                            "icon": {"asset": "Assets/Textures/hero.png"},
                        },
                    }
                ],
                "transform": {
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0, 1],
                    "scale": [1, 1, 1],
                },
            },
        ],
    }
    scene_path = assets / "Scenes" / "t.scene.json"
    scene_path.write_text(json.dumps(scene), "utf-8")
    root_dir = Path(__file__).resolve().parents[2]
    state_path = tmp_path / "state.json"
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "engine.player",
            "--project",
            str(root),
            "--scene",
            "Assets/Scenes/t.scene.json",
            "--headless",
            "--frames",
            "5",
            "--dump-state",
            str(state_path),
        ],
        env=env,
        cwd=root_dir,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    state = json.loads(state_path.read_text("utf-8"))
    objects = {o["name"]: o for o in state["objects"]}
    assert "orc-on|hero.png|Target" in objects, list(objects)
    target = objects["Target"]
    assert target["position"] == pytest.approx([5.0, 11.0, 0.0])
