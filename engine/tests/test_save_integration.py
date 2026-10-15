"""Save/Load через Scene и ScriptComponent (A7a п.4)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from engine.core.save import SaveManager
from engine.core.scene import Scene
from engine.scripting import ScriptComponent, ScriptRuntime

BEHAVIOUR = '''
from engine.scripting import Behaviour, export

class Hero(Behaviour):
    hp: int = export(10, save=True)
    name: str = export("hero")
    buddy: GameObject = export(None, save=True)
    notes = None

    def on_save(self, ctx):
        ctx.set("kills", self.kills)

    def on_load(self, ctx):
        self.loaded_kills = ctx.get("kills")
        self.load_order = [self.hp]

    kills = 0
    loaded_kills = None
'''

FUNCTIONS = '''
coins = 0

def on_save(self, ctx):
    ctx.set("coins", 7)

def on_load(self, ctx):
    self.restored = ctx.get("coins")
'''


def _scene(tmp_path: Path) -> tuple[Scene, ScriptComponent, ScriptComponent]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = ScriptRuntime(project_root=tmp_path, auto_reload=False)
    (tmp_path / "hero.py").write_text(BEHAVIOUR, encoding="utf-8")
    (tmp_path / "fn.py").write_text(FUNCTIONS, encoding="utf-8")
    scene = Scene("level")
    a = scene.create_game_object("A")
    a.uid = "a" * 16
    b = scene.create_game_object("B")
    b.uid = "b" * 16
    hero = a.add_component(ScriptComponent(tmp_path / "hero.py", runtime=runtime))
    fn = b.add_component(ScriptComponent(tmp_path / "fn.py", runtime=runtime))
    scene.save_manager = SaveManager(tmp_path / "saves", version=1)
    return scene, hero, fn


def test_requires_manager(tmp_path: Path) -> None:
    scene = Scene("x")
    with pytest.raises(RuntimeError):
        scene.save("a")
    with pytest.raises(RuntimeError):
        scene.load("a")


def test_behaviour_fields_and_hooks_round_trip(tmp_path: Path) -> None:
    scene, hero, fn = _scene(tmp_path)
    inst = hero.behaviour
    inst.hp = 55
    inst.name = "не сохраняется"
    inst.buddy = fn.game_object
    inst.kills = 4
    scene.state.set("chapter", 2)
    slot = scene.save("s1")

    with zipfile.ZipFile(slot.path) as z:
        data = json.loads(z.read("objects.json"))
    entry = data["a" * 16]["components"]["ScriptComponent#0"]
    assert entry["fields"] == {"hp": 55, "buddy": {"uid": "b" * 16}}
    assert entry["data"] == {"kills": 4}
    assert data["b" * 16]["components"]["ScriptComponent#0"]["data"] == {"coins": 7}

    other, hero2, fn2 = _scene(tmp_path / "second")
    other.save_manager = scene.save_manager
    other.load("s1")
    inst2 = hero2.behaviour
    assert inst2.hp == 55 and inst2.name == "hero"  # name - из сцены/кода
    assert inst2.buddy is fn2.game_object
    assert inst2.loaded_kills == 4
    assert inst2.load_order == [55]  # поля применены до on_load
    assert other.state.get("chapter") == 2
    assert fn2.handle.module.coins == 0


def test_second_save_identical(tmp_path: Path) -> None:
    scene, hero, _ = _scene(tmp_path)
    hero.behaviour.hp = 3
    scene.state.set("k", [1, (2, 3)])
    a = scene.save("a")
    scene.load("a")
    b = scene.save("b")
    with zipfile.ZipFile(a.path) as za, zipfile.ZipFile(b.path) as zb:
        for entry in ("state.json", "objects.json"):
            assert za.read(entry) == zb.read(entry)


def test_hook_exception_does_not_break_load(tmp_path: Path) -> None:
    scene, hero, _ = _scene(tmp_path)

    def boom(ctx):  # noqa: ARG001
        raise RuntimeError("плохой хук")

    hero.behaviour.on_save = boom
    scene.save("s")  # исключение хука логируется, сохранение завершается
    assert scene.save_manager.slot("s") is not None
