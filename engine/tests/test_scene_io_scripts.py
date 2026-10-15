"""Пути скриптов в сцене: в файле — относительно проекта, в памяти — абсолютные."""

from __future__ import annotations

from pathlib import Path

from engine.core.resources import AssetDatabase
from engine.core.scene import Scene
from engine.scene_io import scene_from_dict, scene_to_dict
from engine.scripting import ScriptComponent


def test_script_path_is_relative_to_project(tmp_path: Path) -> None:
    assets_dir = tmp_path / "Assets"
    scripts = assets_dir / "Scripts"
    scripts.mkdir(parents=True)
    script = scripts / "spin.py"
    script.write_text("def on_update(self, delta_time):\n    pass\n", "utf-8")
    database = AssetDatabase(assets_dir, library=tmp_path / "Library")

    scene = Scene("s")
    scene.create_game_object("Spinner").add_component(ScriptComponent(script))

    data = scene_to_dict(scene, database)
    stored = data["objects"][0]["components"][0]["path"]
    assert stored == "Assets/Scripts/spin.py"

    loaded = scene_from_dict(data, database)
    component = loaded.objects[0].get_component(ScriptComponent)
    assert Path(component.path) == script
