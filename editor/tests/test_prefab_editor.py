"""Префабы в редакторе: «Создать → Префаб...» и секция «Префаб» инспектора."""

from __future__ import annotations

import json

from editor.panels.prefab_section import PrefabInstanceSection
from engine.core.prefab_components import PrefabInstance


def _entry(local_id: str, parent: str | None = None) -> dict:
    entry: dict = {
        "local_id": local_id,
        "name": local_id,
        "tag": "Untagged",
        "layer": 1,
        "active": True,
        "transform": {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "components": [],
    }
    if parent is not None:
        entry["parent"] = {"local": parent}
    return entry


def _write_prefab(tmp_path):
    path = tmp_path / "three.prefab.json"
    data = {
        "format": "myengine.prefab",
        "version": 1,
        "name": "three",
        "root": "L0",
        "objects": [_entry("L0"), _entry("L1", "L0"), _entry("L2", "L0")],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_menu_has_prefab_action(window) -> None:
    assert window.action_create_prefab in window.menu_create.actions()
    assert window.action_create_prefab.text() == window.context.tr(
        "menu.create.prefab"
    )


def test_create_prefab_instantiates_and_selects(window, tmp_path) -> None:
    path = _write_prefab(tmp_path)
    count = len(window.context.scene)
    root = window.create_prefab(str(path))
    assert root is not None
    assert len(window.context.scene) == count + 3
    assert window.context.selection.primary is root
    assert root.get_component(PrefabInstance) is not None


def test_inspector_shows_prefab_section(window, tmp_path) -> None:
    root = window.create_prefab(str(_write_prefab(tmp_path)))
    section = window.inspector.prefab_section
    assert isinstance(section, PrefabInstanceSection)
    marker = root.get_component(PrefabInstance)
    assert section.path.text() == (marker.prefab_path or "-")
    assert section.overrides_count.text() == "0"


def test_plain_object_has_no_prefab_section(window) -> None:
    window.create_text()
    assert window.inspector.prefab_section is None


def test_missing_file_does_not_crash(window, tmp_path) -> None:
    count = len(window.context.scene)
    assert window.create_prefab(str(tmp_path / "nope.prefab.json")) is None
    assert len(window.context.scene) == count


def test_broken_file_does_not_crash(window, tmp_path) -> None:
    path = tmp_path / "bad.prefab.json"
    path.write_text("{not json", encoding="utf-8")
    count = len(window.context.scene)
    assert window.create_prefab(str(path)) is None
    assert len(window.context.scene) == count
