"""Overrides префабов в редакторе: жирные поля, «Вернуть», «Применить», иконка."""

from __future__ import annotations

import json

from editor.prefab_actions import REBUILD_WARN_THRESHOLD, apply_override_to_prefab
from engine.core.math3d import Vector3
from engine.core.prefab_components import PrefabInstance
from engine.prefab import get_overrides


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
    path = tmp_path / "pair.prefab.json"
    data = {
        "format": "myengine.prefab",
        "version": 1,
        "name": "pair",
        "root": "L0",
        "objects": [_entry("L0"), _entry("L1", "L0")],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _child(root):
    return next(c for c in root.children if c.name == "L1")


def test_override_shows_buttons_and_bold_label(window, tmp_path) -> None:
    root = window.create_prefab(str(_write_prefab(tmp_path)))
    assert root is not None
    child = _child(root)
    child.transform.position = Vector3(1.0, 2.0, 3.0)
    window.context.selection.select(child)
    window.inspector.refresh()
    section = window.inspector.prefab_section
    assert section is not None
    assert len(section.revert_buttons) == 1
    assert len(section.apply_buttons) == 1
    label = window.inspector._marks[("Transform", "position")]
    assert label is not None and label.font().bold()
    assert not window.inspector._marks[("Transform", "scale")].font().bold()


def test_root_section_lists_all_overrides(window, tmp_path) -> None:
    root = window.create_prefab(str(_write_prefab(tmp_path)))
    _child(root).transform.position = Vector3(1.0, 0.0, 0.0)
    window.context.selection.select(root)
    window.inspector.refresh()
    assert window.inspector.prefab_section.overrides_count.text() == "1"


def test_revert_restores_prefab_value(window, tmp_path) -> None:
    root = window.create_prefab(str(_write_prefab(tmp_path)))
    child = _child(root)
    child.transform.position = Vector3(1.0, 2.0, 3.0)
    window.context.selection.select(child)
    window.inspector.refresh()
    window.inspector.prefab_section.revert_buttons[0].click()
    assert child.transform.position.to_tuple() == (0.0, 0.0, 0.0)
    assert get_overrides(root, assets=window.context.asset_db) == []


def test_apply_writes_file_and_clears_override(window, tmp_path) -> None:
    path = _write_prefab(tmp_path)
    root = window.create_prefab(str(path))
    _child(root).transform.position = Vector3(4.0, 0.0, 0.0)
    record = get_overrides(root, assets=window.context.asset_db)[0]
    count = apply_override_to_prefab(window.context, root, record)
    assert count == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = next(o for o in data["objects"] if o["local_id"] == "L1")
    assert entry["transform"]["position"] == [4.0, 0.0, 0.0]
    rebuilt = next(
        o for o in window.context.scene.objects if o.has_component(PrefabInstance)
    )
    assert _child(rebuilt).transform.position.to_tuple() == (4.0, 0.0, 0.0)
    assert get_overrides(rebuilt, assets=window.context.asset_db) == []


def test_many_instances_warn(window, tmp_path) -> None:
    path = _write_prefab(tmp_path)
    root = None
    for _ in range(REBUILD_WARN_THRESHOLD + 1):
        root = window.create_prefab(str(path))
    assert root is not None
    _child(root).transform.position = Vector3(1.0, 0.0, 0.0)
    record = get_overrides(root, assets=window.context.asset_db)[0]
    warnings: list[str] = []
    calls: list[tuple[int, int]] = []
    apply_override_to_prefab(
        window.context,
        root,
        record,
        progress=lambda d, t: calls.append((d, t)),
        warn=warnings.append,
    )
    assert len(warnings) == 1
    assert calls and calls[-1][0] == calls[-1][1]


def test_hierarchy_icon_for_instance_root(window, tmp_path) -> None:
    from PySide6.QtCore import Qt

    root = window.create_prefab(str(_write_prefab(tmp_path)))
    model = window.hierarchy.model
    model.refresh()
    assert not model.data(model.index_for(root), Qt.DecorationRole).isNull()
    plain = window.create_text()
    model.refresh()
    assert model.data(model.index_for(plain), Qt.DecorationRole) is None
