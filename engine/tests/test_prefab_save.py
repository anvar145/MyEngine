"""save_as_prefab: поддерево -> файл префаба + экземпляр на месте оригинала."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.core.math3d import Quaternion, Vector3
from engine.core.prefab_components import PrefabInstance
from engine.core.resources.asset_db import AssetDatabase
from engine.core.scene import Scene
from engine.prefab import PrefabError
from engine.prefab_save import check_can_save_as_prefab, save_as_prefab
from engine.scene_io import scene_from_dict, scene_to_dict


def _tree(scene: Scene):
    holder = scene.create_game_object("Holder", position=Vector3(0, 2, 0))
    root = scene.create_game_object("Tower", parent=holder, position=Vector3(1, 0, 3))
    root.transform.scale = Vector3(2, 2, 2)
    child = scene.create_game_object("Top", parent=root, position=Vector3(0, 4, 0))
    return holder, root, child


@pytest.fixture
def db(tmp_path: Path):
    (tmp_path / "Assets" / "Prefabs").mkdir(parents=True)
    with AssetDatabase(tmp_path / "Assets") as database:
        yield database


def test_saves_file_and_replaces_with_instance(tmp_path: Path, db) -> None:
    scene = Scene("s")
    holder, root, _child = _tree(scene)
    path = tmp_path / "Assets" / "Prefabs" / "tower.prefab.json"
    instance = save_as_prefab(scene, root, path, assets=db, project_root=tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "Tower" and len(data["objects"]) == 2
    assert instance.parent is holder and instance.name == "Tower"
    assert instance.transform.position.to_tuple() == (1.0, 0.0, 3.0)
    assert instance.transform.scale.to_tuple() == (2.0, 2.0, 2.0)
    marker = instance.get_component(PrefabInstance)
    assert marker is not None and marker.instance_uid
    assert marker.prefab_guid and marker.prefab_path == "Assets/Prefabs/tower.prefab.json"
    assert root.is_destroyed
    assert [c.name for c in instance.children] == ["Top"]
    assert len([o for o in scene.objects if o.name == "Tower"]) == 1


def test_scene_roundtrip_keeps_instance(tmp_path: Path, db) -> None:
    scene = Scene("s")
    _holder, root, _ = _tree(scene)
    path = tmp_path / "Assets" / "tower.prefab.json"
    save_as_prefab(scene, root, path, assets=db, project_root=tmp_path)
    saved = scene_to_dict(scene)
    loaded = scene_from_dict(saved, db)
    assert sorted(o.name for o in loaded.objects) == ["Holder", "Top", "Tower"]


def test_refuses_member_of_instance(tmp_path: Path, db) -> None:
    scene = Scene("s")
    _holder, root, _ = _tree(scene)
    instance = save_as_prefab(scene, root, tmp_path / "Assets" / "t.prefab.json", assets=db, project_root=tmp_path)
    with pytest.raises(PrefabError, match="Unpack"):
        check_can_save_as_prefab(instance)
    with pytest.raises(PrefabError):
        save_as_prefab(
            scene, instance.children[0], tmp_path / "Assets" / "x.prefab.json", project_root=tmp_path
        )
    assert not (tmp_path / "Assets" / "x.prefab.json").exists()


def test_nested_instance_inside_subtree_is_allowed(tmp_path: Path, db) -> None:
    scene = Scene("s")
    _holder, root, _ = _tree(scene)
    inner = save_as_prefab(scene, root, tmp_path / "Assets" / "inner.prefab.json", assets=db, project_root=tmp_path)
    outer = scene.create_game_object("Outer", rotation=Quaternion.identity())
    inner.set_parent(outer, keep_world=False)
    result = save_as_prefab(scene, outer, tmp_path / "Assets" / "outer.prefab.json", assets=db, project_root=tmp_path)
    assert result.get_component(PrefabInstance) is not None
    assert any(o.get_component(PrefabInstance) for o in result.children)


def test_unpack_keeps_objects_uids_and_values(tmp_path: Path, db) -> None:
    from engine.prefab_save import unpack_instance

    scene = Scene("s")
    _holder, root, _child = _tree(scene)
    path = tmp_path / "Assets" / "Prefabs" / "t.prefab.json"
    instance = save_as_prefab(scene, root, path, assets=db, project_root=tmp_path)
    top = instance.children[0]
    top.transform.position = Vector3(0, 5, 0)  # override
    uids = {o.uid for o in scene.objects}
    result = unpack_instance(top)  # член экземпляра: корень ищется вверх
    assert result is instance
    assert instance.get_component(PrefabInstance) is None
    assert {o.uid for o in scene.objects} == uids
    assert top.transform.position.y == 5
    assert "instance_uid" not in json.dumps(scene_to_dict(scene, db))


def test_unpack_rejects_plain_object() -> None:
    from engine.prefab_save import unpack_instance

    scene = Scene("s")
    _holder, root, _child = _tree(scene)
    with pytest.raises(PrefabError):
        unpack_instance(root)


def _script_scene(tmp_path: Path):
    from engine.core.event_bus import EventBus
    from engine.scripting.runtime import ScriptRuntime
    from engine.scripting.script_component import ScriptComponent

    runtime = ScriptRuntime(project_root=tmp_path, auto_reload=False)
    path = tmp_path / "s.py"
    path.write_text("target: GameObject = None\n", "utf-8")
    scene = Scene("s", event_bus=EventBus())
    holder = scene.create_game_object("Holder")
    tower = scene.create_game_object("Tower", parent=holder)
    scene.create_game_object("Top", parent=tower)
    user = scene.create_game_object("User")
    script = user.add_component(ScriptComponent(path, runtime=runtime))
    return scene, tower, script


def test_external_script_ref_follows_new_instance(tmp_path: Path, db) -> None:
    scene, tower, script = _script_scene(tmp_path)
    assert script.set_field("target", tower)
    instance = save_as_prefab(
        scene, tower, tmp_path / "Assets" / "Prefabs" / "t.prefab.json",
        assets=db, project_root=tmp_path,
    )
    assert script.fields["target"] is instance
    assert script.module.target is instance
    assert script.fields["target"].uid == instance.uid
    data = scene_to_dict(scene, db)
    user = next(o for o in data["objects"] if o["name"] == "User")
    refs = [c["fields"]["target"] for c in user["components"] if "fields" in c]
    assert refs == [{"uid": instance.uid}]


def test_child_ref_follows_new_child(tmp_path: Path, db) -> None:
    scene, tower, script = _script_scene(tmp_path)
    top = tower.children[0]
    assert script.set_field("target", top)
    instance = save_as_prefab(
        scene, tower, tmp_path / "Assets" / "Prefabs" / "t.prefab.json",
        assets=db, project_root=tmp_path,
    )
    assert script.fields["target"] is instance.children[0]


def test_pending_uid_ref_is_remapped(tmp_path: Path, db) -> None:
    scene, tower, script = _script_scene(tmp_path)
    script._pending_refs["target"] = tower.uid
    instance = save_as_prefab(
        scene, tower, tmp_path / "Assets" / "Prefabs" / "t.prefab.json",
        assets=db, project_root=tmp_path,
    )
    assert script._pending_refs["target"] == instance.uid


def test_override_value_ref_in_other_instance_is_remapped(tmp_path: Path, db) -> None:
    scene, tower, _script = _script_scene(tmp_path)
    other = scene.create_game_object("Other")
    marker = other.add_component(PrefabInstance("g", "p"))
    marker.overrides.append(
        {"target": "L0", "component": "X", "field": "t", "value": {"uid": tower.uid}}
    )
    old = tower.uid
    instance = save_as_prefab(
        scene, tower, tmp_path / "Assets" / "Prefabs" / "t.prefab.json",
        assets=db, project_root=tmp_path,
    )
    assert marker.overrides[0]["value"] == {"uid": instance.uid}
    assert instance.uid != old


def test_deep_irregular_tree_leaf_ref_follows(tmp_path: Path, db) -> None:
    """Неоднородное дерево: ссылка на лист 3-го уровня и на соседние ветки."""
    scene, tower, script = _script_scene(tmp_path)
    tower.children[0].name = "Top"
    a = scene.create_game_object("A", parent=tower)
    scene.create_game_object("A1", parent=a)
    a2 = scene.create_game_object("A2", parent=a)
    leaf = scene.create_game_object("Leaf", parent=a2)
    scene.create_game_object("B", parent=tower)
    c = scene.create_game_object("C", parent=tower)
    scene.create_game_object("Same", parent=c)
    scene.create_game_object("Same", parent=c)
    assert script.set_field("target", leaf)
    olds = {o.uid for o in [tower, *_all(tower)]}
    instance = save_as_prefab(
        scene, tower, tmp_path / "Assets" / "Prefabs" / "t.prefab.json",
        assets=db, project_root=tmp_path,
    )
    new_leaf = script.fields["target"]
    assert new_leaf.name == "Leaf" and new_leaf.uid not in olds
    assert new_leaf.parent.name == "A2" and new_leaf.parent.parent.name == "A"
    assert new_leaf.parent.parent.parent is instance


def _all(obj):
    for child in obj.children:
        yield child
        yield from _all(child)
