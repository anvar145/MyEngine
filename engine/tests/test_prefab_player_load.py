"""Сцена с экземпляром префаба читается так же, как это делает плеер.

Плеер зовёт ``load_scene_with_sky(path, db)`` / ``scene_from_dict(data, db)``
без ``scene.prefab_loader``: загрузчик берётся из fallback через базу ассетов.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.core.math3d import Vector3
from engine.core.prefab_components import PrefabInstance
from engine.core.resources.asset_db import AssetDatabase
from engine.core.scene import Scene
from engine.prefab import (
    _BASE_CACHE,
    _BASE_CACHE_SIZE,
    get_overrides,
    make_prefab_loader,
)
from engine.scene_io import load_scene_with_sky, save_scene, scene_from_dict
from engine.tests.test_prefab_instantiate import _entry


def _write_prefab(assets_dir: Path) -> Path:
    path = assets_dir / "pair.prefab.json"
    data = {
        "format": "myengine.prefab",
        "version": 1,
        "name": "pair",
        "root": "L0",
        "objects": [_entry("L0"), _entry("L1", "L0")],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_player_style_load_without_scene_loader(tmp_path: Path) -> None:
    assets_dir = tmp_path / "Assets"
    assets_dir.mkdir()
    _write_prefab(assets_dir)
    with AssetDatabase(assets_dir) as db:
        scene = Scene("s")
        scene.prefab_loader = make_prefab_loader(db)
        prefab = scene.prefab_loader("", "Assets/pair.prefab.json")
        root = scene.instantiate(prefab)
        root.children[0].transform.position = Vector3(3.0, 0.0, 0.0)
        target = tmp_path / "level.scene.json"
        save_scene(scene, target, db)

        # как плеер: загрузчика на сцене нет, есть только база ассетов
        loaded, _sky = load_scene_with_sky(target, db)
        assert len(loaded) == 2
        child = next(o for o in loaded.objects if o.name == "L1")
        assert child.transform.position.to_tuple() == (3.0, 0.0, 0.0)

        data = json.loads(target.read_text(encoding="utf-8"))
        again = scene_from_dict(data, db)
        assert len(again) == 2
        marker_root = next(o for o in again.objects if o.has_component(PrefabInstance))
        records = get_overrides(marker_root, assets=db)
        assert [(r["target"], r["component"]) for r in records] == [
            ("L1", "Transform")
        ]


def test_base_cache_holds_only_plain_data(tmp_path: Path) -> None:
    assets_dir = tmp_path / "Assets"
    assets_dir.mkdir()
    _write_prefab(assets_dir)
    scene = Scene("s")
    loader = make_prefab_loader(None, assets_dir)
    scene.prefab_loader = loader
    root = scene.instantiate(loader("", str(assets_dir / "pair.prefab.json")))
    get_overrides(root)
    assert 0 < len(_BASE_CACHE) <= _BASE_CACHE_SIZE
    for base in _BASE_CACHE.values():
        assert all(isinstance(v, dict) for v in base.values())
        json.dumps(base, default=str)  # только данные, без объектов сцены
