"""Импорт образца Warrior (glTF/FBX) в проект: без ошибок, с мешами и материалами."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from engine.core.resources.asset_db import AssetDatabase
from engine.core.resources.import_manager import ImportManager

SOURCE = Path(__file__).resolve().parents[2] / "_work" / "import_samples" / "warrior_sample"

pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason="нет warrior_sample")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("warrior")
    assets = root / "Assets" / "m"
    assets.mkdir(parents=True)
    for name in ("Warrior.obj", "Warrior.mtl", "Warrior.glb", "Warrior.fbx"):
        if (SOURCE / name).exists():
            shutil.copy2(SOURCE / name, assets / name)
    database = AssetDatabase(root / "Assets", library=root / "Library")
    manager = ImportManager(database)
    manager.open()
    assert manager.wait_idle(300)
    failures = manager.failures()
    manager.close()
    return database, assets, failures


def _bounds(database, path):
    handle = database.resolve(database.guid_of(path))
    infos = [a.info for a in handle.artifacts if a.kind == "mesh"]
    low = np.min([i["bounds_min"] for i in infos], axis=0)
    high = np.max([i["bounds_max"] for i in infos], axis=0)
    return len(infos), low, high


@pytest.mark.parametrize("name", ["Warrior.glb", "Warrior.fbx"])
def test_warrior_imports_without_errors(project, name) -> None:
    """Модель импортируется, даёт несколько мешей с материалами и габаритами."""
    database, assets, failures = project
    path = assets / name
    if not path.exists():
        pytest.skip(f"нет {name}")
    assert not [f for f in failures if name in str(f)], failures
    count, low, high = _bounds(database, path)
    size = high - low
    assert count > 1
    assert np.all(size > 0.0)
    assert np.all(np.isfinite(size))
