"""Версии и миграции сохранений (A7a п.3)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from engine.core.save import (
    MIGRATIONS,
    SaveManager,
    SaveVersionError,
    migrate_chain,
    register_migration,
)
from engine.core.scene import Scene


def _forge_version(path: Path, version: int) -> None:
    """Переписать версию в ``meta.json`` и ``version`` (подделка старого сейва)."""
    with zipfile.ZipFile(path) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    meta = json.loads(entries["meta.json"])
    meta["version"] = version
    entries["meta.json"] = json.dumps(meta).encode()
    entries["version"] = str(version).encode()
    with zipfile.ZipFile(path, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)


def _scene() -> Scene:
    scene = Scene("s")
    scene.state.set("gold", 10)
    return scene


def test_global_registry_untouched() -> None:
    assert MIGRATIONS == {}


def test_chain_order_and_shortest() -> None:
    reg: dict = {}
    for a, b in [(1, 2), (2, 3), (1, 3), (3, 4)]:
        register_migration(a, b, reg)(lambda p: p)
    assert len(migrate_chain(1, 3, reg)) == 1  # прямой шаг короче
    assert len(migrate_chain(1, 4, reg)) == 2
    assert migrate_chain(4, 4, reg) == []


def test_chain_missing_and_newer() -> None:
    with pytest.raises(SaveVersionError, match="Нет миграции"):
        migrate_chain(1, 3, {})
    with pytest.raises(SaveVersionError, match="более новой версией"):
        migrate_chain(5, 3, {})


def test_register_validation() -> None:
    reg: dict = {}
    with pytest.raises(ValueError):
        register_migration(2, 2, reg)
    register_migration(1, 2, reg)(lambda p: p)
    with pytest.raises(ValueError):
        register_migration(1, 2, reg)(lambda p: p)


def test_load_old_save_runs_chain(tmp_path: Path) -> None:
    calls: list[str] = []
    reg: dict = {}

    @register_migration(1, 2, reg)
    def _m12(payload: dict) -> dict:
        calls.append("1->2")
        payload["state"]["coins"] = payload["state"].pop("gold")
        return payload

    @register_migration(2, 3, reg)
    def _m23(payload: dict) -> dict:
        calls.append("2->3")
        payload["state"]["coins"] += 1
        return payload

    root = tmp_path / "saves"
    SaveManager(root, version=1).save("s", _scene(), _scene().state)
    mgr = SaveManager(root, version=3, migrations=reg)
    scene = _scene()
    slot = mgr.load("s", scene)
    assert calls == ["1->2", "2->3"]
    assert scene.state.get("coins") == 11 and not scene.state.has("gold")
    assert slot.meta.version == 3
    # файл не тронут до следующего save
    assert mgr.slot("s").meta.version == 1
    mgr.save("s", scene, scene.state)
    assert mgr.slot("s").meta.version == 3


def test_forged_version_triggers_chain(tmp_path: Path) -> None:
    reg: dict = {}
    seen: list[int] = []
    register_migration(4, 5, reg)(lambda p: seen.append(4) or p)
    mgr = SaveManager(tmp_path, version=5, migrations=reg)
    scene = _scene()
    slot = mgr.save("s", scene, scene.state)
    _forge_version(slot.path, 4)
    assert mgr.slot("s").meta.version == 4
    mgr.load("s", _scene())
    assert seen == [4]


def test_migrate_method_returns_payload(tmp_path: Path) -> None:
    reg: dict = {}
    register_migration(1, 2, reg)(lambda p: {**p, "marker": True})
    SaveManager(tmp_path, version=1).save("s", _scene(), _scene().state)
    mgr = SaveManager(tmp_path, version=2, migrations=reg)
    payload = mgr.migrate(mgr.slot("s"), 1, 2)
    assert payload["marker"] and payload["meta"]["version"] == 2
    assert payload["state"] == {"gold": 10}


def test_newer_save_refused_with_message(tmp_path: Path) -> None:
    SaveManager(tmp_path, version=9).save("s", _scene(), _scene().state)
    with pytest.raises(SaveVersionError) as info:
        SaveManager(tmp_path, version=2).load("s", _scene())
    assert "9" in str(info.value) and "Обновите игру" in str(info.value)


def test_missing_chain_refused(tmp_path: Path) -> None:
    SaveManager(tmp_path, version=1).save("s", _scene(), _scene().state)
    with pytest.raises(SaveVersionError, match="Нет миграции"):
        SaveManager(tmp_path, version=2, migrations={}).load("s", _scene())
