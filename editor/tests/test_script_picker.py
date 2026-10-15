"""Встроенный выбор скрипта: только папка scripts с подпапками."""

from __future__ import annotations

from pathlib import Path

from editor.panels.script_picker import ScriptPicker, list_scripts
from engine.scripting import ScriptComponent


def _make_tree(root: Path) -> Path:
    scripts = root / "Assets" / "Scripts"
    (scripts / "sub" / "deep").mkdir(parents=True)
    (scripts / "__pycache__").mkdir()
    (scripts / "a.py").write_text("x = 1\n", encoding="utf-8")
    (scripts / "notes.txt").write_text("", encoding="utf-8")
    (scripts / "sub" / "b.py").write_text("", encoding="utf-8")
    (scripts / "sub" / "deep" / "c.py").write_text("", encoding="utf-8")
    (scripts / "__pycache__" / "a.py").write_text("", encoding="utf-8")
    (root / "outside.py").write_text("", encoding="utf-8")
    return scripts


def test_list_scripts_recurses_and_filters(tmp_path) -> None:
    scripts = _make_tree(tmp_path)
    names = [p.relative_to(scripts).as_posix() for p in list_scripts(scripts)]
    assert names == ["a.py", "sub/b.py", "sub/deep/c.py"]


def test_list_scripts_missing_dir(tmp_path) -> None:
    assert list_scripts(tmp_path / "none") == []


def test_picker_selects_only_files(qapp, tmp_path) -> None:
    scripts = _make_tree(tmp_path)
    dialog = ScriptPicker(scripts, "t")
    folder = dialog.tree.topLevelItem(1)  # sub
    dialog.tree.setCurrentItem(folder)
    assert dialog.selected_path() is None
    leaf = folder.child(0)
    dialog.tree.setCurrentItem(leaf)
    assert dialog.selected_path() is not None


def test_add_script_uses_picker(window, tmp_path, monkeypatch) -> None:
    scripts = _make_tree(tmp_path)
    cube = next(o for o in window.context.scene.objects if o.name == "Cube")
    window.context.selection.select(cube)
    monkeypatch.setattr(
        window.inspector, "_pick_script_path", lambda: scripts / "a.py"
    )
    window.inspector._add_script()  # noqa: SLF001
    assert cube.get_component(ScriptComponent) is not None
