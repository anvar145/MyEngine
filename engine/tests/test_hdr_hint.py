"""``read_hdr_hint``: поле ``hdr`` проекта в три состояния (A14a.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.rendering.dynamic_resolution import read_hdr_hint


def _project(tmp_path: Path, content: str | None) -> Path:
    if content is not None:
        (tmp_path / "g.project.json").write_text(content, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (json.dumps({"name": "g"}), None),
        (json.dumps({"hdr": True}), True),
        (json.dumps({"hdr": False}), False),
        (json.dumps({"hdr": "yes"}), None),
        ("не json", None),
        (None, None),
    ],
)
def test_read_hdr_hint(tmp_path: Path, content: str | None, expected: bool | None) -> None:
    assert read_hdr_hint(_project(tmp_path, content)) is expected
