"""Импорт и отрисовка сэмплов всех форматов (``_work/import_samples``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "render_import_samples.py"
REPORT = ROOT / "_work" / "import_check_report.json"

pytestmark = pytest.mark.skipif(
    not (ROOT / "_work" / "import_samples").is_dir(),
    reason="нет _work/import_samples",
)


def _load_script():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("render_import_samples", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # нужен dataclass-ам с отложенными аннотациями
    spec.loader.exec_module(module)
    return module


def test_import_samples_all_formats() -> None:
    """Каждый формат импортируется; графические — ещё и видны на кадре."""
    module = _load_script()
    code = module.main([])
    report = json.loads(REPORT.read_text("utf-8"))
    failed = {
        ext: item["errors"]
        for ext, item in report["results"].items()
        if item["status"] == "failed"
    }
    assert code == 0 and not failed, failed
    # Pillow — основная зависимость: JPEG/BMP/GIF пропускать нельзя.
    skipped = {e for e, i in report["results"].items() if i["status"] == "skipped"}
    assert not skipped
    assert report["results"]["fbx"]["rendered"] is True
