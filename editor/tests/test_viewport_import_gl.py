"""Вьюпорт редактора показывает модели glTF/FBX из импортированных ассетов."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "_work" / "import_samples"
WARRIOR = SAMPLES / "warrior_sample"


@pytest.fixture(params=["Warrior.glb", "Warrior.fbx"])
def model_window(request, qapp, tmp_path):
    source = WARRIOR / request.param
    if not source.exists():
        pytest.skip("нет образца модели Warrior")
    if request.param.endswith(".fbx"):
        pytest.importorskip("ufbx")
    else:
        pytest.importorskip("pygltflib")

    from editor.context import EditorContext
    from editor.main_window import MainWindow

    models = tmp_path / "Assets" / "Models"
    models.mkdir(parents=True)
    target = models / request.param
    shutil.copy(source, target)
    context = EditorContext(project_root=tmp_path, language="ru")
    window = MainWindow(context)
    window.resize(800, 600)
    window.show()
    qapp.processEvents()
    window.viewport.viewport.update()
    qapp.processEvents()
    if not window.viewport.is_ready:
        window.close()
        pytest.skip("GL-контекст недоступен в этой среде")
    manager = context.start_imports()
    assert manager.wait_idle(120)
    yield window, target
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_imported_model_is_drawn_with_all_parts(model_window) -> None:
    """Модель из ``Library/`` рисуется всеми частями и попадает в кадр."""
    from PySide6.QtGui import QImage

    from engine.core.components import MeshRenderer

    window, path = model_window
    game_object = window.add_model_from_asset(str(path))
    component = game_object.get_component(MeshRenderer)
    viewport = window.viewport

    parts = viewport._parts_for(component)
    assert len(parts) > 1, "ожидалось несколько субмешей"
    bounds = viewport._bounds_for(component)
    assert bounds is not None
    assert (bounds[1] - bounds[0]).length > 0.0

    viewport.frame_selected()
    image = viewport.viewport.grabFramebuffer().convertToFormat(
        QImage.Format.Format_RGBA8888
    )
    raw = np.frombuffer(image.constBits(), dtype=np.uint8)
    pixels = raw.reshape(image.height(), image.width(), 4)[:, :, :3].copy()
    background = pixels[0, 0]
    covered = float(np.mean(np.any(pixels != background, axis=2)))
    assert covered > 0.02, "модель не видна во вьюпорте"
