"""Тени в настоящем Qt-вьюпорте не должны ломать вывод кадра."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def gl_window(qapp, tmp_path):
    from editor.context import EditorContext
    from editor.main_window import MainWindow

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
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def _grab(window) -> np.ndarray:
    from PySide6.QtGui import QImage

    image = window.viewport.viewport.grabFramebuffer().convertToFormat(
        QImage.Format.Format_RGBA8888
    )
    raw = np.frombuffer(image.constBits(), dtype=np.uint8)
    # Копия: буфер принадлежит QImage и освобождается вместе с ним.
    return raw.reshape(image.height(), image.width(), 4)[:, :, :3].copy()


def test_scene_is_visible_after_shadow_pass(gl_window) -> None:
    """После прохода теней геометрия рисуется в FBO виджета, а не мимо него."""
    gl_window.create_light("directional")
    gl_window.viewport.viewport.update()
    pixels = _grab(gl_window)
    background = pixels[0, 0]
    covered = float(np.mean(np.any(pixels != background, axis=2)))
    assert covered > 0.05, "кадр пуст: рисование ушло мимо FBO виджета"


def test_framebuffer_binding_is_restored_after_shadows(gl_window) -> None:
    """Проход теней возвращает привязку к FBO виджета, а не к нулю."""
    from OpenGL import GL

    from engine.rendering.lighting import LightComponent

    widget = gl_window.viewport.viewport
    renderer = widget.renderer
    sun = gl_window.create_light("directional")
    sun.get_component(LightComponent).light.cast_shadow = True
    seen: list[int] = []
    original = renderer.end_frame

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)))
        return result

    renderer.end_frame = spy
    try:
        widget.grabFramebuffer()
    finally:
        del renderer.end_frame
    assert seen, "кадр не отрисован"
    assert renderer.shadow_map is not None, "проход теней не выполнялся"
    assert seen[-1] == widget.defaultFramebufferObject()
