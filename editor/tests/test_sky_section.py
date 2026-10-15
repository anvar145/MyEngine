"""Секция «Небо» в инспекторе: живая правка и отправка в плеер."""

from __future__ import annotations

from editor.panels.sky_section import SkySection
from engine.core.math3d import Vector3


def _sun(window):
    return window.create_light("directional")


def test_sky_section_is_shown_for_sun(window) -> None:
    _sun(window)
    assert hasattr(window.inspector, "sky_section")
    assert isinstance(window.inspector.sky_section, SkySection)


def test_sky_section_edits_document_sky(window) -> None:
    section = SkySection(window.context)
    section.color_buttons["zenith_color"].set_vector(Vector3(0.1, 0.2, 0.3), emit=True)
    section.spins["exposure"].setValue(2.5)
    sky = window.context.active_document.sky
    assert sky.zenith_color == (0.1, 0.2, 0.3)
    assert sky.exposure == 2.5


def test_sky_edit_is_sent_to_active_player(window) -> None:
    sent: list[tuple[str, object]] = []

    class FakePlayMode:
        is_active = True

        def send_setting(self, path, value) -> None:
            sent.append((path, value))

    window.context.play_mode = FakePlayMode()
    section = SkySection(window.context)
    section.color_buttons["horizon_color"].set_vector(Vector3(1, 0, 0), emit=True)
    assert sent == [("sky.horizon_color", (1.0, 0.0, 0.0))]
