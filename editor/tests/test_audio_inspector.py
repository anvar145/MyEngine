"""Инспектор: секции AudioSource/AudioListener и drop клипа (A9a п.5)."""

from __future__ import annotations

from editor.panels.auto_section import AutoSection
from editor.panels.script_fields import ReferenceField
from engine.audio.components import AudioListenerComponent, AudioSourceComponent
from engine.core.math3d import Vector3
from engine.core.resources.asset_ref import AssetRef


def _select(window, *components):
    go = window.context.scene.create_game_object("Sound", position=Vector3(0, 0, 0))
    for component in components:
        go.add_component(component)
    window.context.selection.select(go)
    window.inspector.refresh()
    return go


def _sections(window):
    return {type(s._component): s for s in window.inspector.auto_sections}  # noqa: SLF001


def test_audio_sections_are_built(window) -> None:
    _select(window, AudioSourceComponent(), AudioListenerComponent())
    sections = _sections(window)
    assert set(sections) == {AudioSourceComponent, AudioListenerComponent}
    assert all(isinstance(s, AutoSection) for s in sections.values())


def test_clip_drop_and_clear(window) -> None:
    go = _select(window, AudioSourceComponent())
    section = _sections(window)[AudioSourceComponent]
    field = section.widgets["clip"]
    assert isinstance(field, ReferenceField)
    field.dropped.emit("C:/x/readme.txt")  # не звук — игнорируется
    source = go.get_component(AudioSourceComponent)
    assert source.clip is None
    field.dropped.emit("C:/x/beep.wav")
    assert isinstance(source.clip, AssetRef)
    assert source.clip.path.endswith("beep.wav")
    field.cleared.emit()
    assert source.clip is None
