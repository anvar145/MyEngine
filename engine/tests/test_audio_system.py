"""AudioSource/AudioSystem/self.audio на SoftwareBackend (A9a п.3, п.4)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from engine.audio import (
    AudioListenerComponent,
    AudioSourceComponent,
    AudioSystem,
    SoftwareBackend,
    load_audio_settings,
    save_audio_settings,
)
from engine.core.resources import AssetDatabase
from engine.core.resources.asset_ref import AssetRef
from engine.core.scene import Scene
from engine.scene_io import scene_from_dict, scene_to_dict
from engine.scripting import ScriptComponent
from engine.tests.audio_kit import make_wav


@pytest.fixture
def project(tmp_path: Path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    (assets / "beep.wav").write_bytes(make_wav(0.1, value=0.5))
    db = AssetDatabase(assets)
    db.scan()
    db.reimport_dirty(wait=True)
    yield db
    db.close()


def _world(db):
    backend = SoftwareBackend()
    backend.init(44100, 2)
    scene = Scene()
    system = scene.add_system(AudioSystem(backend, db))
    return scene, system, backend


def _source(scene: Scene, **kwargs) -> AudioSourceComponent:
    obj = scene.create_game_object("Sound")
    return obj.add_component(
        AudioSourceComponent(clip=AssetRef(path="beep.wav"), **kwargs)
    )


def test_autoplay_wav_ends_after_clip(project) -> None:
    scene, _system, backend = _world(project)
    source = _source(scene, autoplay=True)
    scene.update(0.016)
    assert source.is_playing and source.play_state == "playing"
    backend.advance(0.15)
    scene.update(0.016)
    assert not source.is_playing and source.play_state == "stopped"


def test_no_autoplay_stays_silent_until_play(project) -> None:
    scene, _system, _backend = _world(project)
    source = _source(scene)
    scene.update(0.016)
    assert not source.is_playing
    source.play()
    scene.update(0.016)
    assert source.is_playing
    source.stop()
    scene.update(0.016)
    assert not source.is_playing


def test_runtime_gain_change_reaches_backend(project) -> None:
    scene, _system, backend = _world(project)
    source = _source(scene, autoplay=True, loop=True)
    scene.update(0.016)
    voice = source._voice
    assert backend.mixer.final_gain(voice.id) == pytest.approx(1.0)
    source.gain = 0.25
    scene.update(0.016)
    assert backend.mixer.final_gain(voice.id) == pytest.approx(0.25)


def test_music_bus_zero_passes_gain_zero(project) -> None:
    scene, system, backend = _world(project)
    source = _source(scene, autoplay=True, loop=True, bus="music")
    scene.update(0.016)
    system.set_bus_gain("music", 0.0)
    assert backend.mixer.final_gain(source._voice.id) == 0.0
    assert system.get_bus_gain("music") == 0.0


def test_clearing_clip_stops_voice(project) -> None:
    scene, _system, _backend = _world(project)
    source = _source(scene, autoplay=True, loop=True)
    scene.update(0.016)
    assert source.is_playing
    source.clip = None
    scene.update(0.016)
    assert not source.is_playing


def test_inactive_object_is_silent(project) -> None:
    scene, _system, _backend = _world(project)
    source = _source(scene, autoplay=True, loop=True)
    scene.update(0.016)
    source.game_object.active_self = False
    scene.update(0.016)
    assert not source.is_playing


def test_missing_clip_does_not_spam(project, caplog) -> None:
    scene, _system, _backend = _world(project)
    obj = scene.create_game_object("Bad")
    obj.add_component(
        AudioSourceComponent(clip=AssetRef(path="nope.wav"), autoplay=True)
    )
    for _ in range(5):
        scene.update(0.016)
    assert len([r for r in caplog.records if "nope.wav" in r.getMessage()]) <= 1


def test_scene_roundtrip_keeps_format(project) -> None:
    scene, _system, _backend = _world(project)
    _source(scene, gain=0.5, bus="voice", autoplay=True)
    scene.create_game_object("Ear").add_component(AudioListenerComponent())
    scene.update(0.016)  # play_state="playing" не должен попасть в файл
    data = scene_to_dict(scene)
    comps = [c for o in data["objects"] for c in o["components"]]
    saved = next(c for c in comps if c["type"] == "AudioSourceComponent")
    assert "play_state" not in saved and saved["bus"] == "voice"
    assert any(c["type"] == "AudioListenerComponent" for c in comps)
    loaded = scene_from_dict(data)
    again = next(
        c
        for o in loaded.query(AudioSourceComponent)
        for c in o.get_components(AudioSourceComponent)
    )
    assert again.gain == 0.5 and again.bus == "voice" and again.autoplay
    assert again.clip.path == "beep.wav"


def test_script_audio_play_and_buses(project) -> None:
    scene, _system, backend = _world(project)
    script = scene.create_game_object("S").add_component(
        ScriptComponent(source="pass", auto_start=False)
    )
    handle = script.audio.play("beep.wav", gain=0.5, bus="ui")
    assert handle.is_playing
    assert backend.mixer.final_gain(handle.id) == pytest.approx(0.5)
    script.audio.set_bus_gain("ui", 0.5)
    assert backend.mixer.final_gain(handle.id) == pytest.approx(0.25)
    script.audio.master_gain = 0.0
    assert backend.mixer.final_gain(handle.id) == 0.0
    assert script.audio.master_gain == 0.0
    script.audio.stop(handle)
    assert not handle.is_playing
    with pytest.raises(ValueError):
        script.audio.play("beep.wav", bus="nope")


def test_script_audio_without_system_is_noop() -> None:
    scene = Scene()
    script = scene.create_game_object("S").add_component(
        ScriptComponent(source="pass", auto_start=False)
    )
    handle = script.audio.play("beep.wav")
    assert not handle.is_playing
    script.audio.set_bus_gain("music", 0.3)
    assert script.audio.get_bus_gain("music") == pytest.approx(0.3)


def test_settings_file_roundtrip(tmp_path: Path, project) -> None:
    path = tmp_path / "Settings" / "audio.json"
    assert load_audio_settings(path) == {}
    save_audio_settings(path, {"music": 0.4, "master": 0.7})
    assert load_audio_settings(path) == {"music": 0.4, "master": 0.7}
    path.write_text('{"music": 5, "bogus": 1, "sfx": "x"}', encoding="utf-8")
    assert load_audio_settings(path) == {"music": 1.0}
    path.write_text("not json", encoding="utf-8")
    assert load_audio_settings(path) == {}
    _scene, system, backend = _world(project)
    system.apply_settings({"music": 0.4, "master": 0.7, "bogus": 1.0})
    assert backend.mixer.bus_gains["music"] == 0.4 and backend.mixer.master == 0.7


def test_update_cost_of_32_voices(project) -> None:
    scene, system, backend = _world(project)
    for _ in range(32):
        _source(scene, autoplay=True, loop=True)
    scene.update(0.016)
    assert backend.mixer.voice_count == 32
    start = time.perf_counter()
    for _ in range(200):
        system.on_update(scene, 0.016)
    per_frame_ms = (time.perf_counter() - start) / 200 * 1000
    print(f"AudioSystem.update 32 голоса: {per_frame_ms:.3f} мс/кадр")
    assert per_frame_ms <= 0.3
