"""A9b: 3D-звук, слушатель, музыка, SoundCue."""

from __future__ import annotations

import json
import logging
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from engine.audio import (
    Attenuation,
    AudioListenerComponent,
    AudioSourceComponent,
    AudioSystem,
    SoftwareBackend,
    SoundCue,
    attenuation,
    parse_cue,
    parse_playlist,
)
from engine.audio.asset_json import AudioAssetError
from engine.audio.cue import CueClip
from engine.audio.spatial import pan_gains, pan_of
from engine.core.math3d import Quaternion, Vector3
from engine.core.resources import AssetDatabase
from engine.core.resources.asset_ref import AssetRef
from engine.core.scene import Scene
from engine.rendering.camera import CameraComponent
from engine.tests.audio_kit import make_wav


def _ref(name: str) -> dict:
    return {"path": name}


@pytest.fixture
def project(tmp_path: Path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    for name, value, seconds in (
        ("a.wav", 0.2, 0.5),
        ("b.wav", 0.3, 0.5),
        ("c.wav", 0.4, 0.5),
        ("long.wav", 0.5, 3.0),
        ("long2.wav", 0.5, 3.0),
        ("stem.wav", 0.5, 3.0),
    ):
        (assets / name).write_bytes(make_wav(seconds, value=value))
    (assets / "hit.cue.json").write_text(
        json.dumps(
            {
                "clips": [{"clip": _ref(n)} for n in ("a.wav", "b.wav", "c.wav")],
                "avoid_repeat": True,
            }
        ),
        encoding="utf-8",
    )
    (assets / "theme.music.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "clip": _ref("long.wav"),
                        "stem": {"clip": _ref("stem.wav")},
                        "fade_in": 0.0,
                    }
                ],
                "loop": True,
            }
        ),
        encoding="utf-8",
    )
    (assets / "other.music.json").write_text(
        json.dumps({"tracks": [{"clip": _ref("long2.wav"), "fade_in": 1.0}]}),
        encoding="utf-8",
    )
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


def _channels(pcm: bytes) -> tuple[float, float]:
    data = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
    return float(np.abs(data[:, 0]).max()), float(np.abs(data[:, 1]).max())


def _listener(scene: Scene, position=(0.0, 0.0, 0.0)):
    obj = scene.create_game_object("Ears", position=Vector3(*position))
    obj.add_component(AudioListenerComponent())
    return obj


# ----------------------------------------------------------------- формулы
def test_attenuation_linear_endpoints() -> None:
    assert attenuation(0.5, 1.0, 50.0) == pytest.approx(1.0)
    assert attenuation(1.0, 1.0, 50.0) == pytest.approx(1.0)
    assert attenuation(25.5, 1.0, 50.0) == pytest.approx(0.5, abs=0.01)
    assert attenuation(100.0, 1.0, 50.0) == pytest.approx(0.0)


def test_attenuation_models() -> None:
    assert attenuation(4.0, 1.0, 50.0, "inverse") == pytest.approx(0.25)
    expected = 1.0 / (1.0 + 2.0 * np.log(4.0))
    assert attenuation(4.0, 1.0, 50.0, "logarithmic", 2.0) == pytest.approx(expected)
    # логарифмическая и обратная — монотонно убывают
    d = np.linspace(1.0, 50.0, 20)
    for model in ("logarithmic", "inverse", "linear"):
        values = attenuation(d, 1.0, 50.0, model, 1.0)
        assert np.all(np.diff(values) <= 1e-12)


def test_pan_right_source_is_right() -> None:
    right = np.array([1.0, 0.0, 0.0])
    pans = pan_of(np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]), right)
    assert pans[0] == pytest.approx(1.0)
    assert pans[1] == pytest.approx(-1.0)
    assert pans[2] == pytest.approx(0.0, abs=1e-9)
    left_gain, right_gain = pan_gains(1.0)
    assert right_gain > left_gain


# ----------------------------------------------------------------- 3D-микшер
def test_play_at_right_is_louder_in_right_channel(project) -> None:
    scene, system, backend = _world(project)
    _listener(scene)
    scene.update(0.016)
    system.play_at(AssetRef(path="a.wav"), (10.0, 0.0, 0.0))
    backend.advance(0.01)  # рампа
    left, right = _channels(backend.advance(0.01))
    assert right > left * 5


def test_play_at_left_is_louder_in_left_channel(project) -> None:
    scene, system, backend = _world(project)
    _listener(scene)
    scene.update(0.016)
    system.play_at(AssetRef(path="a.wav"), (-10.0, 0.0, 0.0))
    backend.advance(0.01)
    left, right = _channels(backend.advance(0.01))
    assert left > right * 5


def test_distance_attenuates_and_far_is_silent(project) -> None:
    scene, system, backend = _world(project)
    _listener(scene)
    scene.update(0.016)
    near = system.play_at(AssetRef(path="a.wav"), (0.0, 0.0, 0.5))
    assert backend.mixer.spatial_of(near.id)[0] == pytest.approx(1.0)
    near.stop()
    far = system.play_at(AssetRef(path="a.wav"), (0.0, 0.0, 100.0))
    assert backend.mixer.spatial_of(far.id)[0] == pytest.approx(0.0)


def test_listener_orientation_flips_pan(project) -> None:
    scene, system, backend = _world(project)
    ears = _listener(scene)
    ears.transform.rotation = Quaternion.from_axis_angle(Vector3(0, 1, 0), 180.0)
    scene.update(0.016)
    handle = system.play_at(AssetRef(path="a.wav"), (10.0, 0.0, 0.0))
    assert backend.mixer.spatial_of(handle.id)[1] == pytest.approx(-1.0, abs=1e-6)


def test_moving_source_updates_position(project) -> None:
    scene, _system, backend = _world(project)
    _listener(scene)
    obj = scene.create_game_object("Src", position=Vector3(10, 0, 0))
    source = obj.add_component(
        AudioSourceComponent(
            clip=AssetRef(path="long.wav"), spatial=True, loop=True, autoplay=True
        )
    )
    scene.update(0.016)
    assert backend.mixer.spatial_of(source._voice.id)[1] == pytest.approx(1.0)
    obj.transform.position = Vector3(-10, 0, 0)
    scene.update(0.016)
    assert backend.mixer.spatial_of(source._voice.id)[1] == pytest.approx(-1.0)


def test_source_attenuation_fields_reach_mixer(project) -> None:
    scene, _system, backend = _world(project)
    _listener(scene)
    obj = scene.create_game_object("Src", position=Vector3(0, 0, 10))
    source = obj.add_component(
        AudioSourceComponent(
            clip=AssetRef(path="long.wav"),
            spatial=True,
            loop=True,
            autoplay=True,
            min_distance=1.0,
            max_distance=10.0,
        )
    )
    scene.update(0.016)
    assert backend.mixer.spatial_of(source._voice.id)[0] == pytest.approx(0.0)


# ----------------------------------------------------------------- слушатель
def test_listener_follows_camera_without_component(project) -> None:
    scene, system, _backend = _world(project)
    cam = scene.create_game_object("Cam", position=Vector3(3, 4, 5))
    cam.add_component(CameraComponent())
    pose = system.find_listener(scene)
    assert pose is not None
    assert pose[0] == pytest.approx((3.0, 4.0, 5.0))
    cam.transform.position = Vector3(7, 0, 0)
    assert system.find_listener(scene)[0] == pytest.approx((7.0, 0.0, 0.0))


def test_listener_component_wins_over_camera(project) -> None:
    scene, system, _backend = _world(project)
    cam = scene.create_game_object("Cam", position=Vector3(1, 1, 1))
    cam.add_component(CameraComponent())
    _listener(scene, (9.0, 0.0, 0.0))
    assert system.find_listener(scene)[0] == pytest.approx((9.0, 0.0, 0.0))


def test_inactive_listener_is_skipped(project) -> None:
    scene, system, _backend = _world(project)
    ears = _listener(scene, (9.0, 0.0, 0.0))
    ears.get_component(AudioListenerComponent).active = False
    assert system.find_listener(scene) is None


def test_no_listener_warns_once(project, caplog) -> None:
    _scene, system, _backend = _world(project)
    with caplog.at_level(logging.WARNING, logger="engine.audio.system"):
        system.play_at(AssetRef(path="a.wav"), (1.0, 0.0, 0.0))
        system.play_at(AssetRef(path="a.wav"), (2.0, 0.0, 0.0))
    warnings = [r for r in caplog.records if "слушателя" in r.getMessage()]
    assert len(warnings) == 1


# ----------------------------------------------------------------- музыка
def _advance(scene: Scene, backend: SoftwareBackend, seconds: float) -> None:
    step = 0.05
    for _ in range(round(seconds / step)):
        scene.update(step)
        backend.advance(step)


def test_music_plays_and_reports_track(project) -> None:
    scene, system, _backend = _world(project)
    assert system.music.current_track is None
    assert system.music.play(AssetRef(path="theme.music.json"))
    assert system.music.is_playing
    assert system.music.current_track is not None
    assert system.music.current_index == 0
    system.music.stop(0.0)
    assert not system.music.is_playing
    assert system.music.current_track is None
    scene.update(0.016)


def test_music_crossfade_silences_old_track(project) -> None:
    scene, system, backend = _world(project)
    system.music.play(AssetRef(path="theme.music.json"), crossfade=0.0)
    old_layers = list(system.music._layers)  # noqa: SLF001
    scene.update(0.016)
    system.music.play(AssetRef(path="other.music.json"), crossfade=1.0)
    _advance(scene, backend, 1.2)
    assert all(not layer.voice.is_playing for layer in old_layers)
    assert system.music.current_track is not None
    assert system.music.current_track.clip.path == "long2.wav"


def test_music_intensity_controls_stem_gain(project) -> None:
    scene, system, backend = _world(project)
    system.music.play(AssetRef(path="theme.music.json"), crossfade=0.0)
    layer = system.music._layers[0]  # noqa: SLF001
    assert layer.stem_voice is not None
    scene.update(0.016)
    assert backend.mixer.final_gain(layer.stem_voice.id) == pytest.approx(0.0)
    system.music.set_intensity(0.5)
    scene.update(0.016)
    assert backend.mixer.final_gain(layer.stem_voice.id) == pytest.approx(0.5)
    system.music.set_intensity(7.0)  # зажимается в [0, 1]
    assert system.music.intensity == 1.0


def test_music_missing_playlist_is_safe(project) -> None:
    _scene, system, _backend = _world(project)
    assert not system.music.play(AssetRef(path="nope.music.json"))
    assert not system.music.is_playing


def test_music_stops_with_system(project) -> None:
    scene, system, backend = _world(project)
    system.music.play(AssetRef(path="theme.music.json"), crossfade=0.0)
    scene.remove_system(system)
    assert backend.mixer.voice_count == 0


def test_music_playlist_advances_to_next_track(project) -> None:
    scene, system, backend = _world(project)
    playlist = parse_playlist(
        {
            "tracks": [
                {"clip": _ref("a.wav"), "fade_in": 0.0, "fade_out": 0.1},
                {"clip": _ref("b.wav"), "fade_in": 0.1, "fade_out": 0.1},
            ],
            "loop": True,
        }
    )
    system.music.play(playlist, crossfade=0.0)
    assert system.music.current_index == 0
    _advance(scene, backend, 0.45)  # a.wav 0.5 c: переход за fade_out
    assert system.music.current_index == 1


# ----------------------------------------------------------------- SoundCue
def test_cue_100_calls_play_all_clips_without_repeats(project) -> None:
    scene, system, backend = _world(project)
    scene.seed_random(7)
    cue = AssetRef(path="hit.cue.json")
    seen = []
    for _ in range(100):
        handle = system.play_cue(cue)
        assert handle.is_playing
        voices = backend.mixer._voices  # noqa: SLF001
        value = int(voices[handle.id].pcm[0, 0])
        seen.append(value)
        handle.stop()
    assert len(set(seen)) == 3
    assert all(a != b for a, b in pairwise(seen))


def test_cue_is_deterministic_for_seed(project) -> None:
    def run() -> list[int]:
        scene, system, backend = _world(project)
        scene.seed_random(123)
        values = []
        for _ in range(20):
            handle = system.play_cue(AssetRef(path="hit.cue.json"))
            values.append(int(backend.mixer._voices[handle.id].pcm[0, 0]))  # noqa: SLF001
            handle.stop()
        return values

    assert run() == run()


def test_cue_on_source_picks_variant(project) -> None:
    scene, _system, backend = _world(project)
    obj = scene.create_game_object("Src")
    source = obj.add_component(
        AudioSourceComponent(cue=AssetRef(path="hit.cue.json"), autoplay=True)
    )
    scene.update(0.016)
    assert source.is_playing
    assert backend.mixer.voice_count == 1


def test_cue_variation_ranges() -> None:
    import random

    cue = SoundCue(
        (CueClip(AssetRef(path="a.wav")),), randomize_gain=0.2, randomize_pitch=0.1
    )
    rng = random.Random(1)
    for _ in range(200):
        variant = cue.pick(rng)
        assert 0.8 <= variant.gain <= 1.2
        assert 0.9 <= variant.pitch <= 1.1


def test_cue_weights_respected() -> None:
    import random

    cue = parse_cue(
        {
            "clips": [
                {"clip": _ref("a.wav"), "weight": 0.0},
                {"clip": _ref("b.wav"), "weight": 1.0},
            ],
            "avoid_repeat": False,
        }
    )
    rng = random.Random(0)
    assert {cue.pick(rng).index for _ in range(50)} == {1}


# ----------------------------------------------------------------- парсеры
@pytest.mark.parametrize(
    "data",
    [{}, {"clips": []}, {"clips": [{"gain": 1}]}, {"clips": [{"clip": "a", "weight": 0}]}],
)
def test_parse_cue_rejects_bad_data(data) -> None:
    with pytest.raises(AudioAssetError):
        parse_cue(data)


@pytest.mark.parametrize("data", [{}, {"tracks": []}, {"tracks": [{"gain": 1}]}])
def test_parse_playlist_rejects_bad_data(data) -> None:
    with pytest.raises(AudioAssetError):
        parse_playlist(data)


def test_cue_and_music_assets_are_imported(project) -> None:
    for name in ("hit.cue.json", "theme.music.json"):
        guid = project.guid_for_path(name) if hasattr(project, "guid_for_path") else None
        if guid is None:
            guid = AssetRef(path=name).resolve(project)
        assert guid is not None


def test_scene_roundtrip_keeps_new_fields() -> None:
    source = AudioSourceComponent(
        spatial=True,
        min_distance=2.0,
        max_distance=30.0,
        falloff="inverse",
        rolloff=2.0,
        cue=AssetRef(path="hit.cue.json"),
    )
    restored = AudioSourceComponent.from_dict(source.to_dict())
    assert (restored.min_distance, restored.max_distance) == (2.0, 30.0)
    assert (restored.falloff, restored.rolloff) == ("inverse", 2.0)
    assert restored.cue is not None and restored.cue.path == "hit.cue.json"
    assert AudioListenerComponent.from_dict({"active": False}).active is False


def test_attenuation_is_hashable_for_grouping() -> None:
    assert len({Attenuation(), Attenuation(), Attenuation(2.0)}) == 2


# ----------------------------------------------------------------- скрипты
def _script(scene: Scene):
    from engine.scripting import ScriptComponent

    return scene.create_game_object("S").add_component(
        ScriptComponent(source="pass", auto_start=False)
    )


def test_script_play_at_and_cue(project) -> None:
    scene, _system, backend = _world(project)
    _listener(scene)
    scene.update(0.016)
    script = _script(scene)
    handle = script.audio.play_at("a.wav", Vector3(10, 0, 0))
    assert handle.is_playing
    assert backend.mixer.spatial_of(handle.id)[1] == pytest.approx(1.0)
    flat = script.audio.play_cue("hit.cue.json", gain=0.5)
    assert flat.is_playing and backend.mixer.spatial_of(flat.id) is None
    placed = script.audio.play_cue("hit.cue.json", position=(0.0, 0.0, 5.0))
    assert backend.mixer.spatial_of(placed.id) is not None
    with pytest.raises(ValueError):
        script.audio.play_at("a.wav", (0, 0, 0), bus="nope")


def test_script_music_facade(project) -> None:
    scene, system, _backend = _world(project)
    music = _script(scene).audio.music
    assert music.current_track is None
    assert music.play("theme.music.json", crossfade=0.0)
    assert music.current_track is not None
    music.set_intensity(0.3)
    assert system.music.intensity == pytest.approx(0.3)
    music.stop(0.0)
    assert music.current_track is None


def test_script_audio3d_without_system_is_noop() -> None:
    scene = Scene()
    script = _script(scene)
    assert not script.audio.play_at("a.wav", (0, 0, 0)).is_playing
    assert not script.audio.play_cue("hit.cue.json").is_playing
    assert script.audio.music.play("theme.music.json") is False
    script.audio.music.stop()
    script.audio.music.set_intensity(1.0)


# ----------------------------------------------------------------- стоимость
def test_update_cost_of_3d_music_and_cue(project) -> None:
    import time

    scene, system, backend = _world(project)
    _listener(scene)
    for i in range(32):
        obj = scene.create_game_object("S", position=Vector3(i - 16, 0, 5))
        obj.add_component(
            AudioSourceComponent(
                clip=AssetRef(path="long.wav"), spatial=True, loop=True, autoplay=True
            )
        )
    system.music.play(AssetRef(path="theme.music.json"), crossfade=0.0)
    cue_obj = scene.create_game_object("Cue")
    cue_obj.add_component(
        AudioSourceComponent(cue=AssetRef(path="hit.cue.json"), autoplay=True)
    )
    scene.update(0.016)
    assert backend.mixer.voice_count >= 34
    runs = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(200):
            system.on_update(scene, 0.016)
        runs.append((time.perf_counter() - start) / 200 * 1000)
    update_ms = min(runs)
    block = round(0.016 * backend.sample_rate)
    start = time.perf_counter()
    for _ in range(100):
        backend.mixer.render(block)
    mix_ms = (time.perf_counter() - start) / 100 * 1000
    print(f"A9b: update {update_ms:.3f} мс, микшер {mix_ms:.3f} мс/кадр")
    assert update_ms <= 0.5
