"""Аудио-бэкенды и микшер (A9a п.1, п.4)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.audio import (
    IAudioBackend,
    Mixer,
    NullBackend,
    SoftwareBackend,
    create_backend,
    decode_wav,
)
from engine.tests.audio_kit import make_wav


def _stereo(seconds: float = 0.1, value: float = 0.5) -> bytes:
    return decode_wav(make_wav(seconds, channels=2, value=value)).pcm


def _backend() -> SoftwareBackend:
    backend = SoftwareBackend()
    backend.init(44100, 2)
    return backend


def _peak(pcm: bytes) -> int:
    return int(np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32)).max())


def test_backends_follow_protocol() -> None:
    assert isinstance(NullBackend(), IAudioBackend)
    assert isinstance(SoftwareBackend(), IAudioBackend)


def test_null_backend_is_noop() -> None:
    backend = NullBackend()
    backend.init()
    handle = backend.play(b"\x00" * 400, gain=1.0, pan=0.0)
    assert not backend.is_playing(handle) and not handle.is_playing
    handle.stop()
    backend.set_bus_gain("music", 0.0)
    backend.shutdown()


def test_voice_ends_after_clip_duration() -> None:
    backend = _backend()
    handle = backend.play(_stereo(0.1), gain=1.0, pan=0.0)
    assert handle.is_playing
    backend.advance(0.05)
    assert handle.is_playing
    backend.advance(0.1)
    assert not handle.is_playing


def test_loop_keeps_playing_and_stop_ends() -> None:
    backend = _backend()
    handle = backend.play(_stereo(0.1), gain=1.0, pan=0.0, loop=True)
    backend.advance(0.35)
    assert handle.is_playing
    handle.stop()
    assert not handle.is_playing
    handle.stop()  # повторный stop безопасен


def test_gain_is_applied_and_changeable() -> None:
    backend = _backend()
    handle = backend.play(_stereo(1.0, 0.5), gain=1.0, pan=0.0)
    loud = _peak(backend.advance(0.01))
    handle.set_gain(0.5)
    backend.advance(0.01)  # блок рампы: коэффициент доходит до нового значения
    quiet = _peak(backend.advance(0.01))
    assert quiet == pytest.approx(loud / 2, abs=2)


def test_bus_gain_zero_silences_only_that_bus() -> None:
    backend = _backend()
    music = backend.play(_stereo(1.0), gain=1.0, pan=0.0, bus="music")
    backend.set_bus_gain("music", 0.0)
    assert backend.mixer.final_gain(music.id) == 0.0
    assert _peak(backend.advance(0.01)) == 0
    sfx = backend.play(_stereo(1.0), gain=1.0, pan=0.0, bus="sfx")
    assert backend.mixer.final_gain(sfx.id) == 1.0
    assert _peak(backend.advance(0.01)) > 0
    backend.set_bus_gain("music", 1.0)  # шина применяется к уже играющему голосу
    assert backend.mixer.final_gain(music.id) == 1.0


def test_master_and_bus_multiply_and_clamp() -> None:
    backend = _backend()
    handle = backend.play(_stereo(1.0), gain=4.0, pan=0.0, bus="sfx")
    assert backend.mixer.final_gain(handle.id) == 1.0  # clamp, не перегруз
    handle.set_gain(1.0)
    backend.set_bus_gain("sfx", 0.5)
    backend.set_master_gain(0.5)
    assert backend.mixer.final_gain(handle.id) == pytest.approx(0.25)
    backend.set_bus_gain("master", 1.0)
    assert backend.mixer.final_gain(handle.id) == pytest.approx(0.5)


def test_sum_of_voices_is_clipped_not_wrapped() -> None:
    backend = _backend()
    for _ in range(4):
        backend.play(_stereo(1.0, 0.9), gain=1.0, pan=0.0)
    out = np.frombuffer(backend.advance(0.01), dtype=np.int16)
    assert out.min() >= -32768 and (out == 32767).all()


def test_pan_hard_left_mutes_right() -> None:
    backend = _backend()
    backend.play(_stereo(1.0), gain=1.0, pan=-1.0)
    out = np.frombuffer(backend.advance(0.01), dtype=np.int16).reshape(-1, 2)
    assert out[:, 0].max() > 0 and out[:, 1].max() == 0


def test_pitch_two_halves_duration() -> None:
    backend = _backend()
    handle = backend.play(_stereo(0.2), gain=1.0, pan=0.0, pitch=2.0)
    backend.advance(0.08)
    assert handle.is_playing
    backend.advance(0.05)
    assert not handle.is_playing


def test_unknown_bus_rejected() -> None:
    mixer = Mixer()
    with pytest.raises(ValueError):
        mixer.set_bus("nope", 1.0)
    with pytest.raises(ValueError):
        mixer.add(b"\x00" * 8, gain=1.0, pan=0.0, loop=False, bus="nope", pitch=1.0)


def test_empty_pcm_gives_dead_handle() -> None:
    backend = _backend()
    assert not backend.play(b"", gain=1.0, pan=0.0).is_playing


def test_create_backend_factory() -> None:
    assert isinstance(create_backend("null"), NullBackend)
    assert isinstance(create_backend("software"), SoftwareBackend)
    with pytest.raises(ValueError):
        create_backend("nope")
