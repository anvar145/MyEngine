"""AudioClip: декодирование WAV/OGG, приведение формата, LRU-кэш (A9a п.2)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from engine.audio import (
    AudioClip,
    AudioDecodeError,
    ClipCache,
    decode_audio,
    decode_wav,
    load_audio_clip,
)
from engine.core.resources import AssetDatabase
from engine.core.resources.asset_ref import AssetRef
from engine.tests.audio_kit import make_wav

FIXTURES = Path(__file__).parent / "fixtures"


def _samples(clip: AudioClip) -> np.ndarray:
    return np.frombuffer(clip.pcm, dtype=np.int16)


@pytest.mark.parametrize(
    ("bits", "float_format", "expected"),
    [(8, False, 0.5), (16, False, 0.5), (24, False, 0.5), (32, False, 0.5),
     (32, True, 0.5), (64, True, 0.5)],
)
def test_wav_formats_decode_to_int16(bits: int, float_format: bool, expected: float):
    clip = decode_wav(
        make_wav(0.1, rate=22050, channels=2, bits=bits, float_format=float_format)
    )
    assert (clip.sample_rate, clip.channels, clip.frames) == (22050, 2, 2205)
    assert clip.duration == pytest.approx(0.1)
    assert _samples(clip).mean() / 32767 == pytest.approx(expected, abs=0.02)


def test_wav_duration_100ms() -> None:
    clip = decode_audio(make_wav(0.1))
    assert clip.frames == 4410 and clip.channels == 1


def test_bad_data_raises() -> None:
    with pytest.raises(AudioDecodeError):
        decode_audio(b"nonsense")
    with pytest.raises(AudioDecodeError):
        decode_wav(b"RIFF\x00\x00\x00\x00WAVE")


def test_to_format_channels_and_rate_cached() -> None:
    clip = decode_wav(make_wav(0.1, rate=22050, channels=1))
    out = clip.to_format(44100, 2)
    assert len(out) == 4410 * 2 * 2
    assert clip.to_format(44100, 2) is out
    stereo = np.frombuffer(out, dtype=np.int16).reshape(-1, 2)
    assert (stereo[:, 0] == stereo[:, 1]).all()
    same = decode_wav(make_wav(0.1, rate=44100, channels=2))
    assert same.to_format(44100, 2) is same.pcm


REAL_OGG = (
    Path(__file__).parents[2]
    / "projects/PlayModeTest/Assets/Audio/impactBell_heavy_000.ogg"
)


def test_ogg_decodes_through_backend() -> None:
    pytest.importorskip("miniaudio")
    if not REAL_OGG.is_file():
        pytest.skip("нет реального OGG в PlayModeTest")
    clip = decode_audio(REAL_OGG.read_bytes())
    assert (clip.sample_rate, clip.channels) == (44100, 2)
    assert clip.frames == 65276 and clip.duration == pytest.approx(1.48, abs=0.01)


def test_fake_ogg_header_is_decode_error_not_crash() -> None:
    pytest.importorskip("miniaudio")
    # Фикстура импорта — только заголовок без аудиоданных.
    with pytest.raises(AudioDecodeError):
        decode_audio((FIXTURES / "tiny.ogg").read_bytes())


def test_cache_is_lru_with_memory_limit() -> None:
    def clip(seconds: float) -> AudioClip:
        return decode_wav(make_wav(seconds, channels=1))

    one_second = clip(1.0).nbytes
    cache = ClipCache(limit_bytes=int(one_second * 2.5))
    loads: list[str] = []

    def load(name: str, seconds: float = 1.0) -> AudioClip:
        return cache.get(name, lambda: (loads.append(name), clip(seconds))[1])

    load("a"), load("b")
    load("a")  # a свежее b
    load("c")  # вытесняет b
    assert "a" in cache and "c" in cache and "b" not in cache
    assert loads == ["a", "b", "c"]
    assert cache.nbytes <= cache.limit_bytes


def test_load_audio_clip_from_project(tmp_path: Path) -> None:
    assets = tmp_path / "Assets"
    assets.mkdir()
    shutil.copyfile(FIXTURES / "tiny.wav", assets / "tiny.wav")
    with AssetDatabase(assets) as db:
        db.scan()
        db.reimport_dirty(wait=True)
        cache = ClipCache()
        clip = load_audio_clip(db, AssetRef(path="tiny.wav"), cache)
        assert clip is not None and clip.duration == pytest.approx(0.1, abs=0.01)
        guid = db.guid_of(assets / "tiny.wav")
        assert load_audio_clip(db, AssetRef(guid=guid), cache) is clip
        assert load_audio_clip(db, AssetRef(path="missing.wav"), cache) is None
        assert load_audio_clip(db, None, cache) is None
