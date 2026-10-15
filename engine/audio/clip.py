"""``AudioClip``: декодированный звук (int16 PCM) и LRU-кэш клипов.

Декодирование — в рантайме из байт ``audio_clip_bin`` (в ``Library``/``.pak``
лежит исходный сжатый файл): WAV разбирается здесь на numpy, OGG (и прочее) —
через ``miniaudio`` (extra ``[audio]``); без него OGG-клип не загружается
(:class:`AudioDecodeError`, вызывающий код пишет WARNING).
"""

from __future__ import annotations

import logging
import struct
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.core.resources.asset_ref import AssetRef

__all__ = [
    "CLIP_CACHE_LIMIT",
    "AudioClip",
    "AudioDecodeError",
    "ClipCache",
    "decode_audio",
    "decode_wav",
    "load_audio_clip",
]

_log = logging.getLogger("engine.audio")

#: Лимит кэша клипов по умолчанию, байт.
CLIP_CACHE_LIMIT = 64 * 1024 * 1024

_WAVE_PCM = 1
_WAVE_FLOAT = 3
_WAVE_EXTENSIBLE = 0xFFFE


class AudioDecodeError(RuntimeError):
    """Звук не удалось декодировать."""


@dataclass(eq=False)
class AudioClip:
    """Декодированный клип.

    Attributes:
        sample_rate: Частота дискретизации, Гц.
        channels: Число каналов.
        frames: Число кадров (сэмплов на канал).
        pcm: int16, interleaved.
    """

    sample_rate: int
    channels: int
    pcm: bytes
    frames: int = 0
    _formats: dict[tuple[int, int], bytes] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.channels < 1 or self.sample_rate < 1:
            raise AudioDecodeError("некорректный формат клипа")
        self.frames = len(self.pcm) // (2 * self.channels)

    @property
    def duration(self) -> float:
        """Длительность, секунды."""
        return self.frames / self.sample_rate

    @property
    def nbytes(self) -> int:
        """Занятая память: исходный PCM и сохранённые переформатирования."""
        return len(self.pcm) + sum(len(v) for v in self._formats.values())

    def to_format(self, sample_rate: int, channels: int = 2) -> bytes:
        """PCM в формате бэкенда (результат кэшируется в клипе).

        Каналы: моно дублируется, лишние каналы отбрасываются; частота
        приводится линейной интерполяцией.
        """
        key = (int(sample_rate), int(channels))
        cached = self._formats.get(key)
        if cached is not None:
            return cached
        if key == (self.sample_rate, self.channels):
            data = self.pcm
        else:
            data = _convert(self.pcm, self.channels, self.sample_rate, *key[::-1])
        self._formats[key] = data
        return data


def _convert(
    pcm: bytes, channels: int, rate: int, out_channels: int, out_rate: int
) -> bytes:
    samples = np.frombuffer(pcm, dtype=np.int16)
    frames = samples.size // channels
    block = samples[: frames * channels].reshape(frames, channels)
    if channels != out_channels:
        if channels == 1:
            block = np.repeat(block, out_channels, axis=1)
        else:
            block = block[:, :out_channels]
            if block.shape[1] < out_channels:
                block = np.repeat(block, out_channels, axis=1)[:, :out_channels]
    if rate != out_rate and frames > 1:
        count = max(1, round(frames * out_rate / rate))
        positions = np.arange(count, dtype=np.float64) * (rate / out_rate)
        lower = np.minimum(positions.astype(np.int64), frames - 1)
        upper = np.minimum(lower + 1, frames - 1)
        frac = (positions - lower).astype(np.float32)[:, None]
        mixed = block[lower].astype(np.float32) * (1.0 - frac) + block[upper].astype(
            np.float32
        ) * frac
        block = np.rint(mixed).astype(np.int16)
    return np.ascontiguousarray(block).tobytes()


# ------------------------------------------------------------------ WAV
def decode_wav(data: bytes) -> AudioClip:
    """Декодировать WAV (PCM 8/16/24/32 бит или float 32/64) в int16.

    Raises:
        AudioDecodeError: Не WAV, повреждён или формат не поддержан.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioDecodeError("это не WAV")
    fmt: tuple[int, int, int, int] | None = None
    raw: bytes | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        body = offset + 8
        if chunk_id == b"fmt ":
            if size < 16 or body + 16 > len(data):
                raise AudioDecodeError("обрезан блок fmt")
            tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", data, body)
            if tag == _WAVE_EXTENSIBLE and size >= 26 and body + 26 <= len(data):
                (tag,) = struct.unpack_from("<H", data, body + 24)
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            raw = data[body : body + min(size, len(data) - body)]
            break
        offset = body + size + (size & 1)
    if fmt is None or raw is None:
        raise AudioDecodeError("нет блока fmt или data")
    tag, channels, rate, bits = fmt
    if channels == 0 or rate == 0:
        raise AudioDecodeError("некорректные параметры fmt")
    return AudioClip(rate, channels, _to_int16(raw, tag, bits, channels))


def _to_int16(raw: bytes, tag: int, bits: int, channels: int) -> bytes:
    width = bits // 8
    usable = len(raw) - len(raw) % (width * channels) if width else 0
    raw = raw[:usable]
    if tag == _WAVE_PCM and bits == 16:
        return raw
    if tag == _WAVE_PCM and bits == 8:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
    elif tag == _WAVE_PCM and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        samples = (b[:, 1].astype(np.int16)) | (b[:, 2].astype(np.int16) << 8)
    elif tag == _WAVE_PCM and bits == 32:
        samples = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int16)
    elif tag == _WAVE_FLOAT and bits in (32, 64):
        floats = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8")
        samples = np.rint(np.clip(floats, -1.0, 1.0) * 32767.0).astype(np.int16)
    else:
        raise AudioDecodeError(f"формат WAV не поддержан (tag={tag}, bits={bits})")
    return samples.astype("<i2").tobytes()


# ------------------------------------------------------------ общий вход
def decode_audio(data: bytes) -> AudioClip:
    """Декодировать WAV или OGG по сигнатуре.

    Raises:
        AudioDecodeError: Формат неизвестен, повреждён или нет декодера.
    """
    if data[:4] == b"RIFF":
        return decode_wav(data)
    if data[:4] == b"OggS":
        return _decode_with_backend(data)
    raise AudioDecodeError("неизвестный аудиоформат")


def _decode_with_backend(data: bytes) -> AudioClip:
    # Лениво: ядро не требует miniaudio, пока не встретился OGG.
    from engine.audio.backend import AudioBackendError
    from engine.core.resources.importers.audio_importer import probe_audio_bytes

    try:
        from engine.audio.backends.miniaudio_backend import decode_with_miniaudio

        info = probe_audio_bytes(data)
        channels = min(2, max(1, int(info["channels"])))
        rate = int(info["sample_rate"])
        return AudioClip(rate, channels, decode_with_miniaudio(data, rate, channels))
    except AudioBackendError as exc:
        raise AudioDecodeError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - проба заголовка выбросила своё
        raise AudioDecodeError(f"не удалось декодировать OGG: {exc}") from exc


# ------------------------------------------------------------------ кэш
class ClipCache:
    """LRU-кэш клипов по GUID с лимитом по памяти.

    Args:
        limit_bytes: Предельный суммарный ``AudioClip.nbytes``.
    """

    def __init__(self, limit_bytes: int = CLIP_CACHE_LIMIT) -> None:
        self.limit_bytes = int(limit_bytes)
        self._items: OrderedDict[str, AudioClip] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, guid: str) -> bool:
        return guid in self._items

    @property
    def nbytes(self) -> int:
        """Суммарная занятая память."""
        with self._lock:
            return sum(c.nbytes for c in self._items.values())

    def get(self, guid: str, loader: Callable[[], AudioClip]) -> AudioClip:
        """Клип из кэша или ``loader()`` (результат кладётся в кэш)."""
        with self._lock:
            clip = self._items.get(guid)
            if clip is not None:
                self._items.move_to_end(guid)
                return clip
        clip = loader()
        with self._lock:
            self._items[guid] = clip
            self._items.move_to_end(guid)
            self._evict()
        return clip

    def _evict(self) -> None:
        total = sum(c.nbytes for c in self._items.values())
        while total > self.limit_bytes and len(self._items) > 1:
            _, dropped = self._items.popitem(last=False)
            total -= dropped.nbytes

    def clear(self) -> None:
        """Очистить кэш."""
        with self._lock:
            self._items.clear()


def load_audio_clip(
    vfs: Any, ref: AssetRef | str | dict | None, cache: ClipCache | None = None
) -> AudioClip | None:
    """Загрузить клип по ссылке из VFS (``audio_clip_bin``).

    Args:
        vfs: ``VFS``, ``AssetDatabase`` или ``PakReader``.
        ref: ``AssetRef`` (или то, что принимает ``AssetRef.coerce``).
        cache: Кэш клипов; без него клип декодируется каждый раз.

    Returns:
        Клип или ``None``, если ссылка пуста/битая либо декодирование
        не удалось (причина — WARNING в логе).
    """
    asset_ref = AssetRef.coerce(ref)
    if asset_ref is None:
        return None
    from engine.core.resources.vfs import as_assets, is_vfs

    guid = asset_ref.resolve(vfs)
    if guid is None:
        _log.warning("Аудиоклип не найден: %s", asset_ref.path or asset_ref.guid)
        return None

    def read() -> AudioClip:
        data = vfs.read(guid, "audio_clip_bin") if is_vfs(vfs) else None
        if data is None:
            handle = as_assets(vfs).get(guid)
            path = handle.artifact_path("audio_clip_bin") if handle else None
            data = path.read_bytes() if path is not None else None
        if data is None:
            raise AudioDecodeError("в ассете нет артефакта audio_clip_bin")
        return decode_audio(data)

    try:
        return cache.get(guid, read) if cache is not None else read()
    except AudioDecodeError as exc:
        _log.warning("Аудиоклип %s не загружен: %s", asset_ref.path or guid, exc)
        return None
