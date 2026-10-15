"""Импортёр аудио: WAV и OGG (Vorbis/Opus).

Звук на импорте **не декодируется**: проверяется заголовок, а байты кладутся в
``Library`` как ``audio_clip_bin`` — декодирование выполняет аудио-бэкенд
(extra ``[audio]``).
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path
from typing import Any

from engine.core.resources.errors import AssetImportError
from engine.core.resources.importer import Artifact, AssetPreview, ImportContext

__all__ = ["AudioImporter", "probe_audio", "probe_audio_bytes"]

_OGG_TAIL = 65536


def probe_audio(path: Path) -> dict[str, Any]:
    """Прочитать параметры аудиофайла по заголовку.

    Args:
        path: Путь к ``.wav`` или ``.ogg``.

    Returns:
        Словарь ``format``, ``channels``, ``sample_rate`` и ``duration``
        (секунды или ``None``, если длительность не удалось определить).

    Raises:
        AssetImportError: Если формат не распознан или заголовок повреждён.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AssetImportError(f"Не удалось прочитать '{path}': {exc}") from exc
    return probe_audio_bytes(data, path)


def probe_audio_bytes(data: bytes, path: Path | str = "audio") -> dict[str, Any]:
    """Прочитать параметры аудио по байтам (те же поля, что у :func:`probe_audio`).

    Args:
        data: Содержимое ``.wav``/``.ogg``.
        path: Имя для сообщений об ошибках.

    Raises:
        AssetImportError: Если формат не распознан или заголовок повреждён.
    """
    path = Path(path)
    if data.startswith(b"RIFF"):
        return _probe_wav(data, path)
    if data.startswith(b"OggS"):
        return _probe_ogg(data, path)
    raise AssetImportError(f"'{path.name}': неизвестный аудиоформат")


def _probe_wav(data: bytes, path: Path) -> dict[str, Any]:
    """Пройти по RIFF-блокам: ``fmt `` задаёт параметры, ``data`` — длительность."""
    if len(data) < 12 or data[8:12] != b"WAVE":
        raise AssetImportError(f"'{path.name}': это RIFF, но не WAVE")
    fmt: tuple[int, int, int, int] | None = None
    data_bytes: int | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        (chunk_size,) = struct.unpack_from("<I", data, offset + 4)
        body = offset + 8
        if chunk_id == b"fmt ":
            if chunk_size < 16 or body + 16 > len(data):
                raise AssetImportError(f"'{path.name}': обрезан блок fmt")
            tag, channels, rate, _byte_rate, _align, bits = struct.unpack_from(
                "<HHIIHH", data, body
            )
            fmt = (channels, rate, bits, tag)
        elif chunk_id == b"data":
            data_bytes = min(chunk_size, len(data) - body)
            break
        offset = body + chunk_size + (chunk_size & 1)
    if fmt is None:
        raise AssetImportError(f"'{path.name}': нет блока fmt")
    if data_bytes is None:
        raise AssetImportError(f"'{path.name}': нет блока data")
    channels, rate, bits, _tag = fmt
    if channels == 0 or rate == 0 or bits == 0:
        raise AssetImportError(f"'{path.name}': некорректные параметры fmt")
    frame_bytes = channels * ((bits + 7) // 8)
    return {
        "format": "wav",
        "channels": channels,
        "sample_rate": rate,
        "bits_per_sample": bits,
        "duration": data_bytes / (frame_bytes * rate),
    }


def _probe_ogg(data: bytes, path: Path) -> dict[str, Any]:
    """Прочитать идентификационный пакет Vorbis/Opus из первой страницы."""
    if len(data) < 27 or data[4] != 0 or not data[5] & 0x02:
        raise AssetImportError(f"'{path.name}': повреждена первая страница OGG")
    segments = data[26]
    start = 27 + segments
    packet = data[start : start + 64]
    if packet.startswith(b"\x01vorbis") and len(packet) >= 16:
        channels = packet[11]
        (rate,) = struct.unpack_from("<I", packet, 12)
        codec = "vorbis"
    elif packet.startswith(b"OpusHead") and len(packet) >= 12:
        channels = packet[9]
        rate = 48000  # Opus всегда декодируется в 48 кГц
        codec = "opus"
    else:
        raise AssetImportError(f"'{path.name}': OGG без потока Vorbis/Opus")
    if channels == 0 or rate == 0:
        raise AssetImportError(f"'{path.name}': некорректный заголовок {codec}")
    return {
        "format": "ogg",
        "codec": codec,
        "channels": channels,
        "sample_rate": rate,
        "duration": _ogg_duration(data, rate),
    }


def _ogg_duration(data: bytes, rate: int) -> float | None:
    """Длительность по granule position последней страницы (``None``, если нет)."""
    tail_start = max(0, len(data) - _OGG_TAIL)
    index = data.rfind(b"OggS", tail_start)
    if index < 0 or index + 14 > len(data):
        return None
    (granule,) = struct.unpack_from("<q", data, index + 6)
    return granule / rate if granule > 0 else None


class AudioImporter:
    """``.wav``/``.ogg`` → артефакт ``audio_clip_bin`` (исходные байты)."""

    name = "audio"
    extensions = (".wav", ".ogg")
    version = 1

    def default_settings(self) -> dict[str, Any]:
        """Настройки по умолчанию."""
        return {"stream": False}

    def inspect(self, path: Path) -> AssetPreview:
        """Описать клип по заголовку."""
        return AssetPreview("audio", probe_audio(path))

    def import_asset(self, ctx: ImportContext) -> list[Artifact]:
        """Проверить заголовок и сохранить байты клипа в ``clip.bin``."""
        info = probe_audio(ctx.source)
        shutil.copyfile(ctx.source, ctx.output_dir / "clip.bin")
        return [
            ctx.artifact(
                "audio_clip_bin",
                "clip.bin",
                stream=bool(ctx.setting("stream", False)),
                **info,
            )
        ]
