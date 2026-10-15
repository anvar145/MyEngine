"""Аудио-бэкенды и фабрика :func:`create_backend`."""

from __future__ import annotations

import logging

from engine.audio.backend import AudioBackendError, IAudioBackend
from engine.audio.backends.null_backend import NullBackend
from engine.audio.backends.software_backend import SoftwareBackend

__all__ = ["NullBackend", "SoftwareBackend", "create_backend"]

_log = logging.getLogger("engine.audio")


def create_backend(
    kind: str = "auto", sample_rate: int = 44100, channels: int = 2
) -> IAudioBackend:
    """Создать и инициализировать бэкенд.

    Args:
        kind: ``"auto"`` (miniaudio, при неудаче Null с WARNING), ``"miniaudio"``
            (при неудаче исключение), ``"software"`` (микшер без устройства),
            ``"null"``.
        sample_rate: Частота дискретизации вывода.
        channels: Число каналов вывода (стерео).

    Returns:
        Готовый к ``play`` бэкенд.

    Raises:
        AudioBackendError: ``kind="miniaudio"`` и устройство недоступно.
        ValueError: Неизвестный ``kind``.
    """
    if kind == "null":
        backend: IAudioBackend = NullBackend()
    elif kind == "software":
        backend = SoftwareBackend()
    elif kind in ("auto", "miniaudio"):
        from engine.audio.backends.miniaudio_backend import MiniaudioBackend

        backend = MiniaudioBackend()
        try:
            backend.init(sample_rate, channels)
        except AudioBackendError as exc:
            if kind == "miniaudio":
                raise
            _log.warning("Звук отключён: %s", exc)
            backend = NullBackend()
        else:
            return backend
    else:
        raise ValueError(f"неизвестный аудио-бэкенд: {kind!r}")
    backend.init(sample_rate, channels)
    return backend
