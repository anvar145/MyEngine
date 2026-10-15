"""Бэкенд на ``miniaudio`` (extra ``[audio]``): устройство + программный микшер.

Устройство тянет блоки PCM из аудио-потока miniaudio; микшер
(:class:`~engine.audio.mixer.Mixer`) работает под блокировкой, главный поток
лишь добавляет/останавливает голоса.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from engine.audio.backend import AudioBackendError
from engine.audio.backends.software_backend import SoftwareBackend

__all__ = ["MiniaudioBackend", "decode_with_miniaudio"]

_BUFFER_MSEC = 40


def _import() -> Any:
    try:
        import miniaudio
    except ImportError as exc:  # pragma: no cover - зависит от окружения
        raise AudioBackendError("miniaudio не установлен (extra [audio])") from exc
    return miniaudio


def decode_with_miniaudio(data: bytes, sample_rate: int, channels: int) -> bytes:
    """Декодировать WAV/OGG/MP3/FLAC в int16 interleaved заданного формата.

    Raises:
        AudioBackendError: ``miniaudio`` нет или формат не распознан.
    """
    miniaudio = _import()
    try:
        decoded = miniaudio.decode(
            data,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=channels,
            sample_rate=sample_rate,
        )
    except Exception as exc:  # miniaudio.DecodeError и ошибки cffi
        raise AudioBackendError(f"не удалось декодировать звук: {exc}") from exc
    return bytes(decoded.samples)


class MiniaudioBackend(SoftwareBackend):
    """Вывод на аудиоустройство по умолчанию."""

    def __init__(self) -> None:
        super().__init__()
        self._device: Any = None

    def init(self, sample_rate: int = 44100, channels: int = 2) -> None:
        """Открыть устройство вывода и запустить аудио-поток.

        Raises:
            AudioBackendError: Нет пакета или устройства.
        """
        super().init(sample_rate, channels)
        miniaudio = _import()
        try:
            device = miniaudio.PlaybackDevice(
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=2,
                sample_rate=self.sample_rate,
                buffersize_msec=_BUFFER_MSEC,
            )
            generator = self._stream()
            next(generator)
            device.start(generator)
        except Exception as exc:
            self.initialized = False
            raise AudioBackendError(f"нет устройства вывода: {exc}") from exc
        self._device = device

    def _stream(self) -> Generator[bytes, int, None]:
        frames = yield b""
        while True:
            frames = yield self.mixer.render(int(frames))

    def shutdown(self) -> None:
        """Остановить поток и закрыть устройство."""
        device, self._device = self._device, None
        if device is not None:
            try:
                device.stop()
                device.close()
            except Exception:  # noqa: BLE001 - при выходе ошибка устройства не важна
                pass
        super().shutdown()
