"""Программный микшер: голоса, шины, мастер -> int16 стерео.

Не зависит от устройства вывода: :class:`Mixer` отдаёт блоки PCM по запросу
(``render``). Устройство (miniaudio) вызывает ``render`` из аудио-потока, тесты —
из главного, поэтому все операции под блокировкой.

Голос с ``position`` — пространственный: затухание и панораму микшер считает
от слушателя (:meth:`Mixer.set_listener`) для всех 3D-голосов одним проходом
numpy. Коэффициенты каналов между блоками меняются плавно (линейная рампа
внутри блока), поэтому смена громкости/позиции не даёт щелчков.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from engine.audio.backend import BUSES
from engine.audio.spatial import Attenuation, attenuation, pan_gains, pan_of

__all__ = ["Mixer"]


@dataclass(slots=True)
class _Voice:
    pcm: np.ndarray  # (n, 2) int16
    gain: float
    pan: float
    loop: bool
    bus: str
    pitch: float
    pos: float = 0.0  # позиция в кадрах (дробная при pitch != 1)
    position: np.ndarray | None = None  # мировая позиция (3,) 3D-голоса
    atten: Attenuation | None = None
    last_lr: tuple[float, float] | None = None  # коэффициенты прошлого блока


class Mixer:
    """Микшер стерео-голосов фиксированного формата.

    Args:
        sample_rate: Частота дискретизации, Гц.
    """

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = int(sample_rate)
        self.master = 1.0
        self.bus_gains: dict[str, float] = dict.fromkeys(BUSES, 1.0)
        self._voices: dict[int, _Voice] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._listener_pos = np.zeros(3, dtype=np.float64)
        self._listener_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        #: ``{id: (затухание, панорама)}`` 3D-голосов на текущий блок.
        self._spatial: dict[int, tuple[float, float]] = {}

    # ------------------------------------------------------------ голоса
    def add(
        self,
        pcm: bytes,
        *,
        gain: float,
        pan: float,
        loop: bool,
        bus: str,
        pitch: float,
        position: tuple[float, float, float] | None = None,
        atten: Attenuation | None = None,
    ) -> int:
        """Добавить голос; возвращает его id (пустой PCM -> ``0``, без голоса).

        ``position`` делает голос пространственным: ``pan`` игнорируется,
        панорама и затухание считаются от слушателя на каждом блоке.
        """
        data = np.frombuffer(pcm, dtype=np.int16)
        frames = data.size // 2
        if frames == 0:
            return 0
        if bus not in self.bus_gains or bus == "master":
            raise ValueError(f"неизвестная шина: {bus!r}")
        voice = _Voice(
            data[: frames * 2].reshape(frames, 2),
            max(0.0, float(gain)),
            min(1.0, max(-1.0, float(pan))),
            bool(loop),
            bus,
            max(0.01, float(pitch)),
        )
        if position is not None:
            voice.position = np.asarray(position, dtype=np.float64).reshape(3)
            voice.atten = atten if atten is not None else Attenuation()
        with self._lock:
            voice_id = self._next_id
            self._next_id += 1
            self._voices[voice_id] = voice
        return voice_id

    def stop(self, voice_id: int) -> None:
        """Убрать голос."""
        with self._lock:
            self._voices.pop(voice_id, None)

    def set_gain(self, voice_id: int, gain: float) -> None:
        """Сменить громкость источника."""
        with self._lock:
            voice = self._voices.get(voice_id)
            if voice is not None:
                voice.gain = max(0.0, float(gain))

    def set_position(self, voice_id: int, position: tuple[float, float, float]) -> None:
        """Сменить мировую позицию пространственного голоса."""
        with self._lock:
            voice = self._voices.get(voice_id)
            if voice is not None and voice.position is not None:
                voice.position = np.asarray(position, dtype=np.float64).reshape(3)

    def set_listener(
        self,
        position: tuple[float, float, float],
        forward: tuple[float, float, float],
        up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        """Позиция и ориентация слушателя (мировые, левосторонняя система).

        «Вправо» = ``up × forward``; вырожденная ориентация оставляет прежнюю.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        right = np.cross(
            np.asarray(up, dtype=np.float64), np.asarray(forward, dtype=np.float64)
        )
        length = float(np.linalg.norm(right))
        with self._lock:
            self._listener_pos = pos
            if length > 1e-9:
                self._listener_right = right / length

    def spatial_of(self, voice_id: int) -> tuple[float, float] | None:
        """``(затухание, панорама)`` голоса по текущему слушателю или ``None``."""
        with self._lock:
            voice = self._voices.get(voice_id)
            if voice is None or voice.position is None:
                return None
            self._compute_spatial()
            return self._spatial[voice_id]

    def is_playing(self, voice_id: int) -> bool:
        """Голос ещё в микшере."""
        with self._lock:
            return voice_id in self._voices

    @property
    def voice_count(self) -> int:
        """Число активных голосов."""
        with self._lock:
            return len(self._voices)

    def final_gain(self, voice_id: int) -> float | None:
        """Итоговая громкость голоса ``min(1, source * bus * master)`` или ``None``.

        Затухание 3D-голоса сюда не входит: см. :meth:`spatial_of`.
        """
        with self._lock:
            voice = self._voices.get(voice_id)
            return None if voice is None else self._gain_of(voice)

    def _gain_of(self, voice: _Voice) -> float:
        return min(1.0, voice.gain * self.bus_gains[voice.bus] * self.master)

    # ------------------------------------------------------------- шины
    def set_master(self, gain: float) -> None:
        """Громкость мастера, 0…1."""
        with self._lock:
            self.master = min(1.0, max(0.0, float(gain)))

    def set_bus(self, bus: str, gain: float) -> None:
        """Громкость шины, 0…1 (``master`` — то же, что :meth:`set_master`)."""
        if bus not in self.bus_gains:
            raise ValueError(f"неизвестная шина: {bus!r}")
        if bus == "master":
            self.set_master(gain)
            return
        with self._lock:
            self.bus_gains[bus] = min(1.0, max(0.0, float(gain)))

    # ---------------------------------------------------------- рендер
    def render(self, frames: int) -> bytes:
        """Смешать ``frames`` кадров; завершившиеся голоса удаляются.

        Returns:
            int16 interleaved стерео, ``frames * 4`` байт; тишина, если голосов нет.
        """
        out = np.zeros((frames, 2), dtype=np.float32)
        with self._lock:
            self._compute_spatial()
            done: list[int] = []
            for voice_id, voice in self._voices.items():
                if not self._mix(voice_id, voice, out, frames):
                    done.append(voice_id)
            for voice_id in done:
                del self._voices[voice_id]
        np.clip(out, -32768.0, 32767.0, out=out)
        return out.astype(np.int16).tobytes()

    def _compute_spatial(self) -> None:
        """Затухание и панорама всех 3D-голосов одним проходом numpy."""
        ids = [i for i, v in self._voices.items() if v.position is not None]
        self._spatial = {}
        if not ids:
            return
        voices = [self._voices[i] for i in ids]
        offset = np.stack([v.position for v in voices]) - self._listener_pos
        dist = np.linalg.norm(offset, axis=1)
        pans = pan_of(offset, self._listener_right)
        gains = np.empty(len(ids), dtype=np.float64)
        groups: dict[Attenuation, list[int]] = {}
        for k, voice in enumerate(voices):
            groups.setdefault(voice.atten, []).append(k)
        for params, idx in groups.items():
            gains[idx] = attenuation(
                dist[idx],
                params.min_distance,
                params.max_distance,
                params.falloff,
                params.rolloff,
            )
        for k, voice_id in enumerate(ids):
            self._spatial[voice_id] = (float(gains[k]), float(pans[k]))

    def _coefficients(self, voice_id: int, voice: _Voice) -> tuple[float, float]:
        """Коэффициенты левого/правого каналов с учётом шин и 3D."""
        gain = self._gain_of(voice)
        spatial = self._spatial.get(voice_id)
        if spatial is not None:
            gain *= spatial[0]
            pan = spatial[1]
        else:
            pan = voice.pan
        left, right = pan_gains(pan)
        return gain * float(left), gain * float(right)

    def _mix(
        self, voice_id: int, voice: _Voice, out: np.ndarray, frames: int
    ) -> bool:
        """Добавить голос в ``out``; ``False`` — голос закончился."""
        total = voice.pcm.shape[0]
        left, right = self._coefficients(voice_id, voice)
        if voice.pitch == 1.0:
            block = self._block(voice, total, frames)
            n = block.shape[0]
            pos = voice.pos + n
            alive = voice.loop or pos < total
            voice.pos = pos % total if voice.loop else pos
        else:
            positions = voice.pos + np.arange(frames, dtype=np.float64) * voice.pitch
            if voice.loop:
                positions %= total
                valid = frames
            else:
                valid = int(np.searchsorted(positions, total - 1, side="right"))
                positions = positions[:valid]
            lower = positions.astype(np.int64)
            upper = np.minimum(lower + 1, total - 1) if not voice.loop else (
                (lower + 1) % total
            )
            frac = (positions - lower).astype(np.float32)[:, None]
            block = (
                voice.pcm[lower].astype(np.float32) * (1.0 - frac)
                + voice.pcm[upper].astype(np.float32) * frac
            )
            n = valid
            next_pos = voice.pos + frames * voice.pitch
            alive = voice.loop or (valid == frames and next_pos < total - 1)
            voice.pos = next_pos % total if voice.loop else next_pos
        if n:
            prev = voice.last_lr
            voice.last_lr = (left, right)
            if prev is None or prev == (left, right):
                out[:n, 0] += block[:, 0] * left
                out[:n, 1] += block[:, 1] * right
            else:
                # Плавная смена коэффициентов внутри блока: без щелчков.
                ramp = (np.arange(1, n + 1, dtype=np.float32) / n)[:, None]
                start = np.array(prev, dtype=np.float32)
                end = np.array((left, right), dtype=np.float32)
                out[:n] += block * (start + (end - start) * ramp)
        return alive

    @staticmethod
    def _block(voice: _Voice, total: int, frames: int) -> np.ndarray:
        start = int(voice.pos)
        if not voice.loop:
            return voice.pcm[start : start + frames]
        if start + frames <= total:
            return voice.pcm[start : start + frames]
        idx = (start + np.arange(frames)) % total
        return voice.pcm[idx]
