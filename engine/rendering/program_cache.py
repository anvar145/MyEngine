"""Дисковый кэш бинарников шейдерных программ (``glGetProgramBinary``).

Компиляция и линковка крупных шейдеров на слабой встроенной графике занимает
сотни миллисекунд на программу, а вариантов (``INSTANCED``, ``ALPHA_MASK``...)
бывает много. Кэш сохраняет линкованный бинарник драйвера рядом с проектом и
при следующем запуске поднимает программу без компиляции.

Ключ записи — хэш исходников всех стадий (уже с ``#define`` варианта) и
идентификатор драйвера (``GL_VENDOR``/``GL_RENDERER``/``GL_VERSION``):
бинарник чужого драйвера не подойдёт, поэтому при обновлении драйвера кэш
сам «промахивается» и пересобирается. Любой сбой (нет расширения, битый файл,
драйвер отверг бинарник) тихо превращается в обычную компиляцию.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import logging
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from engine.core.resources import ShaderStage

__all__ = ["ProgramBinaryCache", "ProgramCacheStats", "warm_program_cache"]

_log = logging.getLogger("engine.rendering.program_cache")

_MAGIC = b"MYPB"
_HEADER = struct.Struct("<4sII")  # magic, binaryFormat, длина данных


@dataclass(slots=True)
class ProgramCacheStats:
    """Счётчики работы кэша (для логов и тестов)."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    rejected: int = 0


class ProgramBinaryCache:
    """Кэш бинарников программ в каталоге.

    Args:
        directory: Каталог кэша (создаётся при первой записи).
        driver_id: Идентификатор драйвера; по умолчанию читается из активного
            контекста при первом обращении.
    """

    def __init__(self, directory: Path | str, *, driver_id: str | None = None) -> None:
        self.directory = Path(directory)
        self.stats = ProgramCacheStats()
        self._driver_id = driver_id
        self._supported: bool | None = None

    # ------------------------------------------------------------------ #
    # Возможности
    # ------------------------------------------------------------------ #
    @property
    def supported(self) -> bool:
        """Умеет ли активный контекст отдавать и принимать бинарники."""
        if self._supported is None:
            self._supported = self._probe()
        return self._supported

    @property
    def driver_id(self) -> str:
        """Строка драйвера, входящая в ключ."""
        if self._driver_id is None:
            from OpenGL import GL

            parts = (GL.GL_VENDOR, GL.GL_RENDERER, GL.GL_VERSION)
            self._driver_id = "|".join(_gl_string(GL, item) for item in parts)
        return self._driver_id

    def key(self, sources: Mapping[ShaderStage, str]) -> str:
        """Ключ записи: хэш драйвера и исходников стадий (порядок не важен)."""
        digest = hashlib.sha256(self.driver_id.encode("utf-8"))
        for stage in sorted(sources, key=lambda item: item.value):
            digest.update(b"\0" + stage.value.encode() + b"\0")
            digest.update(sources[stage].encode("utf-8"))
        return digest.hexdigest()[:32]

    def path_for(self, sources: Mapping[ShaderStage, str]) -> Path:
        """Файл записи для данного набора исходников."""
        return self.directory / f"{self.key(sources)}.bin"

    # ------------------------------------------------------------------ #
    # Чтение и запись
    # ------------------------------------------------------------------ #
    def load(self, name: str, sources: Mapping[ShaderStage, str]) -> int:
        """Поднять программу из кэша.

        Returns:
            Идентификатор слинкованной программы или ``0``, если записи нет
            или драйвер её отверг (тогда нужна обычная сборка).
        """
        if not self.supported:
            return 0
        path = self.path_for(sources)
        try:
            blob = path.read_bytes()
        except OSError:
            self.stats.misses += 1
            return 0
        handle = self._upload(blob)
        if not handle:
            self.stats.rejected += 1
            _log.debug("Бинарник '%s' отвергнут драйвером, запись удалена", name)
            with contextlib.suppress(OSError):
                path.unlink()
            return 0
        self.stats.hits += 1
        return handle

    def store(self, handle: int, name: str, sources: Mapping[ShaderStage, str]) -> bool:
        """Сохранить бинарник слинкованной программы.

        Returns:
            ``True``, если запись создана.
        """
        if not self.supported:
            return False
        from OpenGL import GL

        try:
            size = int(GL.glGetProgramiv(handle, GL.GL_PROGRAM_BINARY_LENGTH))
            if size <= 0:
                return False
            length = (ctypes.c_int * 1)()
            fmt = (ctypes.c_uint * 1)()
            buffer = (ctypes.c_ubyte * size)()
            GL.glGetProgramBinary(handle, size, length, fmt, buffer)
            payload = bytes(buffer[: length[0]])
            blob = _HEADER.pack(_MAGIC, fmt[0], len(payload)) + payload
            path = self.path_for(sources)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_bytes(blob)
            temp.replace(path)
        except Exception as exc:  # noqa: BLE001 - кэш не должен ронять рендер
            _log.debug("Не удалось сохранить бинарник '%s': %s", name, exc)
            return False
        self.stats.stores += 1
        return True

    def clear(self) -> int:
        """Удалить все записи кэша.

        Returns:
            Число удалённых файлов.
        """
        removed = 0
        if self.directory.is_dir():
            for entry in self.directory.glob("*.bin"):
                try:
                    entry.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #
    @staticmethod
    def _probe() -> bool:
        from OpenGL import GL

        try:
            if not bool(GL.glGetProgramBinary) or not bool(GL.glProgramBinary):
                return False
            return int(GL.glGetIntegerv(GL.GL_NUM_PROGRAM_BINARY_FORMATS)) > 0
        except Exception:  # noqa: BLE001 - расширение недоступно
            return False

    @staticmethod
    def _upload(blob: bytes) -> int:
        from OpenGL import GL

        if len(blob) < _HEADER.size:
            return 0
        magic, binary_format, size = _HEADER.unpack_from(blob)
        payload = blob[_HEADER.size :]
        if magic != _MAGIC or size != len(payload) or not size:
            return 0
        handle = 0
        try:
            handle = int(GL.glCreateProgram())
            data = (ctypes.c_ubyte * size).from_buffer_copy(payload)
            GL.glProgramBinary(handle, binary_format, data, size)
            if GL.glGetProgramiv(handle, GL.GL_LINK_STATUS):
                return handle
        except Exception:  # noqa: BLE001 - любой сбой = промах
            pass
        if handle:
            GL.glDeleteProgram(handle)
        return 0


def warm_program_cache(
    name: str,
    *,
    vertex: str,
    fragment: str,
    variants: Iterable[Mapping[str, object]],
) -> int:
    """Собрать варианты шейдера заранее, чтобы их бинарники попали в кэш.

    Нужен активный кэш (:func:`engine.rendering.shader.set_program_cache`) и
    контекст OpenGL. Собранные программы тут же удаляются.

    Args:
        name: Имя программы.
        vertex: Исходник вершинной стадии.
        fragment: Исходник фрагментной стадии.
        variants: Наборы ключевых слов (``{"INSTANCED": 1}``); пустой набор —
            базовый вариант.

    Returns:
        Число собранных вариантов.
    """
    from engine.rendering.shader import ShaderProgram

    count = 0
    for defines in variants:
        program = ShaderProgram.from_sources(
            name, vertex=vertex, fragment=fragment, defines=dict(defines)
        )
        program.dispose()
        count += 1
    return count


def _gl_string(gl: object, name: object) -> str:
    value = gl.glGetString(name)  # type: ignore[attr-defined]
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
