"""Шейдеры: компиляция, линковка и установка uniform-переменных.

Модуль работает с исходниками GLSL, которые готовит загрузчик ресурсов
(:class:`engine.core.resources.loaders.GlslShaderLoader` уже разрешил
``#include`` и ``#pragma once``). Здесь остаётся компиляция, линковка и
кэширование расположений uniform-переменных.

Example:
    >>> program = ShaderProgram.from_sources(
    ...     "lit",
    ...     vertex=open("assets/shaders/simple.vert").read(),
    ...     fragment=open("assets/shaders/simple.frag").read(),
    ... )
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from engine.core.math3d import Quaternion, Vector3
from engine.core.resources import ShaderAsset, ShaderStage
from engine.rendering.errors import ShaderCompileError, ShaderLinkError

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.program_cache import ProgramBinaryCache

__all__ = [
    "Shader",
    "ShaderProgram",
    "ShaderUniformError",
    "get_program_cache",
    "inject_defines",
    "load_program",
    "set_program_cache",
]

_log = logging.getLogger("engine.rendering.shader")

_gl_module: Any = None


def _gl() -> Any:
    """Модуль ``OpenGL.GL`` (импорт один раз).

    ``from OpenGL import GL`` внутри горячей функции под PySide проходит через
    хук ``feature_import`` и на каждый uniform стоит микросекунды.
    """
    global _gl_module
    if _gl_module is None:
        from OpenGL import GL

        _gl_module = GL
    return _gl_module

_VERSION_DIRECTIVE = re.compile(r"^[ \t]*#version[^\n]*\n", re.MULTILINE)

_GL_STAGE = {
    ShaderStage.VERTEX: "GL_VERTEX_SHADER",
    ShaderStage.FRAGMENT: "GL_FRAGMENT_SHADER",
    ShaderStage.GEOMETRY: "GL_GEOMETRY_SHADER",
    ShaderStage.COMPUTE: "GL_COMPUTE_SHADER",
}

#: Установка «одного вектора» нужной длины: размер → имя функции драйвера.
_VECTOR_SETTERS: dict[int, str] = {
    2: "glUniform2fv",
    3: "glUniform3fv",
    4: "glUniform4fv",
}


def _is_integer_array(value: Any) -> bool:
    """Проверить, что значение — массив целых чисел (для ``glUniform1iv``)."""
    if isinstance(value, np.ndarray):
        return value.dtype.kind in ("i", "u")
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(item, (int, np.integer)) for item in value)
    return False


_program_cache: ProgramBinaryCache | None = None


def set_program_cache(cache: ProgramBinaryCache | None) -> None:
    """Включить (или выключить, ``None``) дисковый кэш бинарников программ.

    Кэш общий для всех программ процесса: ключ записи включает драйвер и
    исходники, поэтому разным контекстам он не мешает.
    """
    global _program_cache
    _program_cache = cache


def get_program_cache() -> ProgramBinaryCache | None:
    """Текущий кэш бинарников или ``None``."""
    return _program_cache


class ShaderUniformError(Exception):
    """Не удалось установить uniform (неизвестный тип или имя)."""


def inject_defines(
    source: str, defines: Mapping[str, object] | None = None
) -> str:
    """Вставить ``#define`` сразу после директивы ``#version``.

    В GLSL директива ``#version`` обязана быть первой строкой, поэтому
    ключевые слова вариантов (``INSTANCED``, ``USE_SHADOW``, ...) нельзя просто
    дописать в начало файла.

    Args:
        source: Исходник шейдера.
        defines: Пары «имя — значение»; ``None`` или пустой словарь ничего не
            меняют. Значение ``None`` даёт ``#define NAME`` без значения.

    Returns:
        Исходник с добавленными определениями.
    """
    if not defines:
        return source
    header = "".join(
        f"#define {name} {value}\n" if value is not None else f"#define {name}\n"
        for name, value in defines.items()
    )
    match = _VERSION_DIRECTIVE.search(source)
    if match is None:
        return header + source
    return source[: match.end()] + header + source[match.end():]


@dataclass(slots=True)
class Shader:
    """Один скомпилированный шейдер (стадия)."""

    stage: ShaderStage
    source: str
    name: str = "shader"
    handle: int = 0

    def compile(self) -> int:
        """Скомпилировать шейдер.

        Returns:
            Идентификатор OpenGL.

        Raises:
            ShaderCompileError: Если компиляция не удалась.
        """
        from OpenGL import GL

        stage_name = _GL_STAGE.get(self.stage)
        if stage_name is None:
            raise ShaderCompileError(f"Стадия {self.stage.value} не поддерживается")
        gl_stage = getattr(GL, stage_name)
        handle = GL.glCreateShader(gl_stage)
        GL.glShaderSource(handle, self.source)
        GL.glCompileShader(handle)
        if not GL.glGetShaderiv(handle, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(handle)
            GL.glDeleteShader(handle)
            text = (
                log.decode("utf-8", errors="replace")
                if isinstance(log, bytes)
                else str(log)
            )
            raise ShaderCompileError(
                f"{self.name} ({self.stage.value}) не скомпилировался:\n{text.strip()}"
            )
        self.handle = int(handle)
        return self.handle

    def delete(self) -> None:
        """Удалить шейдер из драйвера."""
        if self.handle:
            from OpenGL import GL

            GL.glDeleteShader(self.handle)
            self.handle = 0

    def __repr__(self) -> str:
        return f"<Shader '{self.name}' {self.stage.value} id={self.handle}>"


@dataclass(slots=True)
class ShaderProgram:
    """Слинкованная программа с кэшем uniform-переменных."""

    name: str
    sources: Mapping[ShaderStage, str]
    handle: int = 0
    _uniforms: dict[str, int] = field(default_factory=dict, repr=False)
    _values: dict[int, Any] = field(default_factory=dict, repr=False)
    _shaders: list[Shader] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ #
    # Создание
    # ------------------------------------------------------------------ #
    @classmethod
    def from_sources(
        cls,
        name: str,
        *,
        vertex: str,
        fragment: str,
        defines: Mapping[str, object] | None = None,
        **extra: str,
    ) -> ShaderProgram:
        """Собрать программу из исходников стадий.

        Args:
            name: Имя программы (для логов и статистики).
            vertex: Исходник вершинного шейдера.
            fragment: Исходник фрагментного шейдера.
            defines: Ключевые слова варианта (вставляются после ``#version``).
            **extra: Дополнительные стадии (``geometry``, ``compute``).

        Returns:
            Скомпилированная и слинкованная программа.
        """
        sources: dict[ShaderStage, str] = {
            ShaderStage.VERTEX: inject_defines(vertex, defines),
            ShaderStage.FRAGMENT: inject_defines(fragment, defines),
        }
        for stage_name, source in extra.items():
            try:
                stage = ShaderStage(stage_name)
            except ValueError as exc:  # pragma: no cover - защита от опечаток
                raise ShaderUniformError(f"Неизвестная стадия '{stage_name}'") from exc
            sources[stage] = inject_defines(source, defines)
        program = cls(name=name, sources=sources)
        program.build()
        return program

    @classmethod
    def from_assets(
        cls,
        vertex: ShaderAsset,
        fragment: ShaderAsset,
        *,
        name: str | None = None,
        defines: Mapping[str, object] | None = None,
    ) -> ShaderProgram:
        """Собрать программу из загруженных ассетов шейдеров.

        Args:
            vertex: Ассет вершинного шейдера.
            fragment: Ассет фрагментного шейдера.
            name: Имя программы; по умолчанию берётся из имени вершинного шейдера.
            defines: Ключевые слова варианта.

        Returns:
            Скомпилированная программа.
        """
        if vertex.stage is not ShaderStage.VERTEX:
            raise ShaderCompileError(
                f"'{vertex.name}' не является вершинным шейдером "
                f"(стадия {vertex.stage.value})"
            )
        if fragment.stage is not ShaderStage.FRAGMENT:
            raise ShaderCompileError(
                f"'{fragment.name}' не является фрагментным шейдером "
                f"(стадия {fragment.stage.value})"
            )
        return cls.from_sources(
            name or vertex.name,
            vertex=vertex.source,
            fragment=fragment.source,
            defines=defines,
        )

    def with_source(self, stage: ShaderStage, source: str) -> ShaderProgram:
        """Вернуть копию программы с заменённым исходником стадии.

        Удобно для вариантов шейдера (keywords): базовый исходник + ``#define``.

        Args:
            stage: Заменяемая стадия.
            source: Новый исходник.

        Returns:
            Новая (ещё не собранная) программа.
        """
        sources = dict(self.sources)
        sources[stage] = source
        return ShaderProgram(name=self.name, sources=sources)

    # ------------------------------------------------------------------ #
    # Сборка
    # ------------------------------------------------------------------ #
    def build(self) -> int:
        """Скомпилировать стадии и слинковать программу.

        Returns:
            Идентификатор OpenGL.

        Raises:
            ShaderCompileError: Если стадия не скомпилировалась.
            ShaderLinkError: Если линковка не удалась.
        """
        from OpenGL import GL

        if self.handle:
            self.dispose()
        cache = _program_cache
        if cache is not None:
            cached = cache.load(self.name, self.sources)
            if cached:
                self.handle = cached
                self._shaders = []
                self._uniforms.clear()
                self._values.clear()
                _log.debug("Программа из кэша: %s (id=%d)", self.name, cached)
                return cached
        compiled: list[Shader] = []
        try:
            for stage, source in self.sources.items():
                shader = Shader(stage=stage, source=source, name=self.name)
                shader.compile()
                compiled.append(shader)
            handle = int(GL.glCreateProgram())
            if cache is not None and cache.supported:
                GL.glProgramParameteri(
                    handle, GL.GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL.GL_TRUE
                )
            for shader in compiled:
                GL.glAttachShader(handle, shader.handle)
            GL.glLinkProgram(handle)
            if not GL.glGetProgramiv(handle, GL.GL_LINK_STATUS):
                log = GL.glGetProgramInfoLog(handle)
                GL.glDeleteProgram(handle)
                text = (
                    log.decode("utf-8", errors="replace")
                    if isinstance(log, bytes)
                    else str(log)
                )
                raise ShaderLinkError(
                    f"Программа '{self.name}' не слинковалась:\n{text.strip()}"
                )
            for shader in compiled:
                GL.glDetachShader(handle, shader.handle)
            if cache is not None:
                cache.store(handle, self.name, self.sources)
        finally:
            for shader in compiled:
                shader.delete()

        self.handle = handle
        self._shaders = []
        self._uniforms.clear()
        self._values.clear()
        _log.debug("Шейдерная программа '%s' собрана (id=%d)", self.name, handle)
        return handle

    def dispose(self) -> None:
        """Удалить программу из драйвера."""
        if self.handle:
            from OpenGL import GL

            GL.glDeleteProgram(self.handle)
            _log.debug("Программа '%s' удалена", self.name)
        self.handle = 0
        self._uniforms.clear()
        self._values.clear()

    # ------------------------------------------------------------------ #
    # Использование
    # ------------------------------------------------------------------ #
    def use(self, state: object | None = None) -> int:
        """Сделать программу активной.

        Args:
            state: Кэш состояний (:class:`~engine.rendering.state.GLState`).
                Если он передан, привязка идёт через него, и кэш остаётся
                согласованным с драйвером. Без него выполняется «сырой»
                ``glUseProgram``, после чего кэш будет считать активной прежнюю
                программу — тогда uniform-переменные уйдут в чужую программу и
                драйвер ответит ``GL_INVALID_OPERATION``. Поэтому из кода
                рендера всегда передавайте ``state``.

        Returns:
            Идентификатор программы.

        Raises:
            ShaderLinkError: Если программа не собрана.
        """
        if not self.handle:
            raise ShaderLinkError(f"Программа '{self.name}' не собрана")
        use_program = getattr(state, "use_program", None)
        if use_program is not None:
            use_program(self.handle)
            return self.handle
        from OpenGL import GL

        GL.glUseProgram(self.handle)
        if state is None:
            _log.debug(
                "Программа '%s' привязана в обход кэша состояний", self.name
            )
        return self.handle

    def uniform_location(self, name: str) -> int:
        """Вернуть расположение uniform-переменной (с кэшем).

        Args:
            name: Имя переменной в шейдере.

        Returns:
            Расположение или ``-1``, если переменной нет (или её выбросил
            оптимизатор — это не ошибка).
        """
        cached = self._uniforms.get(name)
        if cached is not None:
            return cached
        from OpenGL import GL

        location = int(GL.glGetUniformLocation(self.handle, name))
        self._uniforms[name] = location
        if location < 0:
            _log.debug(
                "Uniform '%s' отсутствует в программе '%s'", name, self.name
            )
        return location

    def set_uniform(self, name: str, value: Any) -> bool:
        """Установить uniform-переменную, подобрав вызов по типу значения.

        Поддерживаются: ``bool``, ``int``, ``float``, ``Vector3``,
        ``Quaternion``, последовательности длиной 2–4, numpy-массивы
        ``(N,)``/``(3,)``/``(4,)``/``(4, 4)``/``(N, 4, 4)``.

        Args:
            name: Имя переменной.
            value: Значение.

        Returns:
            ``True``, если значение установлено; ``False``, если переменной
            в шейдере нет.

        Raises:
            ShaderUniformError: Если тип значения не поддерживается.
        """
        location = self._uniforms.get(name)
        if location is None:
            location = self.uniform_location(name)
        if location < 0:
            return False

        # Значения uniform живут в программе между кадрами: повторная запись
        # того же скаляра/вектора — лишний вызов драйвера (PyOpenGL ~10 мкс).
        kind = type(value)
        key: Any = None
        if kind is float or kind is int or kind is bool:
            key = (kind, value)
        elif kind is Vector3:
            key = (kind, value.x, value.y, value.z)
        elif kind is tuple:
            key = (kind, value)
        if key is not None:
            if self._values.get(location) == key:
                return True
            self._values[location] = key
        GL = _gl()
        if isinstance(value, bool):
            GL.glUniform1i(location, int(value))
            return True
        if isinstance(value, int):
            GL.glUniform1i(location, value)
            return True
        if isinstance(value, float):
            GL.glUniform1f(location, value)
            return True
        if isinstance(value, Vector3):
            GL.glUniform3f(location, value.x, value.y, value.z)
            return True
        if isinstance(value, Quaternion):
            GL.glUniform4f(location, value.x, value.y, value.z, value.w)
            return True
        if isinstance(value, np.ndarray):
            self._set_uniform_array(location, value)
            return True
        if isinstance(value, (tuple, list)):
            components = [float(item) for item in value]
            if len(components) == 2:
                GL.glUniform2f(location, *components)
            elif len(components) == 3:
                GL.glUniform3f(location, *components)
            elif len(components) == 4:
                GL.glUniform4f(location, *components)
            else:
                raise ShaderUniformError(
                    f"'{name}': последовательность длиной "
                    f"{len(components)} не поддерживается"
                )
            return True
        raise ShaderUniformError(
            f"'{name}': тип {type(value).__name__} не поддерживается"
        )

    def _set_uniform_array(self, location: int, value: np.ndarray) -> None:
        """Установить uniform из numpy-массива нужной формы."""
        self._values.pop(location, None)
        GL = _gl()
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            components = array.size
            if components == 2:
                GL.glUniform2fv(location, 1, array)
            elif components == 3:
                GL.glUniform3fv(location, 1, array)
            elif components == 4:
                GL.glUniform4fv(location, 1, array)
            elif components == 16:
                GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, array)
            else:
                raise ShaderUniformError(
                    f"Вектор длины {components} не поддерживается"
                )
            return
        if array.ndim == 2 and array.shape == (4, 4):
            # Матрицы ядра хранятся построчно (row-major), а OpenGL ожидает
            # столбцовую раскладку — передаём транспонированную копию.
            GL.glUniformMatrix4fv(
                location, 1, GL.GL_FALSE, np.ascontiguousarray(array.T)
            )
            return
        if array.ndim == 3 and array.shape[1:] == (4, 4):
            # Паллета матриц (скиннинг): транспонируем каждую.
            count = array.shape[0]
            transposed = np.ascontiguousarray(array.transpose(0, 2, 1))
            GL.glUniformMatrix4fv(location, count, GL.GL_FALSE, transposed)
            return
        raise ShaderUniformError(f"Форма массива {array.shape} не поддерживается")

    def set_uniforms(self, values: Mapping[str, Any]) -> int:
        """Установить несколько uniform-переменных.

        Returns:
            Число установленных переменных.
        """
        applied = 0
        for name, value in values.items():
            if self.set_uniform(name, value):
                applied += 1
        return applied

    def set_uniform_array(self, name: str, value: Any) -> bool:
        """Установить uniform-массив одним вызовом драйвера.

        Отличие от :meth:`set_uniform` в том, что значение трактуется как
        **массив** известного размера: numpy-массив ``(N, 3)`` уходит одним
        ``glUniform3fv`` с ``count = N``, а не как покомпонентная установка.
        Это основной путь передачи света (см.
        :class:`~engine.rendering.lighting.LightManager`): на слабых драйверах
        один вызов на массив заметно дешевле, чем N вызовов на элементы.

        Поддерживаются формы: ``(N,)``, ``(N, 2)``, ``(N, 3)``, ``(N, 4)``,
        ``(N, 4, 4)`` и целочисленные массивы (``glUniform1iv``).

        Args:
            name: Имя uniform-переменной.
            value: Массив значений.

        Returns:
            ``True``, если значение установлено; ``False``, если переменной в
            шейдере нет.

        Raises:
            ShaderUniformError: Если форму массива не поддерживает драйвер.
        """
        location = self.uniform_location(name)
        if location < 0:
            return False
        self._values.pop(location, None)

        from OpenGL import GL

        # Целочисленный массив: типы источников, счётчики.
        if _is_integer_array(value):
            data = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
            GL.glUniform1iv(location, int(data.size), data)
            return True

        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            count = int(array.size)
            # Одномерный массив трактуем как один вектор: удобно передавать
            # одиночные значения через тот же интерфейс, что и массивы.
            if count == 2:
                GL.glUniform2fv(location, 1, array)
                return True
            if count == 3:
                GL.glUniform3fv(location, 1, array)
                return True
            if count == 4:
                GL.glUniform4fv(location, 1, array)
                return True
            if count % 3 == 0:
                GL.glUniform3fv(location, count // 3, array)
                return True
            if count % 4 == 0:
                GL.glUniform4fv(location, count // 4, array)
                return True
            raise ShaderUniformError(
                f"'{name}': массив длины {count} не разложить на vec3/vec4"
            )
        if array.ndim == 2:
            rows, columns = array.shape
            if columns == 2:
                GL.glUniform2fv(location, int(rows), array)
            elif columns == 3:
                GL.glUniform3fv(location, int(rows), array)
            elif columns == 4:
                GL.glUniform4fv(location, int(rows), array)
            else:
                raise ShaderUniformError(
                    f"'{name}': массив (N, {columns}) не поддерживается"
                )
            return True
        if array.ndim == 3 and array.shape[1:] == (4, 4):
            transposed = np.ascontiguousarray(array.transpose(0, 2, 1))
            GL.glUniformMatrix4fv(
                location, int(array.shape[0]), GL.GL_FALSE, transposed
            )
            return True
        raise ShaderUniformError(
            f"'{name}': форма массива {array.shape} не поддерживается"
        )

    @property
    def uniform_names(self) -> tuple[str, ...]:
        """Имена уже запрошенных uniform-переменных."""
        return tuple(self._uniforms)

    def __repr__(self) -> str:
        stages = "+".join(stage.value for stage in self.sources)
        return f"<ShaderProgram '{self.name}' [{stages}] id={self.handle}>"


def load_program(
    name: str,
    vertex_path: Path | str,
    fragment_path: Path | str,
    resources: Any = None,
) -> ShaderProgram:
    """Загрузить и собрать программу из файлов через менеджер ресурсов.

    Args:
        name: Имя программы.
        vertex_path: Путь к вершинному шейдеру.
        fragment_path: Путь к фрагментному шейдеру.
        resources: :class:`~engine.core.resources.ResourceManager`; если не
            задан, создаётся локальный с корнем в каталоге шейдеров.

    Returns:
        Собранная программа.
    """
    from engine.core.resources import ResourceManager

    manager = resources
    if manager is None:
        manager = ResourceManager(roots=[Path(vertex_path).parent])
    vertex = manager.load(vertex_path)
    fragment = manager.load(fragment_path)
    if not isinstance(vertex, ShaderAsset) or not isinstance(fragment, ShaderAsset):
        raise ShaderCompileError("Ожидались ассеты шейдеров (.vert/.frag)")
    return ShaderProgram.from_assets(vertex, fragment, name=name)
