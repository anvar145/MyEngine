"""Кэш состояний OpenGL.

Главный источник потерь на слабых драйверах — лишние переключения состояния.
:class:`GLState` запоминает, что уже установлено, и пропускает повторные вызовы
(``glUseProgram``, ``glEnable``, ``glBindVertexArray`` и т. д.), а также ведёт
счётчики: сколько вызовов реально ушло в драйвер, а сколько подавлено.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.rendering.errors import RenderError

__all__ = ["SCRATCH_TEXTURE_UNIT", "GLState", "StateStats", "select_scratch_unit"]

#: Юнит для «внеочередных» операций с текстурами (загрузка, создание целей).
#: Материалы используют юниты 0–5, поэтому привязка здесь не портит кэш
#: :class:`GLState`: он не узнаёт о ней, а нужные ему юниты остаются нетронутыми.
SCRATCH_TEXTURE_UNIT = 15


def select_scratch_unit() -> None:
    """Сделать активным служебный юнит перед прямым ``glBindTexture``."""
    from OpenGL import GL

    GL.glActiveTexture(GL.GL_TEXTURE0 + SCRATCH_TEXTURE_UNIT)


@dataclass(slots=True)
class StateStats:
    """Счётчики переключений состояния за кадр."""

    calls: int = 0
    redundant: int = 0

    def reset(self) -> None:
        """Обнулить счётчики."""
        self.calls = 0
        self.redundant = 0

    @property
    def saved(self) -> int:
        """Сколько вызовов драйвера удалось не делать."""
        return self.redundant


class GLState:
    """Тонкая обёртка над состоянием контекста с проверкой изменений.

    Все методы можно вызывать без активного контекста: тогда они лишь
    обновляют внутренние значения. Реальные вызовы GL выполняются лениво при
    первом обращении (``OpenGL`` импортируется внутри методов).
    """

    __slots__ = (
        "_blend",
        "_blend_func",
        "_bound_textures",
        "_color_write",
        "_cull_face",
        "_cull_mode",
        "_depth_func",
        "_depth_test",
        "_depth_write",
        "_framebuffer",
        "_default_framebuffer",
        "_front_face",
        "_gl",
        "_program",
        "_scissor",
        "_scissor_rect",
        "_vao",
        "_viewport",
        "stats",
    )

    def __init__(self) -> None:
        self._gl = None
        self.stats = StateStats()
        self._program: int | None = None
        self._vao: int | None = None
        self._framebuffer: int | None = None
        self._default_framebuffer = 0
        self._viewport: tuple[int, int, int, int] | None = None
        self._depth_test: bool | None = None
        self._depth_write: bool | None = None
        self._depth_func: int | None = None
        self._blend: bool | None = None
        self._blend_func: tuple[int, int] | None = None
        self._cull_face: bool | None = None
        self._cull_mode: int | None = None
        self._front_face: int | None = None
        self._scissor: bool | None = None
        self._scissor_rect: tuple[int, int, int, int] | None = None
        self._color_write: bool | None = None
        self._bound_textures: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ #
    # Внутреннее
    # ------------------------------------------------------------------ #
    @property
    def gl(self):
        """Модуль ``OpenGL.GL`` (импортируется при первом обращении)."""
        if self._gl is None:
            from OpenGL import GL

            self._gl = GL
        return self._gl

    def _call(self, name: str, *args: object) -> None:
        """Выполнить вызов GL и учесть статистику."""
        self.stats.calls += 1
        getattr(self.gl, name)(*args)

    def _skip(self) -> None:
        """Отметить, что вызов драйвера не потребовался."""
        self.stats.redundant += 1

    def invalidate(self) -> None:
        """Забыть всё отслеживаемое состояние.

        Вызывается при смене контекста, после ``makeCurrent`` другого
        контекста или при подозрении, что состояние изменили в обход кэша.
        """
        self._program = None
        self._vao = None
        self._framebuffer = None
        self._viewport = None
        self._depth_test = None
        self._depth_write = None
        self._depth_func = None
        self._blend = None
        self._blend_func = None
        self._cull_face = None
        self._cull_mode = None
        self._front_face = None
        self._scissor = None
        self._scissor_rect = None
        self._color_write = None
        self._bound_textures.clear()

    # ------------------------------------------------------------------ #
    # Программа и геометрия
    # ------------------------------------------------------------------ #
    def use_program(self, program: int) -> None:
        """Сделать программу активной."""
        if program == 0:
            raise RenderError("Нельзя использовать программу 0")
        if self._program == program:
            self._skip()
            return
        self._call("glUseProgram", program)
        self._program = program

    def invalidate_program(self) -> None:
        """Забыть, какая программа была привязана.

        Нужно, когда активную программу сменили в обход кэша (например,
        удалили её или вызвали ``glUseProgram`` напрямую): иначе следующий
        :meth:`use_program` посчитает, что привязка уже сделана, и
        uniform-переменные уйдут в чужую программу.
        """
        self._program = None

    def bind_vertex_array(self, vao: int) -> None:
        """Привязать VAO."""
        if self._vao == vao:
            self._skip()
            return
        self._call("glBindVertexArray", vao)
        self._vao = vao

    @property
    def default_framebuffer(self) -> int:
        """Идентификатор «экранного» framebuffer (у ``QOpenGLWidget`` не ноль)."""
        return self._default_framebuffer

    def set_default_framebuffer(self, framebuffer: int) -> None:
        """Запомнить экранный framebuffer и сбросить кэш привязки.

        Оконная система привязывает свой FBO до отрисовки кадра в обход кэша,
        поэтому кэшу привязки доверять после этого нельзя.
        """
        self._default_framebuffer = int(framebuffer)
        self._framebuffer = None

    def bind_default_framebuffer(self) -> None:
        """Привязать экранный framebuffer (не обязательно с номером 0)."""
        self.bind_framebuffer(self._default_framebuffer)

    def bind_framebuffer(self, framebuffer: int = 0) -> None:
        """Привязать framebuffer (0 — экран)."""
        if self._framebuffer == framebuffer:
            self._skip()
            return
        self._call("glBindFramebuffer", self.gl.GL_FRAMEBUFFER, framebuffer)
        self._framebuffer = framebuffer
        # Привязка FBO сбрасывает viewport-состояние драйвера в некоторых
        # реализациях, поэтому перестаём доверять кэшу viewport.
        self._viewport = None

    def bind_texture(self, unit: int, target: int, texture: int) -> None:
        """Привязать текстуру к юниту, не переключая лишний раз."""
        key = (unit, target)
        if self._bound_textures.get(key) == texture:
            self._skip()
            return
        self._call("glActiveTexture", self.gl.GL_TEXTURE0 + unit)
        self._call("glBindTexture", target, texture)
        self._bound_textures[key] = texture

    # ------------------------------------------------------------------ #
    # Растеризация
    # ------------------------------------------------------------------ #
    def set_viewport(self, x: int, y: int, width: int, height: int) -> None:
        """Установить область вывода (отрицательные размеры обрезаются)."""
        width = max(0, int(width))
        height = max(0, int(height))
        requested = (int(x), int(y), width, height)
        if self._viewport == requested:
            self._skip()
            return
        self._call("glViewport", *requested)
        self._viewport = requested

    def set_depth_test(self, enabled: bool = True) -> None:
        """Включить или выключить проверку глубины."""
        if self._depth_test == enabled:
            self._skip()
            return
        self._call("glEnable" if enabled else "glDisable", self.gl.GL_DEPTH_TEST)
        self._depth_test = enabled

    def set_depth_write(self, enabled: bool = True) -> None:
        """Разрешить или запретить запись в буфер глубины."""
        if self._depth_write == enabled:
            self._skip()
            return
        self._call("glDepthMask", self.gl.GL_TRUE if enabled else self.gl.GL_FALSE)
        self._depth_write = enabled

    def set_color_write(self, enabled: bool = True) -> None:
        """Разрешить или запретить запись цвета.

        Отключение нужно проходу теней: он пишет только глубину, и запись
        цвета была бы чистой потерей полосы пропускания (на интегрированной
        графике это существенно — карта 512×512 рисуется целиком).
        """
        if self._color_write == enabled:
            self._skip()
            return
        flag = self.gl.GL_TRUE if enabled else self.gl.GL_FALSE
        self._call("glColorMask", flag, flag, flag, flag)
        self._color_write = enabled

    def invalidate_color_write(self) -> None:
        """Забыть, что известно про маску записи цвета.

        Нужно, когда состояние изменили напрямую (или другой владелец
        контекста): иначе следующий :meth:`set_color_write` решит, что значение
        уже правильное, и не тронет драйвер — а кадр останется без цвета.
        """
        self._color_write = None

    def set_depth_func(self, func: int | None = None) -> None:
        """Задать функцию сравнения глубины (по умолчанию ``GL_LESS``)."""
        target = self.gl.GL_LESS if func is None else func
        if self._depth_func == target:
            self._skip()
            return
        self._call("glDepthFunc", target)
        self._depth_func = target

    def set_blend(self, enabled: bool, func: tuple[int, int] | None = None) -> None:
        """Включить смешивание и (опционально) задать функцию."""
        if self._blend != enabled:
            self._call("glEnable" if enabled else "glDisable", self.gl.GL_BLEND)
            self._blend = enabled
        else:
            self._skip()
        if enabled:
            target = func or (self.gl.GL_SRC_ALPHA, self.gl.GL_ONE_MINUS_SRC_ALPHA)
            if self._blend_func != target:
                self._call("glBlendFunc", *target)
                self._blend_func = target
            else:
                self._skip()

    def set_cull(self, enabled: bool, mode: int | None = None) -> None:
        """Включить отсечение граней и задать отсекаемые грани."""
        if self._cull_face != enabled:
            self._call("glEnable" if enabled else "glDisable", self.gl.GL_CULL_FACE)
            self._cull_face = enabled
        else:
            self._skip()
        if enabled:
            target = self.gl.GL_BACK if mode is None else mode
            if self._cull_mode != target:
                self._call("glCullFace", target)
                self._cull_mode = target
            else:
                self._skip()

    def set_front_face(self, winding: int | None = None) -> None:
        """Задать порядок обхода вершин для лицевых граней (по умолчанию CCW)."""
        target = self.gl.GL_CCW if winding is None else winding
        if self._front_face == target:
            self._skip()
            return
        self._call("glFrontFace", target)
        self._front_face = target

    def set_scissor(self, enabled: bool) -> None:
        """Включить или выключить отсечение по прямоугольнику."""
        if self._scissor == enabled:
            self._skip()
            return
        self._call("glEnable" if enabled else "glDisable", self.gl.GL_SCISSOR_TEST)
        self._scissor = enabled

    def set_scissor_rect(self, x: int, y: int, width: int, height: int) -> None:
        """Задать прямоугольник отсечения (пиксели, начало снизу слева)."""
        rect = (int(x), int(y), int(width), int(height))
        if self._scissor_rect == rect:
            self._skip()
            return
        self._call("glScissor", *rect)
        self._scissor_rect = rect

    # ------------------------------------------------------------------ #
    # Применение типового состояния кадра
    # ------------------------------------------------------------------ #
    def apply_default_3d(self) -> None:
        """Применить состояние по умолчанию для 3D-прохода."""
        self.set_depth_test(True)
        self.set_depth_write(True)
        self.set_depth_func()
        self.set_cull(True)
        self.set_front_face()
        self.set_blend(False)

    def __repr__(self) -> str:
        return (
            f"<GLState program={self._program} vao={self._vao} "
            f"calls={self.stats.calls} saved={self.stats.saved}>"
        )
