"""Граф проходов рендеринга и цели вывода (framebuffer-объекты).

Здесь заложен путь к отложенному шейдингу: :class:`RenderTarget` умеет держать
несколько цветовых привязанностей (MRT) и буфер глубины, а :class:`RenderGraph`
— упорядоченный набор проходов, каждый из которых сам решает, куда рисовать.
Базовый forward-рендерер использует тот же граф, поэтому переход на deferred
не потребует менять интерфейсы.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from engine.rendering.errors import RenderError
from engine.rendering.gl_debug import get_gl as _gl
from engine.rendering.state import select_scratch_unit

if TYPE_CHECKING:  # pragma: no cover - только для типизации
    from engine.rendering.camera import Camera, CameraStack
    from engine.rendering.material import DirectionalLight
    from engine.rendering.renderer import RenderStats
    from engine.rendering.state import GLState

__all__ = ["FrameContext", "RenderGraph", "RenderPass", "RenderTarget"]

_log = logging.getLogger("engine.rendering.graph")


@dataclass(slots=True)
class RenderTarget:
    """Цель вывода: framebuffer с цветовыми текстурами и буфером глубины.

    Args:
        name: Имя цели (для логов).
        width: Ширина в пикселях.
        height: Высота в пикселях.
        color_attachments: Число цветовых привязанностей (1 — обычный кадр,
            3–4 — G-буфер для отложенного шейдинга).
        depth: Создавать буфер глубины.
        samples: Число сэмплов MSAA (0 — без сглаживания).
        hdr: Цвет в ``RGBA16F`` (значения > 1 не обрезаются). Если драйвер не
            собрал такую цель, создаётся ``RGBA8`` с предупреждением; итог
            виден в :attr:`is_hdr`.
    """

    name: str = "target"
    width: int = 1
    height: int = 1
    color_attachments: int = 1
    depth: bool = True
    samples: int = 0
    hdr: bool = False
    _hdr_active: bool = field(default=False, repr=False)
    _framebuffer: int = field(default=0, repr=False)
    _colors: list[int] = field(default_factory=list, repr=False)
    _depth_buffer: int = field(default=0, repr=False)
    _resolve_fbo: int = field(default=0, repr=False)
    _msaa_colors: list[int] = field(default_factory=list, repr=False)
    _msaa_samples: int = field(default=0, repr=False)

    @property
    def msaa_samples(self) -> int:
        """Фактическое число сэмплов (0 — цель без MSAA)."""
        return self._msaa_samples

    @property
    def is_hdr(self) -> bool:
        """``True``, если цвет цели создан в ``RGBA16F``."""
        return self._hdr_active and self._framebuffer != 0

    def _internal_format(self) -> int:
        GL = _gl()
        return GL.GL_RGBA16F if self._hdr_active else GL.GL_RGBA8

    def _pixel_type(self) -> int:
        GL = _gl()
        return GL.GL_HALF_FLOAT if self._hdr_active else GL.GL_UNSIGNED_BYTE

    @property
    def framebuffer(self) -> int:
        """Идентификатор FBO (0 — не создан)."""
        return self._framebuffer

    @property
    def color_textures(self) -> tuple[int, ...]:
        """Идентификаторы цветовых текстур."""
        return tuple(self._colors)

    @property
    def is_created(self) -> bool:
        """``True``, если цель создана в драйвере."""
        return self._framebuffer != 0

    @property
    def aspect(self) -> float:
        """Отношение ширины к высоте."""
        return self.width / max(1, self.height)

    def create(self) -> int:
        """Создать текстуры, буфер глубины и FBO.

        Returns:
            Идентификатор FBO.

        Raises:
            RenderError: Если драйвер не смог собрать цель.

        Note:
            При ``samples > 0`` кадр рисуется в мультисэмпловые рендербуферы, а
            итог сводит :meth:`resolve` в обычные текстуры (их читают
            ``read_pixels`` и шейдеры). Число сэмплов ограничивается
            ``GL_MAX_SAMPLES``; если драйвер не собрал MSAA-цель, она создаётся
            без сглаживания.
        """
        if self._framebuffer:
            return self._framebuffer
        self._hdr_active = self.hdr
        try:
            return self._create_impl()
        except RenderError:
            if not self._hdr_active:
                raise
        _log.warning(
            "RenderTarget '%s': RGBA16F не собран, откат на RGBA8 (без HDR)",
            self.name,
        )
        self._hdr_active = False
        return self._create_impl()

    def _create_impl(self) -> int:
        """Собрать цель в формате, выбранном :meth:`create`."""
        GL = _gl()

        if self.samples > 0 and self._create_msaa():
            return self._framebuffer

        framebuffer = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        self._colors = []
        for index in range(max(1, self.color_attachments)):
            texture = int(GL.glGenTextures(1))
            select_scratch_unit()
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                self._internal_format(),
                self.width,
                self.height,
                0,
                GL.GL_RGBA,
                self._pixel_type(),
                None,
            )
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
            )
            GL.glTexParameteri(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
            )
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0 + index,
                GL.GL_TEXTURE_2D,
                texture,
                0,
            )
            self._colors.append(texture)

        if self.color_attachments > 1:
            attachments = [
                GL.GL_COLOR_ATTACHMENT0 + index
                for index in range(self.color_attachments)
            ]
            GL.glDrawBuffers(self.color_attachments, attachments)

        if self.depth:
            depth_buffer = int(GL.glGenRenderbuffers(1))
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_buffer)
            GL.glRenderbufferStorage(
                GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, self.width, self.height
            )
            GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER,
                GL.GL_DEPTH_ATTACHMENT,
                GL.GL_RENDERBUFFER,
                depth_buffer,
            )
            self._depth_buffer = depth_buffer

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            self.dispose()
            raise RenderError(
                f"RenderTarget '{self.name}' не собран: статус 0x{status:04X}"
            )

        self._framebuffer = framebuffer
        _log.debug(
            "RenderTarget '%s' создан: %dx%d, цветовых привязанностей %d, глубина %s",
            self.name,
            self.width,
            self.height,
            self.color_attachments,
            self.depth,
        )
        return framebuffer

    def _create_msaa(self) -> bool:
        """Собрать цель с MSAA: рисующий FBO и FBO с текстурами для resolve.

        Returns:
            ``True``, если цель собрана; ``False`` — драйвер отказал (всё
            освобождено), и вызывающий создаёт обычную цель.
        """
        GL = _gl()

        samples = min(int(self.samples), int(GL.glGetIntegerv(GL.GL_MAX_SAMPLES)))
        if samples < 2:
            return False
        attachments = max(1, self.color_attachments)

        resolve_fbo = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, resolve_fbo)
        self._colors = []
        for index in range(attachments):
            texture = int(GL.glGenTextures(1))
            select_scratch_unit()
            GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, self._internal_format(), self.width,
                self.height, 0, GL.GL_RGBA, self._pixel_type(), None,
            )  # fmt: skip
            for name, value in (
                (GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR),
                (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
            ):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, name, value)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0 + index,
                GL.GL_TEXTURE_2D, texture, 0,
            )  # fmt: skip
            self._colors.append(texture)
        resolve_status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)

        draw_fbo = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, draw_fbo)
        self._msaa_colors = []
        for index in range(attachments):
            buffer = int(GL.glGenRenderbuffers(1))
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, buffer)
            GL.glRenderbufferStorageMultisample(
                GL.GL_RENDERBUFFER, samples, self._internal_format(),
                self.width, self.height,
            )
            GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0 + index,
                GL.GL_RENDERBUFFER, buffer,
            )  # fmt: skip
            self._msaa_colors.append(buffer)
        if attachments > 1:
            GL.glDrawBuffers(
                attachments,
                [GL.GL_COLOR_ATTACHMENT0 + index for index in range(attachments)],
            )
        if self.depth:
            depth_buffer = int(GL.glGenRenderbuffers(1))
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_buffer)
            GL.glRenderbufferStorageMultisample(
                GL.GL_RENDERBUFFER, samples, GL.GL_DEPTH_COMPONENT24,
                self.width, self.height,
            )  # fmt: skip
            GL.glFramebufferRenderbuffer(
                GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                GL.GL_RENDERBUFFER, depth_buffer,
            )  # fmt: skip
            self._depth_buffer = depth_buffer
        draw_status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        self._framebuffer = draw_fbo
        self._resolve_fbo = resolve_fbo
        if (
            resolve_status != GL.GL_FRAMEBUFFER_COMPLETE
            or draw_status != GL.GL_FRAMEBUFFER_COMPLETE
        ):
            _log.warning(
                "RenderTarget '%s': MSAA x%d не собран (0x%04X/0x%04X), "
                "цель без сглаживания",
                self.name, samples, resolve_status, draw_status,
            )  # fmt: skip
            self.dispose()
            return False
        self._msaa_samples = samples
        _log.debug("RenderTarget '%s' создан с MSAA x%d", self.name, samples)
        return True

    def resolve(self) -> None:
        """Свести мультисэмпловый кадр в текстуры цели.

        Для цели без MSAA ничего не делает. Вызывается в конце кадра
        (``Renderer.end_frame``) и перед чтением пикселей. Привязка FBO
        остаётся такой же, как до вызова (рисующий FBO цели), поэтому кэш
        :class:`GLState` не рассинхронизируется.
        """
        GL = _gl()

        if not self._resolve_fbo or not self._framebuffer:
            return
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._framebuffer)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._resolve_fbo)
        for index in range(len(self._colors)):
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0 + index)
            GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0 + index])
            GL.glBlitFramebuffer(
                0, 0, self.width, self.height, 0, 0, self.width, self.height,
                GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST,
            )  # fmt: skip
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._framebuffer)

    def resize(self, width: int, height: int) -> None:
        """Изменить размер цели (пересоздаёт буферы)."""
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height and self.is_created:
            return
        self.width, self.height = width, height
        self.dispose()
        self.create()

    def bind(self, state: GLState | None = None, *, viewport: bool = True) -> None:
        """Сделать цель активной."""
        GL = _gl()

        self.create()
        if state is not None:
            state.bind_framebuffer(self._framebuffer)
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._framebuffer)
        if viewport:
            if state is not None:
                state.set_viewport(0, 0, self.width, self.height)
            else:
                GL.glViewport(0, 0, self.width, self.height)

    def unbind(self, state: GLState | None = None) -> None:
        """Вернуть вывод на экран."""
        GL = _gl()

        if state is not None:
            state.bind_default_framebuffer()
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    @property
    def read_framebuffer(self) -> int:
        """FBO, из которого читается готовый кадр (после ``resolve``)."""
        return self._resolve_fbo or self._framebuffer

    def read_pixels(self, attachment: int = 0) -> np.ndarray:
        """Прочитать цветовую привязанность в массив ``(H, W, 4)`` uint8.

        Args:
            attachment: Индекс цветовой привязки.

        Returns:
            Массив пикселей в порядке снизу вверх (как в OpenGL).

        Raises:
            RenderError: Если цель не создана или индекс вне диапазона.
        """
        GL = _gl()

        if not self.is_created:
            raise RenderError(f"RenderTarget '{self.name}' не создан")
        if not 0 <= attachment < len(self._colors):
            raise RenderError(
                f"У '{self.name}' нет цветовой привязки {attachment}"
            )
        if self._resolve_fbo:
            self.resolve()
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._resolve_fbo)
        else:
            self.bind()
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0 + attachment)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        raw = GL.glReadPixels(
            0, 0, self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE
        )
        self.unbind()
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)

    def dispose(self) -> None:
        """Освободить буферы цели."""
        GL = _gl()

        for texture in self._colors:
            GL.glDeleteTextures([texture])
        self._colors.clear()
        if self._msaa_colors:
            GL.glDeleteRenderbuffers(len(self._msaa_colors), self._msaa_colors)
            self._msaa_colors = []
        if self._depth_buffer:
            GL.glDeleteRenderbuffers(1, [self._depth_buffer])
            self._depth_buffer = 0
        if self._resolve_fbo:
            GL.glDeleteFramebuffers(1, [self._resolve_fbo])
            self._resolve_fbo = 0
        self._msaa_samples = 0
        if self._framebuffer:
            GL.glDeleteFramebuffers(1, [self._framebuffer])
            self._framebuffer = 0

    def __repr__(self) -> str:
        state = "gpu" if self.is_created else "cpu"
        return (
            f"<RenderTarget '{self.name}' {self.width}x{self.height} "
            f"attachments={self.color_attachments} {state}>"
        )


@dataclass(slots=True)
class FrameContext:
    """Данные, доступные проходам во время выполнения кадра.

    Attributes:
        camera: Активная камера.
        light: Источник света сцены.
        state: Кэш состояний OpenGL.
        viewport: Размер области вывода.
        frame_index: Номер кадра (растёт с каждым ``end_frame``).
        clear_color: Цвет очистки фона.
        stats: Статистика текущего кадра.
        renderer: Ссылка на рендерер (для доступа к программам и очереди).
        stack: Стек камер кадра или ``None``, если камера одна.
    """

    camera: Camera
    light: DirectionalLight
    state: GLState
    viewport: tuple[int, int]
    frame_index: int = 0
    clear_color: tuple[float, float, float, float] = (0.09, 0.10, 0.13, 1.0)
    stats: RenderStats | None = None
    renderer: object | None = None
    stack: CameraStack | None = None

    @property
    def aspect(self) -> float:
        """Отношение сторон вьюпорта."""
        width, height = self.viewport
        return width / max(1, height)

    @property
    def viewport_rect(self) -> tuple[int, int, int, int]:
        """Прямоугольник вывода кадра ``(x, y, width, height)``.

        Проходы, которые меняют цель вывода (карта теней, G-буфер), обязаны
        вернуть этот прямоугольник после себя: кэш состояний помнит viewport,
        и без восстановления сцена рисуется в область чужого размера.
        """
        return (0, 0, *self.viewport)


@runtime_checkable
class RenderPass(Protocol):
    """Проход рендеринга.

    Проход получает :class:`FrameContext` и сам решает, что и куда рисовать:
    базовый forward-путь, теневые карты, G-буфер, пост-обработка — всё это
    одинаковые по форме проходы.
    """

    @property
    def name(self) -> str:
        """Имя прохода (для отладки и графа)."""
        ...

    def execute(self, context: FrameContext) -> None:
        """Выполнить проход."""
        ...


class RenderGraph:
    """Упорядоченный список проходов кадра.

    Граф сознательно простой: проходы выполняются в порядке добавления, а
    зависимости выражаются позицией. Такой формы достаточно для forward-пути,
    теней и G-буфера, а усложнение (автоматические барьеры, транзитные
    ресурсы, aliasing памяти) добавляется позже без смены интерфейса.
    """

    __slots__ = ("_context", "_frame_index", "name", "_passes")

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._passes: list[RenderPass] = []
        self._frame_index = 0
        self._context: FrameContext | None = None

    # ------------------------------------------------------------------ #
    # Состав графа
    # ------------------------------------------------------------------ #
    def add_pass(
        self, render_pass: RenderPass, *, index: int | None = None
    ) -> RenderPass:
        """Добавить проход в конец (или на указанную позицию).

        Args:
            render_pass: Проход, реализующий :class:`RenderPass`.
            index: Позиция вставки; ``None`` — в конец.

        Returns:
            Добавленный проход.
        """
        if index is None:
            self._passes.append(render_pass)
        else:
            self._passes.insert(index, render_pass)
        return render_pass

    def remove_pass(self, render_pass: RenderPass) -> bool:
        """Удалить проход.

        Returns:
            ``True``, если проход был в графе.
        """
        if render_pass in self._passes:
            self._passes.remove(render_pass)
            return True
        return False

    def pass_by_name(self, name: str) -> RenderPass | None:
        """Найти проход по имени."""
        for render_pass in self._passes:
            if render_pass.name == name:
                return render_pass
        return None

    @property
    def passes(self) -> tuple[RenderPass, ...]:
        """Проходы в порядке выполнения."""
        return tuple(self._passes)

    def __len__(self) -> int:
        return len(self._passes)

    def __iter__(self):
        return iter(self._passes)

    # ------------------------------------------------------------------ #
    # Выполнение
    # ------------------------------------------------------------------ #
    def execute(self, context: FrameContext, phase: str = "all") -> int:
        """Выполнить проходы.

        Args:
            context: Данные кадра.
            phase: ``"all"`` — все проходы; ``"main"`` — только проходы без
                атрибута ``after_post`` (рисуются в HDR-цель до пост-обработки);
                ``"late"`` — только проходы с ``after_post = True`` (UI поверх
                уже обработанного кадра, они рисуются в итоговую цель).

        Returns:
            Число выполненных проходов.
        """
        self._context = context
        executed = 0
        for render_pass in list(self._passes):
            late = bool(getattr(render_pass, "after_post", False))
            if (phase == "main" and late) or (phase == "late" and not late):
                continue
            render_pass.execute(context)
            executed += 1
        if phase != "main":
            self._frame_index += 1
        return executed

    @property
    def frame_index(self) -> int:
        """Сколько кадров выполнил граф."""
        return self._frame_index

    def clear(self) -> None:
        """Убрать все проходы."""
        self._passes.clear()

    def __repr__(self) -> str:
        names = ", ".join(render_pass.name for render_pass in self._passes)
        return f"<RenderGraph '{self.name}' [{names}]>"


def make_default_graph(render_pass: RenderPass | None = None) -> RenderGraph:
    """Создать граф из переданных проходов.

    Args:
        render_pass: Проход геометрии (обычно forward-путь рендерера);
            ``None`` оставит граф пустым.

    Returns:
        Готовый граф.
    """
    graph = RenderGraph("default")
    if render_pass is not None:
        graph.add_pass(render_pass)
    return graph


def describe_passes(passes: Sequence[RenderPass]) -> str:
    """Собрать строку с именами проходов (для логов)."""
    return " → ".join(render_pass.name for render_pass in passes) or "<пусто>"
