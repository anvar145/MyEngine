"""Сбор UI-виджетов сцены в :class:`UIBatch` и :class:`TextBatch`.

Общий для плеера и тестов. Раскладку считает :class:`engine.core.ui_layout.UILayout`
(та же, что использует ввод), здесь она превращается в квады и тексты.

Допущения MVP: отсечение (``clip``) действует на квады, но не на текст; картинки
читаются прямо из файла проекта (не через ``AssetDatabase``).

Example:
    >>> filler = UIFiller()
    >>> filler.fill(scene, screen, batch, text_batch, roots=(project_root,))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from engine.core.ui_components import UIButton, UIImage, UIPanel, UIText
from engine.core.ui_layout import UILayout
from engine.rendering.text.atlas import get_atlas
from engine.rendering.text.layout import layout_text
from engine.rendering.texture import GpuTexture

if TYPE_CHECKING:
    from engine.core.scene import Scene
    from engine.rendering.text.batch import TextBatch
    from engine.rendering.ui.batch import UIBatch
    from engine.rendering.ui.screen import UIScreen

__all__ = ["UIFiller"]

_log = logging.getLogger("engine.rendering.ui.scene")


def _with_alpha(
    color: tuple[float, float, float, float], opacity: float
) -> tuple[float, float, float, float]:
    return (color[0], color[1], color[2], color[3] * opacity)


class UIFiller:
    """Наполняет пакеты квадов и текста по UI-сцене.

    Пакеты пересобираются только при изменении раскладки; иначе остаются как есть.
    """

    def __init__(self) -> None:
        self.layout = UILayout()
        self._textures: dict[Path, GpuTexture | None] = {}
        self._roots: tuple[Path, ...] = ()
        self._last: tuple | None = None

    # ------------------------------------------------------------------ #
    def fill(
        self,
        scene: Scene,
        screen: UIScreen,
        batch: UIBatch,
        text_batch: TextBatch,
        roots: tuple[Path, ...] = (),
    ) -> int:
        """Очистить пакеты и добавить виджеты сцены.

        Args:
            scene: Сцена.
            screen: Экран UI (размер холста).
            batch: Пакет квадов.
            text_batch: Пакет текста.
            roots: Каталоги для поиска файлов (шрифты, картинки).

        Returns:
            Число виджетов.
        """
        self._roots = roots
        items = self.layout.update(scene, screen.canvas_size, self.measure)
        key = (id(items), screen.canvas_size, roots)
        if key == self._last and batch.fill_token is not None:
            return len(items)
        batch.clear()
        text_batch.clear()
        canvas_h = screen.canvas_size[1]
        for item in items:
            element = item.element
            if isinstance(element, UIPanel):
                self._panel(batch, item, element)
            elif isinstance(element, UIImage):
                self._image(batch, item, element)
            elif isinstance(element, UIButton):
                self._button(batch, text_batch, item, element, canvas_h)
            elif isinstance(element, UIText):
                self._text(text_batch, item, element, canvas_h)
        batch.fill_token = key
        self._last = key
        return len(items)

    def dispose(self) -> None:
        """Освободить загруженные текстуры картинок."""
        for texture in self._textures.values():
            if texture is not None:
                texture.dispose()
        self._textures.clear()
        self._last = None

    # ------------------------------------------------------------------ #
    # Виджеты
    # ------------------------------------------------------------------ #
    @staticmethod
    def _panel(batch: UIBatch, item, panel: UIPanel) -> None:
        x, y, w, h = item.rect
        border = min(panel.border_width, w / 2.0, h / 2.0)
        if border > 0.0:
            color = _with_alpha(panel.border_color, item.opacity)
            clip = item.clip
            batch.add_quad(x, y, w, border, color, clip=clip)
            batch.add_quad(x, y + h - border, w, border, color, clip=clip)
            batch.add_quad(x, y + border, border, h - 2 * border, color, clip=clip)
            batch.add_quad(
                x + w - border, y + border, border, h - 2 * border, color, clip=clip
            )
        batch.add_quad(
            x + border,
            y + border,
            w - 2 * border,
            h - 2 * border,
            _with_alpha(panel.color, item.opacity),
            clip=item.clip,
        )

    def _image(self, batch: UIBatch, item, image: UIImage) -> None:
        texture = None
        if image.texture_path:
            texture = self._texture(image.texture_path)
            if texture is None:
                return
        x, y, w, h = item.rect
        batch.add_quad(
            x,
            y,
            w,
            h,
            _with_alpha(image.color, item.opacity),
            uv=image.uv_rect,
            texture=texture,
            clip=item.clip,
        )

    def _button(self, batch, text_batch, item, button: UIButton, canvas_h) -> None:
        x, y, w, h = item.rect
        batch.add_quad(
            x, y, w, h, _with_alpha(button.current_color, item.opacity), clip=item.clip
        )
        if not button.text:
            return
        atlas = self._atlas(button.font_path, button.size_px, False, False)
        if atlas is None:
            return
        color = button.text_color
        text_batch.add_text(
            atlas,
            button.text,
            (x + w / 2.0, canvas_h - (y + h / 2.0), 0.0),
            1.0,
            color[:3],
            color[3] * item.opacity,
            "center",
            0.0,
            vertical_align="middle",
            sorting_layer=item.layer,
        )

    def _text(self, text_batch, item, label: UIText, canvas_h) -> None:
        text = label.text
        if not text.text:
            return
        atlas = self._atlas(text.font_path, text.size_px, text.bold, text.italic)
        if atlas is None:
            return
        x, y, w, h = item.rect
        anchor_x = {"left": x, "center": x + w / 2.0, "right": x + w}[text.align]
        anchor_y = {"top": y, "middle": y + h / 2.0, "bottom": y + h}[
            text.vertical_align
        ]
        text_batch.add_text(
            atlas,
            text.text,
            (anchor_x, canvas_h - anchor_y, 0.0),
            1.0,
            text.color,
            text.alpha * item.opacity,
            text.align,
            text.wrap_width,
            vertical_align=text.vertical_align,
            line_spacing=text.line_spacing,
            sorting_layer=item.layer,
            outline_color=text.outline_color,
            outline_width=text.outline_width,
        )

    # ------------------------------------------------------------------ #
    # Ресурсы
    # ------------------------------------------------------------------ #
    def _atlas(self, font_path: str, size_px: int, bold: bool, italic: bool):
        try:
            return get_atlas(
                font_path, size_px, bold=bold, italic=italic, roots=self._roots
            )
        except Exception as error:  # noqa: BLE001
            _log.warning("UI: шрифт '%s' недоступен: %s", font_path, error)
            return None

    def measure(self, label: UIText) -> tuple[float, float]:
        """Размер текста по содержимому (для ``size == (0, 0)``)."""
        text = label.text
        if not text.text:
            return (0.0, 0.0)
        atlas = self._atlas(text.font_path, text.size_px, text.bold, text.italic)
        if atlas is None:
            return (0.0, 0.0)
        layout = layout_text(
            atlas,
            text.text,
            wrap_width=text.wrap_width,
            line_spacing=text.line_spacing,
        )
        return (layout.width, layout.height)

    def _texture(self, path: str) -> GpuTexture | None:
        """Текстура картинки из файла (кэшируется; ``None`` — не найдена)."""
        candidates = [Path(path)] if Path(path).is_absolute() else [
            root / path for root in self._roots
        ]
        for file in candidates:
            if file in self._textures:
                return self._textures[file]
            if not file.is_file():
                continue
            texture = self._load(file)
            self._textures[file] = texture
            return texture
        _log.warning("UI: картинка '%s' не найдена", path)
        return None

    @staticmethod
    def _load(file: Path) -> GpuTexture | None:
        from PIL import Image

        try:
            with Image.open(file) as image:
                pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        except Exception as error:  # noqa: BLE001
            _log.warning("UI: не удалось прочитать '%s': %s", file, error)
            return None
        height, width = pixels.shape[:2]
        return GpuTexture.from_bytes(
            file.name, width, height, np.ascontiguousarray(pixels)
        )
