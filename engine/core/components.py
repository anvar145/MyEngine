"""Базовые компоненты движка: данные, которые понимают системы и редактор.

Компоненты намеренно не знают ни про OpenGL, ни про Qt: они описывают
**что** должно быть нарисовано, а как именно — решает слой рендеринга.
Благодаря этому сцена одинакова в редакторе, в плеере и в headless-тестах.

Example:
    >>> cube = scene.create_game_object("Cube")
    >>> cube.add_component(MeshRenderer, mesh="cube", color=Vector3(0.9, 0.6, 0.2))
"""

from __future__ import annotations

from engine.core.component import Component
from engine.core.math3d import Vector3
from engine.core.resources.asset_ref import AssetRef

__all__ = ["MeshRenderer", "SpriteRenderer", "TextRenderer"]

#: Имена встроенных процедурных мешей (см. ``engine.core.resources.primitives``).
BUILTIN_MESHES: tuple[str, ...] = ("cube", "sphere", "plane", "cylinder")


class MeshRenderer(Component):
    """Описание отрисовки объекта: какой меш и с какими параметрами материала.

    Args:
        mesh: Имя встроенного меша (``cube``, ``sphere``, ``plane``) или путь
            к файлу ассета (``assets/primitives/cube.obj``); удобство
            конструктора, в компоненте хранится ``mesh_ref``.
        color: Базовый цвет (albedo).
        roughness: Шероховатость поверхности.
        metallic: «Металличность» материала.
        visible: Рисовать ли объект.
        cast_shadow: Отбрасывает ли объект тень в карту теней.
        receive_shadow: Затемняется ли объект чужими тенями.
        sorting_layer: Слой сортировки прозрачных (``0`` — взять из материала).
        mesh_guid: GUID ассета меша (удобство конструктора).
        material_guid: GUID ассета ``.mat.json``.
        mesh_ref: Ссылка на меш; перекрывает ``mesh``/``mesh_guid``.

    Attributes:
        mesh_ref: Ссылка на меш (:class:`AssetRef`). Для встроенного примитива
            ``AssetRef(path="cube")`` без GUID. По GUID сцена переживает
            переименование и перемещение файла, путь - запасной вариант.
        mesh: Только чтение: путь/имя из ``mesh_ref``.
        mesh_guid: Только чтение: GUID из ``mesh_ref`` (``""`` - не задан).
        color: Цвет материала.
        roughness: Шероховатость.
        metallic: Металличность.
        visible: Видимость объекта.
        cast_shadow: Отбрасывает ли тень.
        receive_shadow: Принимает ли чужие тени.
        sorting_layer: Слой сортировки прозрачных (``0`` — из материала).
        material_guid: GUID ассета ``.mat.json`` (PBR-материал); пусто —
            материал строится из ``color``/``roughness``/``metallic``.
    """

    __slots__ = (
        "cast_shadow",
        "color",
        "material_guid",
        "mesh_ref",
        "metallic",
        "receive_shadow",
        "roughness",
        "sorting_layer",
        "visible",
    )

    def __init__(
        self,
        mesh: str = "cube",
        color: Vector3 | None = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        *,
        visible: bool = True,
        cast_shadow: bool = True,
        receive_shadow: bool = True,
        sorting_layer: int = 0,
        mesh_guid: str = "",
        material_guid: str = "",
        mesh_ref: AssetRef | None = None,
    ) -> None:
        super().__init__()
        self.sorting_layer = int(sorting_layer)
        self.mesh_ref = mesh_ref or AssetRef(
            guid=str(mesh_guid) or None, path=str(mesh) or "cube"
        )
        self.material_guid = str(material_guid)
        self.color = color or Vector3(0.82, 0.82, 0.86)
        self.roughness = float(roughness)
        self.metallic = float(metallic)
        self.visible = bool(visible)
        self.cast_shadow = bool(cast_shadow)
        self.receive_shadow = bool(receive_shadow)

    @property
    def mesh(self) -> str:
        """Путь к файлу меша или имя примитива (только чтение)."""
        return self.mesh_ref.path or ""

    @property
    def mesh_guid(self) -> str:
        """GUID меша (``""`` - не задан; только чтение)."""
        return self.mesh_ref.guid or ""

    @property
    def is_builtin(self) -> bool:
        """``True``, если используется встроенный процедурный меш."""
        return self.mesh_ref.is_primitive

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент в словарь (для сохранения сцены)."""
        payload: dict[str, object] = {
            "mesh_ref": self.mesh_ref.to_dict(),
            "color": list(self.color.to_tuple()),
            "roughness": self.roughness,
            "metallic": self.metallic,
            "visible": self.visible,
            "cast_shadow": self.cast_shadow,
            "receive_shadow": self.receive_shadow,
        }
        if self.sorting_layer:
            payload["sorting_layer"] = self.sorting_layer
        if self.material_guid:
            payload["material_guid"] = self.material_guid
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MeshRenderer:
        """Восстановить компонент из словаря.

        Читает и старую форму (``mesh`` + ``mesh_guid``).

        Args:
            data: Словарь, полученный из :meth:`to_dict`.

        Returns:
            Новый компонент.
        """
        raw_color = data.get("color") or (0.82, 0.82, 0.86)
        color = Vector3.from_iterable(raw_color)  # type: ignore[arg-type]
        raw_ref = data.get("mesh_ref")
        mesh_ref = AssetRef.from_dict(raw_ref) if isinstance(raw_ref, dict) else None
        return cls(
            mesh=str(data.get("mesh", "cube")),
            mesh_ref=mesh_ref,
            color=color,
            roughness=float(data.get("roughness", 0.5)),  # type: ignore[arg-type]
            metallic=float(data.get("metallic", 0.0)),  # type: ignore[arg-type]
            visible=bool(data.get("visible", True)),
            cast_shadow=bool(data.get("cast_shadow", True)),
            receive_shadow=bool(data.get("receive_shadow", True)),
            sorting_layer=int(data.get("sorting_layer", 0)),  # type: ignore[arg-type]
            mesh_guid=str(data.get("mesh_guid", "")),
            material_guid=str(data.get("material_guid", "")),
        )

    def __repr__(self) -> str:
        return (
            f"<MeshRenderer mesh='{self.mesh}' color={self.color} "
            f"visible={self.visible}>"
        )


class SpriteRenderer(Component):
    """Описание спрайта: кадр атласа, цвет и отражение.

    Положение, поворот вокруг Z и масштаб берутся из ``Transform`` объекта;
    рисует спрайты слой рендеринга (``SpriteBatch``).

    Args:
        sprite: Имя кадра в атласе.
        atlas_guid: GUID ассета ``*.atlas.json`` (удобство конструктора).
        color: Множитель цвета RGBA.
        flip_x: Отразить по горизонтали.
        flip_y: Отразить по вертикали.
        visible: Рисовать ли спрайт.
        sprite_ref: Ссылка на атлас; перекрывает ``atlas_guid``.

    Attributes:
        sprite: Имя кадра в атласе (это не ссылка на ассет).
        sprite_ref: Ссылка на атлас ``*.atlas.json`` или ``None``.
        atlas_guid: Только чтение: GUID из ``sprite_ref`` (``""`` - не задан).
        color: Цвет RGBA.
        flip_x: Отражение по X.
        flip_y: Отражение по Y.
        visible: Видимость.
    """

    __slots__ = ("color", "flip_x", "flip_y", "sprite", "sprite_ref", "visible")

    def __init__(
        self,
        sprite: str = "",
        atlas_guid: str = "",
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        *,
        flip_x: bool = False,
        flip_y: bool = False,
        visible: bool = True,
        sprite_ref: AssetRef | None = None,
    ) -> None:
        super().__init__()
        self.sprite = str(sprite)
        self.sprite_ref = sprite_ref or (
            AssetRef(guid=str(atlas_guid)) if atlas_guid else None
        )
        self.color = tuple(float(c) for c in color)
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)
        self.visible = bool(visible)

    @property
    def atlas_guid(self) -> str:
        """GUID атласа (``""`` - не задан; только чтение)."""
        return (self.sprite_ref.guid if self.sprite_ref else None) or ""

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент в словарь."""
        return {
            "sprite": self.sprite,
            "sprite_ref": self.sprite_ref.to_dict() if self.sprite_ref else None,
            "color": list(self.color),
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "visible": self.visible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SpriteRenderer:
        """Восстановить компонент из словаря, полученного из :meth:`to_dict`."""
        color = data.get("color") or (1.0, 1.0, 1.0, 1.0)
        raw_ref = data.get("sprite_ref")
        sprite_ref = AssetRef.from_dict(raw_ref) if isinstance(raw_ref, dict) else None
        return cls(
            sprite=str(data.get("sprite", "")),
            atlas_guid=str(data.get("atlas_guid", "")),
            sprite_ref=sprite_ref,
            color=tuple(color),  # type: ignore[arg-type]
            flip_x=bool(data.get("flip_x", False)),
            flip_y=bool(data.get("flip_y", False)),
            visible=bool(data.get("visible", True)),
        )

    def __repr__(self) -> str:
        return f"<SpriteRenderer sprite='{self.sprite}' visible={self.visible}>"


#: Допустимые значения горизонтального выравнивания текста.
TEXT_ALIGNS: tuple[str, ...] = ("left", "center", "right")
#: Допустимые значения вертикального выравнивания текста.
TEXT_VERTICAL_ALIGNS: tuple[str, ...] = ("top", "middle", "bottom")


def _rgb(value: object, name: str) -> tuple[float, float, float]:
    try:
        r, g, b = (float(c) for c in value)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        raise ValueError(f"{name}: ожидаются три числа RGB") from None
    return (r, g, b)


class TextRenderer(Component):
    """Описание текста в сцене: строка, шрифт и параметры вёрстки.

    Только данные: растеризацию шрифта и рисование выполняет слой
    рендеринга (``engine.rendering.text``). Положение берётся из ``Transform``.

    Args:
        text: Текст (``\n`` — перенос строки).
        font_path: Путь к TTF/OTF (``""`` — шрифт по умолчанию).
        size_px: Размер глифов в пикселях атласа (> 0).
        color: Цвет RGB.
        alpha: Непрозрачность, 0..1.
        align: Горизонтальное выравнивание: ``left``/``center``/``right``.
        vertical_align: Вертикальное: ``top``/``middle``/``bottom``.
        wrap_width: Ширина переноса в мировых единицах (0 — без переноса).
        line_spacing: Множитель межстрочного интервала (> 0).
        bold: Жирное начертание (синтетическое).
        italic: Курсив (синтетический).
        outline_color: Цвет обводки RGB.
        outline_width: Толщина обводки в пикселях атласа (0 — нет).
        billboard: Всегда лицом к камере.
        units_per_pixel: Мировых единиц на пиксель атласа (> 0).
        visible: Рисовать ли текст.
        sorting_layer: Слой сортировки (больше — позже).

    Raises:
        ValueError: Если значение выходит за допустимые границы.
    """

    __slots__ = (
        "alpha",
        "align",
        "billboard",
        "bold",
        "color",
        "font_path",
        "italic",
        "line_spacing",
        "outline_color",
        "outline_width",
        "size_px",
        "sorting_layer",
        "text",
        "units_per_pixel",
        "vertical_align",
        "visible",
        "wrap_width",
    )

    def __init__(
        self,
        text: str = "Text",
        font_path: str = "",
        size_px: int = 32,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        alpha: float = 1.0,
        align: str = "left",
        vertical_align: str = "top",
        wrap_width: float = 0.0,
        line_spacing: float = 1.0,
        *,
        bold: bool = False,
        italic: bool = False,
        outline_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        outline_width: float = 0.0,
        billboard: bool = True,
        units_per_pixel: float = 0.01,
        visible: bool = True,
        sorting_layer: int = 0,
    ) -> None:
        super().__init__()
        if int(size_px) <= 0:
            raise ValueError("size_px должен быть положительным")
        if float(units_per_pixel) <= 0:
            raise ValueError("units_per_pixel должен быть положительным")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha должна быть в диапазоне [0, 1]")
        if align not in TEXT_ALIGNS:
            raise ValueError(f"align должен быть одним из {TEXT_ALIGNS}")
        if vertical_align not in TEXT_VERTICAL_ALIGNS:
            raise ValueError(
                f"vertical_align должен быть одним из {TEXT_VERTICAL_ALIGNS}"
            )
        if float(wrap_width) < 0:
            raise ValueError("wrap_width не может быть отрицательным")
        if float(line_spacing) <= 0:
            raise ValueError("line_spacing должен быть положительным")
        if float(outline_width) < 0:
            raise ValueError("outline_width не может быть отрицательным")
        self.text = str(text)
        self.font_path = str(font_path)
        self.size_px = int(size_px)
        self.color = _rgb(color, "color")
        self.alpha = float(alpha)
        self.align = align
        self.vertical_align = vertical_align
        self.wrap_width = float(wrap_width)
        self.line_spacing = float(line_spacing)
        self.bold = bool(bold)
        self.italic = bool(italic)
        self.outline_color = _rgb(outline_color, "outline_color")
        self.outline_width = float(outline_width)
        self.billboard = bool(billboard)
        self.units_per_pixel = float(units_per_pixel)
        self.visible = bool(visible)
        self.sorting_layer = int(sorting_layer)

    def to_dict(self) -> dict[str, object]:
        """Сериализовать компонент в словарь."""
        return {
            "text": self.text,
            "font_path": self.font_path,
            "size_px": self.size_px,
            "color": list(self.color),
            "alpha": self.alpha,
            "align": self.align,
            "vertical_align": self.vertical_align,
            "wrap_width": self.wrap_width,
            "line_spacing": self.line_spacing,
            "bold": self.bold,
            "italic": self.italic,
            "outline_color": list(self.outline_color),
            "outline_width": self.outline_width,
            "billboard": self.billboard,
            "units_per_pixel": self.units_per_pixel,
            "visible": self.visible,
            "sorting_layer": self.sorting_layer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TextRenderer:
        """Восстановить компонент; отсутствующие поля берутся по умолчанию."""
        kwargs: dict[str, object] = {}
        for name in cls.__slots__:
            if name in data:
                kwargs[name] = data[name]
        for name in ("color", "outline_color"):
            if name in kwargs:
                kwargs[name] = tuple(kwargs[name])  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]

    #: Глобальный счётчик правок любых ``TextRenderer`` (для кэшей рендера).
    revision = 0

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        TextRenderer.revision += 1

    def __repr__(self) -> str:
        return f"<TextRenderer text={self.text!r} visible={self.visible}>"
