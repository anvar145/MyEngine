"""Material cooker: разбор ``.mat.json`` и упаковка PBR-карт в одну текстуру ORM.

Формат исходника ``.mat.json``::

    {
      "name": "Brick", "shader": "pbr",
      "base_color": [1, 1, 1, 1], "metallic": 0.0, "roughness": 0.8,
      "emissive": [0, 0, 0], "normal_scale": 1.0, "occlusion_strength": 1.0,
      "alpha_mode": "opaque", "alpha_cutoff": 0.5, "double_sided": false,
      "uv_scale": [1, 1], "uv_offset": [0, 0],
      "textures": {"base_color": "brick_albedo.png", "normal": "brick_n.png",
                   "roughness": "brick_r.png", "ao": "brick_ao.png"}
    }

Пути текстур — относительно каталога ``.mat.json``. Все поля необязательны.
Карты ``ao``/``roughness``/``metallic`` (или готовая ``metallic_roughness``
в конвенции glTF) упаковываются в одну RGBA-текстуру: R = occlusion,
G = roughness, B = metallic — так шейдеру нужна одна выборка вместо трёх.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.core.resources.errors import AssetImportError

__all__ = [
    "MATERIAL_VERSION",
    "MaterialSpec",
    "pack_orm",
    "parse_material",
]

MATERIAL_VERSION = 1
ALPHA_MODES = ("opaque", "mask", "blend")
# слот → нужен ли он как отдельная ссылка (иначе входит в упаковку ORM)
LINKED_SLOTS = ("base_color", "normal", "emissive")
PACKED_SLOTS = ("ao", "roughness", "metallic", "metallic_roughness")
_ALIASES = {"albedo": "base_color", "occlusion": "ao", "normal_map": "normal"}
_KEYS = {
    "name", "shader", "version", "base_color", "metallic", "roughness",
    "emissive", "normal_scale", "occlusion_strength", "alpha_mode",
    "alpha_cutoff", "double_sided", "uv_scale", "uv_offset", "textures",
    "sorting_layer",
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    """Проверенное описание PBR-материала.

    Attributes:
        name: Имя материала.
        shader: Имя шейдера (``pbr`` — стандартный).
        base_color: Множитель albedo, RGBA.
        metallic: Множитель металличности (0..1).
        roughness: Множитель шероховатости (0..1).
        emissive: Цвет свечения, RGB (может быть больше 1 для HDR).
        normal_scale: Сила карты нормалей.
        occlusion_strength: Сила ambient occlusion (0..1).
        alpha_mode: ``opaque`` | ``mask`` | ``blend``.
        alpha_cutoff: Порог для ``mask``.
        sorting_layer: Слой сортировки прозрачных (больше — позже, поверх).
        double_sided: Рисовать обе стороны треугольников.
        uv_scale: Тайлинг UV.
        uv_offset: Смещение UV.
        textures: Слот → путь файла (относительно каталога материала).
    """

    name: str = ""
    shader: str = "pbr"
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    alpha_mode: str = "opaque"
    alpha_cutoff: float = 0.5
    sorting_layer: int = 0
    double_sided: bool = False
    uv_scale: tuple[float, float] = (1.0, 1.0)
    uv_offset: tuple[float, float] = (0.0, 0.0)
    textures: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> MaterialSpec:
        """Собрать и проверить описание (неизвестные ключи — ошибка, это опечатки).

        Raises:
            AssetImportError: При некорректных значениях.
        """
        if not isinstance(data, Mapping):
            raise AssetImportError("Материал должен быть JSON-объектом")
        unknown = sorted(set(data) - _KEYS)
        if unknown:
            raise AssetImportError(f"Неизвестные поля материала: {', '.join(unknown)}")
        version = data.get("version", MATERIAL_VERSION)
        if version != MATERIAL_VERSION:
            raise AssetImportError(f"Неподдерживаемая версия материала: {version}")
        alpha_mode = str(data.get("alpha_mode", "opaque")).lower()
        if alpha_mode not in ALPHA_MODES:
            raise AssetImportError(
                f"alpha_mode: допустимы {', '.join(ALPHA_MODES)}, "
                f"получено '{alpha_mode}'"
            )
        try:
            spec = cls(
                name=str(data.get("name", "")),
                shader=str(data.get("shader", "pbr")),
                base_color=_vector(data, "base_color", 4, (1.0, 1.0, 1.0, 1.0)),
                metallic=_unit(data, "metallic", 0.0),
                roughness=_unit(data, "roughness", 1.0),
                emissive=_vector(data, "emissive", 3, (0.0, 0.0, 0.0)),
                normal_scale=_number(data, "normal_scale", 1.0),
                occlusion_strength=_unit(data, "occlusion_strength", 1.0),
                alpha_mode=alpha_mode,
                alpha_cutoff=_unit(data, "alpha_cutoff", 0.5),
                sorting_layer=int(data.get("sorting_layer", 0)),
                double_sided=bool(data.get("double_sided", False)),
                uv_scale=_vector(data, "uv_scale", 2, (1.0, 1.0)),
                uv_offset=_vector(data, "uv_offset", 2, (0.0, 0.0)),
                textures=_textures(data.get("textures", {})),
            )
        except (TypeError, ValueError) as exc:
            raise AssetImportError(f"Некорректный материал: {exc}") from exc
        if min(spec.base_color) < 0 or min(spec.emissive) < 0:
            raise AssetImportError("base_color и emissive не могут быть отрицательными")
        return spec

    @property
    def packed_sources(self) -> dict[str, str]:
        """Карты, которые войдут в упаковку ORM."""
        return {k: v for k, v in self.textures.items() if k in PACKED_SLOTS}

    def to_dict(self) -> dict[str, Any]:
        """Скалярные параметры для ``material.json`` (без текстур)."""
        return {
            "name": self.name,
            "shader": self.shader,
            "base_color": list(self.base_color),
            "metallic": self.metallic,
            "roughness": self.roughness,
            "emissive": list(self.emissive),
            "normal_scale": self.normal_scale,
            "occlusion_strength": self.occlusion_strength,
            "alpha_mode": self.alpha_mode,
            "alpha_cutoff": self.alpha_cutoff,
            "sorting_layer": self.sorting_layer,
            "double_sided": self.double_sided,
            "uv_scale": list(self.uv_scale),
            "uv_offset": list(self.uv_offset),
        }


def parse_material(text: str) -> MaterialSpec:
    """Разобрать текст ``.mat.json``.

    Raises:
        AssetImportError: Если это не JSON или значения некорректны.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise AssetImportError(f"Некорректный JSON материала: {exc}") from exc
    return MaterialSpec.from_mapping(data)


def pack_orm(
    ao: np.ndarray | None = None,
    roughness: np.ndarray | None = None,
    metallic: np.ndarray | None = None,
    metallic_roughness: np.ndarray | None = None,
) -> np.ndarray:
    """Упаковать PBR-карты в RGBA8: R = occlusion, G = roughness, B = metallic.

    Каждая карта — ``uint8`` ``(H, W, C)``; одноканальные берутся из канала 0.
    ``metallic_roughness`` — готовая текстура glTF (G = roughness, B = metallic);
    явные ``roughness``/``metallic`` её перекрывают. Карты разных размеров
    приводятся к самому крупному (ближайший сосед). Недостающий канал — 255,
    то есть «множитель материала действует как есть».

    Raises:
        AssetImportError: Если карт нет или в ``metallic_roughness`` меньше 3 каналов.
    """
    maps = [m for m in (ao, roughness, metallic, metallic_roughness) if m is not None]
    if not maps:
        raise AssetImportError("Нечего упаковывать: нет ни одной PBR-карты")
    if metallic_roughness is not None and metallic_roughness.shape[2] < 3:
        raise AssetImportError("metallic_roughness должна иметь не менее 3 каналов")
    height = max(m.shape[0] for m in maps)
    width = max(m.shape[1] for m in maps)

    def fit(image: np.ndarray) -> np.ndarray:
        rows = np.arange(height) * image.shape[0] // height
        cols = np.arange(width) * image.shape[1] // width
        return image[np.ix_(rows, cols)]

    packed = np.full((height, width, 4), 255, dtype=np.uint8)
    if metallic_roughness is not None:
        fitted = fit(metallic_roughness)
        packed[:, :, 1] = fitted[:, :, 1]
        packed[:, :, 2] = fitted[:, :, 2]
    for channel, image in ((0, ao), (1, roughness), (2, metallic)):
        if image is not None:
            packed[:, :, channel] = fit(image)[:, :, 0]
    return packed


def _number(data: Mapping[str, Any], key: str, default: float) -> float:
    value = float(data.get(key, default))
    if not np.isfinite(value):
        raise ValueError(f"{key}: не конечное число")
    return value


def _unit(data: Mapping[str, Any], key: str, default: float) -> float:
    value = _number(data, key, default)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key}: ожидалось 0..1, получено {value}")
    return value


def _vector(
    data: Mapping[str, Any], key: str, size: int, default: tuple[float, ...]
) -> Any:
    raw = data.get(key, default)
    if not isinstance(raw, (list, tuple)) or len(raw) != size:
        raise ValueError(f"{key}: ожидался массив из {size} чисел")
    values = tuple(float(v) for v in raw)
    if not all(np.isfinite(values)):
        raise ValueError(f"{key}: не конечные числа")
    return values


def _textures(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ValueError("textures: ожидался объект «слот → путь»")
    result: dict[str, str] = {}
    for slot, path in raw.items():
        canonical = _ALIASES.get(slot, slot)
        if canonical not in LINKED_SLOTS + PACKED_SLOTS:
            raise ValueError(f"textures: неизвестный слот '{slot}'")
        if canonical in result:
            raise ValueError(f"textures: слот '{canonical}' указан дважды")
        if not isinstance(path, str) or not path:
            raise ValueError(f"textures.{slot}: ожидался путь к файлу")
        result[canonical] = path
    return result
