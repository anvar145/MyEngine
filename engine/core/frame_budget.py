"""Бюджет кадра по профилю качества: анимация, ИИ, физика (P4).

Профиль качества (``low``/``medium``/``high``) раньше задавал только рендер
(``QualityTier`` в ``engine.rendering.settings``). Бюджет — его вторая половина
для игровых подсистем этапа B:

- ``animation_lod`` — уровень детализации скелетной анимации: 0 — все
  персонажи каждый кадр, 1 — дальние реже, 2 — дальние ещё реже и без
  смешивания слоёв (читает B1);
- ``ai_tick_rate`` — сколько раз в секунду думает ИИ (восприятие, FSM/BT;
  читает B3), между тиками агенты только двигаются;
- ``physics_max_active`` — сколько бодрствующих динамических тел укладывается
  в бюджет; больше — плеер предупреждает в журнале и ``--stats-json``;
- ``target_fps`` — целевая частота кадров профиля (для отчётов и бенчмарков).

Конфиг один для плеера и редактора: :func:`resolve_quality` берёт профиль из
командной строки, затем из ``build.json`` пака, затем из ``*.project.json``
(поле ``quality``), иначе ``medium``; поле ``budget`` проекта переопределяет
отдельные числа профиля. Модуль не зависит от GL и Qt.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

__all__ = [
    "BUDGETS",
    "DEFAULT_TIER",
    "TIERS",
    "FrameBudget",
    "QualityConfig",
    "current_budget",
    "read_project_quality",
    "resolve_quality",
    "set_current_budget",
]

_log = logging.getLogger("engine.core.frame_budget")

#: Имена профилей (совпадают со значениями ``QualityTier``).
TIERS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_TIER = "medium"


@dataclass(frozen=True, slots=True)
class FrameBudget:
    """Числа бюджета одного профиля (см. описание модуля)."""

    target_fps: int = 60
    animation_lod: int = 1
    ai_tick_rate: float = 20.0
    physics_max_active: int = 192

    @property
    def frame_ms(self) -> float:
        """Время кадра при ``target_fps``, мс."""
        return 1000.0 / max(1, self.target_fps)

    @property
    def ai_interval(self) -> float:
        """Секунд между тиками ИИ."""
        return 1.0 / max(1e-6, self.ai_tick_rate)

    @classmethod
    def for_tier(cls, tier: str, **overrides: Any) -> FrameBudget:
        """Бюджет профиля с точечными переопределениями.

        Raises:
            ValueError: Неизвестный профиль.
        """
        try:
            base = BUDGETS[str(tier)]
        except KeyError:
            raise ValueError(f"неизвестный профиль качества: {tier!r}") from None
        return replace(base, **overrides) if overrides else base

    def with_overrides(self, data: Mapping[str, Any]) -> FrameBudget:
        """Копия с полями из словаря (``budget`` проекта).

        Неизвестные поля и значения не того типа пропускаются с предупреждением:
        опечатка в проекте не должна ронять игру.
        """
        known = {f.name for f in fields(self)}
        changes: dict[str, Any] = {}
        for name, value in data.items():
            if name not in known:
                _log.warning("budget: неизвестное поле %r пропущено", name)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _log.warning("budget.%s: ожидалось число, получено %r", name, value)
                continue
            if value < 0:
                _log.warning("budget.%s: отрицательное %r пропущено", name, value)
                continue
            current = getattr(self, name)
            changes[name] = type(current)(value)
        return replace(self, **changes) if changes else self

    def describe(self) -> str:
        """Строка для журнала."""
        return (
            f"{self.target_fps} FPS ({self.frame_ms:.1f} мс), "
            f"animation_lod {self.animation_lod}, ИИ {self.ai_tick_rate:g} Гц, "
            f"физика ≤ {self.physics_max_active} активных тел"
        )

    def to_dict(self) -> dict[str, Any]:
        """Поля как словарь (для ``--stats-json``)."""
        return asdict(self)


#: Бюджеты профилей. ``low`` — ориентир Intel HD 520 при 30 FPS.
BUDGETS: dict[str, FrameBudget] = {
    "low": FrameBudget(
        target_fps=30, animation_lod=2, ai_tick_rate=10.0, physics_max_active=64
    ),
    "medium": FrameBudget(
        target_fps=60, animation_lod=1, ai_tick_rate=20.0, physics_max_active=192
    ),
    "high": FrameBudget(
        target_fps=60, animation_lod=0, ai_tick_rate=30.0, physics_max_active=512
    ),
}


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Итог разрешения профиля: имя, бюджет и откуда взят профиль."""

    tier: str
    budget: FrameBudget
    source: str


def read_project_quality(texts: Iterable[str]) -> tuple[str | None, dict[str, Any]]:
    """Поля ``quality`` и ``budget`` из текстов ``*.project.json``.

    Returns:
        ``(профиль или None, переопределения бюджета)``; берётся первый
        разбираемый файл, как и для остальных полей проекта.
    """
    for text in texts:
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        tier = data.get("quality")
        if tier is not None and tier not in TIERS:
            _log.warning("quality проекта %r неизвестен, берётся профиль по умолчанию",
                         tier)
            tier = None
        budget = data.get("budget")
        return tier, budget if isinstance(budget, dict) else {}
    return None, {}


def _project_texts(project_path: str | Path | None, vfs: Any = None) -> list[str]:
    texts: list[str] = []
    if project_path is not None:
        for file in sorted(Path(project_path).glob("*.project.json")):
            try:
                texts.append(file.read_text(encoding="utf-8"))
            except OSError:
                continue
    if not texts and vfs is not None:
        for key in vfs.list_files(""):
            if "/" not in key and key.endswith(".project.json"):
                data = vfs.read_file(key)
                if data is not None:
                    texts.append(data.decode("utf-8", errors="replace"))
    return texts


def resolve_quality(
    project_path: str | Path | None = None,
    *,
    cli: str | None = None,
    build_info: Mapping[str, Any] | None = None,
    vfs: Any = None,
) -> QualityConfig:
    """Единый для плеера и редактора выбор профиля и бюджета.

    Порядок: ``cli`` → ``build_info["quality"]`` → ``quality`` проекта →
    ``medium``. Поле ``budget`` проекта применяется к любому выбранному профилю.
    """
    project_tier, overrides = read_project_quality(_project_texts(project_path, vfs))
    built = (build_info or {}).get("quality")
    if cli:
        tier, source = cli, "командная строка"
    elif built in TIERS:
        tier, source = str(built), "build.json"
    elif project_tier:
        tier, source = project_tier, "проект"
    else:
        tier, source = DEFAULT_TIER, "по умолчанию"
    budget = FrameBudget.for_tier(tier).with_overrides(overrides)
    return QualityConfig(tier, budget, source)


_current: FrameBudget = BUDGETS[DEFAULT_TIER]


def current_budget() -> FrameBudget:
    """Бюджет процесса (задают плеер и редактор при старте)."""
    return _current


def set_current_budget(budget: FrameBudget) -> None:
    """Задать бюджет процесса (подсистемы читают его через :func:`current_budget`)."""
    global _current
    _current = budget
