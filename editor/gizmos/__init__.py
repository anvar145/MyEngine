"""Гизмо редактора: визуальные манипуляторы и индикаторы в сцене.

Гизмо живут только в редакторе: движок ничего о них не знает.
:mod:`editor.gizmos.camera_gizmo` рисует иконку камеры и её усечённый объём,
:mod:`editor.gizmos.light_gizmo` — источники света с зонами влияния.
"""

from __future__ import annotations

from editor.gizmos.camera_gizmo import CameraGizmo, CameraGizmoStyle
from editor.gizmos.light_gizmo import LightGizmo, LightGizmoStyle, gizmos_for

__all__ = [
    "CameraGizmo",
    "CameraGizmoStyle",
    "LightGizmo",
    "LightGizmoStyle",
    "gizmos_for",
]
