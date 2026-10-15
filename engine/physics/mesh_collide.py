"""Узкая фаза «форма против треугольника» для статических mesh-коллайдеров.

Чистый Python над кортежами float: проверок на запрос единицы-десятки, а
накладные расходы numpy на таких размерах больше самой работы. Нормали везде
направлены от поверхности к форме (наружу из поверхности).
"""

from __future__ import annotations

import math

from engine.physics.bvh import closest_point_triangle

__all__ = ["box_triangle_overlap", "capsule_triangle"]

_EPS = 1e-12


def _seg_seg_dist2(p1, q1, p2, q2):
    """Ближайшие точки двух отрезков (Эрикссон): ``(точка1, точка2, dist²)``."""
    d1x, d1y, d1z = q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2]
    d2x, d2y, d2z = q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2]
    rx, ry, rz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    if a <= _EPS and e <= _EPS:
        s = t = 0.0
    elif a <= _EPS:
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= _EPS:
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > _EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)
    c1 = (p1[0] + d1x * s, p1[1] + d1y * s, p1[2] + d1z * s)
    c2 = (p2[0] + d2x * t, p2[1] + d2y * t, p2[2] + d2z * t)
    dx, dy, dz = c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]
    return c1, c2, dx * dx + dy * dy + dz * dz


def capsule_triangle(p0, p1, radius: float, a, b, c):
    """Пересечение капсулы (ось ``p0 - p1``, радиус) с треугольником ``abc``.

    Returns:
        ``None``, если капсула не касается, иначе ``(nx, ny, nz, depth, px, py,
        pz)``: единичная нормаль от поверхности к капсуле, глубина проникновения
        вдоль неё и точка контакта на треугольнике. Если ось пронзает грань
        насквозь, выталкивание идёт в сторону, где глубина меньше.
    """
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    acx, acy, acz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = aby * acz - abz * acy
    ny = abz * acx - abx * acz
    nz = abx * acy - aby * acx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < _EPS:
        return None  # вырожденный треугольник
    nx, ny, nz = nx / length, ny / length, nz / length
    d0 = nx * (p0[0] - a[0]) + ny * (p0[1] - a[1]) + nz * (p0[2] - a[2])
    d1 = nx * (p1[0] - a[0]) + ny * (p1[1] - a[1]) + nz * (p1[2] - a[2])
    if (d0 > radius and d1 > radius) or (d0 < -radius and d1 < -radius):
        return None  # обе точки оси дальше радиуса по одну сторону плоскости
    if d0 * d1 < 0.0:
        t = d0 / (d0 - d1)
        x = (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t,
             p0[2] + (p1[2] - p0[2]) * t)
        q = closest_point_triangle(x, a, b, c)
        if (q[0] - x[0]) ** 2 + (q[1] - x[1]) ** 2 + (q[2] - x[2]) ** 2 < 1e-18:
            # сторона «+n»: отрицательный конец должен подняться до +radius
            neg = -min(d0, d1)
            pos = max(d0, d1)
            up = radius + neg
            down = radius + pos
            if up <= down:
                return (nx, ny, nz, up, x[0], x[1], x[2])
            return (-nx, -ny, -nz, down, x[0], x[1], x[2])
    best = math.inf
    on_axis = on_tri = None
    for end in (p0, p1):
        q = closest_point_triangle(end, a, b, c)
        dist2 = (end[0] - q[0]) ** 2 + (end[1] - q[1]) ** 2 + (end[2] - q[2]) ** 2
        if dist2 < best:
            best, on_axis, on_tri = dist2, end, q
    for e0, e1 in ((a, b), (b, c), (c, a)):
        s_pt, e_pt, dist2 = _seg_seg_dist2(p0, p1, e0, e1)
        if dist2 < best:
            best, on_axis, on_tri = dist2, s_pt, e_pt
    if best >= radius * radius:
        return None
    dist = math.sqrt(best)
    if dist > 1e-9:
        return (
            (on_axis[0] - on_tri[0]) / dist,
            (on_axis[1] - on_tri[1]) / dist,
            (on_axis[2] - on_tri[2]) / dist,
            radius - dist,
            on_tri[0], on_tri[1], on_tri[2],
        )
    # ось лежит на треугольнике: нормаль грани в сторону середины оси
    mid = ((p0[0] + p1[0]) * 0.5 - a[0]) * nx + ((p0[1] + p1[1]) * 0.5 - a[1]) * ny
    mid += ((p0[2] + p1[2]) * 0.5 - a[2]) * nz
    if mid < 0.0:
        return (-nx, -ny, -nz, radius, on_tri[0], on_tri[1], on_tri[2])
    return (nx, ny, nz, radius, on_tri[0], on_tri[1], on_tri[2])


def box_triangle_overlap(center, axes, half, a, b, c) -> bool:
    """Пересекает ли ориентированный бокс треугольник (SAT по 13 осям).

    Args:
        center: Центр бокса.
        axes: Строки ``axes[i]`` - мировые направления локальных осей бокса.
        half: Половинные размеры по локальным осям.
        a: Вершины треугольника в мировых координатах.
        b: Вторая вершина.
        c: Третья вершина.
    """
    tri = []
    for v in (a, b, c):
        rx, ry, rz = v[0] - center[0], v[1] - center[1], v[2] - center[2]
        tri.append((
            rx * axes[0][0] + ry * axes[0][1] + rz * axes[0][2],
            rx * axes[1][0] + ry * axes[1][1] + rz * axes[1][2],
            rx * axes[2][0] + ry * axes[2][1] + rz * axes[2][2],
        ))
    hx, hy, hz = half
    # оси бокса: сравнение с AABB треугольника
    for k, h in ((0, hx), (1, hy), (2, hz)):
        values = (tri[0][k], tri[1][k], tri[2][k])
        if min(values) > h or max(values) < -h:
            return False
    v0, v1, v2 = tri
    edges = (
        (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]),
        (v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]),
        (v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]),
    )
    # нормаль треугольника
    e0, e1 = edges[0], edges[1]
    nx = e0[1] * e1[2] - e0[2] * e1[1]
    ny = e0[2] * e1[0] - e0[0] * e1[2]
    nz = e0[0] * e1[1] - e0[1] * e1[0]
    off = nx * v0[0] + ny * v0[1] + nz * v0[2]
    if abs(off) > hx * abs(nx) + hy * abs(ny) + hz * abs(nz) + 1e-12:
        return False
    # 9 осей: ребро x ось бокса
    for ex, ey, ez in edges:
        for ax in ((0.0, -ez, ey), (ez, 0.0, -ex), (-ey, ex, 0.0)):
            if ax[0] == 0.0 and ax[1] == 0.0 and ax[2] == 0.0:
                continue
            p0 = ax[0] * v0[0] + ax[1] * v0[1] + ax[2] * v0[2]
            p1 = ax[0] * v1[0] + ax[1] * v1[1] + ax[2] * v1[2]
            p2 = ax[0] * v2[0] + ax[1] * v2[1] + ax[2] * v2[2]
            r = hx * abs(ax[0]) + hy * abs(ax[1]) + hz * abs(ax[2])
            if min(p0, p1, p2) > r + 1e-12 or max(p0, p1, p2) < -r - 1e-12:
                return False
    return True
