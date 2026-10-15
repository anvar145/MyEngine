"""Узкая фаза: специализированные тесты пар форм (без GJK/EPA).

Все функции пар имеют единую сигнатуру
``f(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b) -> ContactManifold | None``
(``rot`` — матрица 3x3 или кватернион xyzw). Нормаль направлена от A к B,
глубина проникновения положительна.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from engine.physics.mathutil import as_matrix, vec3
from engine.physics.shape import BoxShape, CapsuleShape, SphereShape

_EPS = 1e-9
_UP = np.array([0.0, 1.0, 0.0])


@dataclass(slots=True)
class ContactManifold:
    """Контакт пары: общая нормаль (A -> B) и 1..4 точки с глубинами."""

    normal: np.ndarray
    points: np.ndarray  # (k, 3)
    depths: np.ndarray  # (k,)

    @property
    def point(self) -> np.ndarray:
        """Точка самого глубокого проникновения."""
        return self.points[int(np.argmax(self.depths))]

    @property
    def depth(self) -> float:
        """Максимальная глубина проникновения."""
        return float(np.max(self.depths))

    def flipped(self) -> ContactManifold:
        """Тот же контакт при перестановке A и B."""
        return ContactManifold(-self.normal, self.points, self.depths)


def _single(normal, point, depth) -> ContactManifold:
    return ContactManifold(
        np.asarray(normal, dtype=np.float64),
        np.asarray(point, dtype=np.float64).reshape(1, 3),
        np.array([depth], dtype=np.float64),
    )


def _spheres(ca, ra, cb, rb, fallback=_UP) -> ContactManifold | None:
    """Контакт двух шаров с центрами ``ca``/``cb``."""
    d = cb - ca
    dist = float(np.linalg.norm(d))
    if dist >= ra + rb:
        return None
    normal = d / dist if dist > _EPS else np.asarray(fallback, dtype=np.float64)
    depth = ra + rb - dist
    return _single(normal, ca + normal * (ra - depth * 0.5), depth)


def sphere_sphere(sa, pa, ra, sb, pb, rb) -> ContactManifold | None:
    """Шар - шар."""
    return _spheres(vec3(pa), sa.radius, vec3(pb), sb.radius)


# --- шар - параллелепипед ------------------------------------------------


def _sphere_vs_box(center, radius, half, pb, rot) -> ContactManifold | None:
    """Шар (центр ``center``) против бокса; нормаль от шара к боксу."""
    local = rot.T @ (center - pb)
    clamped = np.clip(local, -half, half)
    diff = local - clamped
    dist = float(np.linalg.norm(diff))
    if dist > 1e-12:
        if dist >= radius:
            return None
        normal = -(rot @ (diff / dist))
        return _single(normal, rot @ clamped + pb, radius - dist)
    # Центр внутри бокса: выталкиваем по оси наименьшего проникновения.
    pen = half - np.abs(local)
    k = int(np.argmin(pen))
    sign = 1.0 if local[k] >= 0.0 else -1.0
    out_local = np.zeros(3)
    out_local[k] = sign
    surface = local.copy()
    surface[k] = sign * half[k]
    return _single(-(rot @ out_local), rot @ surface + pb, float(pen[k]) + radius)


def sphere_box(sa, pa, ra, sb, pb, rb) -> ContactManifold | None:
    """Шар - бокс."""
    return _sphere_vs_box(
        vec3(pa), sa.radius, np.asarray(sb.half_extents), vec3(pb), as_matrix(rb)
    )


# --- отрезки и капсулы ---------------------------------------------------


def _closest_on_segment(p, a, b) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom < _EPS:
        return a
    t = min(1.0, max(0.0, float((p - a) @ ab) / denom))
    return a + ab * t


def closest_segment_segment(p1, q1, p2, q2) -> tuple[np.ndarray, np.ndarray]:
    """Ближайшие точки двух отрезков (алгоритм Эриксона)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a <= _EPS and e <= _EPS:
        return p1, p2
    if a <= _EPS:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = float(d1 @ r)
        if e <= _EPS:
            t, s = 0.0, min(1.0, max(0.0, -c / a))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return p1 + d1 * s, p2 + d2 * t


def sphere_capsule(sa, pa, ra, sb, pb, rb) -> ContactManifold | None:
    """Шар - капсула (шар против ближайшей точки оси)."""
    ca = vec3(pa)
    a, b = sb.segment(pb, rb)
    cp = _closest_on_segment(ca, a, b)
    axis = as_matrix(rb)[:, 1]
    ax, ay, az = axis.tolist()
    # axis x (1, 0, 0), затем axis x (0, 0, 1) - без np.cross (вызов ≈0.3 мс).
    perp = np.array([0.0 * az - 0.0 * ay, az * 1.0 - ax * 0.0, ax * 0.0 - ay * 1.0])
    norm = np.linalg.norm(perp)
    if norm < 1e-6:
        perp = np.array([ay * 1.0 - az * 0.0, az * 0.0 - ax * 1.0, ax * 0.0 - ay * 0.0])
        norm = np.linalg.norm(perp)
    return _spheres(ca, sa.radius, cp, sb.radius, perp / norm)


def _cross3(u, v) -> tuple[float, float, float]:
    """Векторное произведение трёх чисел без ``np.cross`` (вызов ≈50 мкс)."""
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def _norm3(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _segment_pair_dist(p1, q1, p2, q2) -> float:
    """Расстояние между отрезками (алгоритм Эриксона) на чистых float.

    Args:
        p1, q1, p2, q2: Концы отрезков, списки из трёх чисел.
    """
    d1 = (q1[0] - p1[0], q1[1] - p1[1], q1[2] - p1[2])
    d2 = (q2[0] - p2[0], q2[1] - p2[1], q2[2] - p2[2])
    r = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]
    e = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]
    f = d2[0] * r[0] + d2[1] * r[1] + d2[2] * r[2]
    if a <= _EPS and e <= _EPS:
        s = t = 0.0
    elif a <= _EPS:
        s, t = 0.0, min(1.0, max(0.0, f / e))
    else:
        c = d1[0] * r[0] + d1[1] * r[1] + d1[2] * r[2]
        if e <= _EPS:
            t, s = 0.0, min(1.0, max(0.0, -c / a))
        else:
            b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]
            denom = a * e - b * b
            s = min(1.0, max(0.0, (b * f - c * e) / denom)) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(1.0, max(0.0, -c / a))
            elif t > 1.0:
                t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    dx = r[0] + d1[0] * s - d2[0] * t
    dy = r[1] + d1[1] * s - d2[1] * t
    dz = r[2] + d1[2] * s - d2[2] * t
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _capsule_capsule_overlap(sa, pa, ra, sb, pb, rb) -> bool:
    """Пересечение двух капсул без построения контакта (марш свипа)."""
    p1, q1 = sa.segment(pa, ra)
    p2, q2 = sb.segment(pb, rb)
    dist = _segment_pair_dist(p1.tolist(), q1.tolist(), p2.tolist(), q2.tolist())
    return dist < sa.radius + sb.radius


def capsule_capsule(sa, pa, ra, sb, pb, rb) -> ContactManifold | None:
    """Капсула - капсула; параллельные оси дают две точки."""
    p1, q1 = sa.segment(pa, ra)
    p2, q2 = sb.segment(pb, rb)
    c1, c2 = closest_segment_segment(p1, q1, p2, q2)
    d1, d2 = q1 - p1, q2 - p2
    d1t, d2t = d1.tolist(), d2.tolist()
    cross = _cross3(d1t, d2t)
    cross_len = _norm3(cross)
    fallback = cross
    if cross_len < 1e-9:
        fallback = _cross3(d1t, (1.0, 0.0, 0.0))
        if _norm3(fallback) < 1e-9:
            fallback = _cross3(d1t, (0.0, 0.0, 1.0))
    fallback = np.asarray(fallback, dtype=np.float64)
    fallback = fallback / np.linalg.norm(fallback)
    main = _spheres(c1, sa.radius, c2, sb.radius, fallback)
    if main is None:
        return None
    a = d1t[0] * d1t[0] + d1t[1] * d1t[1] + d1t[2] * d1t[2]
    if a > _EPS and cross_len < 1e-6 * math.sqrt(a) * _norm3(d2t):
        # Параллельные оси: добавляем вторую точку по границе перекрытия.
        s_vals = [min(1.0, max(0.0, float((p - p1) @ d1) / a)) for p in (p2, q2)]
        if abs(s_vals[0] - s_vals[1]) * math.sqrt(a) > 1e-4:
            pts, deps = [], []
            for s in s_vals:
                ca = p1 + d1 * s
                cb = _closest_on_segment(ca, p2, q2)
                m = _spheres(ca, sa.radius, cb, sb.radius, main.normal)
                if m is not None:
                    pts.append(m.points[0])
                    deps.append(m.depths[0])
            if len(pts) == 2:
                return ContactManifold(main.normal, np.array(pts), np.array(deps))
    return main


def _box_sdf(points, half, pb, rot) -> np.ndarray:
    """Знаковое расстояние от точек (m,3) до бокса (выпуклая функция)."""
    local = (points - pb) @ rot
    d = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside


def _closest_t_outside(a, b, half) -> float | None:
    """Параметр ``t`` точки отрезка ``a + t (b - a)``, ближайшей к боксу.

    Всё в системе координат бокса (``half`` — полуразмеры). Отрезок режется
    плоскостями граней на куски, где ближайшая область бокса постоянна, и на
    каждом куске квадрат расстояния минимизируется аналитически. Если отрезок
    проходит внутри бокса, возвращает ``None`` (нужен знаковый поиск).
    Ничьи (отрезок параллелен грани) усредняются: контакт получается по центру.
    """
    delta = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    cuts = [0.0, 1.0]
    for k in range(3):
        if abs(delta[k]) > 1e-12:
            for plane in (-half[k], half[k]):
                t = (plane - a[k]) / delta[k]
                if 0.0 < t < 1.0:
                    cuts.append(t)
    cuts.sort()
    best = float("inf")
    tied: list[float] = []
    for lo, hi in pairwise(cuts):
        if hi - lo < 1e-12:
            continue
        mid = 0.5 * (lo + hi)
        qa = qb = qc = 0.0
        outside = False
        for k in range(3):
            p = a[k] + mid * delta[k]
            if p > half[k]:
                u = a[k] - half[k]
            elif p < -half[k]:
                u = a[k] + half[k]
            else:
                continue
            outside = True
            qa += delta[k] * delta[k]
            qb += u * delta[k]
            qc += u * u
        if not outside:
            return None
        t = min(hi, max(lo, -qb / qa)) if qa > 1e-18 else mid
        f = (qa * t + 2.0 * qb) * t + qc
        if f < best - 1e-12:
            best, tied = f, [t]
        elif f <= best + 1e-12:
            tied.append(t)
    return sum(tied) / len(tied) if tied else 0.0


def _capsule_inside_exit(la, lb, h, radius, pbv, rot) -> ContactManifold:
    """Контакт капсулы, чья ось пересекает бокс: кратчайший вынос по оси граней.

    Для каждой из 6 граней считается сдвиг капсулы целиком, после которого её
    ось отстоит от грани не меньше ``radius``; берётся минимальный. Результат
    детерминирован (нет зависимости от знака «почти нулевой» координаты) и не
    выталкивает тонкий пол на другую сторону. Нормаль направлена от капсулы к
    боксу, точка — на грани у конца оси, глубже всего зашедшего в бокс.

    Args:
        la, lb: Концы оси в системе координат бокса (списки из 3 чисел).
        h: Полуразмеры бокса.
        radius: Радиус капсулы.
        pbv: Центр бокса в мировых координатах.
        rot: Матрица 3x3 бокса.
    """
    best_push = float("inf")
    best_k, best_sign, best_end = 0, 1.0, la
    for k in range(3):
        lo_end, hi_end = (la, lb) if la[k] <= lb[k] else (lb, la)
        # Сдвиг в +k: нижний конец оси должен уйти за грань h[k] + radius.
        push = h[k] + radius - lo_end[k]
        if push < best_push:
            best_push, best_k, best_sign, best_end = push, k, 1.0, lo_end
        # Сдвиг в -k: верхний конец оси должен уйти за грань -h[k] - radius.
        push = hi_end[k] + h[k] + radius
        if push < best_push:
            best_push, best_k, best_sign, best_end = push, k, -1.0, hi_end
    surface = np.array(best_end, dtype=np.float64)
    surface[best_k] = best_sign * h[best_k]
    out_local = np.zeros(3)
    out_local[best_k] = best_sign
    return _single(-(rot @ out_local), rot @ surface + pbv, best_push)


def capsule_box(sa, pa, ra, sb, pb, rb) -> ContactManifold | None:
    """Капсула - бокс: отрезок оси против бокса, до трёх точек контакта."""
    half = np.asarray(sb.half_extents)
    pbv, rot = vec3(pb), as_matrix(rb)
    p0, p1 = sa.segment(pa, ra)
    seg = p1 - p0
    # Грубая отбраковка: шар вокруг отрезка против описанного шара бокса.
    reach = sa.radius + float(np.linalg.norm(half))
    mid = 0.5 * (p0 + p1) - pbv
    if float(mid @ mid) > (reach + 0.5 * float(np.linalg.norm(seg))) ** 2:
        return None
    local_a = (p0 - pbv) @ rot
    local_b = (p1 - pbv) @ rot
    la, lb, h = local_a.tolist(), local_b.tolist(), half.tolist()
    t_fast = _closest_t_outside(la, lb, h)
    if t_fast is not None:
        # Ось целиком снаружи бокса: если ближайшая точка не ближе радиуса,
        # контакта нет и дорогие проверки шаром не нужны.
        d2 = 0.0
        for k in range(3):
            c = la[k] + t_fast * (lb[k] - la[k])
            if c > h[k]:
                c -= h[k]
            elif c < -h[k]:
                c += h[k]
            else:
                continue
            d2 += c * c
        if d2 >= sa.radius * sa.radius:
            return None
        t_best = t_fast
    else:
        # Ось пересекает бокс: шаром по самой глубокой точке выход неоднозначен
        # (в центре бокса знак выталкивания «дрожит»), поэтому выталкиваем
        # капсулу целиком по кратчайшей грани бокса.
        return _capsule_inside_exit(la, lb, h, sa.radius, pbv, rot)
    candidates = [t_best]
    for t_end in (0.0, 1.0):
        if abs(t_end - t_best) > 1e-4:
            candidates.append(t_end)
    found = []
    for t in candidates:
        m = _sphere_vs_box(p0 + seg * t, sa.radius, half, pbv, rot)
        if m is not None:
            found.append(m)
    if not found:
        return None
    primary = max(found, key=lambda m: m.depth)
    points, depths = [primary.points[0]], [primary.depths[0]]
    for m in found:
        if m is not primary and float(m.normal @ primary.normal) > 0.9:
            points.append(m.points[0])
            depths.append(m.depths[0])
    return ContactManifold(primary.normal, np.array(points), np.array(depths))


def box_capsule(sa, pa, ra, sb, pb, rb) -> ContactManifold | None:
    """Бокс - капсула."""
    m = capsule_box(sb, pb, rb, sa, pa, ra)
    return None if m is None else m.flipped()


# --- бокс - бокс (SAT + clip) ----------------------------------------------


def _clip_polygon(poly, plane_normal, offset):
    """Отсекает полигон полупространством ``plane_normal . x <= offset``."""
    out = []
    n = len(poly)
    nx, ny, nz = plane_normal
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        dc = nx * cur[0] + ny * cur[1] + nz * cur[2] - offset
        dn = nx * nxt[0] + ny * nxt[1] + nz * nxt[2] - offset
        if dc <= 0.0:
            out.append(cur)
        if (dc < 0.0 < dn) or (dn < 0.0 < dc):
            k = dc / (dc - dn)
            out.append((
                cur[0] + (nxt[0] - cur[0]) * k,
                cur[1] + (nxt[1] - cur[1]) * k,
                cur[2] + (nxt[2] - cur[2]) * k,
            ))
    return out


def _reduce_points(points, depths):
    """Оставляет до 4 точек: глубочайшую и наиболее разнесённые."""
    if len(points) <= 4:
        return points, depths
    pts = np.asarray(points)
    i0 = int(np.argmax(depths))
    d0 = np.linalg.norm(pts - pts[i0], axis=1)
    i1 = int(np.argmax(d0))
    edge = pts[i1] - pts[i0]
    cross = np.cross(pts - pts[i0], edge)
    area = np.linalg.norm(cross, axis=1)
    i2 = int(np.argmax(area))
    ref = cross[i2]
    side = cross @ ref
    cand = np.where(side < 0.0, area, -1.0)
    i3 = int(np.argmax(cand))
    chosen = [i0, i1, i2] + ([i3] if cand[i3] > 0.0 else [])
    chosen = list(dict.fromkeys(chosen))
    return [points[i] for i in chosen], [depths[i] for i in chosen]


def _face_contacts(ref_pos, ref_rot, ref_half, ref_axis, inc_pos, inc_rot, inc_half):
    """Клиппинг инцидентной грани по опорной. Нормаль опорной грани — наружу."""
    n_ref = ref_rot[:, ref_axis].copy()
    if float(n_ref @ (inc_pos - ref_pos)) < 0.0:
        n_ref = -n_ref
    center = ref_pos + n_ref * ref_half[ref_axis]
    # Инцидентная грань: наиболее противонаправленная n_ref.
    dots = n_ref @ inc_rot
    k = int(np.argmax(np.abs(dots)))
    n_inc = inc_rot[:, k] * (-1.0 if dots[k] > 0.0 else 1.0)
    inc_center = inc_pos + n_inc * inc_half[k]
    u_i, v_i = [i for i in range(3) if i != k]
    u = inc_rot[:, u_i] * inc_half[u_i]
    v = inc_rot[:, v_i] * inc_half[v_i]
    poly = [
        tuple(p) for p in (
            inc_center + u + v, inc_center - u + v, inc_center - u - v,
            inc_center + u - v,
        )
    ]
    for side_axis in [i for i in range(3) if i != ref_axis]:
        s = ref_rot[:, side_axis]
        h = float(ref_half[side_axis])
        c = float(s @ ref_pos)
        st = tuple(s.tolist())
        poly = _clip_polygon(poly, st, c + h)
        if not poly:
            break
        poly = _clip_polygon(poly, (-st[0], -st[1], -st[2]), -c + h)
        if not poly:
            break
    points, depths = [], []
    cx, cy, cz = center.tolist()
    nx, ny, nz = n_ref.tolist()
    for vx, vy, vz in poly:
        depth = (cx - vx) * nx + (cy - vy) * ny + (cz - vz) * nz
        if depth >= 0.0:
            half_d = depth * 0.5
            points.append((vx + nx * half_d, vy + ny * half_d, vz + nz * half_d))
            depths.append(depth)
    return n_ref, points, depths


def box_box(sa, pa, ra, sb, pb, rb) -> ContactManifold | None:
    """Бокс - бокс: SAT по 15 осям, до 4 точек контакта через клиппинг граней."""
    ha, hb = np.asarray(sa.half_extents), np.asarray(sb.half_extents)
    pa, pb = vec3(pa), vec3(pb)
    rot_a, rot_b = as_matrix(ra), as_matrix(rb)
    t = pb - pa

    ca, cb = rot_a.T[:, None, :], rot_b.T[None, :, :]
    cross = np.empty((3, 3, 3))
    cross[..., 0] = ca[..., 1] * cb[..., 2] - ca[..., 2] * cb[..., 1]
    cross[..., 1] = ca[..., 2] * cb[..., 0] - ca[..., 0] * cb[..., 2]
    cross[..., 2] = ca[..., 0] * cb[..., 1] - ca[..., 1] * cb[..., 0]
    cross = cross.reshape(9, 3)
    lens = np.sqrt(np.einsum("ij,ij->i", cross, cross))
    valid = np.concatenate([np.ones(6, bool), lens > 1e-4])
    axes = np.concatenate([rot_a.T, rot_b.T, cross / np.maximum(lens, 1e-12)[:, None]])
    r_a = np.abs(axes @ rot_a) @ ha
    r_b = np.abs(axes @ rot_b) @ hb
    overlap = r_a + r_b - np.abs(axes @ t)
    overlap = np.where(valid, overlap, np.inf)
    if np.any(overlap < 0.0):
        return None

    bias = np.zeros(15)
    bias[3:6] = 1e-4  # при равенстве предпочитаем грани A
    face_idx = int(np.argmin(overlap[:6] + bias[:6]))
    face_o = float(overlap[face_idx])
    edge_idx = 6 + int(np.argmin(overlap[6:]))
    edge_o = float(overlap[edge_idx])

    if edge_o < 0.95 * face_o - 1e-3:
        n = axes[edge_idx].copy()
        if float(n @ t) < 0.0:
            n = -n
        i, j = divmod(edge_idx - 6, 3)
        pt_a = pa.copy()
        pt_b = pb.copy()
        for k in range(3):
            if k != i:
                pt_a += rot_a[:, k] * ha[k] * (1.0 if rot_a[:, k] @ n > 0 else -1.0)
            if k != j:
                pt_b += rot_b[:, k] * hb[k] * (1.0 if rot_b[:, k] @ n < 0 else -1.0)
        da, db = rot_a[:, i], rot_b[:, j]
        r = pt_a - pt_b
        b = float(da @ db)
        c, f = float(da @ r), float(db @ r)
        denom = 1.0 - b * b
        s = (b * f - c) / denom if denom > 1e-9 else 0.0
        tt = (f - b * c) / denom if denom > 1e-9 else f
        point = 0.5 * ((pt_a + da * s) + (pt_b + db * tt))
        return _single(n, point, edge_o)

    if face_idx < 3:
        n_ref, pts, deps = _face_contacts(pa, rot_a, ha, face_idx, pb, rot_b, hb)
        normal = n_ref
    else:
        n_ref, pts, deps = _face_contacts(pb, rot_b, hb, face_idx - 3, pa, rot_a, ha)
        normal = -n_ref
    if not pts:
        # Вырожденный клиппинг: одна точка по центру пересечения.
        n = axes[face_idx] if float(axes[face_idx] @ t) >= 0.0 else -axes[face_idx]
        return _single(n, 0.5 * (pa + pb), face_o)
    pts, deps = _reduce_points(pts, deps)
    return ContactManifold(normal, np.array(pts), np.array(deps))


# --- диспетчер ---------------------------------------------------------------

_TABLE = {
    (SphereShape, SphereShape): (sphere_sphere, False),
    (SphereShape, BoxShape): (sphere_box, False),
    (BoxShape, SphereShape): (sphere_box, True),
    (SphereShape, CapsuleShape): (sphere_capsule, False),
    (CapsuleShape, SphereShape): (sphere_capsule, True),
    (BoxShape, BoxShape): (box_box, False),
    (BoxShape, CapsuleShape): (box_capsule, False),
    (CapsuleShape, BoxShape): (capsule_box, False),
    (CapsuleShape, CapsuleShape): (capsule_capsule, False),
}


def _capsule_box_hit(local_a, local_b, h, radius) -> bool:
    """Пересечение капсулы с боксом по концам оси в системе координат бокса.

    Args:
        local_a, local_b: Концы оси (списки из 3 чисел) в осях бокса.
        h: Полуразмеры бокса (список из 3 чисел).
        radius: Радиус капсулы.
    """
    t = _closest_t_outside(local_a, local_b, h)
    if t is None:  # ось проходит внутри бокса
        return True
    d2 = 0.0
    for k in range(3):
        c = local_a[k] + t * (local_b[k] - local_a[k])
        if c > h[k]:
            c -= h[k]
        elif c < -h[k]:
            c += h[k]
        else:
            continue
        d2 += c * c
    return d2 < radius * radius


def _capsule_box_overlap(sa, pa, ra, sb, pb, rb) -> bool:
    """Пересечение капсулы с боксом без построения контакта (для триггеров)."""
    half = np.asarray(sb.half_extents)
    pbv, rot = vec3(pb), as_matrix(rb)
    p0, p1 = sa.segment(pa, ra)
    local_a = ((p0 - pbv) @ rot).tolist()
    local_b = ((p1 - pbv) @ rot).tolist()
    return _capsule_box_hit(local_a, local_b, half.tolist(), sa.radius)


def overlaps(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b) -> bool:
    """Пересекаются ли формы (дешевле :func:`collide`: без контактов).

    Нужен триггерам, которым важен лишь факт перекрытия.
    """
    ta, tb = type(shape_a), type(shape_b)
    if ta is CapsuleShape and tb is BoxShape:
        return _capsule_box_overlap(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b)
    if ta is BoxShape and tb is CapsuleShape:
        return _capsule_box_overlap(shape_b, pos_b, rot_b, shape_a, pos_a, rot_a)
    if ta is CapsuleShape and tb is CapsuleShape:
        return _capsule_capsule_overlap(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b)
    return collide(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b) is not None


def collide(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b) -> ContactManifold | None:
    """Контакт произвольной пары форм (нормаль A -> B)."""
    func, swapped = _TABLE[(type(shape_a), type(shape_b))]
    if swapped:
        m = func(shape_b, pos_b, rot_b, shape_a, pos_a, rot_a)
        return None if m is None else m.flipped()
    return func(shape_a, pos_a, rot_a, shape_b, pos_b, rot_b)
