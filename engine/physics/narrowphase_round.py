"""Пакетная узкая фаза пар с шаром и капсулой (numpy).

Функции повторяют скалярные :func:`engine.physics.narrowphase.sphere_sphere`,
``sphere_box``, ``sphere_capsule``, ``capsule_box`` и ``capsule_capsule`` сразу
для множества пар. Скалярные версии остаются эталоном и фолбэком (формы без
пакетной версии).

Детерминизм: результат пары зависит только от её собственных входов, а не от
положения в пакете и его размера. Поэтому нет ``matmul``/``einsum``/``cross``/
``norm`` (BLAS и редукции могут считать «хвосты» иначе) — везде явная
арифметика поэлементных операций, ветвления выражены масками (``np.where``)
с «безопасным» делителем. Порядок точек контакта совпадает со скалярными
функциями (он задаёт слоты warm starting).

Все функции возвращают ``(hit, normal, count, points, depths)``: признак
контакта ``(n,)``, нормаль ``(n, 3)`` от A к B, число точек ``(n,)``, точки
``(n, K, 3)`` и глубины ``(n, K)``; для пар без контакта остальные поля не
определены.
"""

from __future__ import annotations

import numpy as np

from engine.physics.narrowphase_batch import _dot3
from engine.physics.store import KIND_BOX, KIND_CAPSULE, KIND_SPHERE

_EPS = 1e-9
_UP = np.array([0.0, 1.0, 0.0])
_AXIS_X = np.array([1.0, 0.0, 0.0])
_AXIS_Z = np.array([0.0, 0.0, 1.0])


# --- общие помощники ------------------------------------------------------


def _norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(_dot3(a, a))


def _safe(x: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Делитель, в котором «плохие» элементы заменены единицей."""
    return np.where(bad, 1.0, x)


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.broadcast_to(b, a.shape)
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    return np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=1)


def _to_local(rot: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``rot.T @ v`` для пакета матриц ``(n, 3, 3)`` и векторов ``(n, 3)``."""
    return np.stack([_dot3(rot[:, :, i], v) for i in range(3)], axis=1)


def _to_world(rot: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``rot @ v`` для пакета матриц ``(n, 3, 3)`` и векторов ``(n, 3)``."""
    return np.stack([_dot3(rot[:, i, :], v) for i in range(3)], axis=1)


def _segment(pos: np.ndarray, rot: np.ndarray, half_height: np.ndarray):
    """Концы оси капсулы (центры полусфер); ось — локальный Y."""
    axis = rot[:, :, 1] * half_height[:, None]
    return pos - axis, pos + axis


def _closest_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ближайшая к ``p`` точка отрезка ``ab``."""
    ab = b - a
    denom = _dot3(ab, ab)
    degenerate = denom < _EPS
    t = _clamp01(_dot3(p - a, ab) / _safe(denom, degenerate))
    return np.where(degenerate[:, None], a, a + ab * t[:, None])


def _closest_segments(p1, q1, p2, q2):
    """Ближайшие точки двух отрезков (алгоритм Эриксона, как скалярный)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = _dot3(d1, d1), _dot3(d2, d2), _dot3(d2, r)
    a_zero, e_zero = a <= _EPS, e <= _EPS
    safe_a, safe_e = _safe(a, a_zero), _safe(e, e_zero)
    c = _dot3(d1, r)
    b = _dot3(d1, d2)
    denom = a * e - b * b
    ok = denom > 1e-12
    s_gen = np.where(
        ok, _clamp01((b * f - c * e) / _safe(denom, ~ok)), 0.0
    )
    t_gen = (b * s_gen + f) / safe_e
    s_low = _clamp01(-c / safe_a)  # t < 0 (и e == 0)
    s_high = _clamp01((b - c) / safe_a)  # t > 1
    s = np.where(t_gen < 0.0, s_low, np.where(t_gen > 1.0, s_high, s_gen))
    t = np.where(t_gen < 0.0, 0.0, np.where(t_gen > 1.0, 1.0, t_gen))
    s = np.where(a_zero, 0.0, np.where(e_zero, s_low, s))
    t = np.where(
        a_zero, np.where(e_zero, 0.0, _clamp01(f / safe_e)), np.where(e_zero, 0.0, t)
    )
    return p1 + d1 * s[:, None], p2 + d2 * t[:, None]


def _spheres(ca, ra, cb, rb, fallback):
    """Контакт двух шаров: ``(hit, normal, point, depth)``."""
    d = cb - ca
    dist = np.sqrt(_dot3(d, d))
    reach = ra + rb
    ok = dist > _EPS
    normal = np.where(ok[:, None], d / _safe(dist, ~ok)[:, None], fallback)
    depth = reach - dist
    point = ca + normal * (ra - depth * 0.5)[:, None]
    return dist < reach, normal, point, depth


def _single(hit, normal, point, depth):
    n = len(hit)
    return hit, normal, np.ones(n, dtype=np.intp), point[:, None, :], depth[:, None]


# --- шар - шар, шар - бокс, шар - капсула ----------------------------------


def sphere_sphere_batch(pos_a, radius_a, pos_b, radius_b):
    """Шар - шар (нормаль A -> B)."""
    fallback = np.broadcast_to(_UP, pos_a.shape)
    return _single(*_spheres(pos_a, radius_a, pos_b, radius_b, fallback))


def _sphere_vs_box(center, radius, half, pb, rot):
    """Шар против бокса: ``(hit, normal, point, depth)``, нормаль от шара к боксу."""
    local = _to_local(rot, center - pb)
    clamped = np.minimum(np.maximum(local, -half), half)
    diff = local - clamped
    dist = _norm(diff)
    outside = dist > 1e-12
    unit = diff / _safe(dist, ~outside)[:, None]
    normal = -_to_world(rot, unit)
    point = _to_world(rot, clamped) + pb
    depth = radius - dist
    hit = ~outside | (dist < radius)
    inner = np.flatnonzero(~outside)
    if inner.size:  # центр внутри бокса: выталкиваем по оси наименьшего проникновения
        loc, hf, rt = local[inner], half[inner], rot[inner]
        pen = hf - np.abs(loc)
        k = np.where(pen[:, 1] < pen[:, 0], 1, 0)
        k = np.where(pen[:, 2] < np.minimum(pen[:, 0], pen[:, 1]), 2, k)
        rows = np.arange(inner.size)
        sign = np.where(loc[rows, k] >= 0.0, 1.0, -1.0)
        out_local = np.zeros((inner.size, 3))
        out_local[rows, k] = sign
        surface = loc.copy()
        surface[rows, k] = sign * hf[rows, k]
        normal[inner] = -_to_world(rt, out_local)
        point[inner] = _to_world(rt, surface) + pb[inner]
        depth[inner] = pen[rows, k] + radius[inner]
    return hit, normal, point, depth


def sphere_box_batch(pos_s, radius, pos_b, rot_b, half_b):
    """Шар - бокс (нормаль от шара к боксу)."""
    return _single(*_sphere_vs_box(pos_s, radius, half_b, pos_b, rot_b))


def sphere_capsule_batch(pos_s, radius_s, pos_c, rot_c, radius_c, half_height):
    """Шар - капсула (шар против ближайшей точки оси; нормаль от шара)."""
    a, b = _segment(pos_c, rot_c, half_height)
    nearest = _closest_on_segment(pos_s, a, b)
    ay, az = rot_c[:, 1, 1], rot_c[:, 2, 1]
    ax = rot_c[:, 0, 1]
    zero = np.zeros_like(ax)
    perp = np.stack([zero, az, -ay], axis=1)  # ось x (1, 0, 0)
    norm = _norm(perp)
    small = norm < 1e-6
    alt = np.stack([ay, -ax, zero], axis=1)  # ось x (0, 0, 1)
    alt_norm = _norm(alt)
    perp = np.where(
        small[:, None],
        alt / _safe(alt_norm, alt_norm == 0.0)[:, None],
        perp / _safe(norm, norm == 0.0)[:, None],
    )
    return _single(*_spheres(pos_s, radius_s, nearest, radius_c, perp))


# --- капсула - капсула -----------------------------------------------------


def capsule_capsule_batch(
    pos_a, rot_a, radius_a, half_height_a, pos_b, rot_b, radius_b, half_height_b
):
    """Капсула - капсула; параллельные оси дают две точки."""
    n = len(pos_a)
    p1, q1 = _segment(pos_a, rot_a, half_height_a)
    p2, q2 = _segment(pos_b, rot_b, half_height_b)
    c1, c2 = _closest_segments(p1, q1, p2, q2)
    d1, d2 = q1 - p1, q2 - p2
    cross12 = _cross(d1, d2)
    cross_norm = _norm(cross12)
    along_x = _cross(d1, _AXIS_X)
    fallback = np.where((cross_norm < 1e-9)[:, None], along_x, cross12)
    use_z = (cross_norm < 1e-9) & (_norm(along_x) < 1e-9)
    fallback = np.where(use_z[:, None], _cross(d1, _AXIS_Z), fallback)
    length = _norm(fallback)
    fallback = fallback / _safe(length, length == 0.0)[:, None]

    hit, normal, point, depth = _spheres(c1, radius_a, c2, radius_b, fallback)
    points = np.zeros((n, 2, 3))
    depths = np.zeros((n, 2))
    points[:, 0], depths[:, 0] = point, depth
    count = np.ones(n, dtype=np.intp)

    # Параллельные оси: вторая точка по границе перекрытия.
    a = _dot3(d1, d1)
    parallel = (a > _EPS) & (
        cross_norm < 1e-6 * np.sqrt(a) * np.sqrt(_dot3(d2, d2))
    )
    idx = np.flatnonzero(hit & parallel)
    if idx.size:
        ai, d1i, p1i = a[idx], d1[idx], p1[idx]
        s_vals = [
            _clamp01(_dot3(p - p1i, d1i) / ai) for p in (p2[idx], q2[idx])
        ]
        wide = np.abs(s_vals[0] - s_vals[1]) * np.sqrt(ai) > 1e-4
        both = wide.copy()
        extra_pts, extra_deps = [], []
        for s in s_vals:
            on_a = p1i + d1i * s[:, None]
            on_b = _closest_on_segment(on_a, p2[idx], q2[idx])
            h, _, pt, dp = _spheres(
                on_a, radius_a[idx], on_b, radius_b[idx], normal[idx]
            )
            both &= h
            extra_pts.append(pt)
            extra_deps.append(dp)
        sel = idx[both]
        points[sel, 0], points[sel, 1] = extra_pts[0][both], extra_pts[1][both]
        depths[sel, 0], depths[sel, 1] = extra_deps[0][both], extra_deps[1][both]
        count[sel] = 2
    return hit, normal, count, points, depths


# --- капсула - бокс --------------------------------------------------------


def _closest_t_outside(a, b, half):
    """Параметр ``t`` ближайшей к боксу точки отрезка ``a + t (b - a)``.

    Как скалярный ``_closest_t_outside``: отрезок режется плоскостями граней, на
    каждом куске квадрат расстояния минимизируется аналитически, ничьи
    усредняются. Возвращает ``(t, inside)``: ``inside`` — отрезок проходит
    внутри бокса (нужен вынос по грани).
    """
    n = len(a)
    delta = b - a
    cuts = np.ones((n, 8))  # лишние отсечки = 1.0 дают куски нулевой длины
    cuts[:, 0] = 0.0
    col = 2
    for k in range(3):
        d = delta[:, k]
        usable = np.abs(d) > 1e-12
        safe_d = _safe(d, ~usable)
        for plane in (-half[:, k], half[:, k]):
            t = (plane - a[:, k]) / safe_d
            cuts[:, col] = np.where(usable & (t > 0.0) & (t < 1.0), t, 1.0)
            col += 1
    cuts.sort(axis=1)

    best = np.full(n, np.inf)
    total = np.zeros(n)
    tied = np.zeros(n, dtype=np.intp)
    inside = np.zeros(n, dtype=bool)
    for j in range(7):
        lo, hi = cuts[:, j], cuts[:, j + 1]
        active = (hi - lo) >= 1e-12
        mid = 0.5 * (lo + hi)
        qa = np.zeros(n)
        qb = np.zeros(n)
        qc = np.zeros(n)
        outside = np.zeros(n, dtype=bool)
        for k in range(3):
            p = a[:, k] + mid * delta[:, k]
            above = p > half[:, k]
            part = above | (p < -half[:, k])
            u = np.where(above, a[:, k] - half[:, k], a[:, k] + half[:, k])
            dk = delta[:, k]
            outside |= part
            qa += np.where(part, dk * dk, 0.0)
            qb += np.where(part, u * dk, 0.0)
            qc += np.where(part, u * u, 0.0)
        inside |= active & ~outside
        flat = qa <= 1e-18
        t = np.where(
            flat, mid, np.minimum(hi, np.maximum(lo, -qb / _safe(qa, flat)))
        )
        f = (qa * t + 2.0 * qb) * t + qc
        usable = active & outside
        better = usable & (f < best - 1e-12)
        tie = usable & ~better & (f <= best + 1e-12)
        best = np.where(better, f, best)
        total = np.where(better, t, np.where(tie, total + t, total))
        tied = np.where(better, 1, np.where(tie, tied + 1, tied))
    t_best = np.where(tied > 0, total / np.maximum(tied, 1), 0.0)
    return t_best, inside


def _capsule_inside_exit(la, lb, half, radius, pb, rot):
    """Контакт капсулы, чья ось пересекает бокс (как скалярный вариант).

    Returns:
        ``(normal, point, depth)`` — нормаль от капсулы к боксу.
    """
    n = len(la)
    best_push = np.full(n, np.inf)
    best_k = np.zeros(n, dtype=np.intp)
    best_sign = np.ones(n)
    best_end = la.copy()
    for k in range(3):
        swap = (la[:, k] > lb[:, k])[:, None]
        lo_end, hi_end = np.where(swap, lb, la), np.where(swap, la, lb)
        push = half[:, k] + radius - lo_end[:, k]
        better = push < best_push
        best_push = np.where(better, push, best_push)
        best_k = np.where(better, k, best_k)
        best_sign = np.where(better, 1.0, best_sign)
        best_end = np.where(better[:, None], lo_end, best_end)
        push = hi_end[:, k] + half[:, k] + radius
        better = push < best_push
        best_push = np.where(better, push, best_push)
        best_k = np.where(better, k, best_k)
        best_sign = np.where(better, -1.0, best_sign)
        best_end = np.where(better[:, None], hi_end, best_end)
    rows = np.arange(n)
    surface = best_end.copy()
    surface[rows, best_k] = best_sign * half[rows, best_k]
    out_local = np.zeros((n, 3))
    out_local[rows, best_k] = best_sign
    return -_to_world(rot, out_local), _to_world(rot, surface) + pb, best_push


def capsule_box_batch(pos_c, rot_c, radius, half_height, pos_b, rot_b, half_b):
    """Капсула - бокс (нормаль от капсулы к боксу), до трёх точек контакта."""
    n = len(pos_c)
    rows = np.arange(n)
    p0, p1 = _segment(pos_c, rot_c, half_height)
    seg = p1 - p0
    reach = radius + _norm(half_b)
    mid = 0.5 * (p0 + p1) - pos_b
    culled = _dot3(mid, mid) > (reach + 0.5 * _norm(seg)) ** 2
    la = _to_local(rot_b, p0 - pos_b)
    lb = _to_local(rot_b, p1 - pos_b)
    t_fast, inside = _closest_t_outside(la, lb, half_b)

    # Ось целиком снаружи: если ближайшая точка не ближе радиуса, контакта нет.
    d2 = np.zeros(n)
    for k in range(3):
        c = la[:, k] + t_fast * (lb[:, k] - la[:, k])
        high = c > half_b[:, k]
        part = high | (c < -half_b[:, k])
        c = np.where(high, c - half_b[:, k], c + half_b[:, k])
        d2 += np.where(part, c * c, 0.0)
    near = ~inside & (d2 < radius * radius)

    # Кандидаты: ближайшая точка оси и концы (если те заметно отличаются).
    ts = np.stack(
        [t_fast, np.zeros(n), np.ones(n)], axis=1
    )  # (n, 3 кандидата)
    valid = np.stack(
        [np.ones(n, dtype=bool), np.abs(ts[:, 1] - t_fast) > 1e-4,
         np.abs(ts[:, 2] - t_fast) > 1e-4],
        axis=1,
    )
    centers = p0[:, None, :] + seg[:, None, :] * ts[:, :, None]  # (n, 3, 3)

    def rep(x):  # кандидаты идут подряд: (кандидат 0 всех пар, кандидат 1, ...)
        return np.concatenate([x, x, x])

    c_hit, c_normal, c_point, c_depth = _sphere_vs_box(
        np.concatenate([centers[:, 0], centers[:, 1], centers[:, 2]]),
        rep(radius), rep(half_b), rep(pos_b), rep(rot_b),
    )
    found = valid & c_hit.reshape(3, n).T
    c_depth = c_depth.reshape(3, n).T
    c_normal = c_normal.reshape(3, n, 3).transpose(1, 0, 2)
    c_point = c_point.reshape(3, n, 3).transpose(1, 0, 2)

    primary = np.argmax(np.where(found, c_depth, -np.inf), axis=1)
    normal = c_normal[rows, primary]
    points = np.zeros((n, 3, 3))
    depths = np.zeros((n, 3))
    points[:, 0] = c_point[rows, primary]
    depths[:, 0] = c_depth[rows, primary]
    count = np.ones(n, dtype=np.intp)
    for j in range(3):
        add = found[:, j] & (primary != j) & (_dot3(c_normal[:, j], normal) > 0.9)
        sel = np.flatnonzero(add)
        points[sel, count[sel]] = c_point[sel, j]
        depths[sel, count[sel]] = c_depth[sel, j]
        count[sel] += 1
    hit = near & found.any(axis=1)

    exit_rows = np.flatnonzero(inside)
    if exit_rows.size:
        e_normal, e_point, e_depth = _capsule_inside_exit(
            la[exit_rows], lb[exit_rows], half_b[exit_rows], radius[exit_rows],
            pos_b[exit_rows], rot_b[exit_rows],
        )
        normal[exit_rows] = e_normal
        points[exit_rows, 0] = e_point
        depths[exit_rows, 0] = e_depth
        count[exit_rows] = 1
        hit[exit_rows] = True
    return hit & ~culled, normal, count, points, depths


# --- диспетчер по видам форм ------------------------------------------------


def collide_batch(kind_a, pos_a, rot_a, half_a, kind_b, pos_b, rot_b, half_b):
    """Пары одного вида ``(kind_a, kind_b)`` пакетом; нормаль от A к B.

    Параметры форм — в формате ``ColliderStore``: ``half`` у шара ``(r, r, r)``,
    у капсулы ``(r, half_height, 0)``.

    Returns:
        ``(hit, normal, count, points, depths)`` или ``None``, если для вида пары
        пакетной версии нет (бокс - бокс и прочее считаются другим путём).
    """
    sphere, capsule, box = KIND_SPHERE, KIND_CAPSULE, KIND_BOX
    if kind_a == sphere and kind_b == sphere:
        return sphere_sphere_batch(pos_a, half_a[:, 0], pos_b, half_b[:, 0])
    if kind_a == sphere and kind_b == box:
        return sphere_box_batch(pos_a, half_a[:, 0], pos_b, rot_b, half_b)
    if kind_a == box and kind_b == sphere:
        return _flip(sphere_box_batch(pos_b, half_b[:, 0], pos_a, rot_a, half_a))
    if kind_a == sphere and kind_b == capsule:
        return sphere_capsule_batch(
            pos_a, half_a[:, 0], pos_b, rot_b, half_b[:, 0], half_b[:, 1]
        )
    if kind_a == capsule and kind_b == sphere:
        return _flip(sphere_capsule_batch(
            pos_b, half_b[:, 0], pos_a, rot_a, half_a[:, 0], half_a[:, 1]
        ))
    if kind_a == capsule and kind_b == box:
        return capsule_box_batch(
            pos_a, rot_a, half_a[:, 0], half_a[:, 1], pos_b, rot_b, half_b
        )
    if kind_a == box and kind_b == capsule:
        return _flip(capsule_box_batch(
            pos_b, rot_b, half_b[:, 0], half_b[:, 1], pos_a, rot_a, half_a
        ))
    if kind_a == capsule and kind_b == capsule:
        return capsule_capsule_batch(
            pos_a, rot_a, half_a[:, 0], half_a[:, 1],
            pos_b, rot_b, half_b[:, 0], half_b[:, 1],
        )
    return None


def _flip(result):
    """Тот же контакт при перестановке A и B (как ``ContactManifold.flipped``)."""
    hit, normal, count, points, depths = result
    return hit, -normal, count, points, depths
