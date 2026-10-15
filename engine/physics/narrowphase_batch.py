"""Пакетная узкая фаза: бокс - бокс сразу для множества пар (numpy).

:func:`box_box_batch` повторяет скалярный :func:`engine.physics.narrowphase.box_box`
(SAT по 15 осям, ребро-ребро или клиппинг граней) на массивах пар. Скалярная
версия остаётся эталоном и фолбэком.

Детерминизм: результат пары зависит только от её собственных входов, а не от
положения в пакете и его размера. Для этого вместо ``matmul``/``einsum`` (BLAS
может по-разному считать «хвосты») везде явная арифметика поэлементных операций.
Порядок точек контакта совпадает со скалярным ``box_box`` (он задаёт слоты
warm starting и порядок решения контактов).
"""

from __future__ import annotations

import numpy as np

from engine.physics.narrowphase import box_box

_EDGE_BIAS = np.array([0.0, 0.0, 0.0, 1e-4, 1e-4, 1e-4])  # при равенстве — грани A
# Две оси, отличные от k (по возрастанию): k=0 -> (1, 2), k=1 -> (0, 2), k=2 -> (0, 1).
_AXIS_U = np.array([1, 0, 0])
_AXIS_V = np.array([2, 2, 1])
MAX_POINTS = 4


def _dot3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Скалярное произведение по последней оси (3) явной суммой x+y+z."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _clip(poly: np.ndarray, count: np.ndarray, normal: np.ndarray,
          offset: np.ndarray):
    """Отсекает полигоны полупространствами ``normal . x <= offset``.

    Алгоритм Сазерленда — Ходжмана.

    Args:
        poly: Вершины ``(m, cap, 3)`` (первые ``count`` валидны).
        count: Число вершин каждого полигона ``(m,)``.
        normal: Нормали плоскостей ``(m, 3)``.
        offset: Смещения плоскостей ``(m,)``.

    Returns:
        ``(poly, count, overflow)``: полигоны ``(m, cap + 1, 3)`` (выпуклый
        полигон растёт не более чем на вершину за отсечение), новые числа
        вершин и признак переполнения (для таких пар нужен скалярный расчёт).
    """
    m, cap, _ = poly.shape
    idx = np.arange(cap)
    valid = idx[None, :] < count[:, None]
    dist = (
        normal[:, 0:1] * poly[..., 0]
        + normal[:, 1:2] * poly[..., 1]
        + normal[:, 2:3] * poly[..., 2]
    ) - offset[:, None]
    nxt = idx[None, :] + 1
    nxt = np.where(nxt >= count[:, None], 0, nxt)
    dist_next = np.take_along_axis(dist, nxt, axis=1)
    inside = valid & (dist <= 0.0)
    crossing = valid & (
        ((dist < 0.0) & (dist_next > 0.0)) | ((dist_next < 0.0) & (dist > 0.0))
    )
    k = dist / np.where(crossing, dist - dist_next, 1.0)
    poly_next = np.take_along_axis(poly, nxt[..., None], axis=1)
    inter = poly + (poly_next - poly) * k[..., None]

    emit = inside.astype(np.intp) + crossing
    end = np.cumsum(emit, axis=1)
    start = end - emit
    total = end[:, -1]
    out = np.zeros((m, cap + 1, 3))
    rows, cols = np.nonzero(inside)
    out[rows, np.minimum(start[rows, cols], cap)] = poly[rows, cols]
    rows, cols = np.nonzero(crossing)
    slot = start[rows, cols] + inside[rows, cols]
    out[rows, np.minimum(slot, cap)] = inter[rows, cols]
    return out, np.minimum(total, cap + 1), total > cap + 1


def _face_contacts(ref_pos, ref_rot, ref_half, ref_axis, inc_pos, inc_rot, inc_half):
    """Клиппинг инцидентной грани по опорной (как ``narrowphase._face_contacts``).

    Returns:
        ``(n_ref, points, depths, count, overflow)``: нормали опорных граней
        ``(m, 3)``, точки ``(m, 8, 3)`` и глубины ``(m, 8)`` в порядке скалярной
        версии, число точек ``(m,)`` и признак переполнения.
    """
    m = len(ref_pos)
    ar = np.arange(m)
    n_ref = ref_rot[ar, :, ref_axis]
    n_ref = np.where(_dot3(n_ref, inc_pos - ref_pos)[:, None] < 0.0, -n_ref, n_ref)
    center = ref_pos + n_ref * ref_half[ar, ref_axis][:, None]
    dots = (n_ref[:, :, None] * inc_rot).sum(axis=1)
    k = np.argmax(np.abs(dots), axis=1)
    sign = np.where(dots[ar, k] > 0.0, -1.0, 1.0)
    n_inc = inc_rot[ar, :, k] * sign[:, None]
    inc_center = inc_pos + n_inc * inc_half[ar, k][:, None]
    u_i, v_i = _AXIS_U[k], _AXIS_V[k]
    u = inc_rot[ar, :, u_i] * inc_half[ar, u_i][:, None]
    v = inc_rot[ar, :, v_i] * inc_half[ar, v_i][:, None]
    poly = np.stack(
        [inc_center + u + v, inc_center - u + v, inc_center - u - v,
         inc_center + u - v],
        axis=1,
    )
    count = np.full(m, 4, dtype=np.intp)
    overflow = np.zeros(m, dtype=bool)
    for side in (_AXIS_U[ref_axis], _AXIS_V[ref_axis]):
        s = ref_rot[ar, :, side]
        c = _dot3(s, ref_pos)
        h = ref_half[ar, side]
        poly, count, over = _clip(poly, count, s, c + h)
        overflow |= over
        poly, count, over = _clip(poly, count, -s, -c + h)
        overflow |= over

    cap = poly.shape[1]
    depth = _dot3(center[:, None, :] - poly, n_ref[:, None, :])
    keep = (np.arange(cap)[None, :] < count[:, None]) & (depth >= 0.0)
    pts = poly + n_ref[:, None, :] * (depth * 0.5)[..., None]
    pos = np.cumsum(keep, axis=1) - 1
    kept = keep.sum(axis=1)
    out_pts = np.zeros((m, cap, 3))
    out_dep = np.zeros((m, cap))
    rows, cols = np.nonzero(keep)
    out_pts[rows, pos[rows, cols]] = pts[rows, cols]
    out_dep[rows, pos[rows, cols]] = depth[rows, cols]
    return n_ref, out_pts, out_dep, kept, overflow


def _reduce_points_batch(pts: np.ndarray, deps: np.ndarray, kept: np.ndarray):
    """Оставляет до 4 точек у каждого полигона (как ``narrowphase._reduce_points``).

    Выбор тот же: глубочайшая точка, самая дальняя от неё, точка с наибольшей
    площадью треугольника и точка с наибольшей площадью по другую сторону от
    ребра; повторы отбрасываются с сохранением порядка.

    Args:
        pts: Точки ``(w, cap, 3)`` (первые ``kept`` валидны).
        deps: Глубины ``(w, cap)``.
        kept: Число точек каждого полигона ``(w,)``.

    Returns:
        ``(points, depths, count)``: ``(w, 4, 3)``, ``(w, 4)`` и ``(w,)``.
    """
    w, cap, _ = pts.shape
    ar = np.arange(w)
    valid = np.arange(cap)[None, :] < kept[:, None]
    i0 = np.argmax(np.where(valid, deps, -np.inf), axis=1)
    rel = pts - pts[ar, i0][:, None, :]
    dist = np.sqrt(_dot3(rel, rel))
    i1 = np.argmax(np.where(valid, dist, -1.0), axis=1)
    edge = (pts[ar, i1] - pts[ar, i0])[:, None, :]
    cross = np.stack(
        [rel[..., 1] * edge[..., 2] - rel[..., 2] * edge[..., 1],
         rel[..., 2] * edge[..., 0] - rel[..., 0] * edge[..., 2],
         rel[..., 0] * edge[..., 1] - rel[..., 1] * edge[..., 0]],
        axis=-1,
    )
    area = np.sqrt(_dot3(cross, cross))
    i2 = np.argmax(np.where(valid, area, -1.0), axis=1)
    side = _dot3(cross, cross[ar, i2][:, None, :])
    cand = np.where(valid & (side < 0.0), area, -1.0)
    i3 = np.argmax(cand, axis=1)

    chosen = np.stack([i0, i1, i2, i3], axis=1)
    keep = np.stack(
        [np.ones(w, dtype=bool),
         i1 != i0,
         (i2 != i0) & (i2 != i1),
         (cand[ar, i3] > 0.0) & (i3 != i0) & (i3 != i1) & (i3 != i2)],
        axis=1,
    )
    slot = np.cumsum(keep, axis=1) - 1
    out_pts = np.zeros((w, MAX_POINTS, 3))
    out_dep = np.zeros((w, MAX_POINTS))
    rows, cols = np.nonzero(keep)
    src = chosen[rows, cols]
    out_pts[rows, slot[rows, cols]] = pts[rows, src]
    out_dep[rows, slot[rows, cols]] = deps[rows, src]
    return out_pts, out_dep, keep.sum(axis=1)


def box_box_batch(pos_a, rot_a, half_a, pos_b, rot_b, half_b):
    """Контакты бокс - бокс для ``n`` пар сразу.

    Args:
        pos_a, pos_b: Центры ``(n, 3)``.
        rot_a, rot_b: Матрицы поворота ``(n, 3, 3)``.
        half_a, half_b: Полуразмеры ``(n, 3)``.

    Returns:
        ``(hit, normal, count, points, depths)``: признак пересечения ``(n,)``;
        нормаль A -> B ``(n, 3)``; число точек 1..4 (0 у непересекающихся);
        точки ``(n, 4, 3)`` и глубины ``(n, 4)`` (первые ``count`` валидны).
        Совпадает с :func:`~engine.physics.narrowphase.box_box` по каждой паре.
    """
    n = len(pos_a)
    hit = np.zeros(n, dtype=bool)
    normal = np.zeros((n, 3))
    count = np.zeros(n, dtype=np.intp)
    points = np.zeros((n, MAX_POINTS, 3))
    depths = np.zeros((n, MAX_POINTS))
    if n == 0:
        return hit, normal, count, points, depths

    t = pos_b - pos_a
    at = rot_a.transpose(0, 2, 1)  # at[:, i] — i-я ось A
    bt = rot_b.transpose(0, 2, 1)
    ca, cb = at[:, :, None, :], bt[:, None, :, :]
    cross = np.empty((n, 3, 3, 3))
    cross[..., 0] = ca[..., 1] * cb[..., 2] - ca[..., 2] * cb[..., 1]
    cross[..., 1] = ca[..., 2] * cb[..., 0] - ca[..., 0] * cb[..., 2]
    cross[..., 2] = ca[..., 0] * cb[..., 1] - ca[..., 1] * cb[..., 0]
    cross = cross.reshape(n, 9, 3)
    lens = np.sqrt(_dot3(cross, cross))
    axes = np.concatenate(
        [at, bt, cross / np.maximum(lens, 1e-12)[..., None]], axis=1
    )  # (n, 15, 3)
    valid = np.concatenate([np.ones((n, 6), dtype=bool), lens > 1e-4], axis=1)

    def project(rot, half):  # sum_m |axes . rot[:, m]| * half[m]
        proj = (
            axes[:, :, 0:1] * rot[:, None, 0, :]
            + axes[:, :, 1:2] * rot[:, None, 1, :]
            + axes[:, :, 2:3] * rot[:, None, 2, :]
        )
        return (np.abs(proj) * half[:, None, :]).sum(axis=2)

    overlap = project(rot_a, half_a) + project(rot_b, half_b) - np.abs(
        _dot3(axes, t[:, None, :])
    )
    overlap = np.where(valid, overlap, np.inf)
    hit = ~(overlap < 0.0).any(axis=1)
    sel = np.flatnonzero(hit)
    if sel.size == 0:
        return hit, normal, count, points, depths

    ov = overlap[sel]
    axes = axes[sel]
    t = t[sel]
    pa, pb = pos_a[sel], pos_b[sel]
    ra, rb = rot_a[sel], rot_b[sel]
    ha, hb = half_a[sel], half_b[sel]
    m = len(sel)
    ar = np.arange(m)
    face_idx = np.argmin(ov[:, :6] + _EDGE_BIAS, axis=1)
    face_o = ov[ar, face_idx]
    edge_rel = np.argmin(ov[:, 6:], axis=1)
    edge_o = ov[ar, 6 + edge_rel]
    is_edge = edge_o < 0.95 * face_o - 1e-3

    g_normal = np.zeros((m, 3))
    g_count = np.zeros(m, dtype=np.intp)
    g_points = np.zeros((m, MAX_POINTS, 3))
    g_depths = np.zeros((m, MAX_POINTS))
    fallback = np.zeros(m, dtype=bool)

    e = np.flatnonzero(is_edge)
    if e.size:
        nrm = axes[e, 6 + edge_rel[e]]
        nrm = np.where(_dot3(nrm, t[e])[:, None] < 0.0, -nrm, nrm)
        i, j = edge_rel[e] // 3, edge_rel[e] % 3
        rae, rbe, hae, hbe = ra[e], rb[e], ha[e], hb[e]
        pt_a, pt_b = pa[e].copy(), pb[e].copy()
        for k in range(3):
            col = rae[:, :, k]
            sgn = np.where(_dot3(col, nrm) > 0.0, 1.0, -1.0)
            pt_a = np.where((k != i)[:, None],
                            pt_a + col * hae[:, k:k + 1] * sgn[:, None], pt_a)
            col = rbe[:, :, k]
            sgn = np.where(_dot3(col, nrm) < 0.0, 1.0, -1.0)
            pt_b = np.where((k != j)[:, None],
                            pt_b + col * hbe[:, k:k + 1] * sgn[:, None], pt_b)
        ae = np.arange(len(e))
        da, db = rae[ae, :, i], rbe[ae, :, j]
        r = pt_a - pt_b
        b = _dot3(da, db)
        c, f = _dot3(da, r), _dot3(db, r)
        denom = 1.0 - b * b
        ok = denom > 1e-9
        safe = np.where(ok, denom, 1.0)
        s = np.where(ok, (b * f - c) / safe, 0.0)
        tt = np.where(ok, (f - b * c) / safe, f)
        point = 0.5 * ((pt_a + da * s[:, None]) + (pt_b + db * tt[:, None]))
        g_normal[e] = nrm
        g_count[e] = 1
        g_points[e, 0] = point
        g_depths[e, 0] = edge_o[e]

    f_idx = np.flatnonzero(~is_edge)
    if f_idx.size:
        ref_a = face_idx[f_idx] < 3
        axis = np.where(ref_a, face_idx[f_idx], face_idx[f_idx] - 3)
        ra_f, rb_f = ra[f_idx], rb[f_idx]
        pa_f, pb_f = pa[f_idx], pb[f_idx]
        ha_f, hb_f = ha[f_idx], hb[f_idx]
        sel3 = ref_a[:, None]
        n_ref, pts, deps, kept, over = _face_contacts(
            np.where(sel3, pa_f, pb_f), np.where(ref_a[:, None, None], ra_f, rb_f),
            np.where(sel3, ha_f, hb_f), axis,
            np.where(sel3, pb_f, pa_f), np.where(ref_a[:, None, None], rb_f, ra_f),
            np.where(sel3, hb_f, ha_f),
        )
        face_normal = np.where(sel3, n_ref, -n_ref)
        fallback[f_idx] = over | (kept > MAX_POINTS)
        # Вырожденный клиппинг: одна точка по центру пересечения.
        empty = (kept == 0) & ~over
        if empty.any():
            q = f_idx[empty]
            ax = axes[q, face_idx[q]]
            ax = np.where(_dot3(ax, t[q])[:, None] >= 0.0, ax, -ax)
            g_normal[q] = ax
            g_count[q] = 1
            g_points[q, 0] = 0.5 * (pa[q] + pb[q])
            g_depths[q, 0] = face_o[q]
        good = (kept > 0) & (kept <= MAX_POINTS)
        if good.any():
            q = f_idx[good]
            g_normal[q] = face_normal[good]
            g_count[q] = kept[good]
            g_points[q] = pts[good][:, :MAX_POINTS]
            g_depths[q] = deps[good][:, :MAX_POINTS]
        # Больше 4 точек: редукция до 4 (глубочайшая и наиболее разнесённые).
        big = np.flatnonzero((kept > MAX_POINTS) & ~over)
        if big.size:
            q = f_idx[big]
            g_points[q], g_depths[q], g_count[q] = _reduce_points_batch(
                pts[big], deps[big], kept[big]
            )
            g_normal[q] = face_normal[big]
            fallback[q] = False

    # Нештатное переполнение полигона: эталонный скалярный расчёт.
    for q in np.flatnonzero(fallback).tolist():
        o = int(sel[q])
        man = box_box(_Half(half_a[o]), pos_a[o], rot_a[o],
                      _Half(half_b[o]), pos_b[o], rot_b[o])
        k = len(man.points)
        g_normal[q] = man.normal
        g_count[q] = k
        g_points[q, :k] = man.points
        g_depths[q, :k] = man.depths

    normal[sel] = g_normal
    count[sel] = g_count
    points[sel] = g_points
    depths[sel] = g_depths
    return hit, normal, count, points, depths


class _Half:
    """Минимальная «форма» с ``half_extents`` для вызова скалярного ``box_box``."""

    __slots__ = ("half_extents",)

    def __init__(self, half) -> None:
        self.half_extents = half
