"""Sequential-impulse солвер контактов (векторизован раскраской контактов).

Контакты раскрашиваются жадно: в одном цвете нет двух контактов с общим
динамическим телом, поэтому цвет решается одним пакетом numpy и результат
совпадает с последовательным Гаусса-Зейделем в порядке цветов. Раскраска и
порядок обработки детерминированы (зависят только от порядка ``contacts``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from engine.physics.body import RigidBody

DEFAULT_ITERATIONS = 8
BAUMGARTE_BETA = 0.2
PENETRATION_SLOP = 0.005
RESTITUTION_THRESHOLD = 1.0
MAX_DEPENETRATION_SPEED = 4.0


@dataclass(slots=True)
class ContactConstraint:
    """Одна точка контакта между телами ``bodies[body_a]`` и ``bodies[body_b]``.

    ``normal`` направлена от A к B. ``normal_impulse`` и ``tangent_impulse`` на
    входе — накопленные импульсы прошлого кадра (warm start), на выходе —
    результат решения.
    """

    body_a: int
    body_b: int
    normal: np.ndarray
    point: np.ndarray
    depth: float
    friction: float = 0.5
    restitution: float = 0.0
    key: tuple = ()
    normal_impulse: float = 0.0
    tangent_impulse: tuple[float, float] = (0.0, 0.0)


def tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Детерминированный касательный базис (t1, t2) для нормалей (m, 3)."""
    use_x = np.abs(normals[:, 0]) < 0.57735
    ref = np.where(use_x[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    t1 = np.cross(normals, ref)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    return t1, t2


def color_contacts(
    body_a: Sequence[int], body_b: Sequence[int], movable: Sequence[bool]
) -> list[list[int]]:
    """Жадная раскраска: контакты одного цвета не делят подвижные тела."""
    used: dict[int, int] = {}
    colors: list[list[int]] = []
    for i, (a, b) in enumerate(zip(body_a, body_b, strict=True)):
        ma = used.get(a, 0) if movable[a] else 0
        mb = used.get(b, 0) if movable[b] else 0
        mask = ma | mb
        color = (~mask & (mask + 1)).bit_length() - 1
        if color == len(colors):
            colors.append([])
        colors[color].append(i)
        bit = 1 << color
        if movable[a]:
            used[a] = used.get(a, 0) | bit
        if movable[b]:
            used[b] = used.get(b, 0) | bit
    return colors


def solve_island(
    bodies: Sequence[RigidBody],
    contacts: Sequence[ContactConstraint],
    dt: float,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    beta: float = BAUMGARTE_BETA,
    slop: float = PENETRATION_SLOP,
    restitution_threshold: float = RESTITUTION_THRESHOLD,
    max_depenetration_speed: float = MAX_DEPENETRATION_SPEED,
) -> None:
    """Решает контакты, изменяя скорости ``bodies`` на месте.

    Независимые острова можно передавать одним вызовом: они не взаимодействуют.
    """
    m = len(contacts)
    if m == 0:
        return
    n = len(bodies)
    lin = np.array([b.linear_velocity for b in bodies])
    ang = np.array([b.angular_velocity for b in bodies])
    pos = np.array([b.position for b in bodies])
    inv_m = np.array([b.inv_mass for b in bodies])
    inv_i = np.array([b.inv_inertia_world for b in bodies])
    movable = (inv_m > 0.0).tolist()

    ia_l = [c.body_a for c in contacts]
    ib_l = [c.body_b for c in contacts]
    ia = np.array(ia_l)
    ib = np.array(ib_l)
    nrm = np.array([c.normal for c in contacts])
    pnt = np.array([c.point for c in contacts])
    depth = np.array([c.depth for c in contacts])
    mu = np.array([c.friction for c in contacts])
    rest = np.array([c.restitution for c in contacts])
    pn = np.array([c.normal_impulse for c in contacts])
    pt = np.array([c.tangent_impulse for c in contacts]).reshape(m, 2)

    ra = pnt - pos[ia]
    rb = pnt - pos[ib]
    t1, t2 = tangent_basis(nrm)
    ima, imb = inv_m[ia], inv_m[ib]
    iia, iib = inv_i[ia], inv_i[ib]

    def prep(d: np.ndarray):
        rna = np.cross(ra, d)
        rnb = np.cross(rb, d)
        arn = np.einsum("mij,mj->mi", iia, rna)
        brn = np.einsum("mij,mj->mi", iib, rnb)
        k = (
            ima + imb
            + np.einsum("mi,mi->m", rna, arn)
            + np.einsum("mi,mi->m", rnb, brn)
        )
        mass = np.where(k > 1e-12, 1.0 / np.maximum(k, 1e-12), 0.0)
        return rna, rnb, arn, brn, mass

    dirs = [prep(nrm), prep(t1), prep(t2)]
    d_vecs = [nrm, t1, t2]

    # Начальная относительная нормальная скорость (для упругости).
    vn0 = (
        np.einsum("mi,mi->m", lin[ib] - lin[ia], nrm)
        + np.einsum("mi,mi->m", ang[ib], dirs[0][1])
        - np.einsum("mi,mi->m", ang[ia], dirs[0][0])
    )
    bias = np.minimum(
        beta / dt * np.maximum(depth - slop, 0.0), max_depenetration_speed
    )
    bounce = np.where(vn0 < -restitution_threshold, -rest * vn0, 0.0)
    target = np.maximum(bias, bounce)

    # Warm start.
    impulse = (
        pn[:, None] * nrm + pt[:, 0:1] * t1 + pt[:, 1:2] * t2
    )
    ang_a = pn[:, None] * dirs[0][2] + pt[:, 0:1] * dirs[1][2] + pt[:, 1:2] * dirs[2][2]
    ang_b = pn[:, None] * dirs[0][3] + pt[:, 0:1] * dirs[1][3] + pt[:, 1:2] * dirs[2][3]
    np.add.at(lin, ia, -ima[:, None] * impulse)
    np.add.at(lin, ib, imb[:, None] * impulse)
    np.add.at(ang, ia, -ang_a)
    np.add.at(ang, ib, ang_b)

    # Скорости одним массивом (n, 6): [линейная, угловая]. Строки якобиана
    # по направлениям (нормаль, t1, t2): J_a = (-d, -rna), J_b = (d, rnb);
    # отклик скорости на единицу импульса: W_a = (-ima d, -arn), W_b = (imb d, brn).
    vel = np.concatenate([lin, ang], axis=1)
    dd = np.stack(d_vecs, axis=1)
    rna_all = np.stack([x[0] for x in dirs], axis=1)
    rnb_all = np.stack([x[1] for x in dirs], axis=1)
    arn_all = np.stack([x[2] for x in dirs], axis=1)
    brn_all = np.stack([x[3] for x in dirs], axis=1)
    mass_all = np.stack([x[4] for x in dirs], axis=1)
    jac_a = np.concatenate([-dd, -rna_all], axis=2)
    jac_b = np.concatenate([dd, rnb_all], axis=2)
    res_a = np.concatenate([-ima[:, None, None] * dd, -arn_all], axis=2)
    res_b = np.concatenate([imb[:, None, None] * dd, brn_all], axis=2)

    # Пакеты по цветам.
    batches = []
    for idx in color_contacts(ia_l, ib_l, movable):
        s = np.array(idx)
        batches.append(
            (
                s, ia[s], ib[s], mu[s], target[s], mass_all[s],
                jac_a[s], jac_b[s], res_a[s], res_b[s],
            )
        )

    ein_k = "mkj,mj->mk"
    for _ in range(iterations):
        for s, a, b, mu_, target_, mass, ja, jb, wa, wb in batches:
            va = vel[a]
            vb = vel[b]
            # Трение (по текущему накопленному нормальному импульсу).
            old = pt[s]
            rel = np.einsum(ein_k, jb[:, 1:], vb) + np.einsum(ein_k, ja[:, 1:], va)
            new = old - mass[:, 1:] * rel
            limit = mu_ * pn[s]
            length = np.linalg.norm(new, axis=1)
            scale = np.where(length > limit, limit / np.maximum(length, 1e-12), 1.0)
            new = new * scale[:, None]
            pt[s] = new
            delta = new - old
            va = va + np.einsum("mk,mkj->mj", delta, wa[:, 1:])
            vb = vb + np.einsum("mk,mkj->mj", delta, wb[:, 1:])
            # Нормаль.
            vn = np.einsum("mj,mj->m", jb[:, 0], vb) + np.einsum(
                "mj,mj->m", ja[:, 0], va
            )
            old_n = pn[s]
            new_n = np.maximum(old_n + mass[:, 0] * (target_ - vn), 0.0)
            pn[s] = new_n
            dn = (new_n - old_n)[:, None]
            vel[a] = va + dn * wa[:, 0]
            vel[b] = vb + dn * wb[:, 0]

    for i, body in enumerate(bodies):
        if inv_m[i] > 0.0:
            body.linear_velocity = vel[i, :3].copy()
            body.angular_velocity = vel[i, 3:].copy()
    for i, c in enumerate(contacts):
        c.normal_impulse = float(pn[i])
        c.tangent_impulse = (float(pt[i, 0]), float(pt[i, 1]))
    del n
