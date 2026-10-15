"""Пакетный sequential-impulse солвер на строках ``BodyStore``.

Те же уравнения, что у :func:`~engine.physics.solver.solve_island` (оставленного
эталоном), но без фасадов ``RigidBody`` и объектов ``ContactConstraint``:

* контакты приходят массивами (``ContactBatch``), тела — строки ``BodyStore``;
* жадная раскраска даёт тот же порядок Гаусса — Зейделя, что у ``color_contacts``;
  контакты переставляются так, что цвета лежат подряд (срезы без gather/scatter);
* якобианы ``J`` и отклики ``W`` каждого контакта считаются один раз и лежат
  матрицами ``(m, 3, 12)`` по трём направлениям (нормаль, ``t1``, ``t2``), а в
  итерации остаётся пара ``einsum`` на цвет. Контакты цвета малы, и время идёт
  на накладные расходы вызовов numpy, поэтому число вызовов на «цвет x итерацию»
  сведено к минимуму: скорости читаются ``take`` по строкам, пишутся одним
  присваиванием по плоскому индексу, а величины контакта лежат «по направлениям»
  ``(3, k)`` (строки непрерывны, ``[:, None]`` не нужен);
* раскраска переиспользуется, пока набор контактов не изменился (``ColoringCache``);
* влияние шага трения на нормальную скорость заранее свёрнуто в связь ``C``
  (``vn`` после трения = ``vn`` до него + ``delta . C``);
* накопленные импульсы прошлого шага ищутся по ключу контакта (``WarmStart``).

Порядок контактов и раскраска детерминированы: зависят только от порядка пар
узкой фазы. Результат совпадает с ``solve_island`` с точностью округления
(порядок суммирования в warm start и свёртках иной).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from engine.physics.solver import (
    BAUMGARTE_BETA,
    DEFAULT_ITERATIONS,
    MAX_DEPENETRATION_SPEED,
    PENETRATION_SLOP,
    RESTITUTION_THRESHOLD,
)
from engine.physics.store import BodyStore

try:  # einsum без обёртки Python: на малых пакетах она стоит ≈ 40 % вызова
    from numpy._core._multiarray_umath import c_einsum as _einsum
except ImportError:  # numpy 1.x
    try:
        from numpy.core._multiarray_umath import c_einsum as _einsum
    except ImportError:
        _einsum = np.einsum

# Ключ контакта: (cid_a, cid_b, слот) в одном int64. cid — счётчик, не
# переиспользуется и не достигает 2**30; слот — номер точки манифолда (< 8).
_CID_BITS = 30
_SLOT_BITS = 3
MAX_SLOTS = 1 << _SLOT_BITS


def contact_keys(cid_a: np.ndarray, cid_b: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """Ключи контактов ``(cid_a, cid_b, slot)`` как int64 (возрастают с парой)."""
    return (
        (cid_a.astype(np.int64) << (_CID_BITS + _SLOT_BITS))
        | (cid_b.astype(np.int64) << _SLOT_BITS)
        | slot.astype(np.int64)
    )


@dataclass(slots=True)
class ContactBatch:
    """Контакты шага массивами; ``normal`` направлена от тела A к телу B.

    ``body_a``/``body_b`` — строки ``BodyStore``; ``key`` — ключ warm start,
    строго возрастающий (контакты идут в порядке пар узкой фазы).
    """

    body_a: np.ndarray
    body_b: np.ndarray
    normal: np.ndarray
    point: np.ndarray
    depth: np.ndarray
    friction: np.ndarray
    restitution: np.ndarray
    key: np.ndarray

    def __len__(self) -> int:
        return len(self.body_a)


class WarmStart:
    """Накопленные импульсы ``[pn, pt1, pt2]`` прошлого шага по ключу контакта."""

    def __init__(self) -> None:
        self.keys = np.empty(0, dtype=np.int64)
        self.impulse = np.empty((0, 3))

    def clear(self) -> None:
        self.keys = np.empty(0, dtype=np.int64)
        self.impulse = np.empty((0, 3))

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Импульсы для ``keys`` ``(m, 3)``; у новых контактов — нули."""
        out = np.zeros((len(keys), 3))
        size = len(self.keys)
        if size and len(keys):
            pos = np.minimum(np.searchsorted(self.keys, keys), size - 1)
            found = self.keys[pos] == keys
            out[found] = self.impulse[pos[found]]
        return out

    def store(self, keys: np.ndarray, impulse: np.ndarray) -> None:
        """Запоминает импульсы шага (ключи должны быть строго возрастающими)."""
        if len(keys) > 1 and (keys[1:] <= keys[:-1]).any():
            order = np.argsort(keys, kind="stable")
            keys, impulse = keys[order], impulse[order]
        self.keys = keys
        self.impulse = impulse


def assign_colors(
    body_a: list[int], body_b: list[int], movable: list[bool]
) -> list[int]:
    """Номер цвета каждого контакта (та же жадная раскраска, что у ``color_contacts``).

    Args:
        body_a: Индексы первых тел контактов.
        body_b: Индексы вторых тел контактов.
        movable: Подвижность тела по индексу (у неподвижных цвет не занимается).
    """
    used = [0] * len(movable)
    colors: list[int] = []
    append = colors.append
    for a, b in zip(body_a, body_b, strict=True):
        mask = used[a] | used[b]
        color = (~mask & (mask + 1)).bit_length() - 1
        bit = 1 << color
        if movable[a]:
            used[a] |= bit
        if movable[b]:
            used[b] |= bit
        append(color)
    return colors


class ColoringCache:
    """Раскраска прошлого шага: повторяется, пока набор контактов не изменился.

    Ключ — пары тел контактов (в порядке контактов) и их обратные массы
    (подвижность); раскраска зависит только от них, поэтому результат совпадает
    с пересчётом.
    """

    def __init__(self) -> None:
        self._key = b""
        self._perm = np.empty(0, dtype=np.intp)
        self._bounds: list[int] = [0]

    def get(
        self,
        body_a: np.ndarray,
        body_b: np.ndarray,
        inv_mass_a: np.ndarray,
        inv_mass_b: np.ndarray,
        inv_mass: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Перестановка контактов по цветам и границы цветов в ней.

        Args:
            body_a: Строки первых тел контактов.
            body_b: Строки вторых тел контактов.
            inv_mass_a: Обратные массы первых тел (ключ кэша).
            inv_mass_b: Обратные массы вторых тел (ключ кэша).
            inv_mass: Обратные массы всех тел по строкам (подвижность).
        """
        key = (
            body_a.tobytes() + body_b.tobytes()
            + inv_mass_a.tobytes() + inv_mass_b.tobytes()
        )
        if key != self._key:
            movable = (inv_mass > 0.0).tolist()
            colors = np.array(assign_colors(body_a.tolist(), body_b.tolist(), movable))
            self._perm = np.argsort(colors, kind="stable")
            self._bounds = [0, *np.cumsum(np.bincount(colors)).tolist()]
            self._key = key
        return self._perm, self._bounds


_REF_X = np.array([1.0, 0.0, 0.0])
_REF_Y = np.array([0.0, 1.0, 0.0])


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Векторное произведение по последней оси ``(x, y, z)``.

    Те же вычисления, что у ``np.cross`` (результат совпадает побитово), без его
    обвязки на Python: на малых пакетах она в разы дороже арифметики.

    Args:
        a: Первые векторы; формы ``a`` и ``b`` согласуются вещанием.
        b: Вторые векторы **полной** формы результата.
    """
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    out = np.empty(b.shape)
    np.subtract(ay * bz, az * by, out=out[..., 0])
    np.subtract(az * bx, ax * bz, out=out[..., 1])
    np.subtract(ax * by, ay * bx, out=out[..., 2])
    return out


def _directions(normals: np.ndarray) -> np.ndarray:
    """Направления контактов ``(m, 3, xyz)``: нормаль и касательный базис.

    Касательные те же, что у :func:`~engine.physics.solver.tangent_basis`
    (побитово), только без обвязки ``np.cross``/``np.linalg.norm``.
    """
    use_x = np.abs(normals[:, 0]) < 0.57735
    t1 = _cross(normals, np.where(use_x[:, None], _REF_X, _REF_Y))
    t1 /= np.sqrt(np.add.reduce(t1 * t1, axis=1, keepdims=True))
    dirs = np.empty((len(normals), 3, 3))
    dirs[:, 0] = normals
    dirs[:, 1] = t1
    dirs[:, 2] = _cross(normals, t1)
    return dirs


def solve_batch(
    bodies: BodyStore,
    contacts: ContactBatch,
    warm: np.ndarray,
    dt: float,
    iterations: int = DEFAULT_ITERATIONS,
    *,
    beta: float = BAUMGARTE_BETA,
    slop: float = PENETRATION_SLOP,
    restitution_threshold: float = RESTITUTION_THRESHOLD,
    max_depenetration_speed: float = MAX_DEPENETRATION_SPEED,
    coloring: ColoringCache | None = None,
) -> np.ndarray:
    """Решает контакты, меняя ``bodies.vel`` на месте.

    Независимые острова решаются одним вызовом: они не взаимодействуют.

    Args:
        bodies: Хранилище тел (скорости, позы, обратные массы/инерции).
        contacts: Контакты шага.
        warm: Импульсы прошлого шага ``(m, 3)`` (``WarmStart.lookup``).
        dt: Шаг времени.
        iterations: Число итераций Гаусса — Зейделя.
        coloring: Кэш раскраски между шагами (без него раскраска считается заново).

    Returns:
        Накопленные импульсы ``(m, 3)`` в порядке ``contacts``: ``[pn, pt1, pt2]``.
    """
    m = len(contacts)
    if m == 0:
        return np.zeros((0, 3))
    ia, ib = contacts.body_a, contacts.body_b
    vel = bodies.vel
    inv_m = bodies.inv_mass
    ima, imb = inv_m[ia], inv_m[ib]
    iia, iib = bodies.inv_inertia_world[ia], bodies.inv_inertia_world[ib]

    pnt = contacts.point
    ra = pnt - bodies.position[ia]
    rb = pnt - bodies.position[ib]
    dirs = _directions(contacts.normal)  # (m, 3 направления, xyz)
    rna = _cross(ra[:, None, :], dirs)
    rnb = _cross(rb[:, None, :], dirs)
    arn = np.matmul(rna, iia.transpose(0, 2, 1))  # I_a . (ra x d)
    brn = np.matmul(rnb, iib.transpose(0, 2, 1))
    k_eff = (
        (ima + imb)[:, None]
        + _einsum("mdi,mdi->md", rna, arn)
        + _einsum("mdi,mdi->md", rnb, brn)
    )
    mass = np.where(k_eff > 1e-12, 1.0 / np.maximum(k_eff, 1e-12), 0.0)

    # Строки якобиана по направлениям: J_a = (-d, -rna), J_b = (d, rnb); отклик
    # скорости на единицу импульса: W_a = (-ima d, -arn), W_b = (imb d, brn).
    # Скорости пары тел — один вектор (12): [A.lin, A.ang, B.lin, B.ang].
    jac = np.concatenate((-dirs, -rna, dirs, rnb), axis=2)  # (m, 3, 12)
    resp = np.concatenate(
        (-ima[:, None, None] * dirs, -arn, imb[:, None, None] * dirs, brn), axis=2
    )

    pair = np.stack([ia, ib], axis=1)  # (m, 2) строки тел
    # Начальная нормальная скорость (для упругости) — до warm start.
    vn0 = _einsum("mj,mj->m", jac[:, 0], vel.take(pair.reshape(-1), 0).reshape(m, 12))
    bias = np.minimum(
        beta / dt * np.maximum(contacts.depth - slop, 0.0), max_depenetration_speed
    )
    bounce = np.where(vn0 < -restitution_threshold, -contacts.restitution * vn0, 0.0)
    target = np.maximum(bias, bounce)

    imp = np.array(warm, dtype=np.float64)  # (m, 3): [pn, pt1, pt2]
    if imp.any():  # warm start
        dv = _einsum("md,mdj->mj", imp, resp)
        np.add.at(vel, ia, dv[:, :6])
        np.add.at(vel, ib, dv[:, 6:])

    # Раскраска -> перестановка: цвета идут подряд.
    if coloring is None:
        coloring = ColoringCache()
    perm, bounds = coloring.get(ia, ib, ima, imb, inv_m[: bodies.count])

    # Дальше всё лежит в порядке цветов и «по направлениям»: величины контакта
    # (3, m) / (2, m), чтобы срез пакета был набором непрерывных строк.
    rows = pair[perm].reshape(-1)  # строки тел: [a0, b0, a1, b1, ...]
    elem = (rows[:, None] * 6 + np.arange(6)).reshape(-1)  # индексы в плоской vel
    jac = jac[perm]
    resp = resp[perm]
    mass_t = np.ascontiguousarray(mass[perm].T)  # (3, m): нормаль, t1, t2
    imp_t = np.ascontiguousarray(imp[perm].T)  # накопленные импульсы (3, m)
    mu = contacts.friction[perm]
    tgt = target[perm]
    # Связь: изменение нормальной скорости от шага трения (на единицу delta).
    couple = np.ascontiguousarray(_einsum("mj,mtj->tm", jac[:, 0], resp[:, 1:]))

    # Константы-массивы: операнд-массив в ufunc дешевле скаляра (≈0.6 мкс на вызов).
    consts = np.empty((3, m))
    consts[0] = 1e-12
    consts[1] = 1.0
    consts[2] = 0.0

    # Данные пакета - непрерывные массивы: ufunc над ними идёт одним циклом
    # (над срезом строк большего массива - вдвое дороже).
    batches = []
    for lo, hi in pairwise(bounds):
        k = hi - lo
        step = np.empty((3, k))
        batches.append((
            rows[2 * lo: 2 * hi], elem[12 * lo: 12 * hi], jac[lo:hi], resp[lo:hi],
            mass_t[1:, lo:hi].copy(), mass_t[0, lo:hi], mu[lo:hi], tgt[lo:hi],
            couple[:, lo:hi].copy(), imp_t[0, lo:hi], imp_t[1:, lo:hi].copy(),
            step, step[0], step[1:], consts[0, :k], consts[1, :k], consts[2, :k], k,
        ))

    take = vel.take
    flat_vel = vel.reshape(-1)  # vel непрерывна: это вид, а не копия
    einsum = _einsum
    hypot = np.hypot
    minimum = np.minimum
    maximum = np.maximum
    subtract = np.subtract
    for _ in range(iterations):
        for (rows_c, elem_c, jac_c, resp_c, mass_tc, mass_n, mu_c, tgt_c, cpl, pn, pt,
             step, step_n, step_t, eps, one, zero, k) in batches:
            v = take(rows_c, 0).reshape(k, 12)
            rel = einsum("kij,kj->ik", jac_c, v)
            # Трение (по текущему накопленному нормальному импульсу).
            new = pt - mass_tc * rel[1:]
            length = hypot(new[0], new[1])
            new *= minimum(mu_c * pn / maximum(length, eps), one)
            subtract(new, pt, out=step_t)
            pt[...] = new
            # Нормаль: скорость после трения через свёрнутую связь.
            vn = rel[0] + einsum("ik,ik->k", step_t, cpl)
            new_n = maximum(pn + mass_n * (tgt_c - vn), zero)
            subtract(new_n, pn, out=step_n)
            pn[...] = new_n
            v += einsum("dk,kdj->kj", step, resp_c)
            flat_vel[elem_c] = v.reshape(-1)

    for (lo, hi), batch in zip(pairwise(bounds), batches, strict=True):
        imp_t[1:, lo:hi] = batch[10]  # касательные импульсы живут в копии пакета
    out = np.empty((m, 3))
    out[perm] = imp_t.T
    return out
