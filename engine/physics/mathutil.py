"""Мелкие математические помощники физики (numpy, кватернион xyzw)."""

from __future__ import annotations

import numpy as np

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def vec3(value) -> np.ndarray:
    """Приводит значение к ``float64`` массиву формы (3,)."""
    arr = np.asarray(tuple(value) if not isinstance(value, np.ndarray) else value,
                     dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"ожидался вектор из 3 чисел, получено {arr.shape}")
    return arr


def quat_to_matrix(q) -> np.ndarray:
    """Кватернион (x, y, z, w) -> матрица вращения 3x3."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-18:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1 - xx - yy],
        ]
    )


def as_matrix(rotation) -> np.ndarray:
    """Принимает матрицу 3x3 (или 4x4) либо кватернион и возвращает 3x3."""
    arr = np.asarray(rotation, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.shape == (4, 4):
        return arr[:3, :3]
    if arr.shape == (4,):
        return quat_to_matrix(arr)
    raise ValueError(f"неподдерживаемая форма вращения: {arr.shape}")


def quat_mul(a, b) -> np.ndarray:
    """Произведение кватернионов (xyzw): сначала ``b``, затем ``a``."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_normalize(q) -> np.ndarray:
    """Нормализованный кватернион (единичный при нулевой длине)."""
    arr = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        return np.array(IDENTITY_QUAT)
    return arr / n


def quat_integrate(q, angular_velocity, dt: float) -> np.ndarray:
    """Интегрирует вращение мировой угловой скоростью за ``dt``."""
    w = np.asarray(angular_velocity, dtype=np.float64)
    wx, wy, wz = w
    spin = np.array([wx, wy, wz, 0.0])
    dq = quat_mul(spin, q) * (0.5 * dt)
    return quat_normalize(np.asarray(q, dtype=np.float64) + dq)


def quat_from_axis_angle(axis, radians: float) -> np.ndarray:
    """Кватернион поворота вокруг оси (радианы)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    s = np.sin(radians * 0.5)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(radians * 0.5)])


def quats_to_matrices(q: np.ndarray) -> np.ndarray:
    """Пакетное преобразование кватернионов (n, 4) xyzw -> матрицы (n, 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    n2 = np.einsum("ni,ni->n", q, q)
    s = 2.0 / np.maximum(n2, 1e-18)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    out = np.empty((len(q), 3, 3))
    out[:, 0, 0] = 1 - yy - zz
    out[:, 0, 1] = xy - wz
    out[:, 0, 2] = xz + wy
    out[:, 1, 0] = xy + wz
    out[:, 1, 1] = 1 - xx - zz
    out[:, 1, 2] = yz - wx
    out[:, 2, 0] = xz - wy
    out[:, 2, 1] = yz + wx
    out[:, 2, 2] = 1 - xx - yy
    return out


def quats_integrate(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Пакетное интегрирование вращения угловой скоростью (n, 3)."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    # spin = (w, 0); dq = 0.5 * dt * spin * q
    dx = wx * qw + wy * qz - wz * qy
    dy = wy * qw + wz * qx - wx * qz
    dz = wz * qw + wx * qy - wy * qx
    dw = -wx * qx - wy * qy - wz * qz
    out = q + 0.5 * dt * np.stack([dx, dy, dz, dw], axis=1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)
