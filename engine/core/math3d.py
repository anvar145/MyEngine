"""3D-математика ядра: векторы, кватернионы и матрицы 4x4.

Модуль избегает создания матриц в Python-цикле на каждый объект: там, где
нужно обработать сразу много трансформов, используются batch-функции
(:func:`compose_matrices`), работающие с numpy-массивами формы ``(N, 3)``,
``(N, 4)`` и ``(N, 4, 4)``.

Соглашения:
    * кватернион хранится как ``(x, y, z, w)``;
    * матрица — numpy-массив ``(4, 4)`` типа :data:`MATRIX_DTYPE` в row-major
      раскладке; последний столбец содержит сдвиг (column-vector convention:
      ``v' = M @ v``, композиция иерархии ``world = parent_world @ local``);
    * вращение в градусах во всех публичных функциях.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

__all__ = [
    "EPSILON",
    "MATRIX_DTYPE",
    "Quaternion",
    "Vector3",
    "compose_matrices",
    "compose_matrix",
    "extract_translation",
    "identity_matrix",
    "invert_matrix",
    "invert_rigid",
    "look_at_matrix",
    "matrix_to_quaternion",
    "orthographic_matrix",
    "perspective_matrix",
    "quaternion_to_euler",
    "transform_point",
    "trs_from_matrix",
]

EPSILON = 1e-6
MATRIX_DTYPE = np.float32


# --------------------------------------------------------------------------- #
# Вектор
# --------------------------------------------------------------------------- #
class Vector3:
    """Неизменяемый вектор из трёх компонент.

    Вектор ведёт себя как значение: любые арифметические операции возвращают
    новый объект, поэтому его безопасно использовать в компонентах и событиях.

    Example:
        >>> Vector3(1, 2, 3) + Vector3.up() * 2
        Vector3(1.0, 4.0, 3.0)
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # -- конструкторы ------------------------------------------------------ #
    @classmethod
    def zero(cls) -> Vector3:
        """Вектор ``(0, 0, 0)``."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def one(cls) -> Vector3:
        """Вектор ``(1, 1, 1)``."""
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def up(cls) -> Vector3:
        """Единичный вектор вверх ``(0, 1, 0)``."""
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def right(cls) -> Vector3:
        """Единичный вектор вправо ``(1, 0, 0)``."""
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def forward(cls) -> Vector3:
        """Единичный вектор вперёд ``(0, 0, 1)``."""
        return cls(0.0, 0.0, 1.0)

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> Vector3:
        """Собрать вектор из последовательности длиной 3.

        Args:
            values: Любая последовательность из трёх чисел.

        Raises:
            ValueError: Если элементов не три.
        """
        items = tuple(values)
        if len(items) != 3:
            raise ValueError(f"Vector3 ожидает 3 компоненты, получено {len(items)}")
        return cls(items[0], items[1], items[2])

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> Vector3:
        """Скопировать вектор из numpy-массива формы ``(3,)``."""
        data = np.asarray(array, dtype=float).reshape(3)
        return cls(data[0], data[1], data[2])

    # -- свойства ---------------------------------------------------------- #
    @property
    def length(self) -> float:
        """Длина вектора."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def length_squared(self) -> float:
        """Квадрат длины (без вычисления корня)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    @property
    def normalized(self) -> Vector3:
        """Единичный вектор; нулевой вектор возвращается без изменений."""
        length = self.length
        if length <= EPSILON:
            return Vector3.zero()
        inv = 1.0 / length
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    @property
    def is_zero(self) -> bool:
        """``True``, если длина вектора близка к нулю."""
        return self.length_squared <= EPSILON * EPSILON

    def to_tuple(self) -> tuple[float, float, float]:
        """Представить вектор кортежем ``(x, y, z)``."""
        return (self.x, self.y, self.z)

    def to_numpy(self) -> np.ndarray:
        """Представить вектор копией numpy-массива формы ``(3,)``."""
        return np.array((self.x, self.y, self.z), dtype=float)

    # -- операции ---------------------------------------------------------- #
    def dot(self, other: Vector3) -> float:
        """Скалярное произведение."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Векторное произведение."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: Vector3) -> float:
        """Расстояние до другого вектора."""
        return (self - other).length

    def lerp(self, other: Vector3, t: float) -> Vector3:
        """Линейная интерполяция к ``other`` с коэффициентом ``t``."""
        return Vector3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )

    def almost_equals(self, other: Vector3, tolerance: float = 1e-5) -> bool:
        """Сравнить с другим вектором с заданным допуском."""
        return (
            abs(self.x - other.x) <= tolerance
            and abs(self.y - other.y) <= tolerance
            and abs(self.z - other.z) <= tolerance
        )

    def clamp_length(self, max_length: float) -> Vector3:
        """Ограничить длину вектора значением ``max_length``."""
        length = self.length
        if length <= max_length or length <= EPSILON:
            return Vector3(self.x, self.y, self.z)
        scale = max_length / length
        return Vector3(self.x * scale, self.y * scale, self.z * scale)

    def rotated(self, rotation: Quaternion) -> Vector3:
        """Повернуть вектор кватернионом."""
        return rotation.rotate_vector(self)

    # -- dunder ------------------------------------------------------------ #
    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        if abs(scalar) <= EPSILON:
            raise ZeroDivisionError("Деление вектора на ноль")
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Vector3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"Vector3({self.x:.4g}, {self.y:.4g}, {self.z:.4g})"


# --------------------------------------------------------------------------- #
# Кватернион
# --------------------------------------------------------------------------- #
class Quaternion:
    """Неизменяемый кватернион вращения ``(x, y, z, w)``.

    Example:
        >>> Quaternion.from_euler(0, 90, 0).rotate_vector(Vector3.forward())
        Vector3(1, 0, ~0)
    """

    __slots__ = ("x", "y", "z", "w")

    def __init__(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    # -- конструкторы ------------------------------------------------------ #
    @classmethod
    def identity(cls) -> Quaternion:
        """Единичный кватернион (без вращения)."""
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_axis_angle(cls, axis: Vector3, degrees: float) -> Quaternion:
        """Создать кватернион вращения вокруг оси на угол в градусах.

        Args:
            axis: Ось вращения; нормализуется внутри.
            degrees: Угол в градусах (правостороннее правило).

        Raises:
            ValueError: Если ось нулевая.
        """
        unit = axis.normalized
        if unit.is_zero:
            raise ValueError("Ось вращения не может быть нулевой")
        half = math.radians(degrees) * 0.5
        sin_half = math.sin(half)
        return cls(
            unit.x * sin_half, unit.y * sin_half, unit.z * sin_half, math.cos(half)
        )

    @classmethod
    def from_two_vectors(cls, source: Vector3, target: Vector3) -> Quaternion:
        """Создать кватернион кратчайшего поворота от одного направления к другому.

        Args:
            source: Исходное направление; нормализуется внутри.
            target: Целевое направление; нормализуется внутри.

        Returns:
            Единичный кватернион ``q``, для которого ``q.rotate_vector(source)``
            сонаправлен с ``target``. Для сонаправленных векторов - единичный
            поворот, для противоположных - полуоборот вокруг произвольной оси,
            перпендикулярной ``source``.

        Raises:
            ValueError: Если один из векторов нулевой.
        """
        a = source.normalized
        b = target.normalized
        if a.is_zero or b.is_zero:
            raise ValueError("Направление не может быть нулевым")
        cosine = a.dot(b)
        if cosine > 1.0 - 1e-12:
            return cls.identity()
        if cosine < -1.0 + 1e-12:
            helper = Vector3.right() if abs(a.x) < 0.9 else Vector3.up()
            axis = a.cross(helper).normalized
            return cls(axis.x, axis.y, axis.z, 0.0)
        axis = a.cross(b)
        return cls(axis.x, axis.y, axis.z, 1.0 + cosine).normalized

    @classmethod
    def from_euler(
        cls, x_degrees: float = 0.0, y_degrees: float = 0.0, z_degrees: float = 0.0
    ) -> Quaternion:
        """Создать кватернион из углов Эйлера (порядок применения Z, X, Y).

        Порядок совпадает с привычным для игровых редакторов: сначала рыскание
        вокруг Y, затем тангаж вокруг X, затем крен вокруг Z.
        """
        yaw = cls.from_axis_angle(Vector3.up(), y_degrees)
        pitch = cls.from_axis_angle(Vector3.right(), x_degrees)
        roll = cls.from_axis_angle(Vector3.forward(), z_degrees)
        return yaw * pitch * roll

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> Quaternion:
        """Скопировать кватернион из numpy-массива формы ``(4,)`` в порядке xyzw."""
        data = np.asarray(array, dtype=float).reshape(4)
        return cls(data[0], data[1], data[2], data[3])

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> Quaternion:
        """Извлечь вращение из матрицы 4x4 или 3x3."""
        return matrix_to_quaternion(matrix)

    # -- свойства ---------------------------------------------------------- #
    @property
    def length(self) -> float:
        """Норма кватерниона."""
        return math.sqrt(
            self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
        )

    @property
    def normalized(self) -> Quaternion:
        """Нормализованный кватернион; нулевой заменяется единичным."""
        length = self.length
        if length <= EPSILON:
            return Quaternion.identity()
        inv = 1.0 / length
        return Quaternion(self.x * inv, self.y * inv, self.z * inv, self.w * inv)

    @property
    def conjugate(self) -> Quaternion:
        """Сопряжённый кватернион (обратное вращение для единичного)."""
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    @property
    def inverse(self) -> Quaternion:
        """Обратный кватернион."""
        length_squared = self.length * self.length
        if length_squared <= EPSILON:
            return Quaternion.identity()
        inv = 1.0 / length_squared
        return Quaternion(-self.x * inv, -self.y * inv, -self.z * inv, self.w * inv)

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Представить кватернион кортежем ``(x, y, z, w)``."""
        return (self.x, self.y, self.z, self.w)

    def to_numpy(self) -> np.ndarray:
        """Представить кватернион numpy-массивом формы ``(4,)``."""
        return np.array((self.x, self.y, self.z, self.w), dtype=float)

    # -- операции ---------------------------------------------------------- #
    def dot(self, other: Quaternion) -> float:
        """Скалярное произведение кватернионов."""
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def rotate_vector(self, vector: Vector3) -> Vector3:
        """Повернуть вектор этим кватернионом (без промежуточных матриц)."""
        qx, qy, qz, qw = self.x, self.y, self.z, self.w
        vx, vy, vz = vector.x, vector.y, vector.z

        # t = 2 * (q_vec x v)
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        return Vector3(
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx),
        )

    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """Сферическая интерполяция между кватернионами."""
        dot = self.dot(other)
        if dot < 0.0:  # выбрать короткий путь
            other = Quaternion(-other.x, -other.y, -other.z, -other.w)
            dot = -dot
        if dot > 0.9995:
            blended = Quaternion(
                self.x + (other.x - self.x) * t,
                self.y + (other.y - self.y) * t,
                self.z + (other.z - self.z) * t,
                self.w + (other.w - self.w) * t,
            )
            return blended.normalized
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)
        scale_a = math.sin((1.0 - t) * theta) / sin_theta
        scale_b = math.sin(t * theta) / sin_theta
        return Quaternion(
            self.x * scale_a + other.x * scale_b,
            self.y * scale_a + other.y * scale_b,
            self.z * scale_a + other.z * scale_b,
            self.w * scale_a + other.w * scale_b,
        )

    def almost_equals(self, other: Quaternion, tolerance: float = 1e-5) -> bool:
        """Сравнить с другим кватернионом по компонентам с допуском."""
        return (
            abs(self.x - other.x) <= tolerance
            and abs(self.y - other.y) <= tolerance
            and abs(self.z - other.z) <= tolerance
            and abs(self.w - other.w) <= tolerance
        )

    # -- dunder ------------------------------------------------------------ #
    def __mul__(self, other: Quaternion) -> Quaternion:
        """Композиция вращений: ``a * b`` применяет сначала ``b``, затем ``a``."""
        if not isinstance(other, Quaternion):
            return NotImplemented
        return Quaternion(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Quaternion):
            return NotImplemented
        return (self.x, self.y, self.z, self.w) == (other.x, other.y, other.z, other.w)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z, self.w))

    def __repr__(self) -> str:
        return f"Quaternion({self.x:.4g}, {self.y:.4g}, {self.z:.4g}, {self.w:.4g})"


# --------------------------------------------------------------------------- #
# Матрицы
# --------------------------------------------------------------------------- #
def identity_matrix() -> np.ndarray:
    """Единичная матрица 4x4 нужного типа."""
    return np.identity(4, dtype=MATRIX_DTYPE)


def compose_matrix(
    position: Vector3 | None = None,
    rotation: Quaternion | None = None,
    scale: Vector3 | None = None,
) -> np.ndarray:
    """Собрать матрицу ``T * R * S`` из позиции, вращения и масштаба.

    Args:
        position: Сдвиг; по умолчанию нулевой.
        rotation: Вращение; по умолчанию единичное.
        scale: Масштаб; по умолчанию единичный.

    Returns:
        numpy-массив ``(4, 4)`` типа :data:`MATRIX_DTYPE`.
    """
    pos = position or Vector3.zero()
    rot = (rotation or Quaternion.identity()).normalized
    scl = scale or Vector3.one()

    matrix = np.identity(4, dtype=MATRIX_DTYPE)
    x, y, z, w = rot.x, rot.y, rot.z, rot.w

    matrix[0, 0] = (1.0 - 2.0 * (y * y + z * z)) * scl.x
    matrix[0, 1] = (2.0 * (x * y - z * w)) * scl.y
    matrix[0, 2] = (2.0 * (x * z + y * w)) * scl.z
    matrix[1, 0] = (2.0 * (x * y + z * w)) * scl.x
    matrix[1, 1] = (1.0 - 2.0 * (x * x + z * z)) * scl.y
    matrix[1, 2] = (2.0 * (y * z - x * w)) * scl.z
    matrix[2, 0] = (2.0 * (x * z - y * w)) * scl.x
    matrix[2, 1] = (2.0 * (y * z + x * w)) * scl.y
    matrix[2, 2] = (1.0 - 2.0 * (x * x + y * y)) * scl.z
    matrix[0, 3] = pos.x
    matrix[1, 3] = pos.y
    matrix[2, 3] = pos.z
    return matrix


def compose_matrices(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Векторизованно собрать матрицы ``T * R * S`` для набора трансформов.

    Это горячий путь обновления иерархии: вместо цикла по объектам одна
    numpy-операция собирает все локальные матрицы сразу.

    Args:
        positions: Массив ``(N, 3)``.
        rotations: Массив кватернионов ``(N, 4)`` в порядке xyzw.
        scales: Массив ``(N, 3)``.

    Returns:
        Массив ``(N, 4, 4)`` типа :data:`MATRIX_DTYPE`.

    Raises:
        ValueError: Если формы массивов не согласованы.
    """
    pos = np.asarray(positions, dtype=np.float32)
    rot = np.asarray(rotations, dtype=np.float32)
    scl = np.asarray(scales, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions должен иметь форму (N, 3), получено {pos.shape}")
    if rot.shape != (pos.shape[0], 4):
        raise ValueError(f"rotations должен иметь форму (N, 4), получено {rot.shape}")
    if scl.shape != pos.shape:
        raise ValueError(f"scales должен иметь форму {pos.shape}, получено {scl.shape}")

    count = pos.shape[0]
    matrices = np.zeros((count, 4, 4), dtype=MATRIX_DTYPE)
    if count == 0:
        return matrices

    # Нормализация кватернионов: без неё масштаб «утекает» в матрицу вращения.
    lengths = np.linalg.norm(rot, axis=1, keepdims=True)
    lengths[lengths < EPSILON] = 1.0
    rot = rot / lengths

    x, y, z, w = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    sx, sy, sz = scl[:, 0], scl[:, 1], scl[:, 2]

    matrices[:, 0, 0] = (1.0 - 2.0 * (y * y + z * z)) * sx
    matrices[:, 0, 1] = (2.0 * (x * y - z * w)) * sy
    matrices[:, 0, 2] = (2.0 * (x * z + y * w)) * sz
    matrices[:, 1, 0] = (2.0 * (x * y + z * w)) * sx
    matrices[:, 1, 1] = (1.0 - 2.0 * (x * x + z * z)) * sy
    matrices[:, 1, 2] = (2.0 * (y * z - x * w)) * sz
    matrices[:, 2, 0] = (2.0 * (x * z - y * w)) * sx
    matrices[:, 2, 1] = (2.0 * (y * z + x * w)) * sy
    matrices[:, 2, 2] = (1.0 - 2.0 * (x * x + y * y)) * sz
    matrices[:, 0, 3] = pos[:, 0]
    matrices[:, 1, 3] = pos[:, 1]
    matrices[:, 2, 3] = pos[:, 2]
    matrices[:, 3, 3] = 1.0
    return matrices


def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    """Обратить произвольную матрицу 4x4.

    Raises:
        ValueError: Если матрица вырождена.
    """
    source = np.asarray(matrix, dtype=float).reshape(4, 4)
    try:
        result = np.linalg.inv(source)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - защитная ветка
        raise ValueError("Матрица вырождена и не может быть обращена") from exc
    return np.asarray(result, dtype=MATRIX_DTYPE)


def invert_rigid(matrix: np.ndarray) -> np.ndarray:
    """Быстро обратить матрицу без масштабирования (``T * R``).

    Для матриц с масштабом используйте :func:`invert_matrix`.
    """
    source = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    rotation = source[:3, :3]
    translation = source[:3, 3]

    result = np.identity(4, dtype=MATRIX_DTYPE)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def transform_point(matrix: np.ndarray, point: Vector3) -> Vector3:
    """Преобразовать точку матрицей 4x4 (с учётом сдвига)."""
    data = np.asarray(matrix, dtype=float).reshape(4, 4)
    vector = np.array((point.x, point.y, point.z, 1.0), dtype=float)
    result = data @ vector
    return Vector3(result[0], result[1], result[2])


def matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
    """Извлечь кватернион вращения из матрицы 4x4 или 3x3.

    Масштаб из матрицы удаляется (столбцы нормализуются). Для вырожденных
    матриц возвращается единичный кватернион.
    """
    data = np.asarray(matrix, dtype=float)
    # Статичные камеры и источники света раскладывают одну и ту же матрицу
    # каждый кадр; кватернион неизменяем, поэтому результат можно переиспользовать.
    key = data.tobytes()
    cached = _QUATERNION_CACHE.get(key)
    if cached is not None:
        return cached
    result = _matrix_to_quaternion(data)
    if len(_QUATERNION_CACHE) >= _QUATERNION_CACHE_LIMIT:
        _QUATERNION_CACHE.clear()
    _QUATERNION_CACHE[key] = result
    return result


_QUATERNION_CACHE: dict[bytes, Quaternion] = {}
_QUATERNION_CACHE_LIMIT = 512


def _matrix_to_quaternion(data: np.ndarray) -> Quaternion:
    """Вычислить кватернион по матрице (без кэша)."""
    rotation = data[:3, :3] if data.shape == (4, 4) else data.reshape(3, 3)
    lengths = np.linalg.norm(rotation, axis=0)
    lengths[lengths < EPSILON] = 1.0
    rotation = rotation / lengths

    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale

    return Quaternion(x, y, z, w).normalized


def trs_from_matrix(matrix: np.ndarray) -> tuple[Vector3, Quaternion, Vector3]:
    """Разложить матрицу 4x4 обратно на позицию, вращение и масштаб.

    Используется при смене родителя с сохранением мирового положения.
    """
    data = np.asarray(matrix, dtype=float).reshape(4, 4)
    position = Vector3(data[0, 3], data[1, 3], data[2, 3])
    rotation_part = data[:3, :3]
    scale = Vector3(
        float(np.linalg.norm(rotation_part[:, 0])),
        float(np.linalg.norm(rotation_part[:, 1])),
        float(np.linalg.norm(rotation_part[:, 2])),
    )
    rotation = matrix_to_quaternion(rotation_part)
    return position, rotation, scale


def extract_translation(matrix: np.ndarray) -> Vector3:
    """Извлечь сдвиг из матрицы 4x4 (последний столбец)."""
    data = np.asarray(matrix, dtype=float).reshape(4, 4)
    return Vector3(float(data[0, 3]), float(data[1, 3]), float(data[2, 3]))


def quaternion_to_euler(rotation: Quaternion) -> Vector3:
    """Разложить кватернион на углы Эйлера в порядке применения Z, X, Y.

    Обратная операция к :meth:`Quaternion.from_euler`: результат можно
    показывать в инспекторе и возвращать обратно через ``from_euler``.

    Args:
        rotation: Кватернион вращения.

    Returns:
        Углы в градусах ``(x — тангаж, y — рыскание, z — крен)``.
    """
    quaternion = rotation.normalized
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    # Матрица вращения (строки — базисные векторы).
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    sin_pitch = max(-1.0, min(1.0, -r12))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) < 1.0 - 1e-6:
        yaw = math.atan2(r02, r22)
        roll = math.atan2(r10, r11)
    else:  # вырожденный случай: фиксируем крен, вычисляем рыскание
        yaw = math.atan2(-r01, r00)
        roll = 0.0
    return Vector3(math.degrees(pitch), math.degrees(yaw), math.degrees(roll))


def look_at_matrix(
    eye: Vector3, target: Vector3, up: Vector3 | None = None
) -> np.ndarray:
    """Построить видовую матрицу (world → view) для камеры.

    Args:
        eye: Позиция камеры в мире.
        target: Точка, куда смотрит камера.
        up: Вектор «вверх»; по умолчанию мировой ``+Y``.

    Returns:
        Матрица 4x4 в соглашениях движка (column-vector, сдвиг в столбце).

    Raises:
        ValueError: Если направление взгляда совпадает с вектором «вверх».
    """
    up_vector = up or Vector3.up()
    forward = (target - eye).normalized
    if forward.is_zero:
        raise ValueError("Камера не может смотреть в никуда: eye == target")
    right = forward.cross(up_vector)
    if right.is_zero:
        raise ValueError("Вектор up параллелен направлению взгляда")
    right = right.normalized
    true_up = right.cross(forward)

    matrix = np.identity(4, dtype=MATRIX_DTYPE)
    matrix[0, :3] = (right.x, right.y, right.z)
    matrix[1, :3] = (true_up.x, true_up.y, true_up.z)
    matrix[2, :3] = (-forward.x, -forward.y, -forward.z)
    matrix[0, 3] = -right.dot(eye)
    matrix[1, 3] = -true_up.dot(eye)
    matrix[2, 3] = forward.dot(eye)
    return matrix


def perspective_matrix(
    fov_degrees: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """Построить матрицу перспективной проекции.

    Args:
        fov_degrees: Вертикальный угол обзора в градусах.
        aspect: Отношение ширины к высоте.
        near: Ближняя плоскость отсечения.
        far: Дальняя плоскость отсечения.

    Returns:
        Матрица проекции 4x4.

    Raises:
        ValueError: Если параметры вырождены.
    """
    if not 0.0 < fov_degrees < 180.0:
        raise ValueError(f"Некорректный угол обзора: {fov_degrees}")
    if near <= 0.0 or far <= near:
        raise ValueError(f"Некорректные плоскости отсечения: near={near}, far={far}")
    if aspect <= 0.0:
        raise ValueError(f"Некорректный aspect: {aspect}")

    focal = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=MATRIX_DTYPE)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def orthographic_matrix(
    left: float, right: float, bottom: float, top: float, near: float, far: float
) -> np.ndarray:
    """Построить матрицу ортографической проекции.

    Returns:
        Матрица проекции 4x4.

    Raises:
        ValueError: Если размеры или плоскости вырождены.
    """
    if right - left <= EPSILON or top - bottom <= EPSILON:
        raise ValueError("Некорректные размеры ортографической проекции")
    if far - near <= EPSILON:
        raise ValueError("Некорректные плоскости отсечения")

    matrix = np.identity(4, dtype=MATRIX_DTYPE)
    matrix[0, 0] = 2.0 / (right - left)
    matrix[1, 1] = 2.0 / (top - bottom)
    matrix[2, 2] = -2.0 / (far - near)
    matrix[0, 3] = -(right + left) / (right - left)
    matrix[1, 3] = -(top + bottom) / (top - bottom)
    matrix[2, 3] = -(far + near) / (far - near)
    return matrix
