"""Тесты 3D-математики: векторы, кватернионы, матрицы."""

from __future__ import annotations

import numpy as np
import pytest

from engine.core.math3d import (
    Quaternion,
    Vector3,
    compose_matrices,
    compose_matrix,
    invert_matrix,
    invert_rigid,
    matrix_to_quaternion,
    transform_point,
    trs_from_matrix,
)


class TestVector3:
    def test_arithmetic(self) -> None:
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        assert (a + b).to_tuple() == (5.0, 7.0, 9.0)
        assert (b - a).to_tuple() == (3.0, 3.0, 3.0)
        assert (a * 2).to_tuple() == (2.0, 4.0, 6.0)
        assert (2 * a).to_tuple() == (2.0, 4.0, 6.0)
        assert (-a).to_tuple() == (-1.0, -2.0, -3.0)

    def test_length_and_normalize(self) -> None:
        assert Vector3(3, 4, 0).length == 5.0
        assert Vector3(3, 4, 0).length_squared == 25.0
        assert Vector3(3, 4, 0).normalized.length == 1.0
        assert Vector3.zero().normalized == Vector3.zero()

    def test_products(self) -> None:
        assert Vector3(1, 0, 0).dot(Vector3(0, 1, 0)) == 0.0
        assert Vector3(1, 0, 0).cross(Vector3(0, 1, 0)) == Vector3(0, 0, 1)

    def test_helpers(self) -> None:
        assert Vector3.from_iterable([1, 2, 3]) == Vector3(1, 2, 3)
        assert Vector3(1, 2, 3).to_numpy().tolist() == [1.0, 2.0, 3.0]
        assert Vector3(0, 0, 10).clamp_length(5).length == 5.0
        assert list(Vector3(1, 2, 3)) == [1.0, 2.0, 3.0]
        assert len(Vector3()) == 3

    def test_lerp_and_distance(self) -> None:
        assert Vector3(0, 0, 0).lerp(Vector3(10, 0, 0), 0.5) == Vector3(5, 0, 0)
        assert Vector3(0, 0, 0).distance_to(Vector3(0, 3, 4)) == 5.0

    def test_zero_division_is_reported(self) -> None:
        try:
            Vector3(1, 1, 1) / 0.0
        except ZeroDivisionError:
            return
        raise AssertionError("Ожидалось ZeroDivisionError")

    def test_invalid_iterable_length(self) -> None:
        try:
            Vector3.from_iterable([1, 2])
        except ValueError:
            return
        raise AssertionError("Ожидалось ValueError")


class TestQuaternion:
    def test_identity_keeps_vector(self) -> None:
        v = Vector3(1, 2, 3)
        assert Quaternion.identity().rotate_vector(v).almost_equals(v)

    def test_yaw_90_degrees(self) -> None:
        rotated = Quaternion.from_euler(0, 90, 0).rotate_vector(Vector3.forward())
        assert rotated.almost_equals(Vector3(1, 0, 0), 1e-5)

    def test_axis_angle_matches_euler(self) -> None:
        from_axis = Quaternion.from_axis_angle(Vector3.up(), 90)
        from_euler = Quaternion.from_euler(0, 90, 0)
        assert from_axis.dot(from_euler) == pytest.approx(1.0, abs=1e-6)

    def test_composition_order(self) -> None:
        yaw = Quaternion.from_axis_angle(Vector3.up(), 90)
        pitch = Quaternion.from_axis_angle(Vector3.right(), 90)
        combined = pitch * yaw
        assert combined.rotate_vector(Vector3.forward()).almost_equals(
            Vector3(1, 0, 0), 1e-5
        )

    def test_inverse_and_conjugate(self) -> None:
        q = Quaternion.from_euler(30, 45, 60)
        assert (q * q.inverse).almost_equals(Quaternion.identity(), 1e-5)
        assert q.conjugate.almost_equals(q.inverse, 1e-5)

    def test_slerp_midpoint(self) -> None:
        start = Quaternion.identity()
        end = Quaternion.from_axis_angle(Vector3.up(), 90)
        middle = start.slerp(end, 0.5)
        assert middle.rotate_vector(Vector3.forward()).almost_equals(
            Vector3(1, 0, 1).normalized, 1e-4
        )

    def test_zero_axis_is_error(self) -> None:
        try:
            Quaternion.from_axis_angle(Vector3.zero(), 10)
        except ValueError:
            return
        raise AssertionError("Ожидалось ValueError")

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (Vector3(0, 1, 0), Vector3(1, 0, 0)),
            (Vector3(0, 1, 0), Vector3(0, 0, -1)),
            (Vector3(1, 0, 0), Vector3(0, 0, 1)),
            (Vector3(0, 1, 0), Vector3(1, 2, 3).normalized),
            (Vector3(-2, 0.5, 1).normalized, Vector3(0.3, 0.9, -0.2).normalized),
        ],
    )
    def test_from_two_vectors_maps_source_to_target(
        self, source: Vector3, target: Vector3
    ) -> None:
        q = Quaternion.from_two_vectors(source, target)
        assert q.rotate_vector(source).almost_equals(target, 1e-5)
        assert q.length == pytest.approx(1.0, abs=1e-9)

    def test_from_two_vectors_is_minimal_arc(self) -> None:
        """Поворот идёт по кратчайшей дуге: ось ⟂ обоим векторам, угол = углу между ними."""
        source = Vector3(0, 1, 0)
        target = Vector3(0, 1, 1).normalized  # 45°
        q = Quaternion.from_two_vectors(source, target)
        expected = Quaternion.from_axis_angle(Vector3.right(), 45)
        assert q.dot(expected) == pytest.approx(1.0, abs=1e-6)

    def test_from_two_vectors_normalizes_inputs(self) -> None:
        q = Quaternion.from_two_vectors(Vector3(0, 7, 0), Vector3(0, 0, 0.25))
        assert q.rotate_vector(Vector3.up()).almost_equals(Vector3.forward(), 1e-5)

    def test_from_two_vectors_parallel_is_identity(self) -> None:
        q = Quaternion.from_two_vectors(Vector3(0, 1, 0), Vector3(0, 3, 0))
        assert q.almost_equals(Quaternion.identity(), 1e-9)

    def test_from_two_vectors_antiparallel_flips(self) -> None:
        for source in (Vector3(0, 1, 0), Vector3(1, 0, 0), Vector3(0, 0, 1)):
            q = Quaternion.from_two_vectors(source, -source)
            assert q.rotate_vector(source).almost_equals(-source, 1e-5)
            assert q.length == pytest.approx(1.0, abs=1e-9)

    def test_from_two_vectors_zero_vector_is_error(self) -> None:
        with pytest.raises(ValueError):
            Quaternion.from_two_vectors(Vector3.zero(), Vector3.up())
        with pytest.raises(ValueError):
            Quaternion.from_two_vectors(Vector3.up(), Vector3.zero())


class TestMatrices:
    def test_compose_translation(self) -> None:
        matrix = compose_matrix(position=Vector3(1, 2, 3))
        assert transform_point(matrix, Vector3.zero()).almost_equals(
            Vector3(1, 2, 3), 1e-6
        )
        assert matrix[0, 3] == pytest.approx(1.0)
        assert matrix[1, 3] == pytest.approx(2.0)
        assert matrix[2, 3] == pytest.approx(3.0)

    def test_batch_matches_single(self) -> None:
        rotations = [
            Quaternion.from_euler(10, 20, 30),
            Quaternion.from_euler(-45, 0, 15),
        ]
        positions = [Vector3(1, 2, 3), Vector3(-4, 0.5, 7)]
        scales = [Vector3(2, 2, 2), Vector3(1, 3, 1)]

        batch = compose_matrices(
            np.array([p.to_tuple() for p in positions], dtype=np.float32),
            np.array([q.to_tuple() for q in rotations], dtype=np.float32),
            np.array([s.to_tuple() for s in scales], dtype=np.float32),
        )
        for index, (position, rotation, scale) in enumerate(
            zip(positions, rotations, scales, strict=True)
        ):
            single = compose_matrix(position, rotation, scale)
            assert np.allclose(batch[index], single, atol=1e-5)

    def test_batch_validation(self) -> None:
        try:
            compose_matrices(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((3, 4), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
            )
        except ValueError:
            return
        raise AssertionError("Ожидалось ValueError для несогласованных форм")

    def test_empty_batch(self) -> None:
        result = compose_matrices(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
        assert result.shape == (0, 4, 4)

    def test_invert_rigid_roundtrip(self) -> None:
        matrix = compose_matrix(Vector3(2, -3, 4), Quaternion.from_euler(0, 45, 0))
        identity = invert_rigid(matrix) @ matrix
        assert np.allclose(identity, np.identity(4), atol=1e-5)

    def test_trs_roundtrip(self) -> None:
        position = Vector3(1, 2, 3)
        rotation = Quaternion.from_euler(15, 30, 45)
        scale = Vector3(2, 1, 0.5)
        restored_position, restored_rotation, restored_scale = trs_from_matrix(
            compose_matrix(position, rotation, scale)
        )
        assert restored_position.almost_equals(position, 1e-4)
        assert restored_rotation.almost_equals(rotation, 1e-4)
        assert restored_scale.almost_equals(scale, 1e-4)

    def test_invert_matrix_with_scale(self) -> None:
        matrix = compose_matrix(Vector3(1, 1, 1), None, Vector3(2, 4, 8))
        assert np.allclose(invert_matrix(matrix) @ matrix, np.identity(4), atol=1e-5)

    def test_matrix_to_quaternion_removes_scale(self) -> None:
        rotation = Quaternion.from_euler(0, 90, 0)
        matrix = compose_matrix(None, rotation, Vector3(3, 3, 3))
        assert matrix_to_quaternion(matrix).almost_equals(rotation, 1e-5)
