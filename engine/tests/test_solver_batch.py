"""Пакетный солвер на BodyStore против эталона ``solve_island``."""

from __future__ import annotations

import numpy as np
import pytest

from engine.physics import solver_batch as solver_batch_module
from engine.physics.body import RigidBody
from engine.physics.collider import Collider
from engine.physics.mathutil import quat_normalize
from engine.physics.shape import BoxShape
from engine.physics.solver import (
    ContactConstraint,
    color_contacts,
    solve_island,
    tangent_basis,
)
from engine.physics.solver_batch import (
    ColoringCache,
    ContactBatch,
    WarmStart,
    _cross,
    _directions,
    assign_colors,
    contact_keys,
    solve_batch,
)
from engine.physics.world import PhysicsWorld

DT = 1.0 / 60.0


def _random_spec(seed: int) -> tuple[list[dict], list[tuple]]:
    """Тела (динамические, статические, кинематические) и контакты между ними."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 13))
    bodies = []
    for i in range(n):
        kind = "dynamic"
        if i < 2 and rng.random() < 0.6:
            kind = "static" if rng.random() < 0.6 else "kinematic"
        moving = kind != "static"
        bodies.append({
            "body_type": kind,
            "mass": float(rng.uniform(0.5, 5.0)),
            "position": rng.uniform(-2.0, 2.0, 3),
            "rotation": quat_normalize(rng.normal(size=4)),
            "linear_velocity": rng.normal(size=3) * 2.0 if moving else (0, 0, 0),
            "angular_velocity": rng.normal(size=3) * 2.0 if moving else (0, 0, 0),
            "freeze_rotation": kind == "dynamic" and rng.random() < 0.15,
        })
    contacts = []
    for _ in range(int(rng.integers(2, 12))):
        a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
        if bodies[a]["body_type"] != "dynamic" and bodies[b]["body_type"] != "dynamic":
            continue
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        for _slot in range(int(rng.integers(1, 5))):
            warm = rng.random() < 0.5
            contacts.append((
                a, b, normal.copy(),
                (bodies[a]["position"] + bodies[b]["position"]) / 2
                + rng.uniform(-0.4, 0.4, 3),
                float(rng.uniform(0.0, 0.08)),
                float(rng.uniform(0.0, 1.0)),
                float(rng.uniform(0.0, 1.0)),
                (float(rng.uniform(0, 3)), float(rng.normal()), float(rng.normal()))
                if warm else (0.0, 0.0, 0.0),
            ))
    contacts.sort(key=lambda c: (c[0], c[1]))  # порядок пар, как в узкой фазе
    return bodies, contacts


def _build(bodies: list[dict]) -> tuple[PhysicsWorld, list[int]]:
    world = PhysicsWorld()
    handles = []
    for spec in bodies:
        handles.append(world.add_body(
            RigidBody(**spec), Collider(BoxShape((0.5, 0.4, 0.3)))
        ))
    return world, handles


@pytest.mark.parametrize("seed", range(50))
def test_solve_batch_matches_solve_island(seed: int) -> None:
    bodies, contacts = _random_spec(seed)
    if not contacts:
        pytest.skip("нет контактов")
    world_old, handles_old = _build(bodies)
    world_new, handles_new = _build(bodies)

    constraints = [
        ContactConstraint(a, b, n, p, d, mu, rest, normal_impulse=w[0],
                          tangent_impulse=(w[1], w[2]))
        for a, b, n, p, d, mu, rest, w in contacts
    ]
    solve_island([world_old.get_body(h) for h in handles_old], constraints, DT)

    rows = world_new._bodies.row_of_handle[handles_new].astype(np.intp)
    batch = ContactBatch(
        body_a=rows[[c[0] for c in contacts]],
        body_b=rows[[c[1] for c in contacts]],
        normal=np.array([c[2] for c in contacts]),
        point=np.array([c[3] for c in contacts]),
        depth=np.array([c[4] for c in contacts]),
        friction=np.array([c[5] for c in contacts]),
        restitution=np.array([c[6] for c in contacts]),
        key=np.arange(len(contacts), dtype=np.int64),
    )
    warm = np.array([c[7] for c in contacts])
    impulse = solve_batch(world_new._bodies, batch, warm, DT)

    for ho, hn in zip(handles_old, handles_new, strict=True):
        old, new = world_old.get_body(ho), world_new.get_body(hn)
        assert np.allclose(old.linear_velocity, new.linear_velocity, atol=1e-9)
        assert np.allclose(old.angular_velocity, new.angular_velocity, atol=1e-9)
    want = np.array([[c.normal_impulse, *c.tangent_impulse] for c in constraints])
    assert np.allclose(impulse, want, atol=1e-9)


def test_solve_batch_is_repeatable() -> None:
    bodies, contacts = _random_spec(7)

    def run() -> bytes:
        world, handles = _build(bodies)
        rows = world._bodies.row_of_handle[handles].astype(np.intp)
        batch = ContactBatch(
            rows[[c[0] for c in contacts]], rows[[c[1] for c in contacts]],
            np.array([c[2] for c in contacts]), np.array([c[3] for c in contacts]),
            np.array([c[4] for c in contacts]), np.array([c[5] for c in contacts]),
            np.array([c[6] for c in contacts]), np.arange(len(contacts)),
        )
        warm = np.array([c[7] for c in contacts])
        solve_batch(world._bodies, batch, warm, DT)
        return world._bodies.vel[: world._bodies.count].tobytes()

    assert run() == run()


def test_solve_batch_without_contacts_is_noop() -> None:
    world, _ = _build(_random_spec(1)[0])
    empty = ContactBatch(*(np.empty((0, 3)) if i in (2, 3) else np.empty(0)
                           for i in range(8)))
    before = world._bodies.vel.copy()
    assert solve_batch(world._bodies, empty, np.zeros((0, 3)), DT).shape == (0, 3)
    assert np.array_equal(world._bodies.vel, before)


@pytest.mark.parametrize("seed", range(10))
def test_assign_colors_matches_color_contacts(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, m = 12, 60
    a = rng.integers(0, n, m)
    b = (a + rng.integers(1, n, m)) % n
    movable = (rng.random(n) < 0.8).tolist()
    want = np.zeros(m, dtype=int)
    for color, group in enumerate(color_contacts(a.tolist(), b.tolist(), movable)):
        want[group] = color
    got = assign_colors(a.tolist(), b.tolist(), movable)
    assert got == want.tolist()


def test_contact_keys_are_ordered_by_pair_then_slot() -> None:
    cid_a = np.array([1, 1, 1, 2, 2, 7])
    cid_b = np.array([3, 3, 9, 4, 4, 8])
    slot = np.array([0, 1, 0, 0, 3, 2])
    keys = contact_keys(cid_a, cid_b, slot)
    assert (keys[1:] > keys[:-1]).all()
    assert keys.dtype == np.int64


def test_warm_start_lookup_finds_matching_keys_only() -> None:
    cache = WarmStart()
    assert cache.lookup(np.array([5, 6])).tolist() == [[0, 0, 0], [0, 0, 0]]
    cache.store(np.array([3, 5, 9]), np.array([[1.0, 0, 0], [2.0, 1, 1], [3.0, 2, 2]]))
    got = cache.lookup(np.array([1, 5, 6, 9, 20]))
    assert got.tolist() == [[0, 0, 0], [2.0, 1, 1], [0, 0, 0], [3.0, 2, 2], [0, 0, 0]]
    cache.clear()
    assert not cache.lookup(np.array([5])).any()


def test_warm_start_store_sorts_unordered_keys() -> None:
    cache = WarmStart()
    cache.store(np.array([9, 3]), np.array([[9.0, 0, 0], [3.0, 0, 0]]))
    assert cache.lookup(np.array([3, 9]))[:, 0].tolist() == [3.0, 9.0]


def _batch_of(world: PhysicsWorld, handles: list[int], contacts: list[tuple]):
    rows = world._bodies.row_of_handle[handles].astype(np.intp)
    batch = ContactBatch(
        rows[[c[0] for c in contacts]], rows[[c[1] for c in contacts]],
        np.array([c[2] for c in contacts]), np.array([c[3] for c in contacts]),
        np.array([c[4] for c in contacts]), np.array([c[5] for c in contacts]),
        np.array([c[6] for c in contacts]), np.arange(len(contacts)),
    )
    return batch, np.array([c[7] for c in contacts])


def test_cross_is_bitwise_equal_to_numpy_cross() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(500, 3))
    b = rng.normal(size=(500, 3, 3))
    assert _cross(a[:, None, :], b).tobytes() == np.cross(a[:, None, :], b).tobytes()
    c = rng.normal(size=(500, 3))
    assert _cross(a, c).tobytes() == np.cross(a, c).tobytes()


def test_directions_match_tangent_basis_bitwise() -> None:
    rng = np.random.default_rng(4)
    normals = rng.normal(size=(2000, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    axes = np.vstack([np.eye(3), -np.eye(3)])
    threshold = np.array([[0.57735, 0.5, 0.6], [0.57734, 0.6, 0.5], [0.0, 0.6, 0.8]])
    normals = np.vstack([normals, axes, threshold])
    t1, t2 = tangent_basis(normals)
    dirs = _directions(normals)
    assert np.array_equal(dirs[:, 0], normals)
    assert dirs[:, 1].tobytes() == t1.tobytes()
    assert dirs[:, 2].tobytes() == t2.tobytes()


def _coloring_args(rng: np.random.Generator, movable: np.ndarray, m: int = 40):
    n = len(movable)
    a = rng.integers(0, n, m)
    b = (a + rng.integers(1, n, m)) % n
    inv_mass = np.where(movable, 1.0, 0.0)
    return a, b, inv_mass[a], inv_mass[b], inv_mass


def test_coloring_cache_matches_direct_colors() -> None:
    rng = np.random.default_rng(5)
    a, b, ima, imb, inv_mass = _coloring_args(rng, rng.random(10) < 0.8)
    perm, bounds = ColoringCache().get(a, b, ima, imb, inv_mass)
    colors = np.array(assign_colors(a.tolist(), b.tolist(), (inv_mass > 0).tolist()))
    assert np.array_equal(perm, np.argsort(colors, kind="stable"))
    assert bounds == [0, *np.cumsum(np.bincount(colors)).tolist()]


def test_coloring_cache_reuses_result_until_contacts_change(monkeypatch) -> None:
    calls = []
    real = solver_batch_module.assign_colors
    monkeypatch.setattr(
        solver_batch_module, "assign_colors",
        lambda *args: calls.append(1) or real(*args),
    )
    rng = np.random.default_rng(6)
    movable = np.ones(10, dtype=bool)
    a, b, ima, imb, inv_mass = _coloring_args(rng, movable)
    cache = ColoringCache()
    first = cache.get(a, b, ima, imb, inv_mass)
    assert cache.get(a.copy(), b.copy(), ima.copy(), imb.copy(), inv_mass)[0] is first[0]
    assert len(calls) == 1
    b2 = b.copy()
    b2[0] = (b2[0] + 1) % 10 if (b2[0] + 1) % 10 != a[0] else (b2[0] + 2) % 10
    cache.get(a, b2, ima, imb, inv_mass)  # другой набор пар
    assert len(calls) == 2
    static = movable.copy()
    static[int(a[0])] = False  # тело стало неподвижным: меняется раскраска
    inv_static = np.where(static, 1.0, 0.0)
    cache.get(a, b2, inv_static[a], inv_static[b2], inv_static)
    assert len(calls) == 3


@pytest.mark.parametrize("seed", range(20))
def test_solve_batch_with_coloring_cache_is_bitwise_equal(seed: int) -> None:
    bodies, contacts = _random_spec(seed)
    if not contacts:
        pytest.skip("нет контактов")
    cache = ColoringCache()
    results = []
    for coloring in (None, cache, cache):  # второй вызов с кэшем идёт по попаданию
        world, handles = _build(bodies)
        batch, warm = _batch_of(world, handles, contacts)
        impulse = solve_batch(world._bodies, batch, warm, DT, coloring=coloring)
        results.append((world._bodies.vel.tobytes(), impulse.tobytes()))
    assert results[0] == results[1] == results[2]
