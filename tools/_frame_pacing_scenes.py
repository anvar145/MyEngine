"""Сцены стенда frame pacing: персонаж идёт сквозь узкие щели (проект в ``_work``)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core.components import MeshRenderer  # noqa: E402
from engine.core.math3d import Quaternion, Vector3  # noqa: E402
from engine.core.physics_components import (  # noqa: E402
    CharacterControllerComponent,
    ColliderComponent,
    RigidBodyComponent,
)
from engine.core.scene import Scene  # noqa: E402
from engine.rendering.camera import (  # noqa: E402
    CameraComponent,
    quaternion_from_forward,
)
from engine.rendering.lighting import DirectionalLight, LightComponent  # noqa: E402
from engine.rendering.sky import SkySettings  # noqa: E402
from engine.scene_io import save_scene  # noqa: E402
from engine.scripting import ScriptComponent  # noqa: E402

PROJECT = ROOT / "_work" / "frame_pacing" / "proj"
SCRIPTS_SRC = ROOT / "projects" / "PlayModeTest" / "Assets" / "Scripts"
SCRIPTS = PROJECT / "Assets" / "Scripts"
UP = Vector3(0, 1, 0)


def _box(scene: Scene, name: str, pos, size, rot_y: float = 0.0, mass=None):
    obj = scene.create_game_object(
        name,
        position=Vector3(*pos),
        scale=Vector3(*size),
        rotation=Quaternion.from_axis_angle(UP, rot_y),
    )
    obj.add_component(MeshRenderer("cube", color=Vector3(0.6, 0.6, 0.6)))
    obj.add_component(ColliderComponent, shape="box", size=(1.0, 1.0, 1.0))
    if mass is not None:
        obj.add_component(RigidBodyComponent, mass=mass)
    return obj


def _pillar(scene: Scene, name: str, pos, kinematic: bool = False):
    obj = scene.create_game_object(
        name, position=Vector3(*pos), scale=Vector3(0.8, 1.8, 0.8)
    )
    obj.add_component(MeshRenderer("cylinder", color=Vector3(0.8, 0.3, 0.3)))
    obj.add_component(ColliderComponent, shape="capsule", radius=0.4, height=1.8)
    if kinematic:
        obj.add_component(RigidBodyComponent, body_type="kinematic")
    return obj


def _base(name: str) -> Scene:
    scene = Scene(name)
    sun = scene.create_game_object("Sun", position=Vector3(0, 8, 0))
    light = DirectionalLight(
        color=Vector3(1, 0.95, 0.85),
        intensity=1.3,
        direction=Vector3(-0.4, -1.0, -0.3),
        shadow_map_size=1024,
        shadow_distance=40.0,
    )
    sun.transform.rotation = quaternion_from_forward(light.unit_direction)
    sun.add_component(LightComponent, light=light).sync_to_light()
    _box(scene, "Floor", (0, -0.5, 0), (60, 1, 120))
    hero = scene.create_game_object("Player", position=Vector3(0.0, 1.0, 3.0))
    hero.add_component(CharacterControllerComponent)
    hero.add_component(ScriptComponent(SCRIPTS / "player_controller.py"))
    body = scene.create_game_object("Body", scale=Vector3(0.8, 1.8, 0.8))
    body.add_component(MeshRenderer("cylinder", color=Vector3(0.95, 0.6, 0.15)))
    body.set_parent(hero, keep_world=False)
    cam = scene.create_game_object("Camera")
    cam.set_parent(hero, keep_world=False)
    cam.transform.position = Vector3(0.0, 4.0, 6.0)
    comp = CameraComponent()
    comp.camera.fov = 55.0
    comp.camera.name = "Camera"
    cam.add_component(comp)
    comp.sync_to_camera()
    return scene


def _gap_pair(scene, tag: str, z: float, gap: float, rot: float = 0.0, mass=None):
    half = 1.5
    for sign in (-1, 1):
        x = sign * (gap / 2 + half)
        _box(scene, f"{tag}{sign}", (x, 1.5, z), (3, 3, 2), sign * rot, mass)


def build() -> dict[str, Scene]:
    """Сценарии «проход в щель»: ширина щели в метрах (капсула 0.8 м)."""
    out: dict[str, Scene] = {}
    for gap in (0.85, 1.0, 1.3):
        scene = _base("gap")
        _gap_pair(scene, "G", -4.0, gap)
        out[f"gap_{int(gap * 100):03d}"] = scene
    scene = _base("funnel")
    _gap_pair(scene, "F", -4.0, 1.0, rot=35.0)
    out["gap_funnel"] = scene
    scene = _base("slalom")
    for i in range(5):
        _gap_pair(scene, f"S{i}", -4.0 - 5.0 * i, 1.0)
    out["gap_slalom"] = scene
    scene = _base("dyn")
    _gap_pair(scene, "D", -4.0, 1.1, mass=20.0)
    out["gap_dynamic"] = scene
    scene = _base("pillars")
    for sign in (-1, 1):
        _pillar(scene, f"P{sign}", (sign * 0.65, 0.9, -4.0))
    out["gap_pillars"] = scene
    scene = _base("kin")
    for sign in (-1, 1):
        _pillar(scene, f"K{sign}", (sign * 0.65, 0.9, -4.0), kinematic=True)
    out["gap_kinematic"] = scene
    return out


def generate() -> Path:
    """Создать проект стенда в ``_work/frame_pacing/proj`` и вернуть его путь."""
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    for name in ("player_controller.py", "patrol.py"):
        shutil.copy2(SCRIPTS_SRC / name, SCRIPTS / name)
    scenes = PROJECT / "Assets" / "Scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    for name, scene in build().items():
        save_scene(scene, scenes / f"{name}.scene.json", None, SkySettings())
    (PROJECT / "Library").mkdir(exist_ok=True)
    return PROJECT


if __name__ == "__main__":
    print(generate())
