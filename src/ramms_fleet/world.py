"""Builds a MuJoCo world holding one walled arena cell per rover.

Each rover gets its own cell so rovers never interact, and each cell gets its
own clutter level and seed. That per-cell variation is what makes the rovers'
local datasets non-IID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import resources

import mujoco
import numpy as np

from ramms_fleet.crowd import CrowdParams, home_positions

ROVER_PREFIX = "rover{index}/"


def rover_xml() -> str:
    return resources.files("ramms_fleet").joinpath("assets/rover.xml").read_text()


@dataclass(frozen=True)
class EnvConfig:
    """One rover's environment."""

    clutter: float = 0.4
    """Obstacles per square metre of free floor."""
    seed: int = 0
    pedestrians: int = 0
    """Scripted pedestrians walking in the cell."""


@dataclass(frozen=True)
class ArenaLayout:
    cell_size: float = 3.0
    wall_height: float = 0.2
    wall_thickness: float = 0.05


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    """Bounding-circle radius, used to keep reset poses clear."""


@dataclass
class Cell:
    index: int
    center: tuple[float, float]
    config: EnvConfig
    obstacles: list[Obstacle] = field(default_factory=list)
    pedestrian_homes: list[tuple[float, float]] = field(default_factory=list)
    """Each pedestrian body's reference position relative to the cell centre; its slide joints measure from here."""


def cell_centers(count: int, layout: ArenaLayout) -> list[tuple[float, float]]:
    cols = math.ceil(math.sqrt(count))
    return [((i % cols) * layout.cell_size, (i // cols) * layout.cell_size) for i in range(count)]


def build_world(
    configs: list[EnvConfig], layout: ArenaLayout = ArenaLayout(), crowd: CrowdParams = CrowdParams()
) -> tuple[mujoco.MjSpec, list[Cell]]:
    """Returns the composed spec and the cell geometry needed for resets."""
    spec = mujoco.MjSpec()
    spec.modelname = "ramms_fleet"
    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    world = spec.worldbody
    centers = cell_centers(len(configs), layout)
    extent = max(max(c) for c in centers) + layout.cell_size
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[extent / 2, extent / 2, 0.1],
        pos=[extent / 2 - layout.cell_size / 2, extent / 2 - layout.cell_size / 2, 0],
        rgba=[0.55, 0.55, 0.55, 1],
    )
    world.add_light(pos=[extent / 2, extent / 2, 6], dir=[0, 0, -1])

    rover = rover_xml()
    cells = []
    for index, (config, center) in enumerate(zip(configs, centers, strict=True)):
        cell = Cell(index=index, center=center, config=config)
        _add_walls(world, cell, layout)
        _add_obstacles(world, cell, layout)
        frame = world.add_frame(pos=[center[0], center[1], 0])
        # Pedestrians go into the rover's spec so they share its prefix and cell frame.
        child = mujoco.MjSpec.from_string(rover)
        _add_pedestrians(child, cell, layout, crowd)
        spec.attach(child, prefix=ROVER_PREFIX.format(index=index), frame=frame)
        cells.append(cell)
    return spec, cells


def _add_walls(world: mujoco.MjsBody, cell: Cell, layout: ArenaLayout, prefix: str | None = None) -> None:
    prefix = f"cell{cell.index}/" if prefix is None else prefix
    half = layout.cell_size / 2
    t = layout.wall_thickness / 2
    h = layout.wall_height / 2
    cx, cy = cell.center
    for name, pos, size in (
        ("n", [cx, cy + half, h], [half + t, t, h]),
        ("s", [cx, cy - half, h], [half + t, t, h]),
        ("e", [cx + half, cy, h], [t, half + t, h]),
        ("w", [cx - half, cy, h], [t, half + t, h]),
    ):
        world.add_geom(
            name=f"{prefix}wall_{name}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=pos,
            size=size,
            rgba=[0.35, 0.35, 0.40, 1],
        )


def _add_obstacles(world: mujoco.MjsBody, cell: Cell, layout: ArenaLayout, prefix: str | None = None) -> None:
    prefix = f"cell{cell.index}/" if prefix is None else prefix
    rng = np.random.default_rng(cell.config.seed)
    inner = layout.cell_size / 2 - layout.wall_thickness
    count = round(cell.config.clutter * (2 * inner) ** 2)
    cx, cy = cell.center
    for k in range(count):
        x = cx + rng.uniform(-inner + 0.2, inner - 0.2)
        y = cy + rng.uniform(-inner + 0.2, inner - 0.2)
        half_height = rng.uniform(0.08, 0.3)
        if rng.random() < 0.5:
            hx, hy = rng.uniform(0.05, 0.25, size=2)
            yaw = rng.uniform(0, math.pi)
            world.add_geom(
                name=f"{prefix}obstacle{k}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, half_height],
                size=[hx, hy, half_height],
                quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
                rgba=[0.75, 0.45, 0.20, 1],
            )
            radius = math.hypot(hx, hy)
        else:
            radius = rng.uniform(0.05, 0.2)
            world.add_geom(
                name=f"{prefix}obstacle{k}",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[x, y, half_height],
                size=[radius, half_height, 0],
                rgba=[0.70, 0.60, 0.25, 1],
            )
        cell.obstacles.append(Obstacle(x=x, y=y, radius=radius))


def _add_pedestrians(spec: mujoco.MjSpec, cell: Cell, layout: ArenaLayout, params: CrowdParams) -> None:
    """Adds the cell's pedestrians to `spec`, positioned relative to the cell centre.

    Each pedestrian is an upright capsule on two slide joints with a velocity
    actuator per axis, so joint and actuator order is ped0_x, ped0_y, ped1_x, ...
    after the rover's own. The capsule clears the floor, so walking has no friction.
    """
    count = cell.config.pedestrians
    if count == 0:
        return
    cx, cy = cell.center
    obstacles = np.array([[o.x - cx, o.y - cy, o.radius] for o in cell.obstacles]).reshape(-1, 3)
    half = layout.cell_size / 2 - layout.wall_thickness
    cell.pedestrian_homes = home_positions(obstacles, count, half, params, cell.config.seed)
    for k, (x, y) in enumerate(cell.pedestrian_homes):
        body = spec.worldbody.add_body(name=f"ped{k}", pos=[x, y, params.height / 2 + 0.005])
        for axis, direction in (("x", [1, 0, 0]), ("y", [0, 1, 0])):
            body.add_joint(name=f"ped{k}_{axis}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=direction)
        body.add_geom(
            name=f"ped{k}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[params.radius, params.height / 2 - params.radius, 0],
            mass=params.mass,
            rgba=[0.30, 0.34, 0.62, 1],
        )
        for axis in ("x", "y"):
            actuator = spec.add_actuator(
                name=f"ped{k}_{axis}",
                target=f"ped{k}_{axis}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                forcelimited=True,
                forcerange=[-params.max_force, params.max_force],
            )
            actuator.set_to_velocity(kv=params.drive_gain)


def build_cell_xml(
    config: EnvConfig, model_name: str, layout: ArenaLayout = ArenaLayout(), crowd: CrowdParams = CrowdParams()
) -> tuple[str, Cell]:
    """One rover's arena as standalone MJCF, for engines that import MJCF files (RAMMS through URLab).

    Built on rover.xml itself, so body, joint, actuator, and sensor names stay
    exactly as in the rover file (no prefixes, no "/" in names). The floor is a
    finite box rather than a plane: several of these arenas can be loaded into
    one simulation, and MuJoCo planes collide infinitely.
    """
    spec = mujoco.MjSpec.from_string(rover_xml())
    spec.modelname = model_name
    spec.option.timestep = 0.005
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    cell = Cell(index=0, center=(0.0, 0.0), config=config)
    half = layout.cell_size / 2 + layout.wall_thickness
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half, half, 0.05],
        pos=[0, 0, -0.05],
        rgba=[0.55, 0.55, 0.55, 1],
    )
    _add_walls(spec.worldbody, cell, layout, prefix="")
    _add_obstacles(spec.worldbody, cell, layout, prefix="")
    _add_pedestrians(spec, cell, layout, crowd)
    return spec.to_xml(), cell
