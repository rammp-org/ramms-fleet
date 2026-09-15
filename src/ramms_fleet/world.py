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

ROVER_PREFIX = "rover{index}/"


def rover_xml() -> str:
    return resources.files("ramms_fleet").joinpath("assets/rover.xml").read_text()


@dataclass(frozen=True)
class EnvConfig:
    """One rover's environment."""

    clutter: float = 0.4
    """Obstacles per square metre of free floor."""
    seed: int = 0


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


def cell_centers(count: int, layout: ArenaLayout) -> list[tuple[float, float]]:
    cols = math.ceil(math.sqrt(count))
    return [((i % cols) * layout.cell_size, (i // cols) * layout.cell_size) for i in range(count)]


def build_world(configs: list[EnvConfig], layout: ArenaLayout = ArenaLayout()) -> tuple[mujoco.MjSpec, list[Cell]]:
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
        spec.attach(
            mujoco.MjSpec.from_string(rover),
            prefix=ROVER_PREFIX.format(index=index),
            frame=frame,
        )
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


def build_cell_xml(config: EnvConfig, model_name: str, layout: ArenaLayout = ArenaLayout()) -> tuple[str, Cell]:
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
    return spec.to_xml(), cell
