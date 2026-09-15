"""RAMMS backend: the same fleet interface as MujocoFleet, simulated inside RAMMS.

Each rover's arena is exported as standalone MJCF, imported into a dedicated
RAMMS level through URLab, and spawned as its own articulation. The fleet then
starts Play In Editor, switches URLab to direct (client-clocked) stepping, and
drives every rover with one `step` request per control step.

Requires the RAMMS editor to be running with the URLab bridge listening.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ramms_fleet.crowd import CrowdParams
from ramms_fleet.fleet import FleetObs, SensorNoise, make_crowd, pedestrian_circles, sample_clear_pose
from ramms_fleet.ramms.bridge import BridgeError, URLabBridge
from ramms_fleet.spec import RANGE_SENSORS, RoverParams, RoverProfile
from ramms_fleet.world import ArenaLayout, Cell, EnvConfig, build_cell_xml

ACTOR_PREFIX = "fleet_rover"
LIGHT_PREFIX = "fleet_light"
# (actor id, rotation (roll, pitch, yaw) in degrees, intensity in lux): a key light and a softer fill.
LIGHTS = (("fleet_light_key", [0.0, -55.0, 30.0], 8.0), ("fleet_light_fill", [0.0, -35.0, 210.0], 3.0))
TIMESTEP = 0.005
# qpos layout per articulation: free root (7), both wheel hinges, then x, y slides per pedestrian.
PED_QPOS = 9


class RammsFleet:
    backend = "ramms"

    def __init__(
        self,
        configs: list[EnvConfig],
        control_hz: float = 20.0,
        seed: int = 0,
        layout: ArenaLayout = ArenaLayout(),
        params: RoverParams = RoverParams(),
        profiles: list[RoverProfile] | None = None,
        address: str = "tcp://127.0.0.1:5559",
        level: str = "FleetArena",
        scene_dir: Path = Path("results/ramms-scene"),
        gap: float = 0.5,
        camera: bool = False,
        crowd: CrowdParams = CrowdParams(),
    ):
        self.profiles = profiles or [RoverProfile()] * len(configs)
        if len(self.profiles) != len(configs):
            raise ValueError(f"{len(self.profiles)} profiles for {len(configs)} rovers")
        self.num_rovers = len(configs)
        self.layout = layout
        self.params = params
        self.substeps = max(1, round(1.0 / (control_hz * TIMESTEP)))
        self.dt = self.substeps * TIMESTEP
        self.names = [f"{ACTOR_PREFIX}{i}" for i in range(self.num_rovers)]
        self._rng = np.random.default_rng(seed)
        self._noise = SensorNoise(self.profiles, seed, params.max_range)
        self.camera = camera
        self.camera_names = [f"{name}/front" for name in self.names]
        self._images: np.ndarray | None = None
        self._image_time = np.full(len(configs), -np.inf)

        cols = math.ceil(math.sqrt(self.num_rovers))
        pitch = layout.cell_size + gap
        self.cells: list[Cell] = []
        xml_paths: list[Path] = []
        scene_dir = Path(scene_dir).resolve()
        scene_dir.mkdir(parents=True, exist_ok=True)
        for i, config in enumerate(configs):
            xml, local = build_cell_xml(config, self.names[i], layout, crowd)
            center = ((i % cols) * pitch, (i // cols) * pitch)
            # Obstacles are generated around the origin; shift them with the arena.
            for o in local.obstacles:
                o.x += center[0]
                o.y += center[1]
            self.cells.append(
                Cell(
                    index=i,
                    center=center,
                    config=config,
                    obstacles=local.obstacles,
                    pedestrian_homes=local.pedestrian_homes,
                )
            )
            path = scene_dir / f"{self.names[i]}.xml"
            path.write_text(xml)
            xml_paths.append(path)
        self.crowd = make_crowd(self.cells, layout, crowd, seed)
        self._ped_home = [np.array(c.pedestrian_homes, dtype=float).reshape(-1, 2) for c in self.cells]

        self.bridge = URLabBridge(address, timeout_s=120)
        self.bridge.hello()
        self._build_scene(level, xml_paths)
        self._start_simulation()
        self._last: dict = {}

    # ---- setup

    def _build_scene(self, level: str, xml_paths: list[Path]) -> None:
        b = self.bridge
        if b.request("pie_status").get("state") not in ("off", None):
            b.request("stop_pie")
            deadline = time.monotonic() + 60
            while b.request("pie_status").get("state") not in ("off", None):
                if time.monotonic() > deadline:
                    raise TimeoutError("RAMMS Play In Editor did not stop")
                time.sleep(0.5)
        level_path = f"/Game/Levels/{level}"
        if b.request("current_level").get("level_path") != level_path:
            try:
                b.request("load_level", level_path=level_path)
            except BridgeError:
                b.request("create_level", name=level)
        b.request("ensure_manager")

        # Remove fleet rovers left over from an earlier run, then import and
        # spawn this fleet's arenas. Reimporting picks up new obstacle layouts.
        wanted = set(self.names)
        for actor in b.request("snapshot").get("actors", []):
            actor_id = actor.get("actor_id") or ""
            if actor_id.startswith(ACTOR_PREFIX):
                b.request("remove_actor", target=actor_id)
        # A new level has no lights, and cameras would render black.
        for actor in b.request("snapshot").get("actors", []):
            if (actor.get("actor_id") or "").startswith(LIGHT_PREFIX):
                b.request("remove_actor", target=actor["actor_id"])
        for actor_id, rotation, intensity in LIGHTS:
            b.request(
                "spawn_light", kind="directional", actor_id=actor_id, rotation_euler=rotation, intensity=intensity
            )
        for name, path, cell in zip(self.names, xml_paths, self.cells, strict=True):
            blueprint = b.request("import_xml", path=str(path), force_reimport=True)["blueprint_class_path"]
            b.request("spawn_actor", blueprint=blueprint, actor_id=name, location=[cell.center[0], cell.center[1], 0.0])
        b.request("save_level")
        missing = wanted - {a.get("actor_id") for a in b.request("snapshot").get("actors", [])}
        if missing:
            raise RuntimeError(f"rovers missing from the RAMMS level after spawning: {sorted(missing)}")

    def _start_simulation(self, timeout_s: float = 300.0) -> None:
        b = self.bridge
        state = b.request("begin_pie")
        started = time.monotonic()
        while state.get("state") != "ready":
            if state.get("state") in ("compile_failed", "timeout"):
                raise RuntimeError(
                    f"RAMMS Play In Editor failed: {state.get('state')} {state.get('compile_error', '')}"
                )
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("RAMMS Play In Editor did not become ready")
            time.sleep(1.0)
            state = b.request("pie_status")
        b.hello()
        present = {a["actor_id"] for a in b.handshake.get("articulations", [])}
        if not set(self.names) <= present:
            raise RuntimeError(f"articulations missing in RAMMS: {sorted(set(self.names) - present)}")
        b.request("set_mode", mode="direct")
        b.request("set_sim_options", options={"timestep": TIMESTEP})
        for name in self.names:
            b.request("claim_control", articulation=name, ttl_s=0)
        if self.camera:
            self._warm_up_cameras()

    def _warm_up_cameras(self, max_steps: int = 100) -> None:
        """Steps with the wheels stopped until every camera delivers a frame.

        Captures only start once a step asks for them. Reset follows, so the
        warm-up steps do not affect collection.
        """
        idle = {
            name: {"ctrl": [0.0] * (2 + 2 * cell.config.pedestrians)}
            for name, cell in zip(self.names, self.cells, strict=True)
        }
        for _ in range(max_steps):
            reply = self.bridge.request("step", n_steps=1, per_articulation=idle, **self._camera_fields())
            if len(reply.get("cameras", {})) == self.num_rovers:
                self._store_frames(reply)
                return
        raise RuntimeError(
            "RAMMS cameras did not deliver frames; make sure the editor is not starved for CPU and that "
            "'Use Less CPU when in Background' is off"
        )

    def _camera_fields(self) -> dict:
        # render="sync" captures this step's state before replying: slower, but the frame matches the sensors.
        return {"include_cameras": dict.fromkeys(self.camera_names, "latest"), "render": "sync"}

    def _store_frames(self, reply: dict) -> None:
        for i, key in enumerate(self.camera_names):
            frame = reply.get("cameras", {}).get(key)
            if frame is None:
                continue
            height, width = int(frame["height"]), int(frame["width"])
            bgra = np.frombuffer(frame["data"], dtype=np.uint8).reshape(height, width, 4)
            gray = 0.114 * bgra[..., 0] + 0.587 * bgra[..., 1] + 0.299 * bgra[..., 2]
            if self._images is None:
                self._images = np.zeros((self.num_rovers, height, width), dtype=np.uint8)
            self._images[i] = np.clip(gray + 0.5, 0, 255).astype(np.uint8)
            self._image_time[i] = float(frame["sim_time"])

    # ---- fleet interface

    def reset(self, rovers: list[int] | None = None) -> FleetObs:
        """Places rovers at random clear poses in their cells.

        A full reset also zeroes every velocity. URLab's set_qpos does not touch
        velocities, so a partial reset (a tipped rover) keeps its last velocity.
        """
        chosen = range(self.num_rovers) if rovers is None else rovers
        if rovers is None:
            self.bridge.request("reset")
            peds = np.full((self.num_rovers, 0 if self.crowd is None else self.crowd.size, 2), np.nan)
        else:
            peds = self._pedestrian_positions()
        for i in chosen:
            cell = self.cells[i]
            # A full reset scatters the pedestrians too; a partial one leaves them walking.
            placed = self.crowd.place(i) if rovers is None and cell.config.pedestrians else None
            if placed is not None:
                peds[i, : len(placed)] = placed
            x, y, yaw = sample_clear_pose(
                cell, self.layout, self._rng, avoid=pedestrian_circles(cell, peds[i], self.crowd)
            )
            qpos = [x, y, self.params.wheel_radius, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            if placed is not None:
                # The full vector: root, wheel angles, then pedestrian offsets from their homes.
                qpos += [0.0, 0.0, *(placed - self._ped_home[i]).ravel().tolist()]
            self.bridge.request("set_qpos", target=self.names[i], qpos=qpos)
            # A frame from before the reset shows the old pose.
            self._image_time[i] = -np.inf
        self._last = self.bridge.request("forward")
        return self.observe()

    def step(self, commands: np.ndarray) -> FleetObs:
        """Applies (linear m/s, angular rad/s) body-velocity commands, shape (N, 2)."""
        half_track = self.params.track_width / 2
        left = (commands[:, 0] - commands[:, 1] * half_track) / self.params.wheel_radius
        right = (commands[:, 0] + commands[:, 1] * half_track) / self.params.wheel_radius
        per_articulation = {
            name: {"ctrl": [float(lv), float(rv)]} for name, lv, rv in zip(self.names, left, right, strict=True)
        }
        if self.crowd is not None:
            arts = self._last["arts"]
            rover_xy = np.array(
                [np.asarray(arts[n]["qpos"][:2]) - c.center for n, c in zip(self.names, self.cells, strict=True)]
            )
            walk = self.crowd.velocities(self._pedestrian_positions(), rover_xy, self.dt)
            for i, (name, cell) in enumerate(zip(self.names, self.cells, strict=True)):
                per_articulation[name]["ctrl"] += walk[i, : cell.config.pedestrians].ravel().tolist()
        fields = self._camera_fields() if self.camera else {}
        self._last = self.bridge.request("step", n_steps=self.substeps, per_articulation=per_articulation, **fields)
        if self.camera:
            self._store_frames(self._last)
        return self.observe()

    def observe(self) -> FleetObs:
        n = self.num_rovers
        ranges = np.empty((n, len(RANGE_SENSORS)))
        accel = np.empty((n, 3))
        gyro = np.empty((n, 3))
        wheel_vel = np.empty((n, 2))
        bump = np.empty(n, dtype=bool)
        pose = np.empty((n, 3))
        upright = np.empty(n, dtype=bool)
        arts = self._last.get("arts", {})
        for i, name in enumerate(self.names):
            art = arts[name]
            sensors = art["sensors"]
            ranges[i] = [sensors[s][0] for s in RANGE_SENSORS]
            accel[i] = sensors["accel"]
            gyro[i] = sensors["gyro"]
            wheel_vel[i] = [sensors["wheel_left_vel"][0], sensors["wheel_right_vel"][0]]
            bump[i] = sensors["bump"][0] > self.params.bump_force
            x, y, _, qw, qx, qy, qz = art["qpos"][:7]
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            pose[i] = (x - self.cells[i].center[0], y - self.cells[i].center[1], yaw)
            # z component of the chassis up axis in the world frame.
            upright[i] = 1 - 2 * (qx * qx + qy * qy) > 0.5
        ranges[(ranges < 0) | (ranges > self.params.max_range)] = self.params.max_range
        self._noise.apply(ranges, accel, gyro)
        images = image_age = None
        if self.camera and self._images is not None:
            images = self._images.copy()
            image_age = float(self._last.get("time", 0.0)) - self._image_time
        return FleetObs(
            ranges=ranges,
            accel=accel,
            gyro=gyro,
            wheel_vel=wheel_vel,
            bump=bump,
            pose=pose,
            upright=upright,
            images=images,
            image_age=image_age,
            pedestrians=None if self.crowd is None else self._pedestrian_positions(),
        )

    def _pedestrian_positions(self) -> np.ndarray:
        out = np.full((self.num_rovers, 0 if self.crowd is None else self.crowd.size, 2), np.nan)
        arts = self._last.get("arts", {})
        for i, (name, cell) in enumerate(zip(self.names, self.cells, strict=True)):
            count = cell.config.pedestrians
            if count and name in arts:
                slides = np.asarray(arts[name]["qpos"][PED_QPOS : PED_QPOS + 2 * count], dtype=float)
                out[i, :count] = self._ped_home[i] + slides.reshape(count, 2)
        return out

    def close(self, stop_simulation: bool = True) -> None:
        for name in self.names:
            try:
                self.bridge.request("release_control", articulation=name)
            except (BridgeError, TimeoutError):
                pass
        if stop_simulation:
            try:
                self.bridge.request("stop_pie")
            except (BridgeError, TimeoutError):
                pass
        self.bridge.close()
