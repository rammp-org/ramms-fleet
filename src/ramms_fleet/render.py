"""Renders fleet videos, with rovers colored by a model's predicted collision risk.

Replays the rovers of a collected run (same clutter, obstacle seeds, speeds, and
sensor noise) in one shared world, drives them with the exploration policy, and
feeds each rover's own noisy observations to the chosen models every control
step. Rover bodies go from blue (predicted safe) to red (collision predicted
within the label horizon). Rangefinder rays are drawn from the true geometry;
the noise only affects what the rover and models see.

- overview: every rover's cell from above, colored by one model
- compare: one rover's cell side by side, colored by two models on the same run
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ramms_fleet.experiment import RiskModel  # noqa: E402
from ramms_fleet.fleet import MujocoFleet  # noqa: E402
from ramms_fleet.policy import WanderPolicy  # noqa: E402
from ramms_fleet.spec import RoverProfile  # noqa: E402
from ramms_fleet.world import ROVER_PREFIX, EnvConfig  # noqa: E402

SAFE = np.array([0.20, 0.45, 0.80, 1.0])
DANGER = np.array([0.92, 0.15, 0.12, 1.0])
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def fleet_from_run(data_dir: Path, rovers: list[int] | None, seed: int) -> tuple[MujocoFleet, list[dict]]:
    meta = json.loads((data_dir / "meta.json").read_text())
    chosen = [r for r in meta["rovers"] if rovers is None or r["rover"] in rovers]
    configs = [EnvConfig(clutter=r["clutter"], seed=r["env_seed"], pedestrians=r.get("pedestrians", 0)) for r in chosen]
    profiles = [
        RoverProfile(
            cruise_speed=r.get("cruise_speed", RoverProfile.cruise_speed),
            range_noise=r.get("range_noise", 0.0),
            accel_noise=r.get("accel_noise", 0.0),
            gyro_noise=r.get("gyro_noise", 0.0),
        )
        for r in chosen
    ]
    fleet = MujocoFleet(configs, control_hz=meta["control_hz"], seed=seed, shared_world=True, profiles=profiles)
    return fleet, chosen


def _simulate(fleet: MujocoFleet, models: dict[str, list[RiskModel]], seconds: float, seed: int):
    """Yields per control step: the fleet and each model's per-rover risk."""
    speeds = np.array([p.cruise_speed for p in fleet.profiles])
    policy = WanderPolicy(fleet.num_rovers, fleet.dt, seed=seed + 1, cruise_speeds=speeds)
    obs = fleet.reset()
    policy.reset(np.arange(fleet.num_rovers), obs)
    for _ in range(round(seconds / fleet.dt)):
        commands = policy.act(obs)
        features = np.concatenate([obs.ranges, obs.accel, obs.gyro, obs.wheel_vel, commands], axis=1)
        risks = {}
        for name, per_rover in models.items():
            # Local models are one per rover; shared models score every rover.
            if len(per_rover) == 1:
                risks[name] = per_rover[0](features)
            else:
                risks[name] = np.array([m(features[i : i + 1])[0] for i, m in enumerate(per_rover)])
        yield risks, obs
        obs = fleet.step(commands)
        tipped = np.flatnonzero(~obs.upright)
        if len(tipped):
            obs = fleet.reset(list(tipped))
            policy.reset(tipped, obs)
            for per_rover in models.values():
                for m in per_rover:
                    m.reset(tipped if len(per_rover) == 1 else np.array([0]))


def _paint(model: mujoco.MjModel, fleet: MujocoFleet, risk: np.ndarray) -> None:
    for i in range(fleet.num_rovers):
        geom = model.geom(ROVER_PREFIX.format(index=i) + "body").id
        model.geom_rgba[geom] = SAFE + (DANGER - SAFE) * float(np.clip(risk[i], 0, 1))


def _camera(fleet: MujocoFleet, cells: list[int], aspect: float) -> mujoco.MjvCamera:
    centers = np.array([fleet.rovers[i].cell.center for i in cells])
    size = fleet.layout.cell_size
    lo, hi = centers.min(axis=0) - size / 2, centers.max(axis=0) + size / 2
    cam = mujoco.MjvCamera()
    cam.lookat[:2] = (lo + hi) / 2
    cam.elevation, cam.azimuth = -90, 90
    # Fit the grid's height and width into the default 45 degree vertical field of view.
    half_extent = max((hi - lo)[1], (hi - lo)[0] / aspect) / 2
    cam.distance = 1.08 * half_extent / math.tan(math.radians(22.5)) + 0.3
    return cam


def _cell_label_positions(fleet: MujocoFleet, cam: mujoco.MjvCamera, width: int, height: int, cells: list[int]):
    """Approximate pixel position of each cell's top-left corner for a top-down camera."""
    view_h = 2 * (cam.distance - 0.3) * math.tan(math.radians(22.5))
    scale = height / view_h
    size = fleet.layout.cell_size
    out = {}
    for i in cells:
        cx, cy = fleet.rovers[i].cell.center
        # Azimuth 90 from above: world +x points right, +y points up on screen.
        px = width / 2 + (cx - size / 2 - cam.lookat[0]) * scale
        py = height / 2 - (cy + size / 2 - cam.lookat[1]) * scale
        out[i] = (int(px) + 8, int(py) + 6)
    return out


def _describe(meta_row: dict, crowd: bool = False) -> str:
    speed, noise = meta_row.get("cruise_speed", 0.3), meta_row.get("range_noise", 0.0)
    peds = meta_row.get("pedestrians", 0)
    second = f"{peds or 'no'} pedestrian{'s' * (peds != 1)}" if crowd else f"{speed:.2f} m/s, noise {noise:.2f} m"
    return f"rover {meta_row['rover']}  clutter {meta_row['clutter']:.2f}\n{second}"


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    box = draw.multiline_textbbox(xy, text, font=font, spacing=2)
    draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0, 150))
    draw.multiline_text(xy, text, font=font, fill=(255, 255, 255), spacing=2)


def render_overview(args: argparse.Namespace) -> Path:
    fleet, rows = fleet_from_run(args.data, args.rovers, args.seed)
    model, data = fleet.worlds[0]
    name, path = args.model.split("=", 1)
    models = {name: [RiskModel(Path(path), fleet.num_rovers)]}
    width, height = args.width, args.height
    cells = list(range(fleet.num_rovers))
    cam = _camera(fleet, cells, width / height)
    labels = _cell_label_positions(fleet, cam, width, height, cells)
    font, title_font = _font(max(11, height // 55)), _font(max(16, height // 34))
    crowd = any(r.get("pedestrians", 0) for r in rows)

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    model.vis.global_.offwidth, model.vis.global_.offheight = max(width, 640), max(height, 480)
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for risks, _ in _simulate(fleet, models, args.seconds, args.seed):
            _paint(model, fleet, risks[name])
            renderer.update_scene(data, cam, scene_option=opt)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image, "RGBA")
            for i, xy in labels.items():
                _label(draw, xy, _describe(rows[i], crowd), font)
            caption = f"{args.title or name}: blue = predicted safe, red = collision predicted within 0.5 s"
            draw.text((12, height - title_font.size - 12), caption, font=title_font, fill=(255, 255, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))  # fmt: skip
            frames.append(np.asarray(image))
    return _write(frames, args.out, fleet.dt)


def render_compare(args: argparse.Namespace) -> Path:
    fleet, rows = fleet_from_run(args.data, [args.rover], args.seed)
    model, data = fleet.worlds[0]
    loaded = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        loaded[name] = [RiskModel(Path(path.replace("{rover}", f"{args.rover:02d}")), 1)]
    names = list(loaded)
    half = args.width // 2
    cam = _camera(fleet, [0], half / args.height)
    font, small = _font(max(16, args.height // 26)), _font(max(12, args.height // 40))

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
    model.vis.global_.offwidth, model.vis.global_.offheight = max(half, 640), max(args.height, 480)
    frames = []
    with mujoco.Renderer(model, height=args.height, width=half) as renderer:
        for risks, obs in _simulate(fleet, loaded, args.seconds, args.seed):
            panels = []
            for name in names:
                _paint(model, fleet, risks[name])
                renderer.update_scene(data, cam, scene_option=opt)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.text((12, 10), f"{name}: risk {risks[name][0]:.2f}", font=font, fill=(255, 255, 255),
                          stroke_width=2, stroke_fill=(0, 0, 0))  # fmt: skip
                if obs.bump[0]:
                    draw.text((12, 14 + font.size), "BUMP", font=font, fill=(255, 80, 60), stroke_width=2,
                              stroke_fill=(0, 0, 0))  # fmt: skip
                panels.append(np.asarray(image))
            frame = np.concatenate(panels, axis=1)
            image = Image.fromarray(frame)
            ImageDraw.Draw(image).text((12, args.height - small.size - 10), _describe(rows[0]), font=small,
                                       fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))  # fmt: skip
            frames.append(np.asarray(image))
    return _write(frames, args.out, fleet.dt)


def _write(frames: list[np.ndarray], out: Path, dt: float) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = round(1 / dt)
    imageio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=2,
                     pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])  # fmt: skip
    print(f"wrote {out} ({len(frames)} frames at {fps} fps, {out.stat().st_size / 1e6:.1f} MB)")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("overview", "compare"):
        p = sub.add_parser(mode)
        p.add_argument("--data", type=Path, required=True, help="run directory from ramms-fleet-collect")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--seconds", type=float, default=20.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--width", type=int, default=1280)
        p.add_argument("--height", type=int, default=720)
    overview, compare = sub.choices["overview"], sub.choices["compare"]
    overview.add_argument("--model", required=True, help="NAME=PATH to a model.pt scoring every rover")
    overview.add_argument("--rovers", type=int, nargs="+", help="subset of rover indices (default all)")
    overview.add_argument("--title")
    compare.add_argument("--rover", type=int, required=True)
    compare.add_argument(
        "--models", nargs=2, required=True, help="NAME=PATH twice; {rover} in a path becomes the rover index"
    )
    args = parser.parse_args(argv)
    (render_overview if args.mode == "overview" else render_compare)(args)


if __name__ == "__main__":
    main()
