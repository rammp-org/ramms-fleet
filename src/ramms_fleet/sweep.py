"""Repeats the whole experiment over seeds and summarizes across them.

For each seed: collect a fresh dataset (new obstacle layouts and trajectories),
train the local-only and centralized baselines, run federated FedAvg and
FedProx with a process per rover, and evaluate everything. Seeds run in
parallel `--jobs` at a time on separate port ranges. Finished steps are skipped
on re-run, so an interrupted sweep can be resumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BIN = Path(sys.executable).parent


THREADS_PER_JOB = "1"


def _run(step: str, command: list[str], log: Path, done: Path) -> None:
    if done.exists():
        print(f"  skip {step} (done)", flush=True)
        return
    started = time.monotonic()
    # Seeds run in parallel; without a cap every PyTorch process would use every core.
    env = dict(os.environ, OMP_NUM_THREADS=THREADS_PER_JOB)
    with open(log, "w") as f:
        result = subprocess.run([str(c) for c in command], stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0 or not done.exists():
        raise RuntimeError(f"{step} failed (exit {result.returncode}); see {log}")
    print(f"  {step} done in {time.monotonic() - started:.0f}s", flush=True)


def run_seed(seed: int, args: argparse.Namespace) -> Path:
    data = args.data_root / f"seed_{seed}"
    results = args.out / f"seed_{seed}"
    logs = results / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    port_base = args.port_base + 20 * seed
    print(f"seed {seed}: starting", flush=True)

    _run(
        f"seed {seed} collect",
        [
            BIN / "ramms-fleet-collect",
            "--rovers",
            args.rovers,
            "--seconds",
            args.seconds,
            "--seed",
            seed,
            "--out",
            data,
            "--speed",
            *args.speed,
            "--range-noise",
            *args.range_noise,
            "--accel-noise",
            *args.accel_noise,
            "--gyro-noise",
            *args.gyro_noise,
        ],
        logs / "collect.log",
        data / "meta.json",
    )
    _run(
        f"seed {seed} baselines",
        [
            BIN / "ramms-fleet-baseline",
            "--data",
            data,
            "--out",
            results,
            "--epochs",
            args.rounds * args.local_epochs,
            "--seed",
            seed,
            "--label",
            args.label,
            "--horizon-m",
            args.horizon_m,
            "--inputs",
            args.inputs,
        ],
        logs / "baseline.log",
        results / "centralized" / "model.pt",
    )
    for name, mu in federated_methods(args.proximal_mus):
        _run(
            f"seed {seed} {name}",
            [
                BIN / "ramms-fleet-federate",
                "--data",
                data,
                "--out",
                results,
                "--name",
                name,
                "--rounds",
                args.rounds,
                "--local-epochs",
                args.local_epochs,
                "--proximal-mu",
                mu,
                "--evaluate-every",
                args.evaluate_every,
                "--seed",
                seed,
                "--port-base",
                port_base,
                "--label",
                args.label,
                "--horizon-m",
                args.horizon_m,
                "--inputs",
                args.inputs,
            ],
            logs / f"{name}.log",
            results / name / "model.pt",
        )
    (results / "evaluation.json").unlink(missing_ok=True)
    _run(
        f"seed {seed} eval",
        [
            BIN / "ramms-fleet-eval",
            "--data",
            data,
            "--results",
            results,
            "--finetune-epochs",
            args.finetune_epochs,
            "--label",
            args.label,
            "--horizon-m",
            args.horizon_m,
        ],
        logs / "eval.log",
        results / "evaluation.json",
    )
    return results


def federated_methods(proximal_mus: list[float]) -> list[tuple[str, float]]:
    """FedAvg plus one FedProx run per mu; a single mu keeps the plain name `fedprox`."""
    if not proximal_mus:
        return [("fedavg", 0.0)]
    if len(proximal_mus) == 1:
        return [("fedavg", 0.0), ("fedprox", proximal_mus[0])]
    return [("fedavg", 0.0)] + [(f"fedprox-mu{mu:g}", mu) for mu in proximal_mus]


def _stats(values: list[float]) -> dict[str, float]:
    finite = np.array([v for v in values if not math.isnan(v)])
    std = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
    return {"mean": float(finite.mean()), "std": std, "n": int(len(finite))}


def summarize(out: Path, seeds: list[int]) -> dict:
    reports = {s: json.loads((out / f"seed_{s}" / "evaluation.json").read_text()) for s in seeds}
    first = next(iter(reports.values()))["rovers"][0]
    methods = [m for m in first if isinstance(first[m], dict)]

    per_seed = {
        m: [float(np.nanmean([row[m]["auprc"] for row in reports[s]["rovers"]])) for s in seeds] for m in methods
    }
    summary: dict = {"seeds": seeds, "methods": {m: _stats(v) for m, v in per_seed.items()}, "paired": {}}
    fedprox = [m for m in methods if m.startswith("fedprox")]
    pairs = [("fedavg", "local"), ("centralized", "fedavg"), ("fedavg+ft", "fedavg"), ("fedavg+ft", "local")]
    pairs += [(m, "fedavg") for m in fedprox]
    for a, b in pairs:
        if a in per_seed and b in per_seed:
            summary["paired"][f"{a} - {b}"] = _stats([x - y for x, y in zip(per_seed[a], per_seed[b], strict=True)])

    rows = [{"seed": s, **row} for s in seeds for row in reports[s]["rovers"]]
    _write_per_rover_csv(out / "per_rover.csv", rows, methods)
    summary["by_factor"] = _by_factor(rows, methods)

    # Rovers are ordered by clutter in every seed, so average by rover index.
    num_rovers = len(reports[seeds[0]]["rovers"])
    summary["by_rover"] = [
        {
            "rover": i,
            "clutter": reports[seeds[0]]["rovers"][i]["clutter"],
            **{m: _stats([reports[s]["rovers"][i][m]["auprc"] for s in seeds])["mean"] for m in methods},
        }
        for i in range(num_rovers)
    ]

    curves = {}
    for name in [m for m in methods if m.startswith("fed") and not m.endswith("+ft")]:
        by_round: dict[int, list[float]] = {}
        for s in seeds:
            path = out / f"seed_{s}" / name / "rounds.json"
            if path.exists():
                for r, metrics in json.loads(path.read_text()).items():
                    if "auprc" in metrics["evaluate"]:
                        by_round.setdefault(int(r), []).append(metrics["evaluate"]["auprc"])
        curves[name] = {r: _stats(v) for r, v in sorted(by_round.items())}
    summary["federated_eval_auprc_by_round"] = curves

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


FACTORS = ("clutter", "cruise_speed", "range_noise")


def _write_per_rover_csv(path: Path, rows: list[dict], methods: list[str]) -> None:
    columns = ["seed", "rover", "clutter", "cruise_speed", "range_noise", "accel_noise", "gyro_noise"]
    lines = [",".join(columns + [f"{m}_auprc" for m in methods])]
    for row in rows:
        values = [row.get(c) for c in columns] + [row[m]["auprc"] for m in methods]
        lines.append(",".join("" if v is None else f"{v:.6g}" if isinstance(v, float) else str(v) for v in values))
    path.write_text("\n".join(lines) + "\n")


def _by_factor(rows: list[dict], methods: list[str]) -> dict:
    """Mean AUPRC per method in low, middle, and high thirds of each varying rover factor.

    Speed and noise are shuffled across rover indices per seed, so they are
    analysed by value over all rovers of all seeds rather than by index.
    """
    result = {}
    for factor in FACTORS:
        values = np.array([row.get(factor) if row.get(factor) is not None else np.nan for row in rows], float)
        if np.all(np.isnan(values)) or np.nanmax(values) - np.nanmin(values) < 1e-9:
            continue
        edges = np.nanquantile(values, [1 / 3, 2 / 3])
        bins = np.digitize(values, edges)
        groups = []
        for b, label in enumerate(("low", "mid", "high")):
            members = [row for row, k in zip(rows, bins, strict=True) if k == b]
            if not members:
                continue
            group_values = [row[factor] for row in members]
            groups.append(
                {
                    "bin": label,
                    "range": [float(min(group_values)), float(max(group_values))],
                    "rovers": len(members),
                    **{m: float(np.nanmean([row[m]["auprc"] for row in members])) for m in methods},
                }
            )
        result[factor] = groups
    return result


def print_summary(summary: dict) -> None:
    methods = list(summary["methods"])
    width = max(11, *(len(m) for m in methods))
    print(f"\nMean test AUPRC across rovers, {len(summary['seeds'])} seeds (mean ± std across seeds)")
    for m in methods:
        s = summary["methods"][m]
        print(f"  {m:>{width}}: {s['mean']:.3f} ± {s['std']:.3f}")
    print("Paired differences (same seed)")
    for k, s in summary["paired"].items():
        print(f"  {k:>{2 * width + 3}}: {s['mean']:+.3f} ± {s['std']:.3f}")
    for factor, groups in summary.get("by_factor", {}).items():
        print(f"By {factor} (all rovers of all seeds, thirds by value)")
        print(f"  {'bin':>4} {'range':>13} " + " ".join(f"{m:>{width}}" for m in methods))
        for g in groups:
            span = f"{g['range'][0]:.2f}-{g['range'][1]:.2f}"
            print(f"  {g['bin']:>4} {span:>13} " + " ".join(f"{g[m]:>{width}.3f}" for m in methods))
    print("Federated evaluation AUPRC by round (mean over seeds)")
    for name, curve in summary["federated_eval_auprc_by_round"].items():
        print(f"  {name:>{width}}: " + "  ".join(f"r{r}={s['mean']:.3f}" for r, s in curve.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rovers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument(
        "--local-epochs", type=int, default=1, help="per round; baselines train rounds x local-epochs epochs"
    )
    parser.add_argument(
        "--proximal-mus", type=float, nargs="*", default=[0.1], help="one FedProx run per value; none for FedAvg only"
    )
    parser.add_argument("--label", choices=("time", "distance"), default="time")
    parser.add_argument("--inputs", choices=("features", "camera", "both"), default="features")
    parser.add_argument("--horizon-m", type=float, default=0.15, help="travel horizon for --label distance")
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2, help="seeds to run at the same time")
    parser.add_argument("--port-base", type=int, default=9200)
    parser.add_argument("--data-root", type=Path, default=Path("data/sweep"))
    parser.add_argument("--out", type=Path, default=Path("results/sweep"))
    parser.add_argument("--speed", type=float, nargs=2, default=(0.3, 0.3), metavar=("MIN", "MAX"))
    parser.add_argument("--range-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"))
    parser.add_argument("--accel-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"))
    parser.add_argument("--gyro-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--finetune-epochs", type=int, default=0, help="also score federated models fine-tuned per rover"
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    args.data_root, args.out = args.data_root.resolve(), args.out.resolve()
    global THREADS_PER_JOB
    THREADS_PER_JOB = str(max(1, (os.cpu_count() or 1) // max(1, args.jobs)))

    if not args.summarize_only:
        started = time.monotonic()
        with ThreadPoolExecutor(args.jobs) as pool:
            for future in [pool.submit(run_seed, s, args) for s in args.seeds]:
                future.result()
        print(f"sweep finished in {(time.monotonic() - started) / 60:.0f} min")
    print_summary(summarize(args.out, args.seeds))


if __name__ == "__main__":
    main()


def compare_main(argv: list[str] | None = None) -> None:
    """Writes a markdown comparison of several finished sweeps."""
    parser = argparse.ArgumentParser(description="Compare finished sweeps as markdown tables.")
    parser.add_argument("sweeps", nargs="+", help="NAME=RESULTS_DIR, each holding summary.json")
    parser.add_argument("--out", type=Path, help="also write the markdown here")
    args = parser.parse_args(argv)

    summaries = {}
    for item in args.sweeps:
        name, _, path = item.partition("=")
        summaries[name] = json.loads((Path(path) / "summary.json").read_text())
    methods = list(dict.fromkeys(m for s in summaries.values() for m in s["methods"]))
    pairs = list(dict.fromkeys(p for s in summaries.values() for p in s["paired"]))

    def cell(stats: dict | None, signed: bool = False) -> str:
        if not stats or math.isnan(stats["mean"]):
            return ""
        return f"{stats['mean']:{'+' if signed else ''}.3f} ± {stats['std']:.3f}"

    lines = ["## Mean test AUPRC (mean ± std over seeds)", ""]
    lines.append("| Method | " + " | ".join(summaries) + " |")
    lines.append("|---|" + "---:|" * len(summaries))
    for m in methods:
        lines.append(f"| {m} | " + " | ".join(cell(s["methods"].get(m)) for s in summaries.values()) + " |")
    lines += ["", "## Paired differences within a seed", ""]
    lines.append("| Comparison | " + " | ".join(summaries) + " |")
    lines.append("|---|" + "---:|" * len(summaries))
    for p in pairs:
        lines.append(f"| {p} | " + " | ".join(cell(s["paired"].get(p), signed=True) for s in summaries.values()) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
