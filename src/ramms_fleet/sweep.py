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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BIN = Path(sys.executable).parent


def _run(step: str, command: list[str], log: Path, done: Path) -> None:
    if done.exists():
        print(f"  skip {step} (done)", flush=True)
        return
    started = time.monotonic()
    with open(log, "w") as f:
        result = subprocess.run([str(c) for c in command], stdout=f, stderr=subprocess.STDOUT)
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
        ],
        logs / "collect.log",
        data / "meta.json",
    )
    _run(
        f"seed {seed} baselines",
        [BIN / "ramms-fleet-baseline", "--data", data, "--out", results, "--epochs", args.rounds, "--seed", seed],
        logs / "baseline.log",
        results / "centralized" / "model.pt",
    )
    for name, mu in (("fedavg", 0.0), ("fedprox", args.proximal_mu)):
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
                "--proximal-mu",
                mu,
                "--evaluate-every",
                args.evaluate_every,
                "--seed",
                seed,
                "--port-base",
                port_base,
            ],
            logs / f"{name}.log",
            results / name / "model.pt",
        )
    (results / "evaluation.json").unlink(missing_ok=True)
    _run(
        f"seed {seed} eval",
        [BIN / "ramms-fleet-eval", "--data", data, "--results", results],
        logs / "eval.log",
        results / "evaluation.json",
    )
    return results


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
    for a, b in (("fedavg", "local"), ("fedprox", "local"), ("fedprox", "fedavg"), ("centralized", "fedavg")):
        if a in per_seed and b in per_seed:
            summary["paired"][f"{a} - {b}"] = _stats([x - y for x, y in zip(per_seed[a], per_seed[b], strict=True)])

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
    for name in ("fedavg", "fedprox"):
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


def print_summary(summary: dict) -> None:
    methods = list(summary["methods"])
    print(f"\nMean test AUPRC across rovers, {len(summary['seeds'])} seeds (mean ± std across seeds)")
    for m in methods:
        s = summary["methods"][m]
        print(f"  {m:>12}: {s['mean']:.3f} ± {s['std']:.3f}")
    print("Paired differences (same seed)")
    for k, s in summary["paired"].items():
        print(f"  {k:>22}: {s['mean']:+.3f} ± {s['std']:.3f}")
    print("By rover (mean over seeds)")
    print(f"  {'rover':>5} {'clutter':>7} " + " ".join(f"{m:>11}" for m in methods))
    for row in summary["by_rover"]:
        print(f"  {row['rover']:>5} {row['clutter']:>7.2f} " + " ".join(f"{row[m]:>11.3f}" for m in methods))
    print("Federated evaluation AUPRC by round (mean over seeds)")
    for name, curve in summary["federated_eval_auprc_by_round"].items():
        print(f"  {name:>8}: " + "  ".join(f"r{r}={s['mean']:.3f}" for r, s in curve.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rovers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--rounds", type=int, default=50, help="federated rounds; baselines train this many epochs")
    parser.add_argument("--proximal-mu", type=float, default=0.1)
    parser.add_argument("--evaluate-every", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=2, help="seeds to run at the same time")
    parser.add_argument("--port-base", type=int, default=9200)
    parser.add_argument("--data-root", type=Path, default=Path("data/sweep"))
    parser.add_argument("--out", type=Path, default=Path("results/sweep"))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    args.data_root, args.out = args.data_root.resolve(), args.out.resolve()

    if not args.summarize_only:
        started = time.monotonic()
        with ThreadPoolExecutor(args.jobs) as pool:
            for future in [pool.submit(run_seed, s, args) for s in args.seeds]:
                future.result()
        print(f"sweep finished in {(time.monotonic() - started) / 60:.0f} min")
    print_summary(summarize(args.out, args.seeds))


if __name__ == "__main__":
    main()
