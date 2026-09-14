"""Baselines and evaluation for the federated collision-prediction experiment.

`ramms-fleet-baseline` trains the two reference points in one process:
- local: every rover trains its own model on only its own data
- centralized: one model on all rovers' data pooled (an upper bound that
  federated training tries to approach without pooling)

`ramms-fleet-eval` scores every trained model on every rover's held-out test
split. With `--finetune-epochs`, each federated model is also personalized: a
copy is fine-tuned on one rover's own training data and scored on that rover
(method `<name>+ft`), which a real client could do without sharing anything.
The evaluation itself reads all rovers' test data, which a real deployment
could not do; it is the experimenter's view, outside the federated protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from ramms_fleet.learning import LABELS, RoverData, evaluate, load_rover, make_model, train


def rover_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("rover_*.npz"))
    if not files:
        raise SystemExit(f"no rover_*.npz files in {data_dir}")
    return files


def save_model(path: Path, model: torch.nn.Module, input_dim: int, hidden: int, history: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "hidden": hidden, "history": history}, path)


def load_model(path: Path) -> tuple[torch.nn.Module, int]:
    saved = torch.load(path, weights_only=True)
    model = make_model(saved["input_dim"], saved["hidden"])
    model.load_state_dict(saved["state_dict"])
    return model, saved["history"]


def initial_model(input_dim: int, hidden: int, seed: int) -> torch.nn.Module:
    """Every method starts from the same weights for a given seed."""
    torch.manual_seed(seed)
    return make_model(input_dim, hidden)


def run_baselines(
    data_dir: Path,
    out_dir: Path,
    epochs: int,
    hidden: int,
    history: int,
    lr: float,
    seed: int,
    label: str = "time",
    horizon_m: float = 0.15,
) -> None:
    files = rover_files(data_dir)
    datasets = [load_rover(f, history, label=label, horizon_m=horizon_m) for f in files]
    input_dim = datasets[0].input_dim

    for f, data in zip(files, datasets, strict=True):
        model = initial_model(input_dim, hidden, seed)
        loss = train(model, data.x_train, data.y_train, epochs=epochs, lr=lr, seed=seed)
        save_model(out_dir / "local" / f"{f.stem}.pt", model, input_dim, hidden, history)
        print(f"local {f.stem}: {len(data.x_train)} samples, final loss {loss:.4f}")

    x = np.concatenate([d.x_train for d in datasets])
    y = np.concatenate([d.y_train for d in datasets])
    model = initial_model(input_dim, hidden, seed)
    loss = train(model, x, y, epochs=epochs, lr=lr, seed=seed)
    save_model(out_dir / "centralized" / "model.pt", model, input_dim, hidden, history)
    print(f"centralized: {len(x)} samples, final loss {loss:.4f}")


def baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train local-only and centralized baselines.")
    parser.add_argument("--data", type=Path, required=True, help="run directory from ramms-fleet-collect")
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument("--epochs", type=int, default=20, help="match federated rounds x local epochs")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", choices=LABELS, default="time", help="time horizon from collection, or distance")
    parser.add_argument("--horizon-m", type=float, default=0.15, help="travel horizon for --label distance")
    args = parser.parse_args(argv)
    run_baselines(
        args.data, args.out, args.epochs, args.hidden, args.history, args.lr, args.seed, args.label, args.horizon_m
    )


def _score(model: torch.nn.Module, data: RoverData) -> dict[str, float]:
    return evaluate(model, data.x_test, data.y_test)


def _nanmean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def run_evaluation(
    data_dir: Path,
    results_dir: Path,
    finetune_epochs: int = 0,
    lr: float = 1e-3,
    label: str = "time",
    horizon_m: float = 0.15,
) -> dict:
    files = rover_files(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    rover_meta = {r["rover"]: r for r in meta["rovers"]}
    profile_keys = ("clutter", "cruise_speed", "range_noise", "accel_noise", "gyro_noise")
    cache: dict[int, dict[int, RoverData]] = {}

    def data_for(history: int) -> dict[int, RoverData]:
        if history not in cache:
            cache[history] = {i: load_rover(f, history, label=label, horizon_m=horizon_m) for i, f in enumerate(files)}
        return cache[history]

    report: dict = {"rovers": [], "summary": {}}
    # Every subdirectory holding a model.pt is a method trained on all rovers
    # (centralized, federated, fedprox, ...); local/ holds one model per rover.
    shared = sorted(p.parent for p in results_dir.glob("*/model.pt"))
    shared_models = {p.name: load_model(p / "model.pt") for p in shared}
    finetuned = [name for name in shared_models if name.startswith("fed")] if finetune_epochs > 0 else []

    local_own, local_others = [], []
    per_method: dict[str, list[float]] = {name: [] for name in [*shared_models, *(f"{n}+ft" for n in finetuned)]}
    for i, f in enumerate(files):
        row: dict = {"rover": i, **{k: rover_meta.get(i, {}).get(k) for k in profile_keys}}
        for name, (model, history) in shared_models.items():
            data = data_for(history)[i]
            row[name] = _score(model, data)
            per_method[name].append(row[name]["auprc"])
            if name in finetuned:
                personal = copy.deepcopy(model)
                train(personal, data.x_train, data.y_train, epochs=finetune_epochs, lr=lr, seed=0)
                row[f"{name}+ft"] = _score(personal, data)
                per_method[f"{name}+ft"].append(row[f"{name}+ft"]["auprc"])

        local_path = results_dir / "local" / f"{f.stem}.pt"
        if local_path.exists():
            model, history = load_model(local_path)
            data = data_for(history)
            row["local"] = _score(model, data[i])
            others = [_score(model, data[j])["auprc"] for j in data if j != i]
            row["local_on_other_rovers_auprc"] = _nanmean(others)
            local_own.append(row["local"]["auprc"])
            local_others.append(row["local_on_other_rovers_auprc"])
        report["rovers"].append(row)

    for name, values in per_method.items():
        report["summary"][f"{name}_mean_auprc"] = _nanmean(values)
    if local_own:
        report["summary"]["local_mean_auprc"] = _nanmean(local_own)
        report["summary"]["local_on_other_rovers_mean_auprc"] = _nanmean(local_others)

    (results_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    return report


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score trained models on every rover's test split.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--finetune-epochs", type=int, default=0, help="personalize federated models on each rover")
    parser.add_argument("--label", choices=LABELS, default="time")
    parser.add_argument("--horizon-m", type=float, default=0.15)
    args = parser.parse_args(argv)
    report = run_evaluation(
        args.data, args.results, finetune_epochs=args.finetune_epochs, label=args.label, horizon_m=args.horizon_m
    )

    first = report["rovers"][0]
    methods = [m for m in first if isinstance(first[m], dict)]
    print("Test AUPRC per rover (higher is better; positive rate shown for scale)")
    width = max(11, *(len(m) for m in methods))
    print(f"{'rover':>5} {'clutter':>7} {'pos rate':>8} " + " ".join(f"{m:>{width}}" for m in methods))
    for row in report["rovers"]:
        pos_rate = row[methods[0]]["positive_rate"] if methods else float("nan")
        cells = " ".join(f"{row[m]['auprc']:>{width}.3f}" for m in methods)
        print(f"{row['rover']:>5} {row['clutter']:>7.2f} {pos_rate:>8.3f} {cells}")
    for key, value in report["summary"].items():
        print(f"{key}: {value:.3f}")
    print(f"wrote {args.results / 'evaluation.json'}")
