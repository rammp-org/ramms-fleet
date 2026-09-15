"""Flower ServerApp: FedAvg over the rover fleet.

The server never sees rover data. It starts from the same initial weights as
the baselines, aggregates client updates weighted by sample count, and writes
the final global model plus per-round client metrics to `results-dir`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from logging import INFO
from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ramms_fleet.experiment import initial_model, save_model
from ramms_fleet.spec import FEATURES

app = ServerApp()


class PeriodicEvalFedAvg(FedAvg):
    """FedAvg that asks clients to evaluate only every `evaluate_every` rounds and on the last round.

    Every ClientApp task starts a fresh process, so skipping evaluation rounds
    saves most of their cost while keeping a learning curve.
    """

    def __init__(self, evaluate_every: int, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.evaluate_every = max(1, evaluate_every)
        self.num_rounds = num_rounds

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if server_round % self.evaluate_every and server_round != self.num_rounds:
            return []
        return super().configure_evaluate(server_round, arrays, config, grid)


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config
    rovers = int(cfg["num-rovers"])
    history, hidden = int(cfg["history"]), int(cfg["hidden"])
    input_dim = history * len(FEATURES)
    results_dir = Path(str(cfg["results-dir"]))

    inputs = str(cfg["inputs"])
    model = initial_model(input_dim, hidden, int(cfg["seed"]), inputs)
    num_rounds = int(cfg["num-server-rounds"])
    strategy = PeriodicEvalFedAvg(
        evaluate_every=int(cfg["evaluate-every"]),
        num_rounds=num_rounds,
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_available_nodes=rovers,
        min_train_nodes=rovers,
        min_evaluate_nodes=rovers,
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord.from_torch_state_dict(model.state_dict()),
        num_rounds=num_rounds,
        train_config=ConfigRecord({}),
    )

    model.load_state_dict(result.arrays.to_torch_state_dict())
    save_model(results_dir / "model.pt", model, input_dim, hidden, history, inputs)
    rounds = {
        str(r): {
            "train": dict(result.train_metrics_clientapp.get(r, {})),
            "evaluate": dict(result.evaluate_metrics_clientapp.get(r, {})),
        }
        for r in sorted(set(result.train_metrics_clientapp) | set(result.evaluate_metrics_clientapp))
    }
    (results_dir / "rounds.json").write_text(json.dumps(rounds, indent=2))
    log(INFO, "wrote %s", results_dir / "model.pt")
