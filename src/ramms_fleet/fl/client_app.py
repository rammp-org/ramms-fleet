"""Flower ClientApp: one rover's local training and evaluation.

Each SuperNode is started with its own `data-path` node config and runs this app
in a child process, so a client only ever opens its own rover's file. Only
model weights and scalar metrics go back to the server.
"""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ramms_fleet.learning import evaluate, load_rover, make_model, train

app = ClientApp()


def _setup(msg: Message, context: Context):
    # SuperNodes run several ClientApp processes side by side on one machine.
    torch.set_num_threads(1)
    cfg = context.run_config
    data = load_rover(
        str(context.node_config["data-path"]),
        history=int(cfg["history"]),
        label=str(cfg["label"]),
        horizon_m=float(cfg["horizon-m"]),
        inputs=str(cfg["inputs"]),
    )
    model = make_model(data.input_dim, int(cfg["hidden"]), str(cfg["inputs"]), int(cfg["history"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return cfg, data, model


@app.train()
def train_handler(msg: Message, context: Context) -> Message:
    cfg, data, model = _setup(msg, context)
    server_round = int(msg.content["config"]["server-round"])
    loss = train(
        model,
        data.x_train,
        data.y_train,
        epochs=int(cfg["local-epochs"]),
        lr=float(cfg["lr"]),
        seed=int(cfg["seed"]) + server_round,
        proximal_mu=float(cfg["proximal-mu"]),
        images=data.img_train,
    )
    metrics = MetricRecord({"num-examples": len(data.x_train), "train-loss": loss})
    content = RecordDict({"arrays": ArrayRecord.from_torch_state_dict(model.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_handler(msg: Message, context: Context) -> Message:
    _, data, model = _setup(msg, context)
    scores = evaluate(model, data.x_test, data.y_test, data.img_test)
    metrics = MetricRecord(
        {
            "num-examples": scores["num_examples"],
            "auprc": scores["auprc"],
            "auroc": scores["auroc"],
            "bce": scores["bce"],
        }
    )
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)
