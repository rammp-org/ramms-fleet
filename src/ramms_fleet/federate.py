"""Runs federated training with every participant in its own OS process.

Starts a Flower SuperLink (whose ServerApp aggregates weights and holds no
data) and one SuperNode per rover. Each SuperNode is given only its own
rover's file through `--node-config`, and runs every ClientApp task in a child
process. The run is submitted with `flwr run`; afterwards everything is shut
down and the global model is in `<out>/federated/model.pt`.

All processes run as the same user on one machine, so file access is
separated by what each process is told, not by OS permissions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from ramms_fleet.experiment import rover_files

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROGRESS = re.compile(r"\[ROUND \d+/\d+\]|aggregate_evaluate|Strategy execution finished|ERROR|Traceback|wrote ")


def _wait_for_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise TimeoutError(f"nothing listening on port {port} after {timeout:.0f}s")


def _toml(value: object) -> str:
    return f"'{value}'" if isinstance(value, str | Path) else str(value)


def _stop(processes: list[subprocess.Popen]) -> None:
    for p in reversed(processes):
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    deadline = time.monotonic() + 15
    for p in processes:
        try:
            p.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()


def run_federated(args: argparse.Namespace) -> Path:
    if not (PROJECT_ROOT / "pyproject.toml").exists():
        raise SystemExit(f"expected a source checkout with pyproject.toml at {PROJECT_ROOT}")
    files = [f.resolve() for f in rover_files(args.data)]
    out = (args.out / "federated").resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (out / "model.pt").unlink(missing_ok=True)

    serverappio, fleet, control = args.port_base, args.port_base + 1, args.port_base + 2
    flwr_home = out / ".flwr"
    flwr_home.mkdir(exist_ok=True)
    (flwr_home / "config.toml").write_text(
        f'[superlink]\ndefault = "local"\n\n[superlink.local]\naddress = "127.0.0.1:{control}"\ninsecure = true\n'
    )
    bin_dir = Path(sys.executable).parent
    env = dict(os.environ, FLWR_HOME=str(flwr_home), PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    processes: list[subprocess.Popen] = []
    record: dict = {"superlink": None, "supernodes": []}
    try:
        superlink = subprocess.Popen(
            [
                bin_dir / "flower-superlink",
                "--insecure",
                "--serverappio-api-address",
                f"127.0.0.1:{serverappio}",
                "--fleet-api-address",
                f"127.0.0.1:{fleet}",
                "--control-api-address",
                f"127.0.0.1:{control}",
                "--database",
                str(out / "superlink.db"),
            ],
            stdout=open(logs / "superlink.log", "w"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append(superlink)
        record["superlink"] = {"pid": superlink.pid}
        _wait_for_port(control)
        _wait_for_port(fleet)

        for i, path in enumerate(files):
            node = subprocess.Popen(
                [
                    bin_dir / "flower-supernode",
                    "--insecure",
                    "--superlink",
                    f"127.0.0.1:{fleet}",
                    "--clientappio-api-address",
                    f"127.0.0.1:{control + 1 + i}",
                    "--node-config",
                    f"data-path='{path}'",
                ],
                stdout=open(logs / f"supernode_{i:02d}.log", "w"),
                stderr=subprocess.STDOUT,
                env=env,
            )
            processes.append(node)
            record["supernodes"].append({"pid": node.pid, "data_path": str(path)})
            print(f"supernode {i:02d} pid {node.pid} <- {path.name}")
        print(f"superlink pid {superlink.pid}; logs in {logs}")

        run_config = {
            "num-rovers": len(files),
            "num-server-rounds": args.rounds,
            "local-epochs": args.local_epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "history": args.history,
            "proximal-mu": args.proximal_mu,
            "seed": args.seed,
            "results-dir": out,
        }
        (out / "processes.json").write_text(json.dumps({**record, "run_config": run_config}, indent=2, default=str))
        command = [
            bin_dir / "flwr",
            "run",
            str(PROJECT_ROOT),
            "local",
            "--stream",
            "--run-config",
            " ".join(f"{k}={_toml(v)}" for k, v in run_config.items()),
        ]
        started = time.monotonic()
        with open(logs / "run.log", "w") as run_log:
            run = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            for line in run.stdout:
                run_log.write(line)
                if _PROGRESS.search(line):
                    print(line.rstrip())
            run.wait()
        if run.returncode != 0 or not (out / "model.pt").exists():
            raise SystemExit(f"federated run failed (exit {run.returncode}); see {logs}")
        print(f"federated training finished in {time.monotonic() - started:.0f}s")
        return out
    finally:
        _stop(processes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="run directory from ramms-fleet-collect")
    parser.add_argument("--out", type=Path, required=True, help="results directory (federated/ is created in it)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="> 0 turns FedAvg into FedProx")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port-base", type=int, default=9091, help="uses port-base .. port-base + 2 + rovers")
    run_federated(parser.parse_args(argv))


if __name__ == "__main__":
    main()
