"""
Fedroom CLI ("TUI/CLI interaction").

    fedroom room create --config configs/rooms/fashion-room.yaml
    fedroom room list
    fedroom room status fashion-room
    fedroom client join fashion-room --config configs/clients/client-a.yaml
    fedroom client train fashion-room --config configs/clients/client-a.yaml --rounds 5
    fedroom client leave fashion-room --client-id client-a
    fedroom train start fashion-room --rounds 5
    fedroom train status fashion-room
    fedroom model list fashion-room
    fedroom infer fashion-room --version latest --config configs/clients/client-a.yaml

IMPORTANT: `client join` only registers a client -- it does not train.
`client train` (or an equivalent long-running process, e.g.
`python -m client.agent <config> --rounds N`, or a Docker Compose client
container) is what actually polls for selection, downloads the model,
trains locally, and submits. Starting a round (`train start`) with no
client actively training will sit "selected" until the round times out,
fail quorum, and mark every selected client "dropped".

Talks to the coordinator purely over HTTP, so it works identically against a
locally running `uvicorn coordinator.app:app` or a Kubernetes-deployed
coordinator (point --url at the service).
"""

from __future__ import annotations

import json
import time
from typing import Optional

import requests
import typer
import yaml
from rich.console import Console
from rich.table import Table

from client.agent import ClientAgent
from client.config import ClientConfig
from client.model import build_model, model_contract_shapes, torch_state_to_numpy
from coordinator.codec import state_to_b64

app = typer.Typer(help="Fedroom: Federated Learning as a Service CLI")
room_app = typer.Typer(help="Manage federation rooms")
client_app = typer.Typer(help="Join/leave a room as a client")
train_app = typer.Typer(help="Start/inspect training")
model_app = typer.Typer(help="List models/checkpoints")
app.add_typer(room_app, name="room")
app.add_typer(client_app, name="client")
app.add_typer(train_app, name="train")
app.add_typer(model_app, name="model")

console = Console()

DEFAULT_URL = "http://localhost:8000"


def _url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


@room_app.command("create")
def room_create(
    config: str = typer.Option(..., help="Path to room YAML config"),
    url: str = typer.Option(DEFAULT_URL, "--url"),
):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Build a fresh model purely to derive the param_shapes contract + an
    # initial global state (version 0) for this room.
    try:
        model = build_model()
        param_shapes = model_contract_shapes(model)
        initial_state = torch_state_to_numpy(model)
    except ImportError:
        console.print(
            "[yellow]torch not installed; using a tiny synthetic contract[/yellow]"
        )
        import numpy as np

        initial_state = {"w": np.zeros((4,), dtype="float32")}
        param_shapes = {"w": [4]}

    payload = {
        "room_id": cfg["room_id"],
        "model_contract": {
            "model_id": cfg.get("model_contract", "fashion-cnn-v1"),
            "framework": cfg.get("framework", "pytorch"),
            "param_shapes": param_shapes,
            "max_update_norm": cfg.get("aggregation", {}).get("max_update_norm"),
        },
        "preprocessing_contract": cfg.get("preprocessing_contract", "default-v1"),
        "aggregation": cfg.get("aggregation", {}),
        "target_rounds": cfg.get("training", {}).get("rounds", 1),
        "initial_state_b64": state_to_b64(initial_state),
    }
    resp = requests.post(_url(url, "/rooms"), json=payload, timeout=30)
    if resp.status_code >= 400:
        console.print(f"[red]Error {resp.status_code}: {resp.text}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Room created:[/green] {resp.json()}")


@room_app.command("list")
def room_list(url: str = typer.Option(DEFAULT_URL, "--url")):
    resp = requests.get(_url(url, "/rooms"), timeout=10)
    resp.raise_for_status()
    rooms = resp.json()
    table = Table(title="Fedroom Rooms")
    for col in ["room_id", "state", "current_round", "current_version", "clients"]:
        table.add_column(col)
    for r in rooms:
        table.add_row(
            r["room_id"],
            r["state"],
            str(r["current_round"]),
            str(r["current_version"]),
            str(len(r["clients"])),
        )
    console.print(table)


@room_app.command("status")
def room_status(room_id: str, url: str = typer.Option(DEFAULT_URL, "--url")):
    resp = requests.get(_url(url, f"/rooms/{room_id}"), timeout=10)
    if resp.status_code == 404:
        console.print(f"[red]Room '{room_id}' not found[/red]")
        raise typer.Exit(1)
    resp.raise_for_status()
    console.print_json(json.dumps(resp.json()))


@room_app.command("stop")
def room_stop(room_id: str, url: str = typer.Option(DEFAULT_URL, "--url")):
    resp = requests.post(_url(url, f"/rooms/{room_id}/stop"), timeout=10)
    resp.raise_for_status()
    console.print(f"[yellow]Room '{room_id}' stopped[/yellow]")


@client_app.command("join")
def client_join(
    room_id: str,
    config: str = typer.Option(..., help="Client YAML config"),
    url: str = typer.Option(
        None, "--url", help="Override coordinator_url from the config"
    ),
):
    cfg = ClientConfig.from_yaml(config)
    cfg.room_id = room_id
    if url:
        cfg.coordinator_url = url
    agent = ClientAgent(cfg)
    console.print(agent.join())
    console.print(
        "[dim]Joining only registers the client -- it does not train by itself. "
        "Run `fedroom client train` (or `client.agent`) to actually participate "
        "in rounds, or the room will time out waiting for this client.[/dim]"
    )


@client_app.command("leave")
def client_leave(
    room_id: str,
    client_id: str = typer.Option(...),
    url: str = typer.Option(DEFAULT_URL, "--url"),
):
    resp = requests.post(
        _url(url, f"/rooms/{room_id}/leave"), json={"client_id": client_id}, timeout=10
    )
    resp.raise_for_status()
    console.print(f"[yellow]Client '{client_id}' left room '{room_id}'[/yellow]")


@client_app.command("train")
def client_train(
    room_id: str,
    config: str = typer.Option(..., help="Client YAML config"),
    rounds: int = typer.Option(1, help="Number of rounds to participate in"),
    url: str = typer.Option(
        None, "--url", help="Override coordinator_url from the config"
    ),
):
    """Join (if needed) and actually run this client's training loop.

    This is the command that makes rounds progress. `client join` alone
    only registers the client with the coordinator -- nothing then polls
    for selection, downloads the model, trains, and submits on its behalf.
    Without this (or an equivalent process, e.g. `python -m client.agent
    <config> --rounds N`, or a running Docker Compose client container),
    a started round will sit "selected" with zero submissions until
    `round_timeout_seconds` elapses, fail quorum, and mark the client
    "dropped" -- which repeats on every subsequent `train start` call.
    """
    cfg = ClientConfig.from_yaml(config)
    cfg.room_id = room_id
    if url:
        cfg.coordinator_url = url
    agent = ClientAgent(cfg)
    agent.join()
    console.print(
        f"[cyan]Training for up to {rounds} round(s)... "
        f"(polling every {cfg.poll_interval_seconds}s until selected)[/cyan]"
    )
    results = agent.run_rounds(rounds)
    if not results:
        console.print(
            "[red]Completed 0 rounds -- the room may not be started yet "
            "(`train start`), or this client never got selected before "
            "giving up. Check `room status` for the room's state.[/red]"
        )
        raise typer.Exit(1)
    console.print(f"[green]Completed {len(results)}/{rounds} round(s)[/green]")
    for r in results:
        console.print(r)


@train_app.command("start")
def train_start(
    room_id: str,
    rounds: int = typer.Option(5),
    url: str = typer.Option(DEFAULT_URL, "--url"),
):
    resp = requests.post(
        _url(url, f"/rooms/{room_id}/start"), json={"rounds": rounds}, timeout=10
    )
    if resp.status_code >= 400:
        console.print(f"[red]Error {resp.status_code}: {resp.text}[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]Training started for '{room_id}' (target {rounds} rounds)[/green]"
    )


@train_app.command("status")
def train_status(room_id: str, url: str = typer.Option(DEFAULT_URL, "--url")):
    room_status(room_id, url)


@train_app.command("advance")
def train_advance(room_id: str, url: str = typer.Option(DEFAULT_URL, "--url")):
    """Kick off the next round once the previous one has finalized."""
    resp = requests.post(_url(url, f"/rooms/{room_id}/next-round"), timeout=10)
    resp.raise_for_status()
    console.print(resp.json())


@model_app.command("list")
def model_list(room_id: str, url: str = typer.Option(DEFAULT_URL, "--url")):
    resp = requests.get(_url(url, f"/rooms/{room_id}"), timeout=10)
    resp.raise_for_status()
    ck = resp.json()["checkpoints"]
    table = Table(title=f"Checkpoints for '{room_id}'")
    table.add_column("version")
    table.add_column("uri")
    table.add_column("n_clients")
    for c in ck:
        table.add_row(str(c["version"]), c["uri"], str(c["n_clients"]))
    console.print(table)


@app.command("infer")
def infer(
    room_id: str,
    config: str = typer.Option(
        ..., help="Client YAML config to use for local inference"
    ),
    version: str = typer.Option("latest"),
    url: str = typer.Option(
        None, "--url", help="Override coordinator_url from the config"
    ),
):
    cfg = ClientConfig.from_yaml(config)
    cfg.room_id = room_id
    if url:
        cfg.coordinator_url = url
    agent = ClientAgent(cfg)
    agent.join()
    result = agent.infer(version=version)
    console.print(f"[cyan]Inference result:[/cyan] {result}")


if __name__ == "__main__":
    app()
