"""Training checkpoints for resuming an interrupted run (cancelled job, time limit).

The state is written at every evaluation step (atomically: temporary file, then rename)
and holds everything the loop needs to continue bit-for-bit: model and optimizer state,
the random streams, the history and the best checkpoint so far. A finished run deletes it.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_state(path: Path | None, step: int, model, optimizer, streams: dict, **loop) -> None:
    """Write the loop state after ``step``; ``streams`` maps names to numpy Generators."""
    if path is None:
        return
    state = dict(
        step=step,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        streams={name: g.bit_generator.state for name, g in streams.items()},
        torch_rng=torch.get_rng_state(),
        **loop,
    )
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_state(path: Path | None, model, optimizer, streams: dict) -> dict | None:
    """Restore model, optimizer and streams in place; returns the saved loop state or None."""
    if path is None or not path.exists():
        return None
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if set(state["streams"]) != set(streams):
        raise ValueError(f"checkpoint streams {sorted(state['streams'])} != {sorted(streams)}")
    for name, generator in streams.items():
        generator.bit_generator.state = state["streams"][name]
    torch.set_rng_state(state["torch_rng"])
    return state
