"""Shared trainer helpers."""

from __future__ import annotations

import subprocess


def should_trigger_stop_loss(
    epoch: int,
    monitored_loss: float | None,
    threshold: float | None,
    warmup_epochs: int = 0,
) -> bool:
    """Return True when stop-loss threshold should terminate training.

    Args:
        epoch: Zero-based epoch index.
        monitored_loss: Validation loss if available, otherwise test loss.
        threshold: Stop-loss threshold.
        warmup_epochs: Ignore stop-loss during first N epochs.
    """
    if threshold is None or monitored_loss is None:
        return False
    if epoch + 1 <= max(0, warmup_epochs):
        return False
    return monitored_loss > threshold


def ensure_history_lists(history: dict, keys: list[str]) -> None:
    """Ensure each requested history key maps to a list."""
    for key in keys:
        value = history.get(key)
        if isinstance(value, list):
            continue
        if value is None:
            history[key] = []
        else:
            history[key] = list(value)


def _parse_cuda_index(device: str) -> int:
    """Extract CUDA index from device string (e.g. cuda:1 -> 1)."""
    if not device.startswith("cuda"):
        return 0
    parts = device.split(":", 1)
    if len(parts) == 1:
        return 0
    try:
        return max(0, int(parts[1]))
    except ValueError:
        return 0


def get_gpu_epoch_metrics(device: str) -> tuple[float | None, float | None]:
    """Return (memory_used_mb, utilization_pct) for the selected CUDA device.

    Returns (None, None) when GPU telemetry is unavailable.
    """
    if not device.startswith("cuda"):
        return None, None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None, None

    gpu_idx = _parse_cuda_index(device)
    if gpu_idx >= len(lines):
        gpu_idx = 0

    parts = [p.strip() for p in lines[gpu_idx].split(",")]
    if len(parts) < 2:
        return None, None

    try:
        mem_used = float(parts[0])
        util = float(parts[1])
    except ValueError:
        return None, None

    return mem_used, util
