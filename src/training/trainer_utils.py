"""Shared trainer helpers."""


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
