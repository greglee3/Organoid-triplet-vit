from __future__ import annotations


def scheduled_triplet_lambda(
    target_lambda: float,
    epoch: int,
    ce_only_epochs: int,
    ramp_epochs: int,
) -> float:
    """Return the effective Triplet weight for one 1-based epoch.

    Epochs ``1..ce_only_epochs`` use no Triplet gradient. The following
    ``ramp_epochs`` linearly increase the weight, reaching ``target_lambda``
    on the final ramp epoch. A zero-length ramp switches to the target weight
    immediately. Defaults of zero for both schedule arguments preserve the
    historical behavior from epoch 1.
    """

    if target_lambda < 0:
        raise ValueError("target_lambda must be non-negative")
    if epoch < 1:
        raise ValueError("epoch must be 1-based and positive")
    if ce_only_epochs < 0 or ramp_epochs < 0:
        raise ValueError("Triplet schedule epochs must be non-negative")
    if target_lambda == 0.0 or epoch <= ce_only_epochs:
        return 0.0
    if ramp_epochs == 0:
        return float(target_lambda)
    ramp_step = epoch - ce_only_epochs
    ramp_fraction = min(1.0, ramp_step / ramp_epochs)
    return float(target_lambda * ramp_fraction)

