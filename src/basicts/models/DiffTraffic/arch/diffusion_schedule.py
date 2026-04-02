from dataclasses import dataclass


@dataclass
class DiffusionSchedule:
    """
    Placeholder diffusion schedule container.

    The goal in v0 is to reserve the API surface without committing to a full
    diffusion implementation.
    """

    name: str = "linear"
    steps: int = 1
