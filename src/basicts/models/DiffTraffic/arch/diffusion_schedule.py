from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    """
    Placeholder diffusion schedule container.

    The goal in v0 is to reserve the API surface without committing to a full
    diffusion implementation.
    """

    name: str = "linear"
    steps: int = 1

    def __post_init__(self) -> None:
        self.steps = max(int(self.steps), 1)

    def sample_timestep(
        self,
        batch_size: int,
        device: torch.device,
        train: bool = True,
    ) -> torch.Tensor:
        """
        Shape:
            return: [B]
        """

        if self.steps <= 1:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        if train:
            return torch.randint(0, self.steps, (batch_size,), device=device)
        return torch.full((batch_size,), self.steps - 1, device=device, dtype=torch.long)

    def noise_scale(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            timestep: [B]
            return: [B, 1, 1]
        """

        if self.steps <= 1:
            return torch.zeros_like(timestep, dtype=torch.float32).view(-1, 1, 1)
        return (timestep.float() / float(self.steps - 1)).view(-1, 1, 1)
