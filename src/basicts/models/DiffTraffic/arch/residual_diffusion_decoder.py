from typing import Optional

import torch
from torch import nn

from ..config import DiffTrafficV0Config
from .denoise_block import DenoiseBlock
from .diffusion_schedule import DiffusionSchedule
from .residual_refinement_block import ResidualRefinementBlock
from .time_embedding import SinusoidalTimeEmbedding


class ResidualDiffusionDecoder(nn.Module):
    """
    Lightweight residual decoder shell.

    v0 responsibility:
    - accept a fused condition
    - return residual_prediction with the same shape as base_prediction
    - keep diffusion-specific details out of the backbone
    """

    def __init__(self, config: DiffTrafficV0Config):
        super().__init__()
        self.config = config
        self.num_features = config.num_features
        hidden_size = config.residual_hidden_size
        dropout = config.residual_dropout
        diffusion_steps = int(getattr(config, "diffusion_steps", 1))
        self.schedule = DiffusionSchedule(name="linear", steps=diffusion_steps)
        self.timestep_embedding_dim = int(getattr(config, "timestep_embedding_dim", 16))
        self.refinement_layers = max(int(getattr(config, "residual_refinement_layers", 2)), 1)

        self.condition_projection = nn.Sequential(
            nn.Linear(self.num_features * 5, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_features),
        )
        self.time_embedding = SinusoidalTimeEmbedding(self.timestep_embedding_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(self.timestep_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.num_features),
        )
        self.denoise_block = DenoiseBlock(self.num_features, hidden_size, dropout)
        self.refinement_blocks = nn.ModuleList(
            [ResidualRefinementBlock(self.num_features, hidden_size, dropout) for _ in range(self.refinement_layers)]
        )

        last_layer = self.denoise_block.net[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        condition: dict,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        train: Optional[bool] = None,
    ) -> dict:
        """
        Args:
            condition: dict returned by ConditionFusion
            step: optional global step
            epoch: optional epoch index
            train: optional training flag

        Returns:
            dict with:
                residual_prediction: [batch_size, output_len, num_features]
                aux_info: auxiliary metadata.
        """

        base_prediction = condition["base_prediction"]
        temporal_condition = condition.get("temporal_condition", torch.zeros_like(base_prediction))
        delta_condition = condition.get("delta_condition", torch.zeros_like(base_prediction))
        graph_condition = condition.get("graph_condition", torch.zeros_like(base_prediction))
        freq_condition = condition.get("frequency_condition")
        residual_prior = condition.get("residual_prior")
        if freq_condition is None:
            freq_condition = torch.zeros_like(base_prediction)
        if residual_prior is None:
            residual_prior = torch.zeros_like(base_prediction)

        condition_signal = torch.cat(
            [temporal_condition, delta_condition, graph_condition, freq_condition, residual_prior],
            dim=-1,
        )
        condition_signal = self.condition_projection(condition_signal)

        batch_size, output_len, _ = base_prediction.shape
        timestep = self.schedule.sample_timestep(
            batch_size=batch_size,
            device=base_prediction.device,
            train=bool(train),
        )
        timestep_embed = self.time_embedding(timestep)
        timestep_embed = self.timestep_mlp(timestep_embed).unsqueeze(1).expand(-1, output_len, -1)

        noise_scale = self.schedule.noise_scale(timestep).to(base_prediction.device, dtype=base_prediction.dtype)
        noisy_residual = torch.randn_like(base_prediction) * noise_scale
        initial_residual = self.denoise_block(
            noisy_residual=noisy_residual,
            condition_signal=condition_signal,
            timestep_embedding=noise_scale.expand(-1, output_len, -1),
        )
        conditioned_context = condition_signal + timestep_embed
        residual_prediction = initial_residual
        for block in self.refinement_blocks:
            residual_prediction = block(residual_prediction, conditioned_context)
        return {
            "residual_prediction": residual_prediction,
            "aux_info": {
                "step": step,
                "epoch": epoch,
                "train": train,
                "condition_keys": list(condition.keys()),
                "diffusion_steps": self.schedule.steps,
                "refinement_layers": self.refinement_layers,
                "timestep_mean": float(timestep.float().mean().item()),
                "noise_scale_mean": float(noise_scale.mean().item()),
            },
        }

