from dataclasses import dataclass, field
from typing import Literal

from basicts.models.TimeMixer.config.timemixer_config import TimeMixerConfig


@dataclass
class DiffTrafficV0Config(TimeMixerConfig):
    """
    Configuration shell for DiffTraffic-v0.

    v0 keeps only the residual decoder controls that are needed by the current
    forecasting shell.
    """

    residual_hidden_size: int = field(default=256, metadata={"help": "Hidden size for the residual decoder."})
    residual_dropout: float = field(default=0.1, metadata={"help": "Dropout for the residual decoder."})
    use_backbone_state: bool = field(default=False, metadata={"help": "Whether to expose backbone_state to the decoder."})
    condition_mode: Literal["base_only", "base_and_state"] = field(
        default="base_only",
        metadata={"help": "Conditioning mode for the residual decoder."},
    )
    diffusion_steps: int = field(default=1, metadata={"help": "Number of diffusion denoising steps."})
    timestep_embedding_dim: int = field(default=16, metadata={"help": "Timestep embedding dimension for diffusion decoder."})
    condition_hidden_size: int | None = field(
        default=None,
        metadata={"help": "Hidden size for condition encoders. If None, use residual_hidden_size."},
    )
    graph_encoder_hidden_size: int | None = field(
        default=None,
        metadata={"help": "Hidden size for graph condition encoder. If None, use condition_hidden_size."},
    )
    enable_graph_condition: bool = field(default=False, metadata={"help": "Enable graph condition branch."})
    enable_frequency_condition: bool = field(default=False, metadata={"help": "Enable frequency condition branch."})
    enable_residual_prior: bool = field(default=True, metadata={"help": "Enable residual prior condition branch."})
    residual_refinement_layers: int = field(default=2, metadata={"help": "Number of residual refinement blocks in decoder."})
    loss_weight_pred: float = field(default=1.0, metadata={"help": "Weight for prediction supervision term."})
    loss_weight_residual: float = field(default=0.0, metadata={"help": "Weight for residual supervision term."})
    loss_weight_diff: float = field(default=0.0, metadata={"help": "Weight for diffusion regularization term."})
    loss_weight_freq: float = field(default=0.0, metadata={"help": "Weight for frequency consistency term."})
