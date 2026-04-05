from dataclasses import dataclass, field
from typing import Literal

from basicts.models.STAEformer.config.staeformer_config import STAEformerConfig


@dataclass
class DiffTrafficV11Config(STAEformerConfig):
    """
    Configuration for DiffTraffic-v1.1 selective residual refinement.
    """

    condition_hidden_size: int = field(default=64, metadata={"help": "Hidden size shared by condition encoders."})
    residual_hidden_size: int = field(default=128, metadata={"help": "Hidden size for the residual refiner."})
    gate_hidden_size: int | None = field(default=None, metadata={"help": "Hidden size for the residual gate."})
    residual_dropout: float = field(default=0.1, metadata={"help": "Dropout for residual modules."})
    residual_refinement_layers: int = field(default=2, metadata={"help": "Number of residual refinement blocks."})
    residual_refiner_mode: Literal["selective", "gate_only"] = field(
        default="selective",
        metadata={"help": "Selective refiner mode."},
    )
    gate_bias_init: float = field(default=-2.0, metadata={"help": "Initial bias for the residual gate logits."})
    enable_residual_prior: bool = field(default=True, metadata={"help": "Whether to use residual prior conditioning."})
    loss_weight_pred: float = field(default=1.0, metadata={"help": "Weight for prediction supervision."})
    loss_weight_residual: float = field(default=0.5, metadata={"help": "Weight for effective residual supervision."})
    loss_weight_gate: float = field(default=0.01, metadata={"help": "Weight for sparse gate regularization."})
