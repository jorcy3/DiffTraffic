from dataclasses import dataclass, field

from basicts.models.STAEformer.config.staeformer_config import STAEformerConfig


@dataclass
class DiffTrafficV1Config(STAEformerConfig):
    """
    Configuration for DiffTraffic-v1 with STAEformer as deterministic backbone.
    """

    condition_hidden_size: int = field(default=64, metadata={"help": "Hidden size shared by condition encoders."})
    graph_encoder_hidden_size: int | None = field(
        default=None,
        metadata={"help": "Hidden size for the graph condition encoder. Defaults to condition_hidden_size."},
    )
    residual_hidden_size: int = field(default=128, metadata={"help": "Hidden size for the residual refiner."})
    residual_dropout: float = field(default=0.1, metadata={"help": "Dropout used in the residual refiner."})
    residual_refinement_layers: int = field(
        default=2,
        metadata={"help": "Number of residual refinement blocks."},
    )
    enable_graph_condition: bool = field(default=False, metadata={"help": "Whether to use graph conditioning."})
    enable_residual_prior: bool = field(default=True, metadata={"help": "Whether to use residual prior conditioning."})
    loss_weight_pred: float = field(default=1.0, metadata={"help": "Weight for prediction supervision."})
    loss_weight_residual: float = field(default=0.5, metadata={"help": "Weight for residual supervision."})
