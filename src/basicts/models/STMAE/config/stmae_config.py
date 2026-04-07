from dataclasses import dataclass, field

from basicts.models.STAEformer.config import STAEformerConfig


@dataclass
class STMAEConfig(STAEformerConfig):
    """
    Config for the minimal STMAE-style enhancement on top of STAEformer.
    """

    enhancement_value_embedding_dim: int = field(default=32, metadata={"help": "Value embedding size for enhancement."})
    enhancement_tod_embedding_dim: int = field(default=8, metadata={"help": "Time-of-day embedding size."})
    enhancement_dow_embedding_dim: int = field(default=8, metadata={"help": "Day-of-week embedding size."})
    enhancement_node_embedding_dim: int = field(default=16, metadata={"help": "Node embedding size."})
    enhancement_model_dim: int = field(default=64, metadata={"help": "Total enhancement encoder model dim."})
    enhancement_num_heads: int = field(default=4, metadata={"help": "Attention heads in enhancement encoder."})
    enhancement_num_layers: int = field(default=1, metadata={"help": "Number of enhancement encoder blocks."})
    enhancement_feed_forward_dim: int = field(default=128, metadata={"help": "Feed-forward dim in enhancement encoder."})
    enhancement_dropout: float = field(default=0.1, metadata={"help": "Dropout in enhancement encoder."})
    enhancement_scale: float = field(default=0.1, metadata={"help": "Residual scale applied to enhanced inputs."})

    spatial_mask_ratio: float = field(default=0.2, metadata={"help": "Spatial masking ratio."})
    temporal_mask_ratio: float = field(default=0.2, metadata={"help": "Temporal patch masking ratio."})
    temporal_patch_size: int = field(default=3, metadata={"help": "Temporal masking patch size."})
    mask_value: float = field(default=0.0, metadata={"help": "Replacement value for masked inputs."})
    pretrained_enhancer_ckpt: str | None = field(
        default=None,
        metadata={"help": "Optional checkpoint path for loading a pretrained enhancement encoder."},
    )

    loss_weight_prediction: float = field(default=1.0, metadata={"help": "Forecasting loss weight."})
    loss_weight_reconstruction: float = field(default=0.1, metadata={"help": "Masked reconstruction loss weight."})
