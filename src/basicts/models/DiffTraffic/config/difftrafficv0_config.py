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
