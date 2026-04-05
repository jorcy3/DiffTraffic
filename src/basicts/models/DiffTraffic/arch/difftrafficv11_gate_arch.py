from ..config.difftrafficv11_config import DiffTrafficV11Config
from .difftrafficv11_arch import DiffTrafficV11ForForecasting


class DiffTrafficV11GateForForecasting(DiffTrafficV11ForForecasting):
    """
    Gate-only ablation for DiffTraffic-v1.1.
    """

    def __init__(self, config: DiffTrafficV11Config):
        config.residual_refiner_mode = "gate_only"
        super().__init__(config)
