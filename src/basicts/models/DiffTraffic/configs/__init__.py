"""
Deprecated compatibility namespace.

Use `basicts.models.DiffTraffic.config` as the canonical import path.
"""

from ..config import (
    DiffTrafficV0Config,
    DiffTrafficV1Config,
    build_official_difftraffic_naive_forecasting_config,
    build_official_difftraffic_v1_forecasting_config,
)
