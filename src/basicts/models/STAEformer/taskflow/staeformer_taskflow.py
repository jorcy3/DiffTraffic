from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.runners.taskflow import BasicTSForecastingTaskFlow
from basicts.utils.mask import null_val_mask

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class STAEformerForecastingTaskFlow(BasicTSForecastingTaskFlow):
    """
    Thin compatibility wrapper that matches the official STAEformer training semantics:
    scale inputs only, keep targets in raw space, and inverse-transform predictions for metrics.
    """

    def preprocess(self, runner: "BasicTSRunner", data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shape:
            data['inputs']: [B, I, N]
            data['targets']: [B, O, N]
        """

        inputs_mask = null_val_mask(data["inputs"], runner.cfg.null_val)
        targets_mask = null_val_mask(data["targets"], runner.cfg.null_val)

        if runner.scaler is not None:
            data["inputs"] = runner.scaler.transform(data["inputs"], inputs_mask)
            data["scaler_mean"] = runner.scaler.stats["mean"]
            data["scaler_std"] = runner.scaler.stats["std"]

        data["inputs"] = torch.where(
            inputs_mask,
            data["inputs"],
            torch.tensor(runner.cfg.null_to_num, device=data["inputs"].device),
        )
        data["targets"] = torch.where(
            targets_mask,
            data["targets"],
            torch.tensor(runner.cfg.null_to_num, device=data["targets"].device),
        )
        data["targets_mask"] = targets_mask
        return data

    def postprocess(self, runner: "BasicTSRunner", forward_return: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shape:
            forward_return['prediction']: [B, O, N]
        """

        if runner.scaler is not None:
            forward_return["prediction"] = runner.scaler.inverse_transform(forward_return["prediction"])

        return forward_return
