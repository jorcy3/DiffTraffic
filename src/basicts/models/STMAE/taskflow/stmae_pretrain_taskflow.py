from typing import TYPE_CHECKING, Any, Dict

import torch

from basicts.runners.taskflow import BasicTSForecastingTaskFlow
from basicts.utils.mask import null_val_mask

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class STMAEPretrainTaskFlow(BasicTSForecastingTaskFlow):
    """
    Pretraining taskflow for STMAE-style masked reconstruction.
    """

    def preprocess(self, runner: "BasicTSRunner", data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shape:
            data['inputs']: [B, I, N]
        """

        inputs_mask = null_val_mask(data["inputs"], runner.cfg.null_val)

        if runner.scaler is not None:
            data["inputs"] = runner.scaler.transform(data["inputs"], inputs_mask)

        inputs_filled = torch.where(
            inputs_mask,
            data["inputs"],
            torch.tensor(runner.cfg.null_to_num, device=data["inputs"].device),
        )

        data["inputs"] = inputs_filled
        data["targets"] = inputs_filled.clone()
        data["enhancement_targets"] = inputs_filled.clone()
        data["enhancement_valid_mask"] = inputs_mask
        data["targets_mask"] = inputs_mask
        return data

    def postprocess(self, runner: "BasicTSRunner", forward_return: Dict[str, Any]) -> Dict[str, Any]:
        return forward_return
