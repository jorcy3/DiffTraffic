from __future__ import annotations

from functools import wraps

import torch
from easydict import EasyDict


_PATCH_FLAG = "_basicts_checkpoint_compat_enabled"
_LOAD_FLAG = "_basicts_checkpoint_compat_wrapped"


def enable_legacy_checkpoint_compat() -> None:
    """
    Make PyTorch 2.6+ checkpoint loading compatible with EasyTorch-style checkpoints.

    This keeps the new default untouched when callers explicitly pass `weights_only`,
    while preserving the pre-2.6 behavior for legacy BasicTS/EasyTorch checkpoints that
    serialize objects such as `easydict.EasyDict`.
    """

    if getattr(torch, _PATCH_FLAG, False):
        return

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([EasyDict])

    if not getattr(torch.load, _LOAD_FLAG, False):
        original_torch_load = torch.load

        @wraps(original_torch_load)
        def compat_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        setattr(compat_torch_load, _LOAD_FLAG, True)
        torch.load = compat_torch_load

    setattr(torch, _PATCH_FLAG, True)
