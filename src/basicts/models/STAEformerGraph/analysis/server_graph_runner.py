from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.STAEformer.utils import load_staeformer_scaler_stats
from basicts.models.STAEformerGraph import build_official_staeformer_graph_forecasting_config


@dataclass(frozen=True)
class GraphVariant:
    name: str
    graph_hop_radius: int = 2
    graph_bias_init: float = 0.20
    graph_far_bias: float = 0.0
    directional_bias_enabled: bool = False
    directional_hop_radius: int = 2
    directional_bias_init: float = 0.10
    semantic_bias_enabled: bool = False
    semantic_topk: int = 8
    semantic_bias_init: float = 0.10
    semantic_use_abs_corr: bool = True


_VARIANT_MAP: dict[str, GraphVariant] = {
    "physical_only": GraphVariant(
        name="physical_only",
        directional_bias_enabled=False,
        semantic_bias_enabled=False,
    ),
    "physical_directional": GraphVariant(
        name="physical_directional",
        directional_bias_enabled=True,
        directional_hop_radius=2,
        directional_bias_init=0.10,
        semantic_bias_enabled=False,
    ),
    "physical_directional_semantic": GraphVariant(
        name="physical_directional_semantic",
        directional_bias_enabled=True,
        directional_hop_radius=2,
        directional_bias_init=0.10,
        semantic_bias_enabled=True,
        semantic_topk=8,
        semantic_bias_init=0.10,
    ),
}


def _normalize_dataset_dir(data_root: str, dataset_name: str) -> str:
    return str(Path(data_root) / dataset_name)


def _resolve_variant(variant_name: str) -> GraphVariant:
    if variant_name not in _VARIANT_MAP:
        supported = ", ".join(sorted(_VARIANT_MAP))
        raise ValueError(f"Unsupported variant `{variant_name}`. Supported variants: {supported}.")
    return _VARIANT_MAP[variant_name]


def _build_ckpt_root(ckpt_root: str | None, dataset_name: str, variant_name: str) -> str:
    if ckpt_root is not None:
        return ckpt_root
    return str(Path("checkpoints") / "STAEformerGraph" / variant_name / dataset_name)


def build_variant_cfg(
    dataset_name: str,
    data_root: str,
    variant_name: str,
    gpus: str | None,
    ckpt_root: str | None = None,
) -> BasicTSForecastingConfig:
    """
    Build a BasicTS config for a named STAEformerGraph ablation.
    """

    variant = _resolve_variant(variant_name)
    return build_official_staeformer_graph_forecasting_config(
        dataset_name=dataset_name,
        data_file_path=_normalize_dataset_dir(data_root, dataset_name),
        gpus=gpus,
        ckpt_save_dir=_build_ckpt_root(ckpt_root, dataset_name, variant_name),
        graph_hop_radius=variant.graph_hop_radius,
        graph_bias_init=variant.graph_bias_init,
        graph_far_bias=variant.graph_far_bias,
        directional_bias_enabled=variant.directional_bias_enabled,
        directional_hop_radius=variant.directional_hop_radius,
        directional_bias_init=variant.directional_bias_init,
        semantic_bias_enabled=variant.semantic_bias_enabled,
        semantic_topk=variant.semantic_topk,
        semantic_bias_init=variant.semantic_bias_init,
        semantic_use_abs_corr=variant.semantic_use_abs_corr,
    )


def _resolve_checkpoint(path_or_root: str) -> Path:
    candidate = Path(path_or_root)
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint path/root does not exist: {path_or_root}")

    matches = sorted(
        candidate.rglob("*_best_val_MAE.pt"),
        key=lambda item: (item.stat().st_mtime, str(item)),
    )
    if not matches:
        raise FileNotFoundError(f"No '*_best_val_MAE.pt' checkpoint found under: {path_or_root}")
    return matches[-1]


def train_variant(
    dataset_name: str,
    data_root: str,
    variant_name: str,
    gpus: str | None,
    ckpt_root: str | None = None,
) -> Path:
    """
    Train one named STAEformerGraph ablation and return the best checkpoint path.
    """

    cfg = build_variant_cfg(
        dataset_name=dataset_name,
        data_root=data_root,
        variant_name=variant_name,
        gpus=gpus,
        ckpt_root=ckpt_root,
    )
    BasicTSLauncher.launch_training(cfg)
    return _resolve_checkpoint(cfg.ckpt_save_dir)


def evaluate_variant(
    ckpt_path_or_root: str,
    dataset_name: str,
    data_root: str,
    gpus: str | None,
    eval_horizons: list[int] | None = None,
    batch_size: int | None = None,
) -> dict[str, object]:
    """
    Evaluate an existing checkpoint and return serialized metrics.
    """

    resolved_ckpt = _resolve_checkpoint(ckpt_path_or_root)
    cfg = BasicTSForecastingConfig.from_json(str(resolved_ckpt.with_name("cfg.json")))
    cfg.gpus = None
    cfg.gpu_num = 0
    cfg.eval_after_train = False
    cfg.rescale = True
    cfg.eval_horizons = eval_horizons or [3, 6, 12]
    cfg.dataset_name = dataset_name
    cfg.dataset_params["dataset_name"] = dataset_name
    cfg.dataset_params["data_file_path"] = _normalize_dataset_dir(data_root, dataset_name)
    cfg.dataset_params["local"] = True
    cfg.dataset_params["memmap"] = False
    cfg.dataset_params["use_timestamps"] = True
    cfg.stats = load_staeformer_scaler_stats(_normalize_dataset_dir(data_root, dataset_name))

    BasicTSLauncher.launch_evaluation(
        cfg,
        ckpt_path=str(resolved_ckpt),
        gpus=gpus,
        batch_size=batch_size,
    )

    metrics_path = resolved_ckpt.parent / "test_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics_payload = json.load(file)
    return metrics_payload


def run_staeformer_graph_pipeline(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or evaluate STAEformerGraph ablations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a named STAEformerGraph ablation.")
    train_parser.add_argument("--dataset-name", required=True, choices=["METR-LA", "PEMS-BAY"])
    train_parser.add_argument("--data-root", default="datasets")
    train_parser.add_argument("--variant", required=True, choices=sorted(_VARIANT_MAP))
    train_parser.add_argument("--gpus", default=None)
    train_parser.add_argument("--ckpt-root", default=None)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained STAEformerGraph checkpoint.")
    eval_parser.add_argument("--dataset-name", required=True, choices=["METR-LA", "PEMS-BAY"])
    eval_parser.add_argument("--data-root", default="datasets")
    eval_parser.add_argument("--ckpt-path", required=True)
    eval_parser.add_argument("--gpus", default=None)
    eval_parser.add_argument("--batch-size", type=int, default=None)
    eval_parser.add_argument("--eval-horizons", nargs="*", type=int, default=[3, 6, 12])

    args = parser.parse_args(argv)

    if args.command == "train":
        ckpt_path = train_variant(
            dataset_name=args.dataset_name,
            data_root=args.data_root,
            variant_name=args.variant,
            gpus=args.gpus,
            ckpt_root=args.ckpt_root,
        )
        print(json.dumps({"checkpoint_path": str(ckpt_path)}, indent=2))
        return 0

    metrics_payload = evaluate_variant(
        ckpt_path_or_root=args.ckpt_path,
        dataset_name=args.dataset_name,
        data_root=args.data_root,
        gpus=args.gpus,
        eval_horizons=args.eval_horizons,
        batch_size=args.batch_size,
    )
    summary = {
        "overall": metrics_payload.get("overall"),
        **{f"horizon_{horizon}": metrics_payload.get(f"horizon_{horizon}") for horizon in args.eval_horizons},
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_staeformer_graph_pipeline())
