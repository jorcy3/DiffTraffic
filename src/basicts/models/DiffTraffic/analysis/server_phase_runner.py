from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DiffTraffic import (
    build_official_difftraffic_naive_forecasting_config,
    build_official_difftraffic_v11_gate_forecasting_config,
    build_official_difftraffic_v11_forecasting_config,
    build_official_difftraffic_v1_forecasting_config,
)
from basicts.models.STAEformer import build_official_staeformer_forecasting_config


@dataclass
class FrozenResult:
    model_name: str
    dataset: str
    checkpoint_path: str
    run_dir: str
    mae: float
    rmse: float
    mape: float
    note: str


@dataclass
class SuiteRunResult:
    experiment: str
    display_name: str
    dataset: str
    status: str
    ckpt_root: str
    checkpoint_path: str | None
    run_dir: str | None
    mae: float | None
    rmse: float | None
    mape: float | None
    error: str | None = None


_BUILDER_MAP: dict[str, Callable[..., BasicTSForecastingConfig]] = {
    "staeformer": build_official_staeformer_forecasting_config,
    "naive": build_official_difftraffic_naive_forecasting_config,
    "structured": build_official_difftraffic_v1_forecasting_config,
    "gate": build_official_difftraffic_v11_gate_forecasting_config,
    "selective": build_official_difftraffic_v11_forecasting_config,
}

_DISPLAY_NAME_MAP = {
    "staeformer": "STAEformer",
    "naive": "STAEformer + naive residual head",
    "structured": "STAEformer + structured refiner",
    "gate": "STAEformer + gate only",
    "selective": "STAEformer + selective residual refiner",
}

_CKPT_ROOT_MAP = {
    "staeformer": "checkpoints/STAEformer_official",
    "naive": "checkpoints/DiffTrafficV1Naive",
    "structured": "checkpoints/DiffTrafficV1",
    "gate": "checkpoints/DiffTrafficV11Gate",
    "selective": "checkpoints/DiffTrafficV11",
}


def _ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _normalize_dataset_dir(data_root: str, dataset_name: str) -> str:
    return str(Path(data_root) / dataset_name)


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


def _load_cfg_from_checkpoint(ckpt_path: Path) -> BasicTSForecastingConfig:
    cfg_path = ckpt_path.with_name("cfg.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"cfg.json not found beside checkpoint: {ckpt_path}")
    return BasicTSForecastingConfig.from_json(str(cfg_path))


def _prepare_cfg_for_eval(
    cfg: BasicTSForecastingConfig,
    dataset_name: str,
    data_file_path: str,
) -> BasicTSForecastingConfig:
    cfg.gpus = None
    cfg.gpu_num = 0
    cfg.eval_after_train = False
    cfg.rescale = True
    cfg.dataset_name = dataset_name
    cfg.dataset_params["dataset_name"] = dataset_name
    cfg.dataset_params["data_file_path"] = data_file_path
    cfg.dataset_params["local"] = True
    cfg.dataset_params["memmap"] = False
    cfg.dataset_params["use_timestamps"] = True
    return cfg


def _evaluate_checkpoint(
    ckpt_path: str,
    dataset_name: str,
    data_root: str,
    gpus: str | None,
    batch_size: int | None = None,
) -> FrozenResult:
    resolved_ckpt = _resolve_checkpoint(ckpt_path)
    cfg = _prepare_cfg_for_eval(
        _load_cfg_from_checkpoint(resolved_ckpt),
        dataset_name=dataset_name,
        data_file_path=_normalize_dataset_dir(data_root, dataset_name),
    )
    BasicTSLauncher.launch_evaluation(
        cfg,
        ckpt_path=str(resolved_ckpt),
        gpus=gpus,
        batch_size=batch_size,
    )

    metrics_path = resolved_ckpt.parent / "test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected test_metrics.json after evaluation, but missing: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics_payload = json.load(file)

    overall = metrics_payload["overall"]
    return FrozenResult(
        model_name=cfg.model.__name__,
        dataset=dataset_name,
        checkpoint_path=str(resolved_ckpt),
        run_dir=str(resolved_ckpt.parent),
        mae=float(overall["MAE"]),
        rmse=float(overall["RMSE"]),
        mape=float(overall["MAPE"]),
        note="single seed; raw-scale evaluation via STAEformer-compatible taskflow",
    )


def _default_ckpt_root(repo_root: str, experiment_key: str, dataset_name: str) -> str:
    return str(Path(repo_root) / _CKPT_ROOT_MAP[experiment_key] / dataset_name)


def _build_train_cfg(
    experiment_key: str,
    dataset_name: str,
    data_root: str,
    gpus: str | None,
    ckpt_root: str,
) -> BasicTSForecastingConfig:
    builder = _BUILDER_MAP[experiment_key]
    return builder(
        dataset_name=dataset_name,
        data_file_path=_normalize_dataset_dir(data_root, dataset_name),
        gpus=gpus,
        ckpt_save_dir=ckpt_root,
    )


def train_experiment(
    experiment_key: str,
    dataset_name: str,
    data_root: str,
    gpus: str | None,
    ckpt_root: str,
) -> Path:
    cfg = _build_train_cfg(
        experiment_key=experiment_key,
        dataset_name=dataset_name,
        data_root=data_root,
        gpus=gpus,
        ckpt_root=ckpt_root,
    )
    BasicTSLauncher.launch_training(cfg)
    return _resolve_checkpoint(ckpt_root)


def evaluate_experiment(
    experiment_key: str,
    dataset_name: str,
    data_root: str,
    gpus: str | None,
    ckpt_path: str,
    batch_size: int | None = None,
) -> FrozenResult:
    result = _evaluate_checkpoint(
        ckpt_path=ckpt_path,
        dataset_name=dataset_name,
        data_root=data_root,
        gpus=gpus,
        batch_size=batch_size,
    )
    result.model_name = _DISPLAY_NAME_MAP[experiment_key]
    return result


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _render_results_table(results: list[FrozenResult]) -> str:
    header = "| Model | Dataset | MAE | RMSE | MAPE | Checkpoint | Note |"
    separator = "|---|---:|---:|---:|---:|---|---|"
    rows = [
        (
            f"| {item.model_name} | {item.dataset} | {item.mae:.4f} | {item.rmse:.4f} | "
            f"{item.mape:.4f} | `{item.checkpoint_path}` | {item.note} |"
        )
        for item in results
    ]
    return "\n".join([header, separator, *rows])


def _render_suite_table(results: list[SuiteRunResult]) -> str:
    header = "| Experiment | Dataset | Status | MAE | RMSE | MAPE | Checkpoint | Error |"
    separator = "|---|---|---|---:|---:|---:|---|---|"
    rows = []
    for item in results:
        mae = f"{item.mae:.4f}" if item.mae is not None else "-"
        rmse = f"{item.rmse:.4f}" if item.rmse is not None else "-"
        mape = f"{item.mape:.4f}" if item.mape is not None else "-"
        ckpt = f"`{item.checkpoint_path}`" if item.checkpoint_path else "-"
        error = item.error or "-"
        rows.append(
            f"| {item.display_name} | {item.dataset} | {item.status} | {mae} | {rmse} | {mape} | {ckpt} | {error} |"
        )
    return "\n".join([header, separator, *rows])


def _write_markdown_summary(
    path: Path,
    metrla_results: list[FrozenResult],
    pemsbay_results: list[FrozenResult],
) -> None:
    lines = [
        "# DiffTraffic Phase Summary",
        "",
        "## Current Route Status",
        "- mainline: STAEformer deterministic backbone",
        "- paused: residual mainline",
        "- exploratory: existing residual variants kept for reference only",
        "",
        "## METR-LA Frozen Results",
        _render_results_table(metrla_results),
        "",
    ]
    if pemsbay_results:
        lines.extend(
            [
                "## PEMS-BAY Results",
                _render_results_table(pemsbay_results),
                "",
            ]
        )
    lines.extend(
        [
            "## Phase Statement",
            "- STAEformer baseline is the current credible mainline.",
            "- Current residual variants are not supported by the available single-seed evidence.",
            "- Residual refinement remains paused until cross-dataset comparison and route review complete.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_phase_status(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Phase Status",
                "",
                "- mainline: STAEformer deterministic backbone",
                "- paused: residual mainline",
                "- exploratory: residual variants kept for cross-dataset review only",
                "",
                "Current judgment:",
                "- STAEformer baseline stands.",
                "- Current residual variants are not supported.",
                "- The residual mainline is paused pending PEMS-BAY comparison and route review.",
            ]
        ),
        encoding="utf-8",
    )


def _load_metrics_if_available(checkpoint_path: str | None) -> tuple[float | None, float | None, float | None]:
    if checkpoint_path is None:
        return None, None, None
    metrics_path = Path(checkpoint_path).with_name("test_metrics.json")
    if not metrics_path.exists():
        return None, None, None
    with open(metrics_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    overall = payload.get("overall", {})
    mae = overall.get("MAE")
    rmse = overall.get("RMSE")
    mape = overall.get("MAPE")
    return (
        float(mae) if mae is not None else None,
        float(rmse) if rmse is not None else None,
        float(mape) if mape is not None else None,
    )


def freeze_results(
    dataset_name: str,
    data_root: str,
    gpus: str | None,
    output_dir: str,
    checkpoints: dict[str, str],
) -> list[FrozenResult]:
    _ensure_dir(output_dir)
    results: list[FrozenResult] = []
    for experiment_key, ckpt_path in checkpoints.items():
        results.append(
            evaluate_experiment(
                experiment_key=experiment_key,
                dataset_name=dataset_name,
                data_root=data_root,
                gpus=gpus,
                ckpt_path=ckpt_path,
            )
        )

    results_json = {
        "dataset": dataset_name,
        "results": [asdict(item) for item in results],
    }
    filename = f"{dataset_name.lower().replace('-', '_')}_results.json"
    _write_json(Path(output_dir) / filename, results_json)
    return results


def run_phase_pipeline(args: argparse.Namespace) -> None:
    output_dir = _ensure_dir(args.output_dir)
    metrla_results = freeze_results(
        dataset_name="METR-LA",
        data_root=args.data_root,
        gpus=args.gpus,
        output_dir=str(output_dir),
        checkpoints={
            "staeformer": args.metrla_staeformer_ckpt,
            "naive": args.metrla_naive_ckpt,
            "structured": args.metrla_structured_ckpt,
            "gate": args.metrla_gate_ckpt,
        },
    )

    pemsbay_results: list[FrozenResult] = []
    if args.run_pemsbay:
        baseline_ckpt = args.pemsbay_staeformer_ckpt
        if baseline_ckpt is None:
            baseline_root = args.pemsbay_staeformer_ckpt_root or _default_ckpt_root(
                args.repo_root,
                "staeformer",
                "PEMS-BAY",
            )
            baseline_ckpt = str(
                train_experiment(
                    experiment_key="staeformer",
                    dataset_name="PEMS-BAY",
                    data_root=args.data_root,
                    gpus=args.gpus,
                    ckpt_root=baseline_root,
                )
            )

        variant_key = args.pemsbay_variant
        variant_ckpt = args.pemsbay_variant_ckpt
        if variant_ckpt is None:
            variant_root = args.pemsbay_variant_ckpt_root or _default_ckpt_root(
                args.repo_root,
                variant_key,
                "PEMS-BAY",
            )
            variant_ckpt = str(
                train_experiment(
                    experiment_key=variant_key,
                    dataset_name="PEMS-BAY",
                    data_root=args.data_root,
                    gpus=args.gpus,
                    ckpt_root=variant_root,
                )
            )

        pemsbay_results = freeze_results(
            dataset_name="PEMS-BAY",
            data_root=args.data_root,
            gpus=args.gpus,
            output_dir=str(output_dir),
            checkpoints={
                "staeformer": baseline_ckpt,
                variant_key: variant_ckpt,
            },
        )

    _write_markdown_summary(
        output_dir / "phase_summary.md",
        metrla_results=metrla_results,
        pemsbay_results=pemsbay_results,
    )
    _write_json(
        output_dir / "route_status.json",
        {
            "mainline": ["STAEformer"],
            "paused": ["DiffTraffic residual mainline"],
            "exploratory": [
                "naive residual head",
                "structured refiner",
                "gate only",
            ],
        },
    )
    _write_phase_status(output_dir / "phase_status.md")


def run_training_suite(
    *,
    dataset_name: str,
    data_root: str,
    gpus: str | None,
    suite_root: str,
    experiments: list[str],
    continue_on_error: bool,
) -> list[SuiteRunResult]:
    suite_dir = _ensure_dir(suite_root)
    results: list[SuiteRunResult] = []

    for experiment_key in experiments:
        display_name = _DISPLAY_NAME_MAP[experiment_key]
        ckpt_root = str(suite_dir / experiment_key)
        try:
            best_ckpt = train_experiment(
                experiment_key=experiment_key,
                dataset_name=dataset_name,
                data_root=data_root,
                gpus=gpus,
                ckpt_root=ckpt_root,
            )
            mae, rmse, mape = _load_metrics_if_available(str(best_ckpt))
            results.append(
                SuiteRunResult(
                    experiment=experiment_key,
                    display_name=display_name,
                    dataset=dataset_name,
                    status="completed",
                    ckpt_root=ckpt_root,
                    checkpoint_path=str(best_ckpt),
                    run_dir=str(best_ckpt.parent),
                    mae=mae,
                    rmse=rmse,
                    mape=mape,
                )
            )
        except Exception as exc:
            results.append(
                SuiteRunResult(
                    experiment=experiment_key,
                    display_name=display_name,
                    dataset=dataset_name,
                    status="failed",
                    ckpt_root=ckpt_root,
                    checkpoint_path=None,
                    run_dir=None,
                    mae=None,
                    rmse=None,
                    mape=None,
                    error=str(exc),
                )
            )
            if not continue_on_error:
                break

        _write_json(
            suite_dir / "suite_manifest.json",
            {
                "dataset": dataset_name,
                "results": [asdict(item) for item in results],
            },
        )
        (suite_dir / "suite_summary.md").write_text(_render_suite_table(results), encoding="utf-8")

    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified server-side train/eval/freeze pipeline for the paused DiffTraffic residual stage."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one experiment with unified checkpoint compatibility.")
    train_parser.add_argument("--experiment", choices=sorted(_BUILDER_MAP.keys()), required=True)
    train_parser.add_argument("--dataset-name", choices=["METR-LA", "PEMS-BAY"], required=True)
    train_parser.add_argument("--data-root", required=True)
    train_parser.add_argument("--gpus", default=None)
    train_parser.add_argument("--ckpt-root", required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluate one best checkpoint with unified checkpoint compatibility.")
    eval_parser.add_argument("--dataset-name", choices=["METR-LA", "PEMS-BAY"], required=True)
    eval_parser.add_argument("--data-root", required=True)
    eval_parser.add_argument("--gpus", default=None)
    eval_parser.add_argument("--ckpt", required=True)
    eval_parser.add_argument("--batch-size", type=int, default=None)

    freeze_parser = subparsers.add_parser("freeze", help="Freeze one dataset result table from explicit checkpoint paths.")
    freeze_parser.add_argument("--dataset-name", choices=["METR-LA", "PEMS-BAY"], required=True)
    freeze_parser.add_argument("--data-root", required=True)
    freeze_parser.add_argument("--gpus", default=None)
    freeze_parser.add_argument("--output-dir", required=True)
    freeze_parser.add_argument("--staeformer-ckpt", required=True)
    freeze_parser.add_argument("--naive-ckpt", default=None)
    freeze_parser.add_argument("--structured-ckpt", default=None)
    freeze_parser.add_argument("--gate-ckpt", default=None)

    suite_parser = subparsers.add_parser("suite-train", help="Train a full dataset suite of baseline/residual variants.")
    suite_parser.add_argument("--dataset-name", choices=["METR-LA", "PEMS-BAY"], required=True)
    suite_parser.add_argument("--data-root", required=True)
    suite_parser.add_argument("--gpus", default=None)
    suite_parser.add_argument("--suite-root", required=True)
    suite_parser.add_argument(
        "--experiments",
        nargs="+",
        default=["staeformer", "naive", "structured", "gate", "selective"],
        choices=sorted(_BUILDER_MAP.keys()),
    )
    suite_parser.add_argument("--continue-on-error", action="store_true")

    phase_parser = subparsers.add_parser("phase", help="Run the full paused-stage server workflow.")
    phase_parser.add_argument("--repo-root", required=True)
    phase_parser.add_argument("--data-root", required=True)
    phase_parser.add_argument("--gpus", default=None)
    phase_parser.add_argument("--output-dir", required=True)
    phase_parser.add_argument("--metrla-staeformer-ckpt", required=True)
    phase_parser.add_argument("--metrla-naive-ckpt", required=True)
    phase_parser.add_argument("--metrla-structured-ckpt", required=True)
    phase_parser.add_argument("--metrla-gate-ckpt", required=True)
    phase_parser.add_argument("--run-pemsbay", action="store_true")
    phase_parser.add_argument("--pemsbay-variant", choices=["naive", "structured", "gate"], default="naive")
    phase_parser.add_argument("--pemsbay-staeformer-ckpt", default=None)
    phase_parser.add_argument("--pemsbay-variant-ckpt", default=None)
    phase_parser.add_argument("--pemsbay-staeformer-ckpt-root", default=None)
    phase_parser.add_argument("--pemsbay-variant-ckpt-root", default=None)

    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.command == "train":
        best_ckpt = train_experiment(
            experiment_key=args.experiment,
            dataset_name=args.dataset_name,
            data_root=args.data_root,
            gpus=args.gpus,
            ckpt_root=args.ckpt_root,
        )
        print(str(best_ckpt))
        return

    if args.command == "eval":
        result = _evaluate_checkpoint(
            ckpt_path=args.ckpt,
            dataset_name=args.dataset_name,
            data_root=args.data_root,
            gpus=args.gpus,
            batch_size=args.batch_size,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return

    if args.command == "freeze":
        checkpoints = {
            key: value
            for key, value in {
                "staeformer": args.staeformer_ckpt,
                "naive": args.naive_ckpt,
                "structured": args.structured_ckpt,
                "gate": args.gate_ckpt,
            }.items()
            if value is not None
        }
        results = freeze_results(
            dataset_name=args.dataset_name,
            data_root=args.data_root,
            gpus=args.gpus,
            output_dir=args.output_dir,
            checkpoints=checkpoints,
        )
        Path(args.output_dir, "results_summary.md").write_text(
            _render_results_table(results),
            encoding="utf-8",
        )
        return

    if args.command == "phase":
        run_phase_pipeline(args)
        return

    if args.command == "suite-train":
        results = run_training_suite(
            dataset_name=args.dataset_name,
            data_root=args.data_root,
            gpus=args.gpus,
            suite_root=args.suite_root,
            experiments=args.experiments,
            continue_on_error=args.continue_on_error,
        )
        print(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
