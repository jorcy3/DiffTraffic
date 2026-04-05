from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicts.configs import BasicTSForecastingConfig
from basicts.runners.builder import Builder
from basicts.models.STAEformer.utils import load_staeformer_scaler_stats
from basicts.utils.constants import BasicTSMode


@dataclass
class LoadedModelBundle:
    name: str
    ckpt_path: str
    cfg_path: str
    cfg: BasicTSForecastingConfig
    model: torch.nn.Module
    scaler: Any


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_device(data: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def _load_cfg_from_checkpoint(ckpt_path: str) -> BasicTSForecastingConfig:
    cfg_path = Path(ckpt_path).with_name("cfg.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find cfg.json beside checkpoint: {ckpt_path}")
    return BasicTSForecastingConfig.from_json(str(cfg_path))


def _prepare_cfg_for_analysis(cfg: BasicTSForecastingConfig, dataset_name: str, data_file_path: str) -> BasicTSForecastingConfig:
    dataset_params = cfg.dataset_params or {}
    model_config = getattr(cfg, "model_config", None)

    if getattr(cfg, "input_len", None) is None:
        cfg.input_len = dataset_params.get("input_len", getattr(model_config, "input_len", None))
    if getattr(cfg, "output_len", None) is None:
        cfg.output_len = dataset_params.get("output_len", getattr(model_config, "output_len", None))
    if getattr(cfg, "use_timestamps", None) is None:
        cfg.use_timestamps = dataset_params.get("use_timestamps", True)

    cfg.gpus = None
    cfg.dataset_name = dataset_name
    cfg.use_timestamps = True if cfg.use_timestamps is None else cfg.use_timestamps
    resolved_test_batch_size = getattr(cfg, "test_batch_size", None) or getattr(cfg, "batch_size", None) or 64
    cfg.test_batch_size = min(int(resolved_test_batch_size), 64)
    cfg.dataset_params["dataset_name"] = dataset_name
    cfg.dataset_params["data_file_path"] = data_file_path
    cfg.dataset_params["local"] = True
    cfg.dataset_params["memmap"] = False
    cfg.dataset_params["use_timestamps"] = True
    cfg.dataset_params["input_len"] = cfg.input_len
    cfg.dataset_params["output_len"] = cfg.output_len

    if cfg.input_len is None or cfg.output_len is None:
        raise ValueError(
            "Failed to resolve input_len/output_len from cfg.json. "
            f"Resolved input_len={cfg.input_len}, output_len={cfg.output_len}."
        )
    return cfg


def _is_invalid_stat_tensor(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, torch.Tensor):
        return False
    return value.numel() == 0


def _resolve_scaler_stats(
    cfg: BasicTSForecastingConfig,
    checkpoint_dict: dict,
    data_file_path: str,
) -> dict | None:
    stats = checkpoint_dict.get("scaler_stats")
    if isinstance(stats, dict):
        mean = stats.get("mean")
        std = stats.get("std")
        if not _is_invalid_stat_tensor(mean) and not _is_invalid_stat_tensor(std):
            return stats

    cfg_stats = getattr(cfg, "stats", None)
    if isinstance(cfg_stats, dict):
        mean = cfg_stats.get("mean")
        std = cfg_stats.get("std")
        if not _is_invalid_stat_tensor(mean) and not _is_invalid_stat_tensor(std):
            return cfg_stats

    return load_staeformer_scaler_stats(data_file_path)


def _build_scaler(cfg: BasicTSForecastingConfig, checkpoint_dict: dict, data_file_path: str):
    if cfg.scaler is None:
        return None
    scaler = Builder._build_scaler(cfg)
    scaler.stats = _resolve_scaler_stats(cfg, checkpoint_dict, data_file_path)
    return scaler


def load_model_bundle(ckpt_path: str, dataset_name: str, data_file_path: str, device: torch.device) -> LoadedModelBundle:
    checkpoint_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _prepare_cfg_for_analysis(_load_cfg_from_checkpoint(ckpt_path), dataset_name, data_file_path)
    model = cfg.model(cfg.model_config).to(device)
    model.load_state_dict(checkpoint_dict["model_state_dict"], strict=True)
    model.eval()
    return LoadedModelBundle(
        name=cfg.model.__name__,
        ckpt_path=ckpt_path,
        cfg_path=str(Path(ckpt_path).with_name("cfg.json")),
        cfg=cfg,
        model=model,
        scaler=_build_scaler(cfg, checkpoint_dict, data_file_path),
    )


def load_adjacency_matrix(data_file_path: str) -> np.ndarray:
    with open(os.path.join(data_file_path, "adj_mx.pkl"), "rb") as file:
        obj = pickle.load(file, encoding="latin1")
    if isinstance(obj, (list, tuple)) and len(obj) >= 3 and hasattr(obj[2], "shape"):
        return np.asarray(obj[2], dtype=np.float32)
    if hasattr(obj, "shape"):
        return np.asarray(obj, dtype=np.float32)
    raise ValueError(f"Unsupported adjacency file format: {type(obj)}")


def _forward_model(model: torch.nn.Module, data: dict[str, Any], step: int, graph_prior: torch.Tensor | None = None) -> dict[str, Any]:
    params = list(inspect.signature(model.forward).parameters.keys())
    params.remove("inputs")
    kwargs = {k: data[k] for k in params if k in data}
    if "graph_prior" in params and graph_prior is not None:
        kwargs["graph_prior"] = graph_prior
    if "step" in params:
        kwargs["step"] = step
    if "epoch" in params:
        kwargs["epoch"] = 0
    if "train" in params:
        kwargs["train"] = False
    forward_return = model(data["inputs"], **kwargs)
    if isinstance(forward_return, torch.Tensor):
        forward_return = {"prediction": forward_return}
    for k, v in data.items():
        if k not in forward_return:
            forward_return[k] = v
    return forward_return


def _inverse_prediction(tensor: torch.Tensor, scaler) -> torch.Tensor:
    return tensor if scaler is None else scaler.inverse_transform(tensor)


def _inverse_residual(tensor: torch.Tensor, scaler) -> torch.Tensor:
    if scaler is None:
        return tensor
    return tensor * scaler.stats["std"].to(tensor.device)


def _timestamps_to_indices(timestamps: np.ndarray, steps_per_day: int = 288, num_day_in_week: int = 7) -> tuple[np.ndarray, np.ndarray]:
    tod = np.floor(timestamps[..., 0] * steps_per_day).astype(np.int64)
    tod = np.clip(tod, 0, steps_per_day - 1)
    dow = np.floor(timestamps[..., 1] * num_day_in_week + 1e-6).astype(np.int64)
    dow = np.clip(dow, 0, num_day_in_week - 1)
    return tod, dow


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def _rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _mape(prediction: np.ndarray, targets: np.ndarray, eps: float = 1e-5) -> float:
    return float(np.mean(np.abs((prediction - targets) / np.maximum(np.abs(targets), eps))))


def _overall_residual_stats(residual: np.ndarray) -> dict[str, float]:
    flat = np.abs(residual).reshape(-1)
    return {
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
        "abs_mean": float(np.mean(flat)),
        "p50": float(np.quantile(flat, 0.50)),
        "p75": float(np.quantile(flat, 0.75)),
        "p90": float(np.quantile(flat, 0.90)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
    }


def _horizon_stats(residual: np.ndarray) -> list[dict[str, float]]:
    out = []
    for idx in range(residual.shape[1]):
        item = residual[:, idx, :]
        out.append({"horizon": idx + 1, "mae": _mae(item), "rmse": _rmse(item), "variance": float(np.var(item))})
    return out


def _bucket_error_summary(error: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    summary = {}
    for label in list(dict.fromkeys(labels.tolist())):
        mask = labels == label
        if not np.any(mask):
            continue
        bucket = error[mask]
        summary[str(label)] = {
            "mae": _mae(bucket),
            "rmse": _rmse(bucket),
            "p50": float(np.quantile(np.abs(bucket), 0.50)),
            "p90": float(np.quantile(np.abs(bucket), 0.90)),
            "p95": float(np.quantile(np.abs(bucket), 0.95)),
        }
    return summary


def _time_bucket_labels(first_tod: np.ndarray, steps_per_day: int = 288) -> np.ndarray:
    hours = first_tod / steps_per_day * 24.0
    labels = np.full(hours.shape, "off_peak", dtype=object)
    labels[(hours >= 7.0) & (hours < 10.0)] = "morning_peak"
    labels[(hours >= 16.0) & (hours < 19.0)] = "evening_peak"
    return labels


def _time_of_day_heatmap(residual: np.ndarray, targets_timestamps: np.ndarray, steps_per_day: int = 288) -> np.ndarray:
    tod, _ = _timestamps_to_indices(targets_timestamps, steps_per_day=steps_per_day)
    hour_bins = np.clip(np.floor(tod / (steps_per_day / 24)).astype(np.int64), 0, 23)
    abs_residual = np.abs(residual).mean(axis=2)
    heatmap = np.zeros((24, residual.shape[1]), dtype=np.float64)
    for hour in range(24):
        for horizon in range(residual.shape[1]):
            mask = hour_bins[:, horizon] == hour
            if np.any(mask):
                heatmap[hour, horizon] = abs_residual[mask, horizon].mean()
    return heatmap


def _dynamic_features(inputs: np.ndarray) -> dict[str, np.ndarray]:
    deltas = inputs[:, 1:, :] - inputs[:, :-1, :]
    recent = deltas[:, -min(3, deltas.shape[1]) :, :]
    slope = np.mean(np.abs((inputs[:, -1, :] - inputs[:, -4, :]) / 3.0), axis=1) if inputs.shape[1] >= 4 else np.mean(np.abs(deltas), axis=(1, 2))
    recent_mean = np.mean(np.abs(recent), axis=(1, 2))
    volatility = np.mean(np.std(recent, axis=1), axis=1)
    return {
        "last_delta": np.mean(np.abs(deltas[:, -1, :]), axis=1),
        "recent_mean_delta": recent_mean,
        "local_slope": slope,
        "volatility": volatility,
        "change_intensity": recent_mean + volatility,
    }


def _three_way_bucket(values: np.ndarray) -> np.ndarray:
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.full(values.shape, "medium_change", dtype=object)
    labels[values <= q1] = "low_change"
    labels[values >= q2] = "high_change"
    return labels


def _graph_stats(residual: np.ndarray, adjacency: np.ndarray) -> dict[str, float]:
    series = residual[:, 0, :]
    corr = np.corrcoef(series, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = adjacency > 0
    np.fill_diagonal(adjacency, False)
    non_adj = ~adjacency
    np.fill_diagonal(non_adj, False)
    sync_adj = float(np.mean(np.abs(corr[adjacency]))) if np.any(adjacency) else 0.0
    sync_non_adj = float(np.mean(np.abs(corr[non_adj]))) if np.any(non_adj) else 0.0
    lag_adj = 0.0
    lag_non_adj = 0.0
    if series.shape[0] > 1:
        current = series[1:]
        previous = series[:-1]
        lag_corr = np.zeros_like(corr)
        for i in range(series.shape[1]):
            for j in range(series.shape[1]):
                lag_corr[i, j] = _safe_corr(current[:, i], previous[:, j])
        lag_adj = float(np.mean(np.abs(lag_corr[adjacency]))) if np.any(adjacency) else 0.0
        lag_non_adj = float(np.mean(np.abs(lag_corr[non_adj]))) if np.any(non_adj) else 0.0
    abs_series = np.abs(series)
    threshold = np.quantile(abs_series, 0.90)
    active = abs_series >= threshold
    density = adjacency.mean()
    cluster_scores = []
    for sample_active in active:
        if sample_active.sum() < 2:
            continue
        sub_adj = adjacency[np.ix_(sample_active, sample_active)]
        possible = sample_active.sum() * (sample_active.sum() - 1)
        if possible > 0:
            cluster_scores.append(float(sub_adj.sum() / possible))
    cluster_ratio = float(np.mean(cluster_scores) / max(density, 1e-8)) if cluster_scores else 0.0
    return {
        "neighbor_sync_corr": sync_adj,
        "non_neighbor_sync_corr": sync_non_adj,
        "neighbor_lag_corr": lag_adj,
        "non_neighbor_lag_corr": lag_non_adj,
        "error_cluster_ratio": cluster_ratio,
    }


def _moving_average(signal: np.ndarray, window: int = 3) -> np.ndarray:
    pad = window // 2
    padded = np.pad(signal, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    trend = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        trend[:, i, :] = padded[:, i : i + window, :].mean(axis=1)
    return trend


def _frequency_stats(residual: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    trend = _moving_average(residual, window=3)
    high = residual - trend
    total_energy = np.mean(np.square(residual), axis=(1, 2))
    high_energy = np.mean(np.square(high), axis=(1, 2))
    ratio = high_energy / np.maximum(total_energy, 1e-8)
    return {
        "high_frequency_ratio_mean": float(np.mean(ratio)),
        "high_frequency_ratio_p50": float(np.quantile(ratio, 0.50)),
        "high_frequency_ratio_p90": float(np.quantile(ratio, 0.90)),
        "trend_energy_mean": float(np.mean(np.square(trend))),
        "high_frequency_energy_mean": float(np.mean(np.square(high))),
    }, ratio


def _plot_histogram(values: np.ndarray, output_path: Path, title: str, xlabel: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        return

    plt.figure(figsize=(8, 4))
    plt.hist(values.reshape(-1), bins=80, color="#2b6cb0", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_horizon_mae(errors: dict[str, np.ndarray], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        return

    horizons = np.arange(1, next(iter(errors.values())).shape[1] + 1)
    plt.figure(figsize=(8, 4))
    for name, error in errors.items():
        plt.plot(horizons, np.mean(np.abs(error), axis=(0, 2)), marker="o", label=name)
    plt.xlabel("horizon")
    plt.ylabel("MAE")
    plt.title("Residual / Forecast Error by Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_heatmap(heatmap: np.ndarray, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        return

    plt.figure(figsize=(9, 5))
    plt.imshow(heatmap, aspect="auto", origin="lower", cmap="YlOrRd")
    plt.colorbar(label="mean |residual|")
    plt.xlabel("horizon")
    plt.ylabel("hour of day")
    plt.title("Time-of-Day × Horizon Residual Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_bucket_summary(summary: dict[str, dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        return

    labels = list(summary.keys())
    mae_values = [summary[label]["mae"] for label in labels]
    rmse_values = [summary[label]["rmse"] for label in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, mae_values, width, label="MAE")
    plt.bar(x + width / 2, rmse_values, width, label="RMSE")
    plt.xticks(x, labels)
    plt.title("Residual by Change-Intensity Bucket")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_graph_corr(graph_stats: dict[str, float], output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ModuleNotFoundError:
        return

    labels = ["sync_neighbor", "sync_non_neighbor", "lag_neighbor", "lag_non_neighbor"]
    values = [
        graph_stats["neighbor_sync_corr"],
        graph_stats["non_neighbor_sync_corr"],
        graph_stats["neighbor_lag_corr"],
        graph_stats["non_neighbor_lag_corr"],
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.xticks(rotation=15)
    plt.ylabel("mean |correlation|")
    plt.title("Neighbor vs Non-Neighbor Residual Correlation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _bucket_compare(predictions: dict[str, np.ndarray], targets: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    return {name: _bucket_error_summary(targets - pred, labels) for name, pred in predictions.items()}


def _model_metrics(prediction: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residual = targets - prediction
    return {"mae": _mae(residual), "rmse": _rmse(residual), "mape": _mape(prediction, targets)}


def _write_report(output_dir: Path, dataset_name: str, bundles: dict[str, LoadedModelBundle], summary: dict[str, Any]) -> None:
    report = f"""# Residual Diagnostics Report

## 1. 任务边界
本轮只做残差诊断，不改模型。

## 2. 数据与 checkpoint
- dataset: `{dataset_name}`
- baseline checkpoint: `{bundles['baseline'].ckpt_path}`
- naive checkpoint: `{bundles['naive'].ckpt_path if 'naive' in bundles else 'N/A'}`
- refiner checkpoint: `{bundles['refiner'].ckpt_path if 'refiner' in bundles else 'N/A'}`
- 评估口径: raw-scale residual analysis on test split
- rescale=True: yes

## 3. 残差是否有结构
### horizon
- 结论: {summary['horizon']['conclusion']}
- 证据: hardest horizons = {summary['horizon']['hardest_horizons']}, monotonic = {summary['horizon']['is_monotonic']}

### temporal
- 结论: {summary['temporal']['conclusion']}
- 证据: {json.dumps(summary['temporal']['peak_buckets'], ensure_ascii=False)}

### dynamic
- 结论: {summary['dynamic']['conclusion']}
- 证据: {json.dumps(summary['dynamic']['correlations'], ensure_ascii=False)}

### graph
- 结论: {summary['graph']['conclusion']}
- 证据: {json.dumps(summary['graph']['stats'], ensure_ascii=False)}

### frequency
- 结论: {summary['frequency']['conclusion']}
- 证据: {json.dumps(summary['frequency']['stats'], ensure_ascii=False)}

## 4. 当前 structured refiner 为什么无效
- weak correlation: {summary['why_refiner_failed']['weak_correlation']}
- coarse expression: {summary['why_refiner_failed']['coarse_expression']}
- backbone absorption: {summary['why_refiner_failed']['backbone_absorption']}
- capacity mismatch: {summary['why_refiner_failed']['capacity_mismatch']}

## 5. 对下一步的建议
- 主建议: {summary['next_step']}
- fail-fast 判断: **{summary['judgment']}**
    """
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def _collect_outputs(
    bundle: LoadedModelBundle,
    dataset_name: str,
    data_file_path: str,
    device: torch.device,
    max_batches: int | None = None,
    inject_graph_prior: bool = False,
) -> dict[str, np.ndarray]:
    dataset = bundle.cfg.dataset_type(
        dataset_name=dataset_name,
        input_len=bundle.cfg.input_len,
        output_len=bundle.cfg.output_len,
        mode=BasicTSMode.TEST,
        use_timestamps=True,
        local=True,
        data_file_path=data_file_path,
        memmap=False,
    )
    loader = DataLoader(dataset, batch_size=bundle.cfg.test_batch_size, shuffle=False)
    runner = SimpleNamespace(cfg=bundle.cfg, scaler=bundle.scaler)
    graph_prior = None
    if inject_graph_prior:
        graph_prior = torch.tensor(load_adjacency_matrix(data_file_path), dtype=torch.float32, device=device)

    gathered: dict[str, list[np.ndarray]] = {
        "inputs": [],
        "inputs_timestamps": [],
        "targets": [],
        "targets_timestamps": [],
        "prediction": [],
        "base_prediction": [],
        "residual_prediction": [],
    }

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break

            raw_inputs = batch["inputs"].float()
            raw_inputs_timestamps = batch["inputs_timestamps"].float()
            raw_targets = batch["targets"].float()
            raw_targets_timestamps = batch["targets_timestamps"].float()

            processed = bundle.cfg.taskflow.preprocess(
                runner,
                {
                    "inputs": raw_inputs.clone(),
                    "inputs_timestamps": raw_inputs_timestamps.clone(),
                    "targets": raw_targets.clone(),
                    "targets_timestamps": raw_targets_timestamps.clone(),
                },
            )
            processed = _to_device(processed, device)
            forward_return = _forward_model(bundle.model, processed, step=step, graph_prior=graph_prior)

            prediction = _inverse_prediction(forward_return["prediction"], bundle.scaler).cpu().numpy()
            if "base_prediction" in forward_return:
                base_prediction = _inverse_prediction(forward_return["base_prediction"], bundle.scaler).cpu().numpy()
            else:
                base_prediction = prediction.copy()
            if "residual_prediction" in forward_return:
                residual_prediction = _inverse_residual(forward_return["residual_prediction"], bundle.scaler).cpu().numpy()
            else:
                residual_prediction = np.zeros_like(prediction)

            gathered["inputs"].append(raw_inputs.numpy())
            gathered["inputs_timestamps"].append(raw_inputs_timestamps.numpy())
            gathered["targets"].append(raw_targets.numpy())
            gathered["targets_timestamps"].append(raw_targets_timestamps.numpy())
            gathered["prediction"].append(prediction)
            gathered["base_prediction"].append(base_prediction)
            gathered["residual_prediction"].append(residual_prediction)

    return {key: np.concatenate(value, axis=0) for key, value in gathered.items()}


def _fail_fast_judgment(horizon_signal: bool, temporal_signal: bool, dynamic_signal: bool, graph_signal: bool, frequency_signal: bool) -> str:
    score = sum([horizon_signal, temporal_signal, dynamic_signal, graph_signal, frequency_signal])
    if score >= 3:
        return "A. residual clearly structured"
    if score >= 2:
        return "B. residual weakly structured"
    return "C. residual mostly noise after STAEformer"


def run_residual_diagnostics(
    *,
    dataset_name: str,
    data_file_path: str,
    baseline_ckpt: str,
    output_dir: str,
    naive_ckpt: str | None = None,
    refiner_ckpt: str | None = None,
    device: str = "cpu",
    max_batches: int | None = None,
    inject_graph_prior: bool = False,
) -> dict[str, Any]:
    out_dir = _ensure_dir(output_dir)
    device_obj = torch.device(device)
    bundles: dict[str, LoadedModelBundle] = {
        "baseline": load_model_bundle(baseline_ckpt, dataset_name, data_file_path, device_obj),
    }
    if naive_ckpt:
        bundles["naive"] = load_model_bundle(naive_ckpt, dataset_name, data_file_path, device_obj)
    if refiner_ckpt:
        bundles["refiner"] = load_model_bundle(refiner_ckpt, dataset_name, data_file_path, device_obj)

    outputs = {
        name: _collect_outputs(
            bundle,
            dataset_name=dataset_name,
            data_file_path=data_file_path,
            device=device_obj,
            max_batches=max_batches,
            inject_graph_prior=inject_graph_prior,
        )
        for name, bundle in bundles.items()
    }

    baseline = outputs["baseline"]
    baseline_residual = baseline["targets"] - baseline["base_prediction"]
    baseline_error = baseline["targets"] - baseline["prediction"]

    overall_stats = _overall_residual_stats(baseline_residual)
    horizon_stats = _horizon_stats(baseline_residual)
    horizon_mae = np.array([item["mae"] for item in horizon_stats])
    hardest_horizons = (np.argsort(-horizon_mae)[:3] + 1).tolist()
    is_monotonic = bool(np.all(np.diff(horizon_mae) >= -1e-6))

    tod_index, dow_index = _timestamps_to_indices(baseline["targets_timestamps"])
    peak_labels = _time_bucket_labels(tod_index[:, 0])
    weekday_labels = np.where(np.isin(dow_index[:, 0], [5, 6]), "weekend", "weekday")
    peak_summary = _bucket_error_summary(baseline_error, peak_labels)
    weekday_summary = _bucket_error_summary(baseline_error, weekday_labels)
    heatmap = _time_of_day_heatmap(baseline_residual, baseline["targets_timestamps"])

    dynamic_features = _dynamic_features(baseline["inputs"])
    dynamic_corr = {name: _safe_corr(values, np.mean(np.abs(baseline_error), axis=(1, 2))) for name, values in dynamic_features.items()}
    change_labels = _three_way_bucket(dynamic_features["change_intensity"])
    dynamic_bucket_summary = _bucket_error_summary(baseline_error, change_labels)

    graph_stats = _graph_stats(baseline_residual, load_adjacency_matrix(data_file_path))
    frequency_stats, high_freq_ratio = _frequency_stats(baseline_residual)

    model_metrics = {name: _model_metrics(out["prediction"], out["targets"]) for name, out in outputs.items()}
    model_errors = {name: out["targets"] - out["prediction"] for name, out in outputs.items()}
    temporal_model_comparison = _bucket_compare({name: out["prediction"] for name, out in outputs.items()}, baseline["targets"], peak_labels)
    dynamic_model_comparison = _bucket_compare({name: out["prediction"] for name, out in outputs.items()}, baseline["targets"], change_labels)

    refiner_local_wins = {}
    if "refiner" in outputs:
        ref_h_mae = np.mean(np.abs(outputs["refiner"]["targets"] - outputs["refiner"]["prediction"]), axis=(0, 2))
        base_h_mae = np.mean(np.abs(outputs["baseline"]["targets"] - outputs["baseline"]["prediction"]), axis=(0, 2))
        refiner_local_wins["better_than_baseline_horizons"] = int(np.sum(ref_h_mae < base_h_mae))
        if "naive" in outputs:
            naive_h_mae = np.mean(np.abs(outputs["naive"]["targets"] - outputs["naive"]["prediction"]), axis=(0, 2))
            refiner_local_wins["better_than_naive_horizons"] = int(np.sum(ref_h_mae < naive_h_mae))

    horizon_signal = horizon_mae[-1] > horizon_mae[0] * 1.20
    temporal_signal = max(v["mae"] for v in peak_summary.values()) > min(v["mae"] for v in peak_summary.values()) * 1.10
    dynamic_signal = max(v["mae"] for v in dynamic_bucket_summary.values()) > min(v["mae"] for v in dynamic_bucket_summary.values()) * 1.10
    graph_signal = graph_stats["neighbor_sync_corr"] > graph_stats["non_neighbor_sync_corr"] * 1.05
    frequency_signal = frequency_stats["high_frequency_ratio_mean"] >= 0.50

    weak_correlation = "supported" if sum([horizon_signal, temporal_signal, dynamic_signal, graph_signal, frequency_signal]) <= 1 else "partially supported"
    coarse_expression = "supported" if ("refiner" in outputs and refiner_local_wins.get("better_than_baseline_horizons", 0) == 0 and sum([temporal_signal, dynamic_signal, graph_signal]) >= 2) else "not primary"
    backbone_absorption = "supported" if model_metrics["baseline"]["mae"] <= min(model_metrics.get("naive", model_metrics["baseline"])["mae"], model_metrics.get("refiner", model_metrics["baseline"])["mae"]) else "partially supported"
    capacity_mismatch = "secondary hypothesis" if coarse_expression == "supported" and weak_correlation != "supported" else "not yet supported"

    next_step = "继续 refinement，但要重构条件与目标" if coarse_expression == "supported" else ("graph branch 暂不应推进" if not graph_signal else "residual 主线整体需要止损")
    summary = {
        "dataset": dataset_name,
        "overall_residual_stats": overall_stats,
        "horizon_stats": horizon_stats,
        "overall_model_metrics": model_metrics,
        "temporal_model_comparison": temporal_model_comparison,
        "dynamic_model_comparison": dynamic_model_comparison,
        "refiner_local_wins": refiner_local_wins,
        "horizon": {"conclusion": "Residual grows with horizon." if horizon_signal else "Residual does not show a strong monotonic horizon trend.", "hardest_horizons": hardest_horizons, "is_monotonic": is_monotonic},
        "temporal": {"conclusion": "Temporal context matters." if temporal_signal else "Temporal context signal is limited.", "peak_buckets": peak_summary, "weekday_weekend": weekday_summary},
        "dynamic": {"conclusion": "Dynamic change intensity is strongly associated with residual." if dynamic_signal else "Dynamic signal is weak or only mildly associated with residual.", "correlations": dynamic_corr, "bucket_summary": dynamic_bucket_summary},
        "graph": {"conclusion": "Residual has graph-aware structure." if graph_signal else "Graph-aware residual structure is weak under current evidence.", "stats": graph_stats},
        "frequency": {"conclusion": "Residual is high-frequency dominated." if frequency_signal else "Residual is not strongly high-frequency dominated.", "stats": frequency_stats},
        "why_refiner_failed": {"weak_correlation": weak_correlation, "coarse_expression": coarse_expression, "backbone_absorption": backbone_absorption, "capacity_mismatch": capacity_mismatch},
        "next_step": next_step,
        "judgment": _fail_fast_judgment(horizon_signal, temporal_signal, dynamic_signal, graph_signal, frequency_signal),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "overall_metrics.json").write_text(json.dumps(model_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_histogram(np.abs(baseline_residual), out_dir / "residual_overall_histogram.png", "Residual Overall Histogram", "|residual|")
    _plot_horizon_mae(model_errors, out_dir / "residual_mae_by_horizon.png")
    _plot_heatmap(heatmap, out_dir / "time_of_day_residual_heatmap.png")
    _plot_bucket_summary(dynamic_bucket_summary, out_dir / "dynamic_bucket_comparison.png")
    _plot_graph_corr(graph_stats, out_dir / "neighbor_vs_nonneighbor_correlation.png")
    _plot_histogram(high_freq_ratio, out_dir / "residual_high_frequency_ratio.png", "Residual High-Frequency Ratio", "high-frequency energy ratio")
    _write_report(out_dir, dataset_name, bundles, summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run residual diagnostics for STAEformer / DiffTraffic checkpoints.")
    parser.add_argument("--dataset-name", required=True, choices=["METR-LA", "PEMS-BAY"])
    parser.add_argument("--data-file-path", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--naive-ckpt", default=None)
    parser.add_argument("--refiner-ckpt", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--inject-graph-prior", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run_residual_diagnostics(
        dataset_name=args.dataset_name,
        data_file_path=args.data_file_path,
        baseline_ckpt=args.baseline_ckpt,
        naive_ckpt=args.naive_ckpt,
        refiner_ckpt=args.refiner_ckpt,
        output_dir=args.output_dir,
        device=args.device,
        max_batches=args.max_batches,
        inject_graph_prior=args.inject_graph_prior,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
