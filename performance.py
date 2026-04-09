import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf


@dataclass
class AcceptanceCriteria:
    max_iou_drop: float = 0.01
    max_accuracy_drop: float = 0.005
    max_count_bias_increase: float = 1.0
    min_speedup_train_epoch: float = 1.1
    min_speedup_inference_latency: float = 1.1


@dataclass
class TrialMetrics:
    trial_id: str
    train_time_per_epoch: float
    total_train_time: float
    time_to_best_val_loss: float
    time_to_target_iou: Optional[float]
    gpu_memory_mb: Optional[float]
    cpu_memory_mb: Optional[float]
    inference_latency_ms: float
    throughput_images_per_sec: float
    model_size_mb: float
    mean_iou: float
    binary_iou: float
    counting_bias: float
    count_agreement_spread: float
    bottlenecks: Dict[str, float] = field(default_factory=dict)


class StageTimer:
    def __init__(self, sink: Dict[str, float], key: str):
        self.sink = sink
        self.key = key
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.sink[self.key] = self.sink.get(self.key, 0.0) + (time.perf_counter() - self.start)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_model_size_mb(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024)


def hardware_software_info() -> Dict[str, Any]:
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [gpu.name for gpu in gpus]
    return {
        "timestamp_utc": timestamp_utc(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "gpu_count": len(gpus),
        "gpu_names": gpu_names,
    }


def save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def aggregate_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trials:
        return {"trial_count": 0, "means": {}, "stddev": {}}

    numeric_keys = [
        "train_time_per_epoch",
        "total_train_time",
        "time_to_best_val_loss",
        "inference_latency_ms",
        "throughput_images_per_sec",
        "model_size_mb",
        "mean_iou",
        "binary_iou",
        "counting_bias",
        "count_agreement_spread",
    ]

    means: Dict[str, float] = {}
    stddev: Dict[str, float] = {}
    for key in numeric_keys:
        vals = [float(t[key]) for t in trials if t.get(key) is not None]
        if vals:
            means[key] = float(statistics.fmean(vals))
            stddev[key] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0

    bottleneck_totals: Dict[str, float] = {}
    for t in trials:
        for k, v in t.get("bottlenecks", {}).items():
            bottleneck_totals[k] = bottleneck_totals.get(k, 0.0) + float(v)

    prioritized_bottlenecks = sorted(
        (
            {"stage": k, "avg_seconds": v / len(trials)}
            for k, v in bottleneck_totals.items()
        ),
        key=lambda x: x["avg_seconds"],
        reverse=True,
    )

    return {
        "trial_count": len(trials),
        "means": means,
        "stddev": stddev,
        "prioritized_bottlenecks": prioritized_bottlenecks,
    }


def evaluate_acceptance(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    criteria: Optional[AcceptanceCriteria] = None,
) -> Dict[str, Any]:
    criteria = criteria or AcceptanceCriteria()

    b = baseline.get("means", baseline)
    c = candidate.get("means", candidate)

    def safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 1.0
        return numerator / denominator

    iou_drop = float(b.get("mean_iou", 0.0)) - float(c.get("mean_iou", 0.0))
    acc_drop = float(b.get("binary_iou", 0.0)) - float(c.get("binary_iou", 0.0))
    bias_increase = abs(float(c.get("counting_bias", 0.0))) - abs(float(b.get("counting_bias", 0.0)))

    train_speedup = safe_ratio(float(b.get("train_time_per_epoch", 0.0)), float(c.get("train_time_per_epoch", 0.0)))
    infer_speedup = safe_ratio(float(b.get("inference_latency_ms", 0.0)), float(c.get("inference_latency_ms", 0.0)))

    checks = {
        "quality_iou_preserved": iou_drop <= criteria.max_iou_drop,
        "quality_binary_iou_preserved": acc_drop <= criteria.max_accuracy_drop,
        "counting_bias_within_tolerance": bias_increase <= criteria.max_count_bias_increase,
        "train_speed_target_met": train_speedup >= criteria.min_speedup_train_epoch,
        "inference_speed_target_met": infer_speedup >= criteria.min_speedup_inference_latency,
    }

    return {
        "criteria": asdict(criteria),
        "computed": {
            "iou_drop": iou_drop,
            "binary_iou_drop": acc_drop,
            "counting_bias_increase": bias_increase,
            "train_speedup": train_speedup,
            "inference_speedup": infer_speedup,
        },
        "checks": checks,
        "accepted": all(checks.values()),
    }
