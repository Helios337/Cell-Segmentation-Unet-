import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from performance import AcceptanceCriteria, aggregate_trials, evaluate_acceptance, hardware_software_info, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproducible baseline/candidate benchmark trials")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-variant", choices=["baseline", "light", "tiny"], default="baseline")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--postprocess-mode", choices=["fast", "high_accuracy"], default="high_accuracy")
    parser.add_argument("--output-dir", type=str, default="benchmark_runs")
    parser.add_argument("--baseline-report", type=str, default=None)
    parser.add_argument("--report-name", type=str, default="benchmark_report.json")
    return parser.parse_args()


def run_command(cmd: List[str], cwd: str) -> None:
    try:
        completed = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command failed ({e.returncode}): {' '.join(cmd)}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e
    if completed.stdout:
        print(completed.stdout)


def _read_json(path: Path) -> Dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_trial_metrics(training: Dict[str, Any], evaluation: Dict[str, Any], trial_id: str) -> Dict[str, Any]:
    t = training.get("training", {})
    e = evaluation.get("metrics", {})
    profiling = {}
    profiling.update(training.get("profiling", {}))
    profiling.update(evaluation.get("profiling", {}))

    return {
        "trial_id": trial_id,
        "train_time_per_epoch": t.get("train_time_per_epoch", 0.0),
        "total_train_time": t.get("total_train_time", 0.0),
        "time_to_best_val_loss": t.get("time_to_best_val_loss", 0.0),
        "time_to_target_iou": t.get("time_to_target_iou", None),
        "gpu_memory_mb": t.get("gpu_memory_mb", None),
        "cpu_memory_mb": t.get("cpu_memory_mb", None),
        "inference_latency_ms": e.get("inference_latency_ms", 0.0),
        "throughput_images_per_sec": e.get("throughput_images_per_sec", 0.0),
        "model_size_mb": t.get("model_size_mb", 0.0),
        "mean_iou": e.get("mean_iou", 0.0),
        "binary_iou": t.get("binary_iou", 0.0),
        "counting_bias": e.get("counting_bias", 0.0),
        "count_agreement_spread": e.get("count_agreement_spread", 0.0),
        "bottlenecks": profiling,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    output_root = (repo_root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_trials = []

    for i in range(args.trials):
        trial_id = f"trial_{i + 1:02d}"
        trial_dir = output_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "train.py",
            "--img-size",
            str(args.img_size),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed + i),
            "--model-variant",
            args.model_variant,
            "--output-dir",
            str(trial_dir),
        ]
        if args.max_samples is not None:
            train_cmd += ["--max-samples", str(args.max_samples)]
        if args.mixed_precision:
            train_cmd += ["--mixed-precision"]

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--output-dir",
            str(trial_dir),
            "--postprocess-mode",
            args.postprocess_mode,
        ]

        print(f"▶ Running {trial_id}: training")
        run_command(train_cmd, cwd=str(repo_root))
        print(f"▶ Running {trial_id}: evaluation")
        run_command(eval_cmd, cwd=str(repo_root))

        training = _read_json(trial_dir / "training_summary.json")
        evaluation = _read_json(trial_dir / "evaluation_metrics.json")
        all_trials.append(_collect_trial_metrics(training, evaluation, trial_id=trial_id))

    aggregate = aggregate_trials(all_trials)

    report: Dict[str, Any] = {
        "frozen_config": {
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "seed_base": args.seed,
            "trials": args.trials,
            "model_variant": args.model_variant,
            "mixed_precision": args.mixed_precision,
            "postprocess_mode": args.postprocess_mode,
            "max_samples": args.max_samples,
        },
        "hardware_software": hardware_software_info(),
        "trials": all_trials,
        "aggregate": aggregate,
    }

    if args.baseline_report:
        baseline = _read_json(Path(args.baseline_report))
        report["acceptance"] = evaluate_acceptance(
            baseline=baseline.get("aggregate", baseline),
            candidate=aggregate,
            criteria=AcceptanceCriteria(),
        )

    save_json(str(output_root / args.report_name), report)
    print(f"✅ Benchmark report saved at {output_root / args.report_name}")


if __name__ == "__main__":
    main()
