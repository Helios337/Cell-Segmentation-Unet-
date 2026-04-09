import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow import keras
from tqdm import tqdm

from performance import StageTimer, save_json
from utils import calculate_iou, count_cells_watershed

# Set Plot Style
plt.style.use("seaborn-v0_8-whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model with profiling")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--postprocess-mode", choices=["fast", "high_accuracy"], default="high_accuracy")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()

    profile = {}
    print("🔄 Loading Model and Test Data...")
    try:
        with StageTimer(profile, "model_load"):
            model = keras.models.load_model(os.path.join(args.output_dir, "best_model.keras"))
        with StageTimer(profile, "test_data_load"):
            X_test = np.load(os.path.join(args.output_dir, "X_test.npy"))
            y_test = np.load(os.path.join(args.output_dir, "y_test.npy"))
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        print("Did you run train.py first?")
        return

    print("📊 Generating Predictions...")
    with StageTimer(profile, "prediction"):
        pred_t0 = time.perf_counter()
        preds = model.predict(X_test, verbose=1)
        prediction_time = time.perf_counter() - pred_t0

    gt_counts, pred_counts, ious = [], [], []

    print("🧮 Calculating Metrics...")
    post_t0 = time.perf_counter()
    for i in tqdm(range(len(X_test))):
        pc, _ = count_cells_watershed(preds[i], threshold=args.threshold, mode=args.postprocess_mode)
        gc, _ = count_cells_watershed(y_test[i], threshold=args.threshold, mode=args.postprocess_mode)
        iou = calculate_iou(y_test[i], preds[i], threshold=args.threshold)

        gt_counts.append(gc)
        pred_counts.append(pc)
        ious.append(iou)
    profile["postprocess_counting"] = time.perf_counter() - post_t0

    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)
    ious = np.array(ious)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2)

    # 1. IoU Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(ious, bins=20, kde=True, color="purple", ax=ax1)
    ax1.set_title(f"Segmentation Quality (Mean IoU: {np.mean(ious):.3f})")
    ax1.set_xlabel("IoU Score")

    # 2. Bland-Altman Plot
    ax2 = fig.add_subplot(gs[0, 1])
    means = (gt_counts + pred_counts) / 2
    diffs = pred_counts - gt_counts
    mean_diff = np.mean(diffs)
    std = np.std(diffs)

    ax2.scatter(means, diffs, alpha=0.6)
    ax2.axhline(mean_diff, color="red", label=f"Bias: {mean_diff:.2f}")
    ax2.axhline(mean_diff + 1.96 * std, color="gray", linestyle="--")
    ax2.axhline(mean_diff - 1.96 * std, color="gray", linestyle="--")
    ax2.set_title("Bland-Altman (Counting Agreement)")
    ax2.set_ylabel("Diff (Pred - GT)")
    ax2.set_xlabel("Mean Count")
    ax2.legend()

    plt.tight_layout()
    report_path = os.path.join(args.output_dir, "evaluation_report.png")
    plt.savefig(report_path)

    inference_latency_ms = (prediction_time / max(len(X_test), 1)) * 1000
    throughput = len(X_test) / max(prediction_time, 1e-8)

    evaluation_summary = {
        "postprocess_mode": args.postprocess_mode,
        "threshold": args.threshold,
        "metrics": {
            "mean_iou": float(np.mean(ious)),
            "counting_bias": float(mean_diff),
            # 3.92 = 2 * 1.96, width of 95% Bland-Altman limits of agreement.
            "count_agreement_spread": float(3.92 * std),
            "inference_latency_ms": float(inference_latency_ms),
            "throughput_images_per_sec": float(throughput),
        },
        "profiling": profile,
        "artifacts": {
            "evaluation_report": report_path,
        },
    }

    save_json(os.path.join(args.output_dir, "evaluation_metrics.json"), evaluation_summary)
    print("✅ Evaluation complete. Report saved as 'evaluation_report.png'")


if __name__ == "__main__":
    main()
