import argparse
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import mixed_precision

from data_loader import RealBiologicalLoader
from model import build_deep_unet
from performance import StageTimer, hardware_software_info, save_json, safe_model_size_mb


@dataclass
class TrainConfig:
    img_size: int = 128
    batch_size: int = 16
    epochs: int = 20
    seed: int = 42
    model_variant: str = "baseline"
    mixed_precision_enabled: bool = False
    max_samples: Optional[int] = None
    iou_target: float = 0.80
    output_dir: str = "."


VARIANTS: Dict[str, Dict] = {
    "baseline": {
        "base_filters": 32,
        "depth": 4,
        "dropout": 0.1,
        "bottleneck_dropout": 0.3,
        "use_separable": False,
    },
    "light": {
        "base_filters": 24,
        "depth": 4,
        "dropout": 0.05,
        "bottleneck_dropout": 0.2,
        "use_separable": True,
    },
    "tiny": {
        "base_filters": 16,
        "depth": 3,
        "dropout": 0.05,
        "bottleneck_dropout": 0.15,
        "use_separable": True,
    },
}


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self._epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._epoch_start)


def set_global_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_augmentation(seed: int) -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomRotation(0.25, fill_mode="reflect", seed=seed),
            keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
            keras.layers.RandomZoom(0.2, 0.2, fill_mode="reflect", seed=seed),
            keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        ],
        name="augmentation",
    )


def create_dataset_pipeline(X, y, batch_size, seed, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=seed, reshuffle_each_iteration=True)

        aug = build_augmentation(seed)

        def _augment(img, mask):
            concat = tf.concat([img, mask], axis=-1)
            concat = aug(concat, training=True)
            return concat[..., :3], concat[..., 3:4]

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train cell segmentation model with profiling")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-variant", choices=list(VARIANTS.keys()), default="baseline")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--iou-target", type=float, default=0.80)
    parser.add_argument("--output-dir", type=str, default=".")
    return parser.parse_args()


def _memory_metrics_mb() -> Dict[str, Optional[float]]:
    metrics = {"gpu_memory_mb": None, "cpu_memory_mb": None}
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        metrics["gpu_memory_mb"] = info.get("peak", 0) / (1024 * 1024)
    except Exception:
        metrics["gpu_memory_mb"] = None
    return metrics


def train(config: TrainConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    set_global_determinism(config.seed)

    if config.mixed_precision_enabled:
        mixed_precision.set_global_policy("mixed_float16")

    profile: Dict[str, float] = {}

    with StageTimer(profile, "dataset_load_total"):
        loader = RealBiologicalLoader()
        X, y = loader.load_dataset(
            img_size=(config.img_size, config.img_size),
            max_samples=config.max_samples,
            profile=profile,
        )

    if X is None:
        raise RuntimeError("Dataset loading failed")

    with StageTimer(profile, "dataset_split"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=config.seed
        )

    np.save(os.path.join(config.output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(config.output_dir, "y_test.npy"), y_test)
    print("✅ Test set saved for evaluation.")

    with StageTimer(profile, "pipeline_build"):
        train_ds = create_dataset_pipeline(X_train, y_train, config.batch_size, config.seed, training=True)
        val_ds = create_dataset_pipeline(X_test, y_test, config.batch_size, config.seed, training=False)

    variant_cfg = VARIANTS[config.model_variant]
    with StageTimer(profile, "model_build_compile"):
        model = build_deep_unet(
            input_shape=(config.img_size, config.img_size, 3),
            **variant_cfg,
        )

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor="val_loss",
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(config.output_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=1),
    ]

    time_history = TimeHistory()
    callbacks.append(time_history)

    print("🚀 Starting Training...")
    with StageTimer(profile, "train_fit"):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=1,
        )

    with StageTimer(profile, "model_save"):
        model.save(os.path.join(config.output_dir, "best_model.keras"))

    mem = _memory_metrics_mb()
    model_size = safe_model_size_mb(os.path.join(config.output_dir, "best_model.keras"))

    val_loss = history.history.get("val_loss", [])
    val_iou = history.history.get("val_iou", [])

    best_epoch_idx = int(np.argmin(val_loss)) if val_loss else 0
    time_to_best_val_loss = (
        float(sum(time_history.epoch_times[: best_epoch_idx + 1])) if time_history.epoch_times else 0.0
    )

    time_to_target_iou = None
    for idx, metric in enumerate(val_iou):
        if metric >= config.iou_target:
            time_to_target_iou = float(sum(time_history.epoch_times[: idx + 1]))
            break

    summary = {
        "config": asdict(config),
        "hardware_software": hardware_software_info(),
        "dataset": {
            "total_samples": int(len(X)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "split": "85/15",
        },
        "training": {
            "epochs_completed": int(len(history.history.get("loss", []))),
            "train_time_per_epoch": float(np.mean(time_history.epoch_times)) if time_history.epoch_times else 0.0,
            "total_train_time": float(sum(time_history.epoch_times)) if time_history.epoch_times else 0.0,
            "time_to_best_val_loss": time_to_best_val_loss,
            "time_to_target_iou": time_to_target_iou,
            "best_val_loss": float(np.min(val_loss)) if val_loss else None,
            "best_val_iou": float(np.max(val_iou)) if val_iou else None,
            "binary_iou": float(val_iou[-1]) if val_iou else None,
            "gpu_memory_mb": mem["gpu_memory_mb"],
            "cpu_memory_mb": mem["cpu_memory_mb"],
            "model_size_mb": model_size,
        },
        "profiling": profile,
    }

    save_json(os.path.join(config.output_dir, "training_summary.json"), summary)
    print("✅ Training Complete. Model saved as 'best_model.keras'")
    return summary


def main():
    args = parse_args()
    config = TrainConfig(
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        model_variant=args.model_variant,
        mixed_precision_enabled=args.mixed_precision,
        max_samples=args.max_samples,
        iou_target=args.iou_target,
        output_dir=args.output_dir,
    )
    train(config)


if __name__ == "__main__":
    main()
