"""
End-to-end cell segmentation pipeline using U-Net with ResNet50 encoder.

Trains on real fluorescence microscopy data (BBBC038/BBBC039),
evaluates with comprehensive metrics, and visualizes results.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import CellSegmentationTool
from data_handler import BBBCDataLoader
from utils import augment_image, prepare_dataset


def plot_training_history(history):
    """Plot training and validation loss + dice coefficient over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].plot(history.history["loss"], label="Training Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history.history["dice_coef"], label="Training Dice")
    axes[1].plot(history.history["val_dice_coef"], label="Validation Dice")
    axes[1].set_title("Dice Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].legend()

    axes[2].plot(history.history["iou_metric"], label="Training IoU")
    axes[2].plot(history.history["val_iou_metric"], label="Validation IoU")
    axes[2].set_title("IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


def load_data(args):
    """Load training data based on the specified source."""
    data_loader = BBBCDataLoader(dataset_name=args.data_source)

    if args.data_source == "BBBC038":
        data_loader.download_dataset(save_dir="./data")
        image_dir = "./data"
        X = data_loader.load_real_data(image_dir=image_dir, img_size=tuple(args.img_size))
        y = np.zeros((len(X), args.img_size[0], args.img_size[1], 1), dtype=np.float32)
        print(f"Loaded {len(X)} images from BBBC038")
        return X, y
    elif args.data_source == "BBBC039":
        data_loader.download_dataset(save_dir="./data")
        image_dir = os.path.join("./data", "images")
        mask_dir = os.path.join("./data", "masks")
        X, y = data_loader.load_real_data(
            image_dir=image_dir, annotation_dir=mask_dir, img_size=tuple(args.img_size)
        )
        print(f"Loaded {len(X)} images from BBBC039")
        return X, y
    else:
        X, y = data_loader.load_synthetic_data(n_samples=args.n_samples)
        if len(y.shape) == 3:
            y = np.expand_dims(y, axis=-1)
        return X, y


def run_training(args):
    """Run the full training pipeline."""
    print("=" * 60)
    print("Cell Segmentation Tool - U-Net Training")
    print("=" * 60)

    print("\n[1/6] Building U-Net model with ResNet50 encoder ...")
    cell_tool = CellSegmentationTool(
        input_shape=tuple(args.img_size) + (3,),
    )
    cell_tool.build_unet()
    cell_tool.compile_model(learning_rate=args.lr_phase1)
    cell_tool.model.summary()

    print("\n[2/6] Loading data ...")
    X, y = load_data(args)

    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=-1)

    print(f"  Total samples: {len(X)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1.0 - args.train_split, random_state=42
    )
    val_ratio = args.val_split / (args.val_split + args.test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_ratio, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    if args.augment:
        print("  Augmentation: enabled")
        X_train_aug = []
        y_train_aug = []
        for i in range(len(X_train)):
            img_aug, mask_aug = augment_image(X_train[i], y_train[i, :, :, 0])
            X_train_aug.append(img_aug)
            y_train_aug.append(mask_aug)
        X_train = np.array(X_train_aug)
        y_train = np.expand_dims(np.array(y_train_aug), axis=-1)
        print(f"  Augmented training samples: {len(X_train)}")

    total_epochs = args.epochs_phase1 + args.epochs_phase2
    print(f"\n[3/6] Training U-Net ({total_epochs} epochs total, "
          f"{args.epochs_phase1} frozen + {args.epochs_phase2} fine-tune) ...")
    history = cell_tool.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=total_epochs,
        batch_size=args.batch_size,
        freeze_encoder_epochs=args.epochs_phase1,
    )

    print("\n[4/6] Evaluating on test set ...")
    metrics = cell_tool.evaluate_model(X_test, y_test, threshold=args.threshold)
    print(f"  Test Loss:       {metrics['loss']:.4f}")
    print(f"  Test Dice Coef:  {metrics['dice_coefficient']:.4f}")
    print(f"  Test IoU:        {metrics['iou']:.4f}")
    print(f"  Test Accuracy:   {metrics['binary_accuracy']:.4f}")
    print(f"  Avg Count MAE:   {metrics['average']['count_mae']:.2f}")
    print(f"  Avg Precision:   {metrics['average']['precision']:.4f}")
    print(f"  Avg Recall:      {metrics['average']['recall']:.4f}")
    print(f"  Avg F1:          {metrics['average']['f1']:.4f}")

    print("\n[5/6] Post-processing optimization ...")
    if args.optimize_thresholds:
        test_idx = np.random.randint(0, len(X_test))
        test_image = X_test[test_idx]
        test_mask = y_test[test_idx, :, :, 0]
        raw_pred = cell_tool.predict_segmentation(test_image)
        best_thresh, best_min_size, best_count, best_iou = \
            cell_tool.optimize_post_processing(raw_pred, test_mask)
        print(f"  Best threshold: {best_thresh}")
        print(f"  Best min_size:  {best_min_size}")
        print(f"  Best IoU:       {best_iou:.4f}")
        print(f"  Best count:     {best_count}")

    print("\n[6/6] Visualizing results ...")
    test_idx = np.random.randint(0, len(X_test))
    test_image = X_test[test_idx]
    test_mask = y_test[test_idx, :, :, 0]

    if args.tta:
        print("  Using test-time augmentation ...")
        raw_prediction = cell_tool.predict_segmentation_tta(test_image)
    else:
        raw_prediction = cell_tool.predict_segmentation(test_image)

    final_count, final_labels = cell_tool.post_process_and_count(
        raw_prediction, threshold=args.threshold, min_size=args.min_size
    )
    cell_tool.visualize_results(
        test_image, test_mask, raw_prediction, final_labels, final_count
    )
    print(f"  Predicted cell count: {final_count}")

    print("\n--- Training History ---")
    plot_training_history(history)

    if args.save_to_drive:
        cell_tool.save_checkpoint_to_drive(
            filepath="best_model.keras",
            drive_path=args.drive_path,
        )

    print("\nPipeline complete!")
    return cell_tool, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Cell Segmentation U-Net Training and Evaluation"
    )
    parser.add_argument("--mode", choices=["train", "eval", "predict"],
                        default="train", help="Run mode")
    parser.add_argument("--data-source", type=str, default="BBBC038",
                        help="Dataset source (BBBC038, BBBC039, synthetic)")
    parser.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                        help="Input image size (height width)")
    parser.add_argument("--epochs-phase1", type=int, default=10,
                        help="Epochs with frozen encoder")
    parser.add_argument("--epochs-phase2", type=int, default=20,
                        help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr-phase1", type=float, default=1e-3,
                        help="Learning rate for phase 1")
    parser.add_argument("--lr-phase2", type=float, default=1e-4,
                        help="Learning rate for phase 2")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable on-the-fly augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable augmentation")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Post-processing threshold")
    parser.add_argument("--min-size", type=int, default=50,
                        help="Minimum cell size for post-processing")
    parser.add_argument("--optimize-thresholds", action="store_true", default=True,
                        help="Optimize threshold and min_size via grid search")
    parser.add_argument("--no-optimize", dest="optimize_thresholds",
                        action="store_false",
                        help="Skip threshold optimization")
    parser.add_argument("--tta", action="store_true", default=True,
                        help="Use test-time augmentation")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="Disable test-time augmentation")
    parser.add_argument("--save-to-drive", action="store_true", default=False,
                        help="Save model to Google Drive")
    parser.add_argument("--drive-path", type=str,
                        default="/content/drive/MyDrive/cell_segmentation/best_model.keras",
                        help="Google Drive path for model save")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of synthetic samples (for demo mode)")
    parser.add_argument("--train-split", type=float, default=0.7,
                        help="Train split ratio")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Test split ratio")

    args = parser.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()