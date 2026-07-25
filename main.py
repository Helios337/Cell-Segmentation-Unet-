"""
End-to-end cell segmentation demonstration using U-Net.

Generates synthetic data, trains a U-Net model, evaluates it, and
visualizes cell segmentation and counting results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import CellSegmentationTool
from data_handler import BBBCDataLoader


def plot_training_history(history):
    """Plot training and validation loss + dice coefficient over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['dice_coef'], label='Training Dice')
    axes[1].plot(history.history['val_dice_coef'], label='Validation Dice')
    axes[1].set_title('Dice Coefficient')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    """Run the full cell segmentation pipeline."""
    print("=" * 60)
    print("Cell Segmentation Tool - U-Net Demo")
    print("=" * 60)

    # 1. Initialize model
    print("\n[1/5] Building U-Net model...")
    cell_tool = CellSegmentationTool(input_shape=(256, 256, 3))
    cell_tool.build_unet()
    cell_tool.compile_model()
    cell_tool.model.summary()

    # 2. Generate synthetic data
    print("\n[2/5] Generating synthetic cell images...")
    data_loader = BBBCDataLoader()
    X, y = data_loader.load_synthetic_data(n_samples=200)

    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=-1)

    # 3. Split into train/val/test
    print("\n[3/5] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # 4. Train
    print("\n[4/5] Training U-Net (25 epochs)...")
    history = cell_tool.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=25,
        batch_size=8,
    )

    # 5. Evaluate
    print("\n[5/5] Evaluating on test set...")
    metrics = cell_tool.evaluate_model(X_test, y_test)
    print(f"  Test Loss:       {metrics['loss']:.4f}")
    print(f"  Test Dice Coef:  {metrics['dice_coefficient']:.4f}")
    print(f"  Test Accuracy:   {metrics['binary_accuracy']:.4f}")

    # 6. Visualize a random test prediction
    print("\n--- Sample Prediction ---")
    test_idx = np.random.randint(0, len(X_test))
    test_image = X_test[test_idx]
    test_mask = y_test[test_idx, :, :, 0]

    raw_prediction = cell_tool.predict_segmentation(test_image)
    final_count, final_labels = cell_tool.post_process_and_count(raw_prediction)
    cell_tool.visualize_results(
        test_image, test_mask, raw_prediction, final_labels, final_count
    )
    print(f"Predicted cell count: {final_count}")

    # 7. Plot training curves
    print("\n--- Training History ---")
    plot_training_history(history)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()