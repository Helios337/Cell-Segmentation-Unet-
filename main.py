# main.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import CellSegmentationTool
from data_handler import BBBCDataLoader

def main():
    """Main function to demonstrate the cell segmentation tool."""
    print("Initializing Cell Segmentation Tool...")
    cell_tool = CellSegmentationTool(input_shape=(256, 256, 3))

    print("Building and compiling U-Net model...")
    model = cell_tool.build_unet()
    cell_tool.compile_model()
    print("Model Summary:")
    model.summary()

    print("\nLoading training data...")
    data_loader = BBBCDataLoader() [cite: 36]
    X, y = data_loader.load_synthetic_data(n_samples=200)

    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=-1)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    print("\nStarting model training...")
    history = cell_tool.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=25,
        batch_size=8
    ) [cite: 37]

    print("\nEvaluating model performance...")
    metrics = cell_tool.evaluate_model(X_test, y_test)
    print(f"Test Results -> Loss: {metrics['loss']:.4f} | Dice Coef: {metrics['dice_coefficient']:.4f}") [cite: 38]

    print("\nDemonstrating prediction and cell counting on a test image...")
    test_idx = np.random.randint(0, len(X_test))
    test_image = X_test[test_idx]
    test_mask = y_test[test_idx, :, :, 0]
    
    raw_prediction = cell_tool.predict_segmentation(test_image)
    final_cell_count, final_labeled_mask = cell_tool.post_process_and_count(raw_prediction) [cite: 39]
    cell_tool.visualize_results(test_image, test_mask, raw_prediction, final_labeled_mask, final_cell_count)
    print(f"Predicted cell count for the sample image: {final_cell_count}")

    print("\nPlotting training history...")
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['dice_coef'], label='Training Dice Coef') [cite: 39, 40]
    plt.plot(history.history['val_dice_coef'], label='Validation Dice Coef') [cite: 40]
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()