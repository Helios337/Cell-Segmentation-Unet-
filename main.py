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
    data_loader = BBBCDataLoader()
    # Ensure this matches the arguments in your corrected data_handler.py
    X, y = data_loader.load_synthetic_data(n_samples=200)

    # Ensure y has the channel dimension if it's missing (N, H, W) -> (N, H, W, 1)
    if len(y.shape) == 3:
        y = np.expand_dims(y, axis=-1)

    # Split data into Train, Validation, and Test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    print("\nStarting model training...")
    # Note: Ensure the variable names in model.py match 'train_model' arguments
    history = cell_tool.train_model(
        X_train, y_train,
        X_val, y_val,
        epochs=25,
        batch_size=8
    )

    print("\nEvaluating model performance...")
    # Ensure 'evaluate_model' returns a dictionary with these keys
    metrics = cell_tool.evaluate_model(X_test, y_test)
    # Using .get() is safer in case keys differ slightly
    loss = metrics.get('loss', 0)
    dice = metrics.get('dice_coefficient', metrics.get('dice_coef', 0))
    print(f"Test Results -> Loss: {loss:.4f} | Dice Coef: {dice:.4f}")

    print("\nDemonstrating prediction and cell counting on a test image...")
    # Pick a random image from the test set
    test_idx = np.random.randint(0, len(X_test))
    test_image = X_test[test_idx]
    # Ground truth mask (remove channel dim for visualization if needed)
    test_mask = y_test[test_idx]
    if test_mask.shape[-1] == 1:
        test_mask = test_mask[:, :, 0]
    
    # Predict
    raw_prediction = cell_tool.predict_segmentation(test_image)
    
    # Post-process (Watershed)
    final_cell_count, final_labeled_mask = cell_tool.post_process_and_count(raw_prediction)
    
    # Visualize
    cell_tool.visualize_results(test_image, test_mask, raw_prediction, final_labeled_mask, final_cell_count)
    print(f"Predicted cell count for the sample image: {final_cell_count}")

    print("\nPlotting training history...")
    plt.figure(figsize=(14, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('loss', []), label='Training Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Dice Coefficient
    plt.subplot(1, 2, 2)
    # Check if the key is 'dice_coef' or 'dice_coefficient' in history
    dice_key = 'dice_coef' if 'dice_coef' in history.history else 'dice_coefficient'
    plt.plot(history.history.get(dice_key, []), label='Training Dice Coef')
    plt.plot(history.history.get(f'val_{dice_key}', []), label='Validation Dice Coef')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
