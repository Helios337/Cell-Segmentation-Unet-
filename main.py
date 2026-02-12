import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import CellSegmentationTool
from data_handler import BBBCDataLoader
import utils

def main():
    parser = argparse.ArgumentParser(description="Cell Segmentation Pipeline")
    parser.add_argument('--mode', type=str, default='synthetic', choices=['synthetic', 'real'], 
                        help='Choose data mode: "synthetic" or "real" (BBBC dataset)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    args = parser.parse_args()

    print(f"Initializing Cell Segmentation Tool in [{args.mode.upper()}] mode...")
    
    # 1. Prepare Data
    data_loader = BBBCDataLoader()
    
    if args.mode == 'synthetic':
        X, y = data_loader.load_synthetic_data(n_samples=500)
        # Add channel dim to masks if needed
        if len(y.shape) == 3: y = np.expand_dims(y, axis=-1)
    else:
        # Real Data Logic
        save_dir = "./data"
        extract_path = data_loader.download_dataset(save_dir)
        # Note: BBBC005 usually has a specific folder structure. Adjust 'image_dir' as needed based on extraction.
        # For this demo, we assume images are in the extracted root.
        image_dir = os.path.join(extract_path, "BBBC005_v1_images")
        
        # NOTE: BBBC005 synthetic dataset often doesn't come with direct "mask" files in the same zip 
        # in the same format. For this code to run "real" mode fully, you need ground truth masks.
        # If masks are missing, we default to synthetic to prevent crash, or you must provide annotation_dir.
        if os.path.exists(image_dir):
            # Try to load images. If no masks, we can't train "real" without them.
            # This block assumes you have masks in a 'masks' subdir or similar.
            # For demonstration, we will revert to synthetic if no masks found, or user needs to supply path.
            print("Real data directory found. (Requires corresponding masks for training).")
            # Placeholder for actual mask loading logic if you have the ground truth separate
            print("Warning: Standard BBBC005 zip contains images. Ground truth is often separate.")
            print("Reverting to Synthetic for training demonstration to ensure execution.")
            X, y = data_loader.load_synthetic_data(n_samples=500)
            if len(y.shape) == 3: y = np.expand_dims(y, axis=-1)
        else:
            print("Data path not found. Generating synthetic...")
            X, y = data_loader.load_synthetic_data(n_samples=500)
            if len(y.shape) == 3: y = np.expand_dims(y, axis=-1)

    # 2. Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 3. Data Augmentation (Scientific Best Practice)
    # We only augment training data, not validation/test
    data_gen_args = dict(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, 
                         zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the flow methods
    seed = 1
    image_generator = image_datagen.flow(X_train, batch_size=8, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=8, seed=seed)
    train_generator = zip(image_generator, mask_generator)

    # 4. Model Setup
    cell_tool = CellSegmentationTool(input_shape=(256, 256, 3))
    model = cell_tool.build_unet()
    cell_tool.compile_model()
    
    # 5. Training
    print("Starting training with Data Augmentation...")
    history = cell_tool.train_model(
        train_gen=train_generator,
        val_data=(X_val, y_val),
        epochs=args.epochs,
        steps_per_epoch=len(X_train) // 8
    )

    # 6. Evaluation
    print("\nEvaluating...")
    metrics = cell_tool.evaluate_model(X_test, y_test)
    print(f"Test Metrics: {metrics}")

    # 7. Visualization
    test_idx = np.random.randint(0, len(X_test))
    test_img = X_test[test_idx]
    test_mask = y_test[test_idx][:,:,0]
    
    pred = cell_tool.predict_segmentation(test_img)
    count, labeled_mask = cell_tool.post_process_and_count(pred)
    
    cell_tool.visualize_results(test_img, test_mask, pred, labeled_mask, count)

    # Plot History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['iou_coef'], label='Train IoU')
    plt.plot(history.history['val_iou_coef'], label='Val IoU')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
