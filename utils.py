# utils.py
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_bbbc_images(image_dir, annotation_dir=None, img_size=(256, 256)):
    """Processes real BBBC images and (optionally) annotations."""
    image_files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('.jpg', '.png', '.tif', '.tiff'))]
    processed_images = []
    processed_masks = []
    for img_file in tqdm(image_files, desc="Processing BBBC images"):
        img_path = os.path.join(image_dir, img_file) [cite: 41]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) [cite: 42]
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        processed_images.append(img)
        if annotation_dir:
            mask_path = os.path.join(annotation_dir, img_file) [cite: 43]
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, img_size)
                mask = (mask > 0).astype(np.float32)
                processed_masks.append(mask)
            else:
                processed_masks.append(np.zeros(img_size, dtype=np.float32)) [cite: 44]
    if annotation_dir:
        return np.array(processed_images), np.expand_dims(np.array(processed_masks), axis=-1)
    return np.array(processed_images)

def save_results_to_csv(image_names, cell_counts, output_path="cell_counts.csv"):
    """Saves cell counting results to a CSV file."""
    results_df = pd.DataFrame({
        'image_name': image_names,
        'predicted_cell_count': cell_counts,
        'timestamp': pd.Timestamp.now()
    })
    results_df.to_csv(output_path, index=False) [cite: 45]
    print(f"\nResults saved to {output_path}")