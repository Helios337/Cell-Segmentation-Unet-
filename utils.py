import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_bbbc_images(image_dir, annotation_dir=None, img_size=(256, 256)):
    """Processes real BBBC images and annotations with correct interpolation."""
    valid_exts = ('.jpg', '.png', '.tif', '.tiff')
    image_files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(valid_exts)]
    
    processed_images = []
    processed_masks = []
    
    for img_file in tqdm(image_files, desc="Processing BBBC images"):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            continue
            
        # Handle grayscale vs RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        processed_images.append(img)
        
        if annotation_dir:
            mask_path = os.path.join(annotation_dir, img_file)
            # Some datasets have different extensions for masks, handle if needed
            if not os.path.exists(mask_path):
                 # Try replacing extension if mask name differs only by ext
                 pre, _ = os.path.splitext(img_file)
                 mask_path = os.path.join(annotation_dir, pre + ".png")

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # CRITICAL: Use Nearest Neighbor for masks to avoid interpolating 0 and 1 into 0.5
                mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.float32)
                processed_masks.append(mask)
            else:
                processed_masks.append(np.zeros(img_size, dtype=np.float32))
                
    if annotation_dir:
        return np.array(processed_images), np.expand_dims(np.array(processed_masks), axis=-1)
    return np.array(processed_images)

def save_results_to_csv(image_names, cell_counts, output_path="cell_counts.csv"):
    results_df = pd.DataFrame({
        'image_name': image_names,
        'predicted_cell_count': cell_counts,
        'timestamp': pd.Timestamp.now()
    })
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
