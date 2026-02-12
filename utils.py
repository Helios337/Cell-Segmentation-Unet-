import numpy as np
from skimage import measure, morphology, feature
from skimage.segmentation import watershed
from scipy import ndimage

def count_cells_watershed(pred_mask, threshold=0.5):
    """
    Counts cells using Watershed algorithm.
    Args:
        pred_mask (np.array): Prediction mask (H, W) or (H, W, 1)
        threshold (float): Binarization threshold
    Returns:
        count (int): Number of cells
        labels (np.array): Labeled mask
    """
    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask[:, :, 0]
        
    # 1. Threshold
    binary = (pred_mask > threshold).astype(np.uint8)
    
    # 2. Noise Removal
    clean = morphology.remove_small_objects(binary.astype(bool), min_size=20)
    
    # 3. Distance Transform
    distance = ndimage.distance_transform_edt(clean)
    
    # 4. Find Peaks
    coords = feature.peak_local_max(distance, min_distance=5, labels=clean)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = measure.label(mask)
    
    # 5. Watershed
    labels = watershed(-distance, markers, mask=clean)
    
    return len(np.unique(labels)) - 1, labels

def calculate_iou(y_true, y_pred, threshold=0.5):
    """Calculates Intersection over Union."""
    y_pred_bin = (y_pred > threshold)
    y_true_bin = (y_true > threshold)
    
    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)
