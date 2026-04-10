import numpy as np
from scipy import ndimage
from skimage import feature, measure, morphology
from skimage.segmentation import watershed


"""Post-processing presets tuned for accuracy-speed tradeoffs.

- high_accuracy: lower threshold and watershed-based separation for touching nuclei.
- fast: stricter threshold + connected-components for lower latency.
"""

POSTPROCESS_PRESETS = {
    "high_accuracy": {"threshold": 0.5, "area_threshold": 20, "min_distance": 5, "use_watershed": True},
    "fast": {"threshold": 0.55, "area_threshold": 30, "min_distance": 8, "use_watershed": False},
}


def _resolve_preset(mode, threshold, min_size, min_distance):
    if mode not in POSTPROCESS_PRESETS:
        raise ValueError(f"Unknown mode '{mode}'. Valid: {list(POSTPROCESS_PRESETS.keys())}")

    preset = POSTPROCESS_PRESETS[mode].copy()
    if threshold is not None:
        preset["threshold"] = threshold
    if min_size is not None:
        preset["area_threshold"] = min_size
    if min_distance is not None:
        preset["min_distance"] = min_distance
    if preset["area_threshold"] <= 0 or preset["min_distance"] <= 0:
        raise ValueError("area_threshold and min_distance must be positive")
    return preset


def count_cells_watershed(pred_mask, threshold=0.5, min_size=20, min_distance=5, mode="high_accuracy"):
    """
    Counts cells from a prediction mask.
    Supports 'high_accuracy' (watershed) and 'fast' (connected-components) presets.
    If no local maxima are detected in high_accuracy mode, no watershed markers are seeded.
    """
    if len(pred_mask.shape) == 3:
        pred_mask = pred_mask[:, :, 0]

    cfg = _resolve_preset(mode=mode, threshold=threshold, min_size=min_size, min_distance=min_distance)

    binary = (pred_mask > cfg["threshold"]).astype(np.uint8)
    clean = morphology.area_opening(binary.astype(bool), area_threshold=cfg["area_threshold"])

    if not cfg["use_watershed"]:
        labels = measure.label(clean)
        return int(labels.max()), labels

    distance = ndimage.distance_transform_edt(clean)
    coords = feature.peak_local_max(distance, min_distance=cfg["min_distance"], labels=clean)
    mask = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        mask[tuple(coords.T)] = True
    markers = measure.label(mask)

    labels = watershed(-distance, markers, mask=clean)
    return len(np.unique(labels)) - 1, labels


def calculate_iou(y_true, y_pred, threshold=0.5):
    """Calculates Intersection over Union."""
    y_pred_bin = y_pred > threshold
    y_true_bin = y_true > threshold

    intersection = np.sum(y_true_bin * y_pred_bin)
    union = np.sum(y_true_bin) + np.sum(y_pred_bin) - intersection
    return (intersection + 1e-6) / (union + 1e-6)
