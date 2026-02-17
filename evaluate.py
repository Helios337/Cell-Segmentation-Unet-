import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tqdm import tqdm
from src.utils import count_cells_watershed, calculate_iou

# Set Plot Style
plt.style.use('seaborn-v0_8-whitegrid')

def main():
    print("Loading Model and Test Data...")
    try:
        model = keras.models.load_model('best_model.keras')
        X_test = np.load("X_test.npy")
        y_test = np.load("y_test.npy")
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Did you run train.py first?")
        return

    print("Generating Predictions...")
    preds = model.predict(X_test, verbose=1)

    gt_counts, pred_counts, ious = [], [], []

    print("Calculating Metrics...")
    for i in tqdm(range(len(X_test))):
        pc, _ = count_cells_watershed(preds[i])
        gc, _ = count_cells_watershed(y_test[i])
        iou = calculate_iou(y_test[i], preds[i])
        
        gt_counts.append(gc)
        pred_counts.append(pc)
        ious.append(iou)

    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)
    ious = np.array(ious)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 2)

    # 1. IoU Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(ious, bins=20, kde=True, color='purple', ax=ax1)
    ax1.set_title(f"Segmentation Quality (Mean IoU: {np.mean(ious):.3f})")
    ax1.set_xlabel("IoU Score")

    # 2. Bland-Altman Plot
    ax2 = fig.add_subplot(gs[0, 1])
    means = (gt_counts + pred_counts) / 2
    diffs = pred_counts - gt_counts
    mean_diff = np.mean(diffs)
    std = np.std(diffs)
    
    ax2.scatter(means, diffs, alpha=0.6)
    ax2.axhline(mean_diff, color='red', label=f'Bias: {mean_diff:.2f}')
    ax2.axhline(mean_diff + 1.96*std, color='gray', linestyle='--')
    ax2.axhline(mean_diff - 1.96*std, color='gray', linestyle='--')
    ax2.set_title("Bland-Altman (Counting Agreement)")
    ax2.set_ylabel("Diff (Pred - GT)")
    ax2.set_xlabel("Mean Count")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("evaluation_report.png")
    print("Evaluation complete. Report saved as 'evaluation_report.png'")

if __name__ == "__main__":
    main()
