"""
Data loader for BBBC datasets and synthetic cell image generation.
"""

import numpy as np
import cv2
import os
import zipfile
import requests
from tqdm import tqdm


BBBC_URLS = {
    "BBBC005": "https://data.broadinstitute.org/bbbc/BBBC005/BBBC005_v1_images.zip",
    "BBBC004": "https://data.broadinstitute.org/bbbc/BBBC004/BBBC004_v1_images.zip",
}


class BBBCDataLoader:
    """
    Handles loading of BBBC biomedical datasets and synthetic data generation.

    Usage::

        loader = BBBCDataLoader(dataset_name="BBBC005")
        loader.download_dataset(save_dir="./data")
        X, y = loader.load_synthetic_data(n_samples=200)
    """

    def __init__(self, dataset_name="BBBC005"):
        self.dataset_name = dataset_name

    def download_dataset(self, save_dir="./data"):
        """Download a BBBC dataset zip file from the Broad Institute."""
        os.makedirs(save_dir, exist_ok=True)

        if self.dataset_name not in BBBC_URLS:
            print(f"Unknown dataset: {self.dataset_name}. Available: {list(BBBC_URLS.keys())}")
            return

        zip_url = BBBC_URLS[self.dataset_name]
        zip_filename = os.path.basename(zip_url)
        zip_path = os.path.join(save_dir, zip_filename)

        if os.path.exists(zip_path):
            print(f"{zip_filename} already downloaded at {zip_path}.")
            return

        print(f"Downloading {self.dataset_name} from {zip_url} ...")
        try:
            response = requests.get(zip_url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=self.dataset_name) as pbar:
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            return

        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"Extraction complete. Files in: {save_dir}")

    def load_synthetic_data(self, n_samples=100, img_size=(256, 256)):
        """Generate synthetic cell-like images for demonstration."""
        print(f"Generating {n_samples} synthetic cell images...")
        X, y = [], []
        for _ in tqdm(range(n_samples), desc="Synthetic data"):
            img = np.zeros((*img_size, 3), dtype=np.uint8)
            mask = np.zeros(img_size, dtype=np.uint8)
            n_cells = np.random.randint(5, 25)
            for _ in range(n_cells):
                cx = np.random.randint(20, img_size[0] - 20)
                cy = np.random.randint(20, img_size[1] - 20)
                radius = np.random.randint(10, 25)
                brightness = np.random.randint(100, 200)
                highlight = np.random.randint(brightness + 10, 255)
                cv2.circle(img, (cx, cy), radius, (brightness,) * 3, -1)
                cv2.circle(img, (cx, cy), int(radius * 0.6), (highlight,) * 3, -1)
                cv2.circle(mask, (cx, cy), radius, 255, -1)
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            X.append(img.astype(np.float32) / 255.0)
            y.append(mask.astype(np.float32) / 255.0)
        return np.array(X), np.array(y)