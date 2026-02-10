import numpy as np
import cv2
import os
import zipfile
import requests
from tqdm import tqdm

class BBBCDataLoader:
    """Data loader for BBBC datasets and synthetic data generation."""

    def __init__(self, dataset_name="BBBC005"):
        self.dataset_name = dataset_name
        self.base_url = f"https://bbbc.broadinstitute.org/{dataset_name}/"

    def download_dataset(self, save_dir="./data"):
        """
        Downloads a BBBC dataset zip file.
        """
        os.makedirs(save_dir, exist_ok=True)
        zip_filename = "BBBC005_v1_images.zip"
        zip_url = os.path.join(self.base_url, zip_filename)
        zip_path = os.path.join(save_dir, zip_filename)

        if os.path.exists(zip_path):
            print(f"{zip_filename} already downloaded.")
        else:
            print(f"Downloading {self.dataset_name} dataset from {zip_url} ...")
            try:
                response = requests.get(zip_url, stream=True, timeout=30)
                if response.status_code == 200:
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print("Download complete.")
                else:
                    print(f"Failed to download. Status code: {response.status_code}")
                    return
            except requests.exceptions.RequestException as e:
                print(f"An error occurred during download: {e}")
                return

        print("Extracting files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)
        print("Extraction complete.")

    def load_synthetic_data(self, n_samples=100, img_size=(256, 256)):
        """Generates synthetic cell-like data for demonstration."""
        print("Generating synthetic cell data for demonstration...")
        X = []
        y = []
        for _ in tqdm(range(n_samples)):
            img = np.zeros((*img_size, 3), dtype=np.uint8)
            mask = np.zeros(img_size, dtype=np.uint8)
            
            # Randomly determine number of cells
            n_cells = np.random.randint(5, 25)
            
            for _ in range(n_cells):
                center = (np.random.randint(20, img_size[0] - 20), np.random.randint(20, img_size[1] - 20))
                radius = np.random.randint(10, 25)
                color1 = np.random.randint(100, 200)
                color2 = np.random.randint(color1 + 10, 255)
                
                # Draw cell body
                cv2.circle(img, center, radius, (color1, color1, color1), -1)
                # Draw cell nucleus (slightly different color)
                cv2.circle(img, center, int(radius * 0.6), (color2, color2, color2), -1)
                # Draw mask
                cv2.circle(mask, center, radius, 255, -1)
            
            # Add noise to make it look more realistic
            noise = np.random.normal(0, 15, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            X.append(img.astype(np.float32) / 255.0)
            y.append(mask.astype(np.float32) / 255.0)
            
        return np.array(X), np.array(y)
