import os
import time
import zipfile
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm


class RealBiologicalLoader:
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def download_data(self, profile: Optional[Dict[str, float]] = None) -> Optional[str]:
        """Downloads and extracts the BBBC038 dataset."""
        profile = profile if profile is not None else {}

        url = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
        zip_path = os.path.join(self.base_dir, "stage1_train.zip")
        extract_path = os.path.join(self.base_dir, "stage1_train")

        if os.path.exists(extract_path):
            print(f"✅ Data found in {extract_path}")
            return extract_path

        print("⬇️ Downloading BBBC038 Dataset...")
        try:
            t0 = time.perf_counter()
            r = requests.get(url, stream=True, timeout=120)
            if r.status_code != 200:
                raise ConnectionError(f"Failed to download. Status: {r.status_code}")

            with open(zip_path, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading"):
                    f.write(chunk)
            profile["download"] = profile.get("download", 0.0) + (time.perf_counter() - t0)

            print("📦 Extracting dataset...")
            t1 = time.perf_counter()
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_path)
            profile["extract"] = profile.get("extract", 0.0) + (time.perf_counter() - t1)
            return extract_path
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def load_dataset(
        self,
        img_size: Tuple[int, int] = (128, 128),
        max_samples: Optional[int] = None,
        profile: Optional[Dict[str, float]] = None,
    ):
        """Parses images and merges mask files."""
        profile = profile if profile is not None else {}

        t0 = time.perf_counter()
        data_path = self.download_data(profile=profile)
        profile["dataset_lookup"] = profile.get("dataset_lookup", 0.0) + (time.perf_counter() - t0)
        if not data_path:
            return None, None

        image_ids = sorted(next(os.walk(data_path))[1])
        if max_samples is not None:
            image_ids = image_ids[: max(0, max_samples)]
        X, y = [], []

        print(f"🔄 Processing {len(image_ids)} images...")

        prep_time = 0.0
        mask_merge_time = 0.0

        for id_ in tqdm(image_ids):
            path = os.path.join(data_path, id_)

            t_prep = time.perf_counter()
            img_path = os.path.join(path, "images", id_ + ".png")
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prep_time += time.perf_counter() - t_prep

            t_mask = time.perf_counter()
            mask_dir = os.path.join(path, "masks")
            masks = np.zeros(img_size, dtype=np.uint8)

            if os.path.exists(mask_dir):
                for mask_file in sorted(os.listdir(mask_dir)):
                    m_path = os.path.join(mask_dir, mask_file)
                    mask_ = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                    if mask_ is None:
                        continue
                    mask_ = cv2.resize(mask_, img_size, interpolation=cv2.INTER_NEAREST)
                    masks = np.maximum(masks, mask_)
            mask_merge_time += time.perf_counter() - t_mask

            X.append(img.astype(np.float32) / 255.0)
            y.append((masks > 0).astype(np.float32))

        profile["preprocessing"] = profile.get("preprocessing", 0.0) + prep_time
        profile["mask_merge"] = profile.get("mask_merge", 0.0) + mask_merge_time

        return np.array(X), np.expand_dims(np.array(y), axis=-1)
