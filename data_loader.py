import os
import cv2
import requests
import zipfile
import numpy as np
from tqdm import tqdm

class RealBiologicalLoader:
    def __init__(self, base_dir="./data"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def download_data(self):
        """Downloads and extracts the BBBC038 dataset."""
        url = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
        zip_path = os.path.join(self.base_dir, "stage1_train.zip")
        extract_path = os.path.join(self.base_dir, "stage1_train")
        
        if os.path.exists(extract_path):
            print(f"Data found in {extract_path}")
            return extract_path
            
        print("Downloading BBBC038 Dataset...")
        try:
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                raise ConnectionError(f"Failed to download. Status: {r.status_code}")
            
            with open(zip_path, 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), desc="Downloading"):
                    f.write(chunk)
            
            print("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_path)
            return extract_path
        except Exception as e:
            print(f"Error: {e}")
            return None

    def load_dataset(self, img_size=(128, 128)):
        """Parses images and merges mask files."""
        data_path = self.download_data()
        if not data_path: 
            return None, None
        
        image_ids = next(os.walk(data_path))[1]
        X, y = [], []
        
        print(f"Processing {len(image_ids)} images...")
        
        for id_ in tqdm(image_ids):
            path = os.path.join(data_path, id_)
            
            # Load Image
            img_path = os.path.join(path, 'images', id_ + '.png')
            img = cv2.imread(img_path)
            if img is None: 
                continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load & Merge Masks
            mask_dir = os.path.join(path, 'masks')
            masks = np.zeros(img_size, dtype=np.uint8)
            
            if os.path.exists(mask_dir):
                for mask_file in os.listdir(mask_dir):
                    m_path = os.path.join(mask_dir, mask_file)
                    mask_ = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
                    if mask_ is None: 
                        continue
                    mask_ = cv2.resize(mask_, img_size, interpolation=cv2.INTER_NEAREST)
                    masks = np.maximum(masks, mask_)
            
            X.append(img.astype(np.float32) / 255.0)
            y.append((masks > 0).astype(np.float32))
            
        return np.array(X), np.expand_dims(np.array(y), axis=-1)
