import os
import tempfile
import unittest

import cv2
import numpy as np

from data_loader import RealBiologicalLoader


class TestDataLoader(unittest.TestCase):
    def test_load_dataset_from_local_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "stage1_train")
            os.makedirs(root, exist_ok=True)

            sample_id = "sample_001"
            image_dir = os.path.join(root, sample_id, "images")
            mask_dir = os.path.join(root, sample_id, "masks")
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            img = np.zeros((16, 16, 3), dtype=np.uint8)
            img[2:10, 2:10] = 255
            cv2.imwrite(os.path.join(image_dir, f"{sample_id}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            mask = np.zeros((16, 16), dtype=np.uint8)
            mask[3:8, 3:8] = 255
            cv2.imwrite(os.path.join(mask_dir, "mask_1.png"), mask)

            loader = RealBiologicalLoader(base_dir=tmp)
            X, y = loader.load_dataset(img_size=(16, 16), max_samples=1)

            self.assertEqual(X.shape, (1, 16, 16, 3))
            self.assertEqual(y.shape, (1, 16, 16, 1))
            self.assertTrue((X >= 0.0).all() and (X <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
