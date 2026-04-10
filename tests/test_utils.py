import unittest

import numpy as np

from utils import calculate_iou, count_cells_watershed


class TestUtils(unittest.TestCase):
    def test_iou_identical_masks(self):
        mask = np.zeros((8, 8, 1), dtype=np.float32)
        mask[2:5, 2:5, 0] = 1.0
        self.assertAlmostEqual(calculate_iou(mask, mask), 1.0, places=6)

    def test_iou_empty_masks(self):
        mask = np.zeros((8, 8, 1), dtype=np.float32)
        self.assertAlmostEqual(calculate_iou(mask, mask), 1.0, places=6)

    def test_count_fast_and_high_accuracy(self):
        mask = np.zeros((32, 32, 1), dtype=np.float32)
        mask[4:10, 4:10, 0] = 1.0
        mask[18:26, 18:26, 0] = 1.0

        fast_count, _ = count_cells_watershed(mask, mode="fast")
        acc_count, _ = count_cells_watershed(mask, mode="high_accuracy")

        self.assertGreaterEqual(fast_count, 2)
        self.assertGreaterEqual(acc_count, 2)


if __name__ == "__main__":
    unittest.main()
