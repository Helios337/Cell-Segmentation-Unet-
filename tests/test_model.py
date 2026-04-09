import unittest

import numpy as np

from model import build_deep_unet


class TestModel(unittest.TestCase):
    def test_build_baseline_shape(self):
        model = build_deep_unet(input_shape=(64, 64, 3), base_filters=16, depth=3)
        x = np.random.rand(2, 64, 64, 3).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, 64, 64, 1))

    def test_build_separable_variant(self):
        model = build_deep_unet(input_shape=(64, 64, 3), base_filters=8, depth=2, use_separable=True)
        x = np.random.rand(1, 64, 64, 3).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (1, 64, 64, 1))


if __name__ == "__main__":
    unittest.main()
