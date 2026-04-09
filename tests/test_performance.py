import unittest

from performance import aggregate_trials, evaluate_acceptance


class TestPerformance(unittest.TestCase):
    def test_aggregate_trials(self):
        trials = [
            {
                "train_time_per_epoch": 2.0,
                "total_train_time": 10.0,
                "time_to_best_val_loss": 8.0,
                "inference_latency_ms": 20.0,
                "throughput_images_per_sec": 50.0,
                "model_size_mb": 10.0,
                "mean_iou": 0.82,
                "binary_iou": 0.82,
                "counting_bias": 0.5,
                "count_agreement_spread": 3.0,
                "bottlenecks": {"train_fit": 10.0},
            },
            {
                "train_time_per_epoch": 1.8,
                "total_train_time": 9.0,
                "time_to_best_val_loss": 7.0,
                "inference_latency_ms": 18.0,
                "throughput_images_per_sec": 55.0,
                "model_size_mb": 9.0,
                "mean_iou": 0.81,
                "binary_iou": 0.81,
                "counting_bias": 0.6,
                "count_agreement_spread": 3.2,
                "bottlenecks": {"train_fit": 9.0},
            },
        ]
        out = aggregate_trials(trials)
        self.assertEqual(out["trial_count"], 2)
        self.assertIn("train_fit", [x["stage"] for x in out["prioritized_bottlenecks"]])

    def test_acceptance(self):
        baseline = {
            "means": {
                "train_time_per_epoch": 2.0,
                "inference_latency_ms": 20.0,
                "mean_iou": 0.85,
                "binary_iou": 0.85,
                "counting_bias": 0.2,
            }
        }
        candidate = {
            "means": {
                "train_time_per_epoch": 1.7,
                "inference_latency_ms": 17.0,
                "mean_iou": 0.845,
                "binary_iou": 0.845,
                "counting_bias": 0.8,
            }
        }
        out = evaluate_acceptance(baseline, candidate)
        self.assertIn("accepted", out)
        self.assertIn("checks", out)


if __name__ == "__main__":
    unittest.main()
