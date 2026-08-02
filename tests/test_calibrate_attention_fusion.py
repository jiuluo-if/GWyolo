from __future__ import annotations

import unittest

from scripts.calibrate_attention_fusion import (
    ScoreRow,
    choose_policy,
    event_metrics,
    validation_metrics,
)


class AttentionFusionCalibrationTests(unittest.TestCase):
    def test_validation_metrics_use_asymmetric_or(self) -> None:
        rows = [
            ScoreRow("positive-primary", 0.2, 0.0, True),
            ScoreRow("positive-attention", 0.0, 0.4, True),
            ScoreRow("negative", 0.0, 0.3, False),
        ]
        metrics = validation_metrics(rows, 0.14, 0.36)
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]),
            (2, 0, 0, 1),
        )

    def test_policy_selection_does_not_need_gw5_metrics(self) -> None:
        rows = [
            {
                "primary_threshold": 0.20,
                "secondary_threshold": 0.40,
                "tp": 2,
                "fp": 0,
                "secondary_invocations": 2,
            },
            {
                "primary_threshold": 0.14,
                "secondary_threshold": 0.36,
                "tp": 2,
                "fp": 0,
                "secondary_invocations": 3,
            },
        ]
        selected = choose_policy(rows, max_false_positives=0)
        self.assertEqual(selected["primary_threshold"], 0.14)
        self.assertEqual(selected["secondary_threshold"], 0.36)

    def test_event_metrics_use_any_detector_and_exclude_catalogue_rows(self) -> None:
        rows = [
            ScoreRow("GW240101_000000-v1-H1-qscan", 0.0, 0.4, None),
            ScoreRow("GW240101_000000-v1-L1-qscan", 0.0, 0.0, None),
            ScoreRow("GW240102_000000-v1-H1-qscan", 0.8, 0.0, None),
        ]
        metrics = event_metrics(rows, {"GW240101_000000"}, 0.14, 0.36)
        self.assertEqual(metrics["eligible_events"], 1)
        self.assertEqual(metrics["recalled_events"], 1)
        self.assertEqual(metrics["detector_image_hits"], 2)


if __name__ == "__main__":
    unittest.main()
