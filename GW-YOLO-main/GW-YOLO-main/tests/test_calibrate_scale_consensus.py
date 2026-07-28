from __future__ import annotations

import unittest

from scripts.calibrate_multiscale_rescue import ScoreRow
from scripts.calibrate_scale_consensus import (
    choose_policy,
    consensus_decision,
    validation_metrics,
)


def row(
    image: str,
    truth: bool,
    baseline_640: float,
    attention_640: float,
    baseline_512: float,
    baseline_768: float,
) -> ScoreRow:
    return ScoreRow(
        image=image,
        truth_has_chirp=truth,
        baseline_640=baseline_640,
        attention_640=attention_640,
        baseline_512=baseline_512,
        baseline_768=baseline_768,
    )


class ScaleConsensusCalibrationTests(unittest.TestCase):
    def test_consensus_requires_both_scales(self) -> None:
        consensus = row("consensus", True, 0.0, 0.0, 0.2, 0.1)
        single_scale = row("single", True, 0.0, 0.0, 0.8, 0.0)
        self.assertEqual(
            consensus_decision(consensus, 0.14, 0.36, 0.15, 0.05)[1],
            "scale_consensus",
        )
        self.assertFalse(
            consensus_decision(single_scale, 0.14, 0.36, 0.15, 0.05)[0]
        )

    def test_consensus_can_recover_positive_without_single_scale_fp(self) -> None:
        rows = [
            row("positive", True, 0.0, 0.0, 0.16, 0.08),
            row("single-scale-negative", False, 0.0, 0.0, 0.55, 0.02),
        ]
        metrics = validation_metrics(rows, 0.15, 0.07)
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]),
            (1, 0, 0, 1),
        )
        self.assertEqual(metrics["consensus_rescues"], 1)

    def test_search_prefers_conservative_thresholds_after_metrics(self) -> None:
        rows = [
            row("positive", True, 0.0, 0.0, 0.16, 0.08),
            row("negative", False, 0.0, 0.0, 0.55, 0.02),
        ]
        selected, grid = choose_policy(
            rows, (0.01, 0.05, 0.07, 0.15), max_false_positives=0
        )
        self.assertEqual(len(grid), 16)
        self.assertEqual(selected["tp"], 1)
        self.assertEqual(selected["fp"], 0)
        self.assertEqual(selected["threshold_512"], 0.15)
        self.assertEqual(selected["threshold_768"], 0.07)


if __name__ == "__main__":
    unittest.main()
