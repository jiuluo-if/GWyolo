from __future__ import annotations

import unittest

from scripts.calibrate_multiscale_rescue import (
    ScoreRow,
    choose_policy,
    choose_rescue_policy,
    policy_decision,
    threshold_grid,
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


class MultiscaleRescueCalibrationTests(unittest.TestCase):
    def test_policy_uses_fixed_stage_order(self) -> None:
        primary = row("primary", True, 0.2, 0.8, 0.9, 0.9)
        attention = row("attention", True, 0.1, 0.4, 0.9, 0.9)
        multiscale = row("multiscale", True, 0.1, 0.2, 0.4, 0.3)
        miss = row("miss", True, 0.1, 0.2, 0.1, 0.1)
        self.assertEqual(policy_decision(primary, 0.14, 0.36, 0.35)[1], "baseline_640")
        self.assertEqual(policy_decision(attention, 0.14, 0.36, 0.35)[1], "attention_640")
        self.assertEqual(
            policy_decision(multiscale, 0.14, 0.36, 0.35)[1],
            "baseline_multiscale",
        )
        self.assertFalse(policy_decision(miss, 0.14, 0.36, 0.35)[0])

    def test_validation_metrics_count_invocations_and_rescues(self) -> None:
        rows = [
            row("primary", True, 0.2, 0.0, 0.0, 0.0),
            row("attention", True, 0.1, 0.4, 0.0, 0.0),
            row("multiscale", True, 0.1, 0.2, 0.4, 0.0),
            row("negative", False, 0.0, 0.0, 0.0, 0.0),
        ]
        metrics = validation_metrics(rows, 0.14, 0.36, 0.35)
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]),
            (3, 0, 0, 1),
        )
        self.assertEqual(metrics["attention_invocations"], 3)
        self.assertEqual(metrics["multiscale_invocations"], 2)
        self.assertEqual(metrics["attention_rescues"], 1)
        self.assertEqual(metrics["multiscale_rescues"], 1)

    def test_choose_policy_obeys_fp_constraint(self) -> None:
        rows = [
            row("positive", True, 0.1, 0.1, 0.3, 0.0),
            row("negative", False, 0.1, 0.1, 0.2, 0.0),
        ]
        selected, _ = choose_policy(rows, (0.15, 0.25), max_false_positives=0)
        self.assertEqual(selected["tp"], 1)
        self.assertEqual(selected["fp"], 0)
        self.assertEqual(selected["multiscale_threshold"], 0.25)

    def test_rescue_search_locks_accepted_thresholds(self) -> None:
        rows = [
            row("positive", True, 0.1, 0.1, 0.3, 0.0),
            row("negative", False, 0.1, 0.1, 0.2, 0.0),
        ]
        selected, _ = choose_rescue_policy(
            rows,
            (0.15, 0.25),
            max_false_positives=0,
            baseline_threshold=0.14,
            attention_threshold=0.36,
        )
        self.assertEqual(selected["baseline_threshold"], 0.14)
        self.assertEqual(selected["attention_threshold"], 0.36)
        self.assertEqual(selected["multiscale_threshold"], 0.25)

    def test_threshold_grid_is_exact_and_validated(self) -> None:
        self.assertEqual(threshold_grid(1, 3), (0.01, 0.02, 0.03))
        with self.assertRaises(ValueError):
            threshold_grid(5, 4)


if __name__ == "__main__":
    unittest.main()
