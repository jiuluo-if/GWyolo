from __future__ import annotations

import unittest

from scripts.benchmark_scale_consensus_cascade import short_circuit_decision


class ScaleConsensusCascadeTests(unittest.TestCase):
    def test_primary_short_circuits_all_rescue_stages(self) -> None:
        decision, stage = short_circuit_decision(
            0.2,
            None,
            None,
            None,
            baseline_threshold=0.14,
            attention_threshold=0.36,
            threshold_512=0.15,
            threshold_768=0.07,
        )
        self.assertTrue(decision)
        self.assertEqual(stage, "baseline_640")

    def test_attention_short_circuits_scale_stages(self) -> None:
        decision, stage = short_circuit_decision(
            0.0,
            0.4,
            None,
            None,
            baseline_threshold=0.14,
            attention_threshold=0.36,
            threshold_512=0.15,
            threshold_768=0.07,
        )
        self.assertTrue(decision)
        self.assertEqual(stage, "attention_640")

    def test_scale_consensus_needs_both_scores(self) -> None:
        accepted = short_circuit_decision(
            0.0,
            0.0,
            0.16,
            0.08,
            baseline_threshold=0.14,
            attention_threshold=0.36,
            threshold_512=0.15,
            threshold_768=0.07,
        )
        rejected = short_circuit_decision(
            0.0,
            0.0,
            0.16,
            None,
            baseline_threshold=0.14,
            attention_threshold=0.36,
            threshold_512=0.15,
            threshold_768=0.07,
        )
        self.assertEqual(accepted, (True, "scale_consensus"))
        self.assertEqual(rejected, (False, "miss"))


if __name__ == "__main__":
    unittest.main()
