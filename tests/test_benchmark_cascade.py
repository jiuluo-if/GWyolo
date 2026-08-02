from __future__ import annotations

import unittest

from scripts.benchmark_cascade import (
    asymmetric_decisions,
    chunks,
    classification_metrics,
    event_metrics,
    merge_scores,
)


class CascadeBenchmarkTests(unittest.TestCase):
    def test_chunks_never_exceeds_requested_size(self) -> None:
        values = list(range(5))
        self.assertEqual(list(chunks(values, 2)), [[0, 1], [2, 3], [4]])

    def test_merge_scores_implements_or_ensemble(self) -> None:
        primary = {"a": 0.8, "b": 0.1}
        secondary = {"a": 0.2, "b": 0.7}
        self.assertEqual(merge_scores(primary, secondary), {"a": 0.8, "b": 0.7})

    def test_asymmetric_decisions_use_independent_thresholds(self) -> None:
        primary = {"primary_hit": 0.2, "attention_hit": 0.1, "miss": 0.1}
        secondary = {"primary_hit": 0.1, "attention_hit": 0.4, "miss": 0.3}
        decisions = asymmetric_decisions(primary, secondary, 0.14, 0.36)
        self.assertEqual(
            decisions,
            {"primary_hit": True, "attention_hit": True, "miss": False},
        )

    def test_classification_metrics(self) -> None:
        scores = {"tp": 0.8, "fn": 0.1, "fp": 0.7, "tn": 0.0}
        truth = {"tp": True, "fn": True, "fp": False, "tn": False}
        metrics = classification_metrics(scores, truth, 0.5)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]), (1, 1, 1, 1))
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["false_positive_rate"], 0.5)

    def test_event_metrics_uses_any_detector(self) -> None:
        scores = {
            "GW240101_000000-v1-H1-qscan": 0.1,
            "GW240101_000000-v1-L1-qscan": 0.8,
            "GW240102_000000-v1-H1-qscan": 0.2,
        }
        catalogue = {"GW240101_000000": True, "GW240102_000000": True}
        metrics = event_metrics(scores, catalogue, 0.5)
        self.assertEqual(metrics["recalled_events"], 1)
        self.assertEqual(metrics["event_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
