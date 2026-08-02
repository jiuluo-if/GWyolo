from __future__ import annotations

import unittest

from scripts.select_operating_points import Profile, choose_profile, pareto_front


def row(name: str, recall: float, hit_rate: float, wall: float, threshold: float = 0.25):
    return {
        "experiment": name,
        "event_recall": recall,
        "detector_image_hit_rate": hit_rate,
        "wall_seconds": wall,
        "threshold": threshold,
    }


class OperatingPointTests(unittest.TestCase):
    def test_pareto_front_removes_strictly_dominated_point(self) -> None:
        fast = row("fast", 0.90, 0.30, 20.0)
        dominated = row("dominated", 0.85, 0.35, 25.0)
        higher_recall = row("higher", 0.95, 0.40, 30.0)
        names = {item["experiment"] for item in pareto_front([fast, dominated, higher_recall])}
        self.assertEqual(names, {"fast", "higher"})

    def test_profile_chooses_highest_recall_within_budgets(self) -> None:
        candidates = [
            row("fast", 0.90, 0.30, 20.0),
            row("slow", 0.95, 0.35, 50.0),
            row("too_many_hits", 0.99, 0.50, 20.0),
        ]
        selected = choose_profile(candidates, Profile("balanced", 0.40, 60.0))
        self.assertIsNotNone(selected)
        self.assertEqual(selected["experiment"], "slow")


if __name__ == "__main__":
    unittest.main()
