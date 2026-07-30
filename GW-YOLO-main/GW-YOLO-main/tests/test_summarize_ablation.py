import unittest

from scripts.summarize_ablation import summarize


class SummarizeAblationTests(unittest.TestCase):
    def test_summarize_groups_three_complete_seeds(self):
        rows = [
            {"model": f"baseline-seed{seed}", "chirp_recall": str(0.7 + seed / 100)}
            for seed in (0, 1, 2)
        ]
        summary = summarize(rows, {0, 1, 2})
        self.assertEqual(summary["baseline"]["种子"], [0, 1, 2])
        self.assertEqual(summary["baseline"]["chirp_recall"]["样本数"], 3)

    def test_summarize_rejects_missing_seed(self):
        rows = [{"model": "p4-seed0", "chirp_recall": "0.7"}]
        with self.assertRaises(ValueError):
            summarize(rows, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
