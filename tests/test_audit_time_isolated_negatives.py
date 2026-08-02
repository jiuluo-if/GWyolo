from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from scripts.audit_time_isolated_negatives import (
    audit,
    paired_windows,
    poisson_upper_mean,
    row_eligibility,
)


def detector_row(
    detector: str,
    *,
    index: int = 0,
    start_seconds: int = 0,
    approved: bool = True,
) -> dict[str, object]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        seconds=start_seconds
    )
    end = start + timedelta(seconds=4)
    return {
        "window_id": f"O2_{detector}_{index}",
        "image": f"{detector.lower()}-{index}",
        "image_hash": f"hash-{detector}-{index}",
        "detector": detector,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "start": start,
        "end": end,
        "gwtc_veto_passed": True,
        "time_isolated_verified": True,
        "human_review_status": "approved" if approved else "pending",
        "human_reviewers": "reviewer_a;reviewer_b" if approved else "",
        "human_reviewed_at": "2026-01-02T00:00:00+00:00" if approved else "",
        "human_review_reason": "independent agreement" if approved else "",
    }


def paired_rows(
    *,
    index: int = 0,
    start_seconds: int = 0,
    approved: bool = True,
) -> list[dict[str, object]]:
    return [
        detector_row(
            detector,
            index=index,
            start_seconds=start_seconds,
            approved=approved,
        )
        for detector in ("H1", "L1")
    ]


class TimeIsolatedNegativeAuditTests(unittest.TestCase):
    def test_zero_alert_upper_bound_uses_network_exposure(self) -> None:
        manifest = paired_rows()
        predictions = {"h1-0": False, "l1-0": False}
        summary, _, _ = audit(manifest, predictions, 0.95)
        expected = 2.995732273553991 / (4.0 / 86400.0)
        self.assertEqual(summary["false_alarms_per_day"], 0.0)
        self.assertAlmostEqual(
            float(summary["one_sided_upper_false_alarms_per_day"]),
            expected,
            places=4,
        )

    def test_consecutive_network_hits_merge_into_one_alert(self) -> None:
        manifest = [
            *paired_rows(index=0, start_seconds=0),
            *paired_rows(index=1, start_seconds=4),
            *paired_rows(index=2, start_seconds=12),
        ]
        predictions = {
            "h1-0": True,
            "l1-0": False,
            "h1-1": False,
            "l1-1": True,
            "h1-2": True,
            "l1-2": False,
        }
        summary, rows, _ = audit(manifest, predictions, 0.95)
        self.assertEqual(summary["false_alarms"], 2)
        self.assertEqual(
            [row["starts_new_alert"] for row in rows],
            [True, False, True],
        )

    def test_unapproved_pair_is_excluded(self) -> None:
        eligible, ineligible = paired_windows(
            [
                *paired_rows(index=0),
                *paired_rows(index=1, start_seconds=4, approved=False),
            ]
        )
        self.assertEqual(len(eligible), 1)
        self.assertEqual(len(ineligible), 1)
        self.assertIn("human_review_not_approved", ineligible[0]["reason"])

    def test_approved_row_requires_two_reviewers(self) -> None:
        row = detector_row("H1")
        row["human_reviewers"] = "reviewer_a"
        accepted, reasons = row_eligibility(row)
        self.assertFalse(accepted)
        self.assertIn("fewer_than_two_reviewers", reasons)

    def test_mismatched_predictions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            audit(
                paired_rows(),
                {"h1-0": False, "unexpected": False},
                0.95,
            )

    def test_poisson_upper_mean_is_exact_for_zero_alerts(self) -> None:
        self.assertAlmostEqual(
            poisson_upper_mean(0, 0.95),
            -__import__("math").log(0.05),
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
