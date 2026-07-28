"""Calibrate a cross-scale consensus rescue after the frozen attention cascade.

The accepted baseline@640=0.14 and Attention Residual@640=0.36 thresholds stay
fixed. A new rescue fires only when both baseline@512 and baseline@768 exceed
their validation-selected thresholds. Requiring agreement across scales is
intended to distinguish a repeatable weak chirp from a single-scale artefact.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Iterable

try:
    from scripts.calibrate_multiscale_rescue import (
        ScoreRow,
        read_eligible_events,
        read_gw5_rows,
        read_validation_rows,
        threshold_grid,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from calibrate_multiscale_rescue import (
        ScoreRow,
        read_eligible_events,
        read_gw5_rows,
        read_validation_rows,
        threshold_grid,
    )


def consensus_decision(
    row: ScoreRow,
    baseline_threshold: float,
    attention_threshold: float,
    threshold_512: float,
    threshold_768: float,
) -> tuple[bool, str]:
    if row.baseline_640 >= baseline_threshold:
        return True, "baseline_640"
    if row.attention_640 >= attention_threshold:
        return True, "attention_640"
    if row.baseline_512 >= threshold_512 and row.baseline_768 >= threshold_768:
        return True, "scale_consensus"
    return False, "miss"


def validation_metrics(
    rows: Iterable[ScoreRow],
    threshold_512: float,
    threshold_768: float,
    baseline_threshold: float = 0.14,
    attention_threshold: float = 0.36,
) -> dict[str, int | float]:
    tp = fp = fn = tn = 0
    attention_invocations = consensus_invocations = 0
    primary_hits = attention_rescues = consensus_rescues = 0
    for row in rows:
        if row.baseline_640 < baseline_threshold:
            attention_invocations += 1
            if row.attention_640 < attention_threshold:
                consensus_invocations += 1
        decision, stage = consensus_decision(
            row,
            baseline_threshold,
            attention_threshold,
            threshold_512,
            threshold_768,
        )
        if stage == "baseline_640":
            primary_hits += 1
        elif stage == "attention_640":
            attention_rescues += 1
        elif stage == "scale_consensus":
            consensus_rescues += 1
        truth = bool(row.truth_has_chirp)
        if truth and decision:
            tp += 1
        elif truth:
            fn += 1
        elif decision:
            fp += 1
        else:
            tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "attention_invocations": attention_invocations,
        "consensus_invocations": consensus_invocations,
        "primary_hits": primary_hits,
        "attention_rescues": attention_rescues,
        "consensus_rescues": consensus_rescues,
    }


def choose_policy(
    rows: Iterable[ScoreRow],
    thresholds: Iterable[float],
    max_false_positives: int,
    baseline_threshold: float = 0.14,
    attention_threshold: float = 0.36,
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    rows = list(rows)
    thresholds = tuple(thresholds)
    grid: list[dict[str, int | float]] = []
    for threshold_512, threshold_768 in product(thresholds, repeat=2):
        metrics = validation_metrics(
            rows,
            threshold_512,
            threshold_768,
            baseline_threshold,
            attention_threshold,
        )
        grid.append(
            {
                "baseline_threshold": baseline_threshold,
                "attention_threshold": attention_threshold,
                "threshold_512": threshold_512,
                "threshold_768": threshold_768,
                **metrics,
            }
        )
    feasible = [
        row for row in grid if int(row["fp"]) <= max_false_positives
    ]
    if not feasible:
        raise ValueError("no consensus threshold pair satisfies the FP constraint")
    feasible.sort(
        key=lambda row: (
            -int(row["tp"]),
            int(row["fp"]),
            -(float(row["threshold_512"]) + float(row["threshold_768"])),
            -float(row["threshold_512"]),
            -float(row["threshold_768"]),
        )
    )
    return feasible[0], grid


def gw5_metrics(
    rows: Iterable[ScoreRow],
    eligible_events: set[str],
    threshold_512: float,
    threshold_768: float,
    baseline_threshold: float = 0.14,
    attention_threshold: float = 0.36,
) -> dict[str, int | float | list[str]]:
    rows = list(rows)
    recalled: set[str] = set()
    detector_image_hits = 0
    attention_invocations = consensus_invocations = 0
    primary_hits = attention_rescues = consensus_rescues = 0
    cascade_inference_ms = full_inference_ms = 0.0
    for row in rows:
        cascade_inference_ms += row.inference_ms_baseline_640
        full_inference_ms += (
            row.inference_ms_baseline_640
            + row.inference_ms_attention_640
            + row.inference_ms_baseline_512
            + row.inference_ms_baseline_768
        )
        if row.baseline_640 < baseline_threshold:
            attention_invocations += 1
            cascade_inference_ms += row.inference_ms_attention_640
            if row.attention_640 < attention_threshold:
                consensus_invocations += 1
                cascade_inference_ms += (
                    row.inference_ms_baseline_512 + row.inference_ms_baseline_768
                )
        decision, stage = consensus_decision(
            row,
            baseline_threshold,
            attention_threshold,
            threshold_512,
            threshold_768,
        )
        if decision:
            detector_image_hits += 1
            if row.event in eligible_events:
                recalled.add(str(row.event))
        if stage == "baseline_640":
            primary_hits += 1
        elif stage == "attention_640":
            attention_rescues += 1
        elif stage == "scale_consensus":
            consensus_rescues += 1
    return {
        "eligible_events": len(eligible_events),
        "recalled_events": len(recalled),
        "event_recall": len(recalled) / len(eligible_events),
        "missed_events": sorted(eligible_events - recalled),
        "detector_images": len(rows),
        "detector_image_hits": detector_image_hits,
        "detector_image_hit_rate": detector_image_hits / len(rows),
        "attention_invocations": attention_invocations,
        "consensus_invocations": consensus_invocations,
        "primary_hits": primary_hits,
        "attention_rescues": attention_rescues,
        "consensus_rescues": consensus_rescues,
        "estimated_cascade_inference_ms": cascade_inference_ms,
        "estimated_full_inference_ms": full_inference_ms,
        "estimated_inference_saved_fraction": (
            1.0 - cascade_inference_ms / full_inference_ms
            if full_inference_ms
            else 0.0
        ),
    }


def write_csv(path: Path, rows: list[dict[str, int | float]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-640", type=Path, required=True)
    parser.add_argument("--validation-512", type=Path, required=True)
    parser.add_argument("--validation-768", type=Path, required=True)
    parser.add_argument("--gw5-predictions", type=Path, required=True)
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=Path("docs/gw5_recall_details.csv"),
    )
    parser.add_argument("--max-false-positives", type=int, default=3)
    parser.add_argument("--grid-start", type=int, default=1)
    parser.add_argument("--grid-stop", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    baseline_threshold = 0.14
    attention_threshold = 0.36
    validation_rows = read_validation_rows(
        args.validation_640,
        args.validation_512,
        args.validation_768,
    )
    thresholds = threshold_grid(args.grid_start, args.grid_stop)
    selected, grid = choose_policy(
        validation_rows,
        thresholds,
        args.max_false_positives,
        baseline_threshold,
        attention_threshold,
    )
    gw5_rows = read_gw5_rows(args.gw5_predictions)
    eligible_events = read_eligible_events(args.catalogue)
    blind_audit = gw5_metrics(
        gw5_rows,
        eligible_events,
        float(selected["threshold_512"]),
        float(selected["threshold_768"]),
        baseline_threshold,
        attention_threshold,
    )
    accepted = (
        int(selected["tp"]) == 69
        and int(selected["fp"]) <= args.max_false_positives
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "calibration_grid.csv", grid)
    payload = {
        "selection_scope": "fixed validation only; GW5 excluded from threshold selection",
        "search": {
            "grid_start": args.grid_start / 100,
            "grid_stop": args.grid_stop / 100,
            "grid_step": 0.01,
            "combinations": len(thresholds) ** 2,
            "max_false_positives": args.max_false_positives,
            "locked_baseline_threshold": baseline_threshold,
            "locked_attention_threshold": attention_threshold,
            "tie_break": "max TP, min FP, highest combined consensus thresholds",
        },
        "selected_policy": {
            "order": [
                "baseline@640",
                "attention@640",
                "baseline@512 AND baseline@768",
            ],
            "accepted": accepted,
            "decision": (
                "accept: validation reached 69/69 positives within FP constraint"
                if accepted
                else "reject: validation did not reach 69/69 within FP constraint"
            ),
            **selected,
        },
        "gw5_blind_audit": blind_audit,
        "inputs": {
            "validation_rows": len(validation_rows),
            "gw5_rows": len(gw5_rows),
            "eligible_events": len(eligible_events),
            "validation_640": str(args.validation_640),
            "validation_512": str(args.validation_512),
            "validation_768": str(args.validation_768),
            "gw5_predictions": str(args.gw5_predictions),
            "catalogue": str(args.catalogue),
        },
    }
    (args.output_dir / "selected_policy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
