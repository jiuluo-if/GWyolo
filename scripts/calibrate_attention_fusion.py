"""根据已保存分数校准非对称 baseline→Attention 级联。

阈值只由验证集分数选择；GW5 分数在策略冻结后才评估，绝不参与策略排序。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


EVENT_PATTERN = re.compile(r"(GW\d{6}_\d{6})")


@dataclass(frozen=True)
class ScoreRow:
    image: str
    primary: float
    secondary: float
    truth: bool | None


def read_scores(path: Path) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth_text = row.get("truth_has_chirp", "").strip().lower()
            truth = None if not truth_text else truth_text == "true"
            rows.append(
                ScoreRow(
                    image=row["image"],
                    primary=float(row["primary_confidence"]),
                    secondary=float(row["secondary_confidence"]),
                    truth=truth,
                )
            )
    if not rows:
        raise ValueError(f"no predictions found in {path}")
    return rows


def read_eligible_events(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["事件"]
        for row in rows
        if row["纳入召回统计"].strip().lower() == "true"
    }


def thresholds(start: float, stop: float, step: float) -> tuple[float, ...]:
    if not 0.0 <= start <= stop <= 1.0 or step <= 0.0:
        raise ValueError("threshold range must be ordered within [0, 1]")
    count = int(round((stop - start) / step))
    return tuple(round(start + index * step, 10) for index in range(count + 1))


def decide(row: ScoreRow, primary_threshold: float, secondary_threshold: float) -> bool:
    return (
        row.primary >= primary_threshold
        or row.secondary >= secondary_threshold
    )


def validation_metrics(
    rows: list[ScoreRow],
    primary_threshold: float,
    secondary_threshold: float,
) -> dict[str, int | float]:
    if any(row.truth is None for row in rows):
        raise ValueError("validation predictions must include truth_has_chirp")
    tp = fp = fn = tn = 0
    for row in rows:
        predicted = decide(row, primary_threshold, secondary_threshold)
        if predicted and row.truth:
            tp += 1
        elif predicted:
            fp += 1
        elif row.truth:
            fn += 1
        else:
            tn += 1
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "secondary_invocations": sum(
            row.primary < primary_threshold for row in rows
        ),
    }


def event_metrics(
    rows: list[ScoreRow],
    eligible_events: set[str],
    primary_threshold: float,
    secondary_threshold: float,
) -> dict[str, int | float | list[str]]:
    recalled: set[str] = set()
    image_hits = 0
    for row in rows:
        if not decide(row, primary_threshold, secondary_threshold):
            continue
        image_hits += 1
        match = EVENT_PATTERN.search(row.image)
        if match and match.group(1) in eligible_events:
            recalled.add(match.group(1))
    missed = sorted(eligible_events - recalled)
    return {
        "eligible_events": len(eligible_events),
        "recalled_events": len(recalled),
        "event_recall": len(recalled) / len(eligible_events),
        "detector_image_hits": image_hits,
        "secondary_invocations": sum(
            row.primary < primary_threshold for row in rows
        ),
        "missed_events": missed,
    }


def choose_policy(
    rows: list[dict[str, int | float]],
    max_false_positives: int,
) -> dict[str, int | float]:
    feasible = [row for row in rows if row["fp"] <= max_false_positives]
    if not feasible:
        raise ValueError("no threshold pair satisfies the false-positive constraint")
    # 召回优先的 Neyman-Pearson 策略：先最大化验证集 TP、最小化 FP，再使用
    # 仍满足约束的最低阈值。最终并列规则偏好更少的注意力调用，且不参考 GW5。
    return min(
        feasible,
        key=lambda row: (
            -int(row["tp"]),
            int(row["fp"]),
            float(row["primary_threshold"]),
            float(row["secondary_threshold"]),
            int(row["secondary_invocations"]),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-predictions", type=Path, required=True)
    parser.add_argument("--gw5-predictions", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--threshold-start", type=float, default=0.01)
    parser.add_argument("--threshold-stop", type=float, default=0.60)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--max-false-positives", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validation = read_scores(args.validation_predictions)
    gw5 = read_scores(args.gw5_predictions)
    eligible_events = read_eligible_events(args.catalogue)
    grid = []
    values = thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)
    for primary_threshold in values:
        for secondary_threshold in values:
            row: dict[str, int | float] = {
                "primary_threshold": primary_threshold,
                "secondary_threshold": secondary_threshold,
            }
            row.update(
                validation_metrics(
                    validation, primary_threshold, secondary_threshold
                )
            )
            grid.append(row)

    selected = choose_policy(grid, args.max_false_positives)
    gw5_metrics = event_metrics(
        gw5,
        eligible_events,
        float(selected["primary_threshold"]),
        float(selected["secondary_threshold"]),
    )
    payload = {
        "selection_protocol": {
            "selection_data": str(args.validation_predictions),
            "gw5_used_for_selection": False,
            "objective": "maximize validation TP, constrain FP, use lowest stable thresholds",
            "max_false_positives": args.max_false_positives,
            "threshold_start": args.threshold_start,
            "threshold_stop": args.threshold_stop,
            "threshold_step": args.threshold_step,
        },
        "selected_policy": selected,
        "gw5_evaluation": gw5_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "calibration_grid.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    (args.output_dir / "selected_policy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
