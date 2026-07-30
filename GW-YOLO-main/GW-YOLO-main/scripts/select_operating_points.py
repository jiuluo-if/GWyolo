"""在明确成本预算下选择 Pareto 有效的 GW5 工作点。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Profile:
    name: str
    max_hit_rate: float
    max_wall_seconds: float


NUMERIC_FIELDS = {
    "threshold": float,
    "eligible_events": int,
    "recalled_events": int,
    "event_recall": float,
    "detector_images": int,
    "detector_image_hits": int,
    "detector_image_hit_rate": float,
    "wall_seconds": float,
    "mean_inference_ms": float,
}


def parse_profile(value: str) -> Profile:
    try:
        name, hit_rate, wall_seconds = value.split(":", 2)
        profile = Profile(name, float(hit_rate), float(wall_seconds))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "profile must use NAME:MAX_IMAGE_HIT_RATE:MAX_WALL_SECONDS"
        ) from exc
    if not profile.name or not 0.0 <= profile.max_hit_rate <= 1.0:
        raise argparse.ArgumentTypeError("profile name and hit rate must be valid")
    if profile.max_wall_seconds <= 0.0:
        raise argparse.ArgumentTypeError("profile wall time must be positive")
    return profile


def read_summary(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field, converter in NUMERIC_FIELDS.items():
                row[field] = converter(raw[field])
            rows.append(row)
    if not rows:
        raise ValueError(f"summary contains no rows: {path}")
    return rows


def dominates(left: dict[str, object], right: dict[str, object]) -> bool:
    """当 left 在所有目标不差且至少一项更好时返回真。"""
    left_values = (
        float(left["event_recall"]),
        -float(left["detector_image_hit_rate"]),
        -float(left["wall_seconds"]),
    )
    right_values = (
        float(right["event_recall"]),
        -float(right["detector_image_hit_rate"]),
        -float(right["wall_seconds"]),
    )
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_front(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    candidates = list(rows)
    return [
        row
        for row in candidates
        if not any(other is not row and dominates(other, row) for other in candidates)
    ]


def choose_profile(
    rows: Iterable[dict[str, object]], profile: Profile
) -> dict[str, object] | None:
    feasible = [
        row
        for row in rows
        if float(row["detector_image_hit_rate"]) <= profile.max_hit_rate
        and float(row["wall_seconds"]) <= profile.max_wall_seconds
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            float(row["event_recall"]),
            -float(row["detector_image_hit_rate"]),
            -float(row["wall_seconds"]),
            float(row["threshold"]),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        type=parse_profile,
        required=True,
        help="NAME:MAX_IMAGE_HIT_RATE:MAX_WALL_SECONDS",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_summary(args.summary)
    frontier = sorted(
        pareto_front(rows),
        key=lambda row: (
            float(row["detector_image_hit_rate"]),
            float(row["wall_seconds"]),
            -float(row["event_recall"]),
        ),
    )
    selections = []
    for profile in args.profile:
        selected = choose_profile(rows, profile)
        selections.append(
            {
                "profile": profile.name,
                "max_detector_image_hit_rate": profile.max_hit_rate,
                "max_wall_seconds": profile.max_wall_seconds,
                "selected": selected,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"profiles": selections, "pareto_front": frontier}
    (args.output_dir / "operating_points.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "pareto_front.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frontier[0]))
        writer.writeheader()
        writer.writerows(frontier)

    lines = [
        "# 批次 02：成本约束工作点校准",
        "",
        "## 选择规则",
        "",
        "在每个预设的探测器图像命中率与总推理耗时上限内，优先选择事件召回率最高的工作点；",
        "召回相同时依次选择更低图像命中率、更低耗时和更高阈值。",
        "",
        "| 档位 | 命中率上限 | 耗时上限 | 选择 | 阈值 | 召回 | 图像命中率 | 耗时 |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in selections:
        selected = item["selected"]
        if selected is None:
            lines.append(
                f"| {item['profile']} | {item['max_detector_image_hit_rate']:.0%} | "
                f"{item['max_wall_seconds']:.0f}s | 无可行项 | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {item['profile']} | {item['max_detector_image_hit_rate']:.0%} | "
            f"{item['max_wall_seconds']:.0f}s | {selected['experiment']} | "
            f"{selected['threshold']:.2f} | "
            f"{selected['recalled_events']}/{selected['eligible_events']} "
            f"({selected['event_recall']:.2%}) | "
            f"{selected['detector_image_hit_rate']:.2%} | "
            f"{selected['wall_seconds']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "## 发布约束",
            "",
            "这些档位是研究用预算示例，不代表业务已接受对应复核量。生产阈值必须在独立负样本集",
            "上增加 Precision、每小时误报数或 false alarms/day 约束后重新校准。",
            "",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(selections, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
