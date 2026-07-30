"""校准仅使用验证集的 Attention Residual + 多尺度救援级联。

策略按下列固定顺序评估：

1. baseline@640；
2. 主模型未过阈值时调用 Attention Residual@640；
3. 前两级均未过阈值时调用 baseline@512 和 baseline@768。

阈值只从带标签验证集选择。GW5 分数在策略冻结后才读取，仅用于冻结的事件级审计。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Iterable


EVENT_PATTERN = re.compile(r"(GW\d{6}_\d{6})")


@dataclass(frozen=True)
class ScoreRow:
    image: str
    truth_has_chirp: bool | None
    baseline_640: float
    attention_640: float
    baseline_512: float
    baseline_768: float
    event: str | None = None
    inference_ms_baseline_640: float = 0.0
    inference_ms_attention_640: float = 0.0
    inference_ms_baseline_512: float = 0.0
    inference_ms_baseline_768: float = 0.0

    @property
    def multiscale_score(self) -> float:
        return max(self.baseline_512, self.baseline_768)


def threshold_grid(start: int = 1, stop: int = 60) -> tuple[float, ...]:
    """返回精确到两位小数的阈值网格。"""
    if start < 0 or stop < start or stop > 100:
        raise ValueError("threshold grid must satisfy 0 <= start <= stop <= 100")
    return tuple(value / 100 for value in range(start, stop + 1))


def _read_cascade_predictions(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth_text = row.get("truth_has_chirp", "").strip().lower()
            truth = None if not truth_text else truth_text == "true"
            rows[row["image"]] = {
                "truth": truth,
                "primary": float(row["primary_confidence"]),
                "secondary": float(row["secondary_confidence"]),
            }
    if not rows:
        raise ValueError(f"prediction file contains no rows: {path}")
    return rows


def read_validation_rows(
    predictions_640: Path,
    predictions_512: Path,
    predictions_768: Path,
) -> list[ScoreRow]:
    at_640 = _read_cascade_predictions(predictions_640)
    at_512 = _read_cascade_predictions(predictions_512)
    at_768 = _read_cascade_predictions(predictions_768)
    image_sets = (set(at_640), set(at_512), set(at_768))
    if len({frozenset(values) for values in image_sets}) != 1:
        raise ValueError("validation prediction files do not contain identical images")

    rows: list[ScoreRow] = []
    for image in sorted(at_640):
        truth = at_640[image]["truth"]
        if truth is None:
            raise ValueError(f"validation truth is missing for {image}")
        for source in (at_512, at_768):
            if source[image]["truth"] != truth:
                raise ValueError(f"validation truth disagrees for {image}")
        rows.append(
            ScoreRow(
                image=image,
                truth_has_chirp=bool(truth),
                baseline_640=float(at_640[image]["primary"]),
                attention_640=float(at_640[image]["secondary"]),
                baseline_512=float(at_512[image]["primary"]),
                baseline_768=float(at_768[image]["primary"]),
            )
        )
    return rows


def policy_decision(
    row: ScoreRow,
    baseline_threshold: float,
    attention_threshold: float,
    multiscale_threshold: float,
    *,
    enable_multiscale: bool = True,
) -> tuple[bool, str]:
    if row.baseline_640 >= baseline_threshold:
        return True, "baseline_640"
    if row.attention_640 >= attention_threshold:
        return True, "attention_640"
    if enable_multiscale and row.multiscale_score >= multiscale_threshold:
        return True, "baseline_multiscale"
    return False, "miss"


def validation_metrics(
    rows: Iterable[ScoreRow],
    baseline_threshold: float,
    attention_threshold: float,
    multiscale_threshold: float,
    *,
    enable_multiscale: bool = True,
) -> dict[str, int | float]:
    rows = list(rows)
    tp = fp = fn = tn = 0
    attention_invocations = multiscale_invocations = 0
    primary_hits = attention_rescues = multiscale_rescues = 0
    for row in rows:
        if row.baseline_640 < baseline_threshold:
            attention_invocations += 1
            if enable_multiscale and row.attention_640 < attention_threshold:
                multiscale_invocations += 1
        decision, stage = policy_decision(
            row,
            baseline_threshold,
            attention_threshold,
            multiscale_threshold,
            enable_multiscale=enable_multiscale,
        )
        if stage == "baseline_640":
            primary_hits += 1
        elif stage == "attention_640":
            attention_rescues += 1
        elif stage == "baseline_multiscale":
            multiscale_rescues += 1
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
        "multiscale_invocations": multiscale_invocations,
        "primary_hits": primary_hits,
        "attention_rescues": attention_rescues,
        "multiscale_rescues": multiscale_rescues,
    }


def choose_policy(
    rows: Iterable[ScoreRow],
    thresholds: Iterable[float],
    max_false_positives: int,
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    rows = list(rows)
    thresholds = tuple(thresholds)
    candidates: list[dict[str, int | float]] = []
    best_tp = -1
    for baseline_threshold, attention_threshold, multiscale_threshold in product(
        thresholds, repeat=3
    ):
        metrics = validation_metrics(
            rows,
            baseline_threshold,
            attention_threshold,
            multiscale_threshold,
        )
        if int(metrics["fp"]) > max_false_positives:
            continue
        tp = int(metrics["tp"])
        candidate = {
            "baseline_threshold": baseline_threshold,
            "attention_threshold": attention_threshold,
            "multiscale_threshold": multiscale_threshold,
            **metrics,
        }
        if tp > best_tp:
            best_tp = tp
            candidates = [candidate]
        elif tp == best_tp:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("no threshold combination satisfies the FP constraint")

    candidates.sort(
        key=lambda item: (
            int(item["fp"]),
            int(item["multiscale_invocations"]),
            int(item["attention_invocations"]),
            -float(item["baseline_threshold"]),
            -float(item["attention_threshold"]),
            -float(item["multiscale_threshold"]),
        )
    )
    return candidates[0], candidates


def choose_rescue_policy(
    rows: Iterable[ScoreRow],
    thresholds: Iterable[float],
    max_false_positives: int,
    baseline_threshold: float = 0.14,
    attention_threshold: float = 0.36,
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    """锁定已接受的两级策略，只校准新增阶段。"""
    rows = list(rows)
    candidates: list[dict[str, int | float]] = []
    best_tp = -1
    for multiscale_threshold in thresholds:
        metrics = validation_metrics(
            rows,
            baseline_threshold,
            attention_threshold,
            multiscale_threshold,
        )
        if int(metrics["fp"]) > max_false_positives:
            continue
        candidate = {
            "baseline_threshold": baseline_threshold,
            "attention_threshold": attention_threshold,
            "multiscale_threshold": multiscale_threshold,
            **metrics,
        }
        tp = int(metrics["tp"])
        if tp > best_tp:
            best_tp = tp
            candidates = [candidate]
        elif tp == best_tp:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("no rescue threshold satisfies the FP constraint")

    # 保守的并列规则：在 TP 和 FP 相同后，优先选择新增阶段的最高阈值。
    # 此排序特意不使用 GW5。
    candidates.sort(
        key=lambda item: (
            int(item["fp"]),
            -float(item["multiscale_threshold"]),
        )
    )
    return candidates[0], candidates


def read_gw5_rows(path: Path) -> list[ScoreRow]:
    required = {
        "baseline@512",
        "baseline@640",
        "baseline@768",
        "attention@640",
    }
    scores: dict[str, dict[str, tuple[float, float, str]]] = {
        name: {} for name in required
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            experiment = row["experiment"]
            if experiment not in required:
                continue
            scores[experiment][row["image"]] = (
                float(row["max_confidence"]),
                float(row["inference_ms"]),
                row["event"],
            )
    image_sets = [set(values) for values in scores.values()]
    if not image_sets or len({frozenset(values) for values in image_sets}) != 1:
        raise ValueError("GW5 prediction members do not contain identical images")

    rows: list[ScoreRow] = []
    for image in sorted(image_sets[0]):
        event = scores["baseline@640"][image][2]
        if not EVENT_PATTERN.fullmatch(event):
            raise ValueError(f"invalid GW5 event name for {image}: {event}")
        rows.append(
            ScoreRow(
                image=image,
                truth_has_chirp=None,
                baseline_640=scores["baseline@640"][image][0],
                attention_640=scores["attention@640"][image][0],
                baseline_512=scores["baseline@512"][image][0],
                baseline_768=scores["baseline@768"][image][0],
                event=event,
                inference_ms_baseline_640=scores["baseline@640"][image][1],
                inference_ms_attention_640=scores["attention@640"][image][1],
                inference_ms_baseline_512=scores["baseline@512"][image][1],
                inference_ms_baseline_768=scores["baseline@768"][image][1],
            )
        )
    return rows


def read_eligible_events(path: Path) -> set[str]:
    events: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["纳入召回统计"].strip().lower() == "true":
                events.add(row["事件"])
    if not events:
        raise ValueError(f"catalogue contains no eligible events: {path}")
    return events


def gw5_metrics(
    rows: Iterable[ScoreRow],
    eligible_events: set[str],
    baseline_threshold: float,
    attention_threshold: float,
    multiscale_threshold: float,
    *,
    enable_multiscale: bool = True,
) -> dict[str, int | float | list[str]]:
    rows = list(rows)
    recalled: set[str] = set()
    hit_images = 0
    attention_invocations = multiscale_invocations = 0
    primary_hits = attention_rescues = multiscale_rescues = 0
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
            if enable_multiscale and row.attention_640 < attention_threshold:
                multiscale_invocations += 1
                cascade_inference_ms += (
                    row.inference_ms_baseline_512 + row.inference_ms_baseline_768
                )
        decision, stage = policy_decision(
            row,
            baseline_threshold,
            attention_threshold,
            multiscale_threshold,
            enable_multiscale=enable_multiscale,
        )
        if decision:
            hit_images += 1
            if row.event in eligible_events:
                recalled.add(str(row.event))
        if stage == "baseline_640":
            primary_hits += 1
        elif stage == "attention_640":
            attention_rescues += 1
        elif stage == "baseline_multiscale":
            multiscale_rescues += 1
    missed = sorted(eligible_events - recalled)
    return {
        "eligible_events": len(eligible_events),
        "recalled_events": len(recalled),
        "event_recall": len(recalled) / len(eligible_events),
        "missed_events": missed,
        "detector_images": len(rows),
        "detector_image_hits": hit_images,
        "detector_image_hit_rate": hit_images / len(rows),
        "attention_invocations": attention_invocations,
        "multiscale_invocations": multiscale_invocations,
        "primary_hits": primary_hits,
        "attention_rescues": attention_rescues,
        "multiscale_rescues": multiscale_rescues,
        "estimated_cascade_inference_ms": cascade_inference_ms,
        "estimated_full_inference_ms": full_inference_ms,
        "estimated_inference_saved_fraction": (
            1.0 - cascade_inference_ms / full_inference_ms
            if full_inference_ms
            else 0.0
        ),
    }


def write_candidates(
    path: Path, candidates: list[dict[str, int | float]], limit: int = 200
) -> None:
    rows = candidates[:limit]
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
    validation_rows = read_validation_rows(
        args.validation_640,
        args.validation_512,
        args.validation_768,
    )
    thresholds = threshold_grid(args.grid_start, args.grid_stop)
    locked_baseline_threshold = 0.14
    locked_attention_threshold = 0.36
    selected, candidates = choose_rescue_policy(
        validation_rows,
        thresholds,
        args.max_false_positives,
        baseline_threshold=locked_baseline_threshold,
        attention_threshold=locked_attention_threshold,
    )

    gw5_rows = read_gw5_rows(args.gw5_predictions)
    eligible_events = read_eligible_events(args.catalogue)
    gw5 = gw5_metrics(
        gw5_rows,
        eligible_events,
        float(selected["baseline_threshold"]),
        float(selected["attention_threshold"]),
        float(selected["multiscale_threshold"]),
    )
    previous_validation = validation_metrics(
        validation_rows,
        locked_baseline_threshold,
        locked_attention_threshold,
        1.01,
        enable_multiscale=False,
    )
    previous_gw5 = gw5_metrics(
        gw5_rows,
        eligible_events,
        locked_baseline_threshold,
        locked_attention_threshold,
        1.01,
        enable_multiscale=False,
    )
    accepted = (
        int(selected["tp"]) > int(previous_validation["tp"])
        and int(selected["fp"]) <= args.max_false_positives
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_candidates(args.output_dir / "top_validation_candidates.csv", candidates)
    payload = {
        "selection_scope": "fixed validation only; GW5 excluded from threshold selection",
        "search": {
            "grid_start": args.grid_start / 100,
            "grid_stop": args.grid_stop / 100,
            "grid_step": 0.01,
            "locked_baseline_threshold": locked_baseline_threshold,
            "locked_attention_threshold": locked_attention_threshold,
            "combinations": len(thresholds),
            "max_false_positives": args.max_false_positives,
            "max_tp_candidate_count": len(candidates),
            "saved_candidate_limit": min(200, len(candidates)),
        },
        "selected_policy": {
            "order": [
                "baseline@640",
                "attention@640",
                "baseline@512+baseline@768",
            ],
            "accepted": accepted,
            "decision": (
                "accept: validation TP improved without exceeding FP constraint"
                if accepted
                else "reject: no validation TP improvement under the FP constraint"
            ),
            **selected,
        },
        "previous_policy": {
            "baseline_threshold": locked_baseline_threshold,
            "attention_threshold": locked_attention_threshold,
            "multiscale_threshold": None,
            "validation": previous_validation,
            "gw5": previous_gw5,
        },
        "gw5_blind_audit": gw5,
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
