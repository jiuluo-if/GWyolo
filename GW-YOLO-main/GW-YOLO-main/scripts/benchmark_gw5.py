"""Benchmark GW5 event recall across models, image sizes, and late fusions.

The benchmark intentionally reports detector-image hit rate as a review-workload
proxy. GW5 is a positive-event catalogue, so it cannot measure a true
false-positive rate without a separate negative dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence


EVENT_PATTERN = re.compile(r"(GW\d{6}_\d{6})")
DEFAULT_THRESHOLDS = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    weights: Path


@dataclass(frozen=True)
class Prediction:
    experiment: str
    model: str
    imgsz: int
    image: str
    event: str
    confidences: tuple[float, ...]
    inference_ms: float

    @property
    def max_confidence(self) -> float:
        return max(self.confidences, default=0.0)


def parse_model_spec(value: str) -> ModelSpec:
    """Parse NAME=WEIGHTS_PATH without restricting '=' in the path."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use NAME=WEIGHTS_PATH")
    name, weights = value.split("=", 1)
    if not name.strip() or not weights.strip():
        raise argparse.ArgumentTypeError("model name and weights path must be non-empty")
    return ModelSpec(name=name.strip(), weights=Path(weights.strip()))


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(sorted({float(item) for item in value.split(",") if item.strip()}))
    if not thresholds or any(item < 0.0 or item > 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in [0, 1]")
    return thresholds


def read_catalogue(path: Path) -> dict[str, dict[str, str | bool]]:
    """Read the stable event/eligibility fields from an earlier PDF audit CSV."""
    events: dict[str, dict[str, str | bool]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            event = row["事件"]
            events[event] = {
                "eligible": row["纳入召回统计"].strip().lower() == "true",
                "network_snr": row.get("网络 SNR", ""),
                "mass_1": row.get("质量 1 (太阳质量)", ""),
                "mass_2": row.get("质量 2 (太阳质量)", ""),
            }
    if not events:
        raise ValueError(f"catalogue contains no events: {path}")
    return events


def run_inference(
    spec: ModelSpec,
    imgsz: int,
    source: Path,
    conf_floor: float,
    device: str,
) -> tuple[list[Prediction], float]:
    """Run one low-threshold pass so all requested thresholds share predictions."""
    from ultralytics import YOLO

    if not spec.weights.is_file():
        raise FileNotFoundError(spec.weights)
    experiment = f"{spec.name}@{imgsz}"
    model = YOLO(spec.weights)
    started = time.perf_counter()
    results = model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf_floor,
        classes=[0],
        batch=1,
        device=device,
        stream=True,
        save=False,
        verbose=False,
    )
    predictions: list[Prediction] = []
    for result in results:
        image = Path(result.path).stem
        event_match = EVENT_PATTERN.search(image)
        if not event_match:
            continue
        confidences: tuple[float, ...] = ()
        if result.boxes is not None and len(result.boxes):
            class_ids = result.boxes.cls.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            confidences = tuple(
                float(score) for class_id, score in zip(class_ids, scores) if int(class_id) == 0
            )
        predictions.append(
            Prediction(
                experiment=experiment,
                model=spec.name,
                imgsz=imgsz,
                image=image,
                event=event_match.group(1),
                confidences=confidences,
                inference_ms=float(result.speed.get("inference", 0.0)),
            )
        )
    return predictions, time.perf_counter() - started


def fuse_predictions(
    experiment: str,
    members: Sequence[tuple[Sequence[Prediction], float]],
) -> tuple[list[Prediction], float]:
    """Late-fuse member decisions by retaining their maximum chirp confidence."""
    by_image: dict[str, list[Prediction]] = defaultdict(list)
    for predictions, _ in members:
        for prediction in predictions:
            by_image[prediction.image].append(prediction)

    fused: list[Prediction] = []
    for image, candidates in sorted(by_image.items()):
        best = max(candidates, key=lambda item: item.max_confidence)
        fused.append(
            Prediction(
                experiment=experiment,
                model="+".join(sorted({item.model for item in candidates})),
                imgsz=0,
                image=image,
                event=best.event,
                confidences=(best.max_confidence,) if best.max_confidence else (),
                inference_ms=sum(item.inference_ms for item in candidates),
            )
        )
    return fused, sum(wall_seconds for _, wall_seconds in members)


def summarize_predictions(
    predictions: Sequence[Prediction],
    catalogue: dict[str, dict[str, str | bool]],
    thresholds: Iterable[float],
    wall_seconds: float,
    kind: str,
    members: str,
) -> list[dict[str, str | int | float]]:
    eligible_events = {event for event, row in catalogue.items() if row["eligible"]}
    excluded_events = set(catalogue) - eligible_events
    event_max: dict[str, float] = defaultdict(float)
    for prediction in predictions:
        event_max[prediction.event] = max(event_max[prediction.event], prediction.max_confidence)

    rows: list[dict[str, str | int | float]] = []
    for threshold in thresholds:
        hit_images = {
            item.image for item in predictions if item.max_confidence >= threshold
        }
        eligible_hit_images = {
            item.image
            for item in predictions
            if item.event in eligible_events and item.max_confidence >= threshold
        }
        recalled = {event for event in eligible_events if event_max[event] >= threshold}
        excluded_hits = {event for event in excluded_events if event_max[event] >= threshold}
        inference_values = [item.inference_ms for item in predictions]
        rows.append(
            {
                "experiment": predictions[0].experiment if predictions else members,
                "kind": kind,
                "members": members,
                "threshold": threshold,
                "eligible_events": len(eligible_events),
                "recalled_events": len(recalled),
                "event_recall": len(recalled) / len(eligible_events),
                "excluded_event_hits": len(excluded_hits),
                "detector_images": len(predictions),
                "detector_image_hits": len(hit_images),
                "eligible_detector_image_hits": len(eligible_hit_images),
                "detector_image_hit_rate": len(hit_images) / len(predictions),
                "wall_seconds": wall_seconds,
                "mean_inference_ms": fmean(inference_values) if inference_values else 0.0,
            }
        )
    return rows


def write_outputs(
    output_dir: Path,
    raw_runs: dict[str, tuple[list[Prediction], float, str, str]],
    summary_rows: list[dict[str, str | int | float]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for predictions, _, _, _ in raw_runs.values():
        for item in predictions:
            prediction_rows.append(
                {
                    "experiment": item.experiment,
                    "model": item.model,
                    "imgsz": item.imgsz,
                    "image": item.image,
                    "event": item.event,
                    "max_confidence": item.max_confidence,
                    "chirp_count_at_floor": len(item.confidences),
                    "confidences": ";".join(f"{value:.8f}" for value in item.confidences),
                    "inference_ms": item.inference_ms,
                }
            )

    def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "threshold_summary.csv", summary_rows)
    payload = {"metadata": metadata, "threshold_results": summary_rows}
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    threshold_025 = [
        row for row in summary_rows if abs(float(row["threshold"]) - 0.25) < 1e-9
    ]
    report_lines = [
        "# GW5 优化批次对比报告",
        "",
        "## 评估口径",
        "",
        "- 事件级召回：质量字段完整的事件中，任一探测器图像检出 chirp。",
        "- 探测器图像命中率：需要人工复核的工作量代理，不是真实误报率。",
        "- 所有阈值共用同一次低阈值推理结果，避免重复推理造成口径漂移。",
        "",
        "## 固定阈值 0.25",
        "",
        "| 实验 | 类型 | 召回 | 事件召回率 | 图像命中数 | 图像命中率 | 推理耗时(s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in threshold_025:
        report_lines.append(
            f"| {row['experiment']} | {row['kind']} | "
            f"{row['recalled_events']}/{row['eligible_events']} | "
            f"{float(row['event_recall']):.2%} | {row['detector_image_hits']} | "
            f"{float(row['detector_image_hit_rate']):.2%} | "
            f"{float(row['wall_seconds']):.2f} |"
        )
    report_lines.extend(
        [
            "",
            "## 结论约束",
            "",
            "GW5 目录只包含已知事件，缺少纯负样本。因此降低阈值或做并集融合带来的召回提升，",
            "必须与固定验证集精度以及独立负样本集联合判断后，才能称为可发布优化。",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--imgsz", action="append", type=int, required=True)
    parser.add_argument("--source", type=Path, default=Path("imgs/gw5.0"))
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=Path("docs/gw5_recall_details.csv"),
        help="Existing PDF audit CSV; only stable catalogue fields are read.",
    )
    parser.add_argument("--conf-floor", type=float, default=0.01)
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalogue = read_catalogue(args.catalogue)
    raw_runs: dict[str, tuple[list[Prediction], float, str, str]] = {}
    base_members: dict[tuple[str, int], tuple[list[Prediction], float]] = {}

    for spec in args.model:
        for imgsz in sorted(set(args.imgsz)):
            predictions, wall_seconds = run_inference(
                spec=spec,
                imgsz=imgsz,
                source=args.source,
                conf_floor=args.conf_floor,
                device=args.device,
            )
            key = f"{spec.name}@{imgsz}"
            base_members[(spec.name, imgsz)] = (predictions, wall_seconds)
            raw_runs[key] = (predictions, wall_seconds, "single", key)
            print(f"completed {key}: {len(predictions)} images in {wall_seconds:.2f}s")

    sizes = sorted(set(args.imgsz))
    if len(sizes) > 1:
        for spec in args.model:
            member_runs = [base_members[(spec.name, size)] for size in sizes]
            key = f"{spec.name}@multiscale"
            predictions, wall_seconds = fuse_predictions(key, member_runs)
            members = "+".join(f"{spec.name}@{size}" for size in sizes)
            raw_runs[key] = (predictions, wall_seconds, "multiscale", members)

    if len(args.model) > 1:
        for size in sizes:
            member_runs = [base_members[(spec.name, size)] for spec in args.model]
            key = f"ensemble@{size}"
            predictions, wall_seconds = fuse_predictions(key, member_runs)
            members = "+".join(f"{spec.name}@{size}" for spec in args.model)
            raw_runs[key] = (predictions, wall_seconds, "model_ensemble", members)
        if len(sizes) > 1:
            member_runs = list(base_members.values())
            key = "ensemble@multiscale"
            predictions, wall_seconds = fuse_predictions(key, member_runs)
            members = "+".join(
                f"{spec.name}@{size}" for spec in args.model for size in sizes
            )
            raw_runs[key] = (predictions, wall_seconds, "full_ensemble", members)

    summary_rows: list[dict[str, str | int | float]] = []
    for predictions, wall_seconds, kind, members in raw_runs.values():
        summary_rows.extend(
            summarize_predictions(
                predictions=predictions,
                catalogue=catalogue,
                thresholds=args.thresholds,
                wall_seconds=wall_seconds,
                kind=kind,
                members=members,
            )
        )

    metadata = {
        "source": str(args.source),
        "catalogue": str(args.catalogue),
        "conf_floor": args.conf_floor,
        "thresholds": args.thresholds,
        "models": [
            {"name": spec.name, "weights": str(spec.weights)} for spec in args.model
        ],
        "image_sizes": sizes,
        "device": args.device,
    }
    write_outputs(args.output_dir, raw_runs, summary_rows, metadata)
    print(f"report written to {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
