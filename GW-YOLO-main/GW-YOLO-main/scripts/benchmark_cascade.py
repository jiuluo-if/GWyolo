"""评测主模型到次模型的 chirp 推理级联。

主模型处理每张图像；仅当主模型的 chirp 置信度低于工作阈值时才调用次模型。
在该阈值下，级联与完整 OR 融合拥有相同的二元决策，同时避免对已被主模型接受的
图像执行次模型推理。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EVENT_PATTERN = re.compile(r"(GW\d{6}_\d{6})")
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)


@dataclass(frozen=True)
class TimedScores:
    scores: dict[str, float]
    wall_seconds: float
    mean_inference_ms: float


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    paths = [
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"no supported images found in {source}")
    return sorted(paths)


def read_truth(labels_dir: Path, images: Iterable[Path]) -> dict[str, bool]:
    truth: dict[str, bool] = {}
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        has_chirp = False
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields and fields[0] == "0":
                    has_chirp = True
                    break
        truth[image.stem] = has_chirp
    return truth


def classification_metrics(
    scores: dict[str, float],
    truth: dict[str, bool],
    threshold: float,
) -> dict[str, int | float]:
    tp = fp = fn = tn = 0
    for image, expected in truth.items():
        predicted = scores.get(image, 0.0) >= threshold
        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and expected:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "accuracy": (tp + tn) / len(truth) if truth else 0.0,
    }


def classification_metrics_from_decisions(
    decisions: dict[str, bool],
    truth: dict[str, bool],
) -> dict[str, int | float]:
    """根据已校准的二元决策计算图像级指标。"""
    return classification_metrics(
        {image: float(decisions.get(image, False)) for image in truth},
        truth,
        0.5,
    )


def event_metrics(
    scores: dict[str, float],
    catalogue: dict[str, bool],
    threshold: float,
) -> dict[str, int | float]:
    event_max = {event: 0.0 for event in catalogue}
    image_hits = 0
    for image, score in scores.items():
        match = EVENT_PATTERN.search(image)
        if match:
            event = match.group(1)
            event_max[event] = max(event_max.get(event, 0.0), score)
        if score >= threshold:
            image_hits += 1
    eligible = {event for event, is_eligible in catalogue.items() if is_eligible}
    recalled = sum(event_max.get(event, 0.0) >= threshold for event in eligible)
    return {
        "eligible_events": len(eligible),
        "recalled_events": recalled,
        "event_recall": recalled / len(eligible) if eligible else 0.0,
        "detector_images": len(scores),
        "detector_image_hits": image_hits,
        "detector_image_hit_rate": image_hits / len(scores) if scores else 0.0,
    }


def event_metrics_from_decisions(
    decisions: dict[str, bool],
    catalogue: dict[str, bool],
) -> dict[str, int | float]:
    """在保持任一探测器命中规则的前提下计算事件召回率。"""
    return event_metrics(
        {image: float(decision) for image, decision in decisions.items()},
        catalogue,
        0.5,
    )


def read_catalogue(path: Path) -> dict[str, bool]:
    catalogue: dict[str, bool] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            catalogue[row["事件"]] = row["纳入召回统计"].strip().lower() == "true"
    return catalogue


def merge_scores(
    primary: dict[str, float], secondary: dict[str, float]
) -> dict[str, float]:
    return {
        image: max(score, secondary.get(image, 0.0))
        for image, score in primary.items()
    }


def asymmetric_decisions(
    primary: dict[str, float],
    secondary: dict[str, float],
    primary_threshold: float,
    secondary_threshold: float,
) -> dict[str, bool]:
    """使用各自独立校准的置信度阈值融合两个模型。"""
    return {
        image: score >= primary_threshold
        or secondary.get(image, 0.0) >= secondary_threshold
        for image, score in primary.items()
    }


def secondary_sources(
    images: Sequence[Path], primary: dict[str, float], threshold: float
) -> list[Path]:
    return [image for image in images if primary.get(image.stem, 0.0) < threshold]


def chunks(items: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict_scores(
    model,
    sources: Sequence[Path],
    imgsz: int,
    conf_floor: float,
    device: str,
    chunk_size: int,
    half: bool,
):
    if not sources:
        return TimedScores({}, 0.0, 0.0)
    started = time.perf_counter()
    scores: dict[str, float] = {}
    inference_times: list[float] = []
    # Ultralytics 会把 Python 图像路径列表视为一个内存批次。显式分块可避免
    # 意外形成全数据集批次并触发注意力模块 OOM，同时仍允许受控微批处理。
    for source_chunk in chunks(sources, chunk_size):
        source = (
            str(source_chunk[0])
            if len(source_chunk) == 1
            else [str(path) for path in source_chunk]
        )
        results = model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf_floor,
            classes=[0],
            batch=chunk_size,
            device=device,
            half=half,
            stream=False,
            save=False,
            verbose=False,
        )
        for result in results:
            score = 0.0
            if result.boxes is not None and len(result.boxes):
                class_ids = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                score = max(
                    (
                        float(confidence)
                        for class_id, confidence in zip(class_ids, confidences)
                        if int(class_id) == 0
                    ),
                    default=0.0,
                )
            scores[Path(result.path).stem] = score
            inference_times.append(float(result.speed.get("inference", 0.0)))
    return TimedScores(
        scores=scores,
        wall_seconds=time.perf_counter() - started,
        mean_inference_ms=sum(inference_times) / len(inference_times),
    )


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(sorted({float(item) for item in value.split(",") if item.strip()}))
    if not thresholds or any(item < 0.0 or item > 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be comma-separated values in [0, 1]")
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-floor", type=float, default=0.01)
    parser.add_argument("--operating-threshold", type=float, default=0.25)
    parser.add_argument(
        "--secondary-threshold",
        type=float,
        help="attention-model threshold; defaults to --operating-threshold",
    )
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    from ultralytics import YOLO

    args = build_parser().parse_args()
    secondary_threshold = (
        args.secondary_threshold
        if args.secondary_threshold is not None
        else args.operating_threshold
    )
    if not 0.0 <= secondary_threshold <= 1.0:
        raise ValueError("secondary threshold must be in [0, 1]")
    images = image_paths(args.source)
    primary_model = YOLO(args.primary)
    secondary_model = YOLO(args.secondary)

    primary = predict_scores(
        primary_model,
        images,
        args.imgsz,
        args.conf_floor,
        args.device,
        args.chunk_size,
        args.half,
    )
    secondary_full = predict_scores(
        secondary_model,
        images,
        args.imgsz,
        args.conf_floor,
        args.device,
        args.chunk_size,
        args.half,
    )
    cascade_inputs = secondary_sources(
        images, primary.scores, args.operating_threshold
    )
    secondary_cascade = predict_scores(
        secondary_model,
        cascade_inputs,
        args.imgsz,
        args.conf_floor,
        args.device,
        args.chunk_size,
        args.half,
    )

    full_ensemble = merge_scores(primary.scores, secondary_full.scores)
    cascade = merge_scores(primary.scores, secondary_cascade.scores)
    full_decisions = asymmetric_decisions(
        primary.scores,
        secondary_full.scores,
        args.operating_threshold,
        secondary_threshold,
    )
    cascade_decisions = asymmetric_decisions(
        primary.scores,
        secondary_cascade.scores,
        args.operating_threshold,
        secondary_threshold,
    )
    if full_decisions != cascade_decisions:
        raise AssertionError("cascade decisions differ from full OR ensemble")

    timings = {
        "primary_wall_seconds": primary.wall_seconds,
        "secondary_full_wall_seconds": secondary_full.wall_seconds,
        "full_ensemble_wall_seconds": primary.wall_seconds + secondary_full.wall_seconds,
        "secondary_cascade_wall_seconds": secondary_cascade.wall_seconds,
        "cascade_wall_seconds": primary.wall_seconds + secondary_cascade.wall_seconds,
        "secondary_invocations": len(cascade_inputs),
        "secondary_invocation_rate": len(cascade_inputs) / len(images),
        "cascade_time_saved_fraction": 1.0
        - (primary.wall_seconds + secondary_cascade.wall_seconds)
        / (primary.wall_seconds + secondary_full.wall_seconds),
    }

    strategies = {
        "primary": primary.scores,
        "secondary": secondary_full.scores,
        "full_ensemble": full_ensemble,
    }
    rows: list[dict[str, object]] = []
    truth = read_truth(args.labels_dir, images) if args.labels_dir else None
    catalogue = read_catalogue(args.catalogue) if args.catalogue else None
    for strategy, scores in strategies.items():
        for threshold in args.thresholds:
            row: dict[str, object] = {
                "strategy": strategy,
                "threshold": threshold,
                "secondary_threshold": threshold,
            }
            if truth is not None:
                row.update(classification_metrics(scores, truth, threshold))
            if catalogue is not None:
                row.update(event_metrics(scores, catalogue, threshold))
            rows.append(row)

    cascade_row: dict[str, object] = {
        "strategy": "cascade",
        "threshold": args.operating_threshold,
        "secondary_threshold": secondary_threshold,
    }
    if truth is not None:
        cascade_row.update(
            classification_metrics_from_decisions(cascade_decisions, truth)
        )
    if catalogue is not None:
        cascade_row.update(event_metrics_from_decisions(cascade_decisions, catalogue))
    rows.append(cascade_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "threshold_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    prediction_rows = []
    for image in images:
        name = image.stem
        prediction_rows.append(
            {
                "image": name,
                "truth_has_chirp": truth.get(name, "") if truth is not None else "",
                "primary_confidence": primary.scores.get(name, 0.0),
                "secondary_confidence": secondary_full.scores.get(name, 0.0),
                "secondary_invoked": name in secondary_cascade.scores,
                "cascade_confidence": cascade.get(name, 0.0),
                "cascade_decision": cascade_decisions.get(name, False),
            }
        )
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    payload = {
        "metadata": {
            "primary": str(args.primary),
            "secondary": str(args.secondary),
            "source": str(args.source),
            "labels_dir": str(args.labels_dir) if args.labels_dir else None,
            "catalogue": str(args.catalogue) if args.catalogue else None,
            "imgsz": args.imgsz,
            "conf_floor": args.conf_floor,
            "operating_threshold": args.operating_threshold,
            "secondary_threshold": secondary_threshold,
            "chunk_size": args.chunk_size,
            "half": args.half,
            "images": len(images),
        },
        "timings": timings,
        "threshold_metrics": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"timings": timings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
