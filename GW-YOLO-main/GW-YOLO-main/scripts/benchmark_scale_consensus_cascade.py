"""Benchmark the accepted short-circuit scale-consensus cascade.

Execution order:

1. baseline@640 on every image;
2. Attention Residual@640 only after a primary miss;
3. baseline@512 only after both earlier stages miss;
4. baseline@768 only when baseline@512 passes its consensus threshold.

The last step is a logical short circuit: it preserves the calibrated
``baseline@512 AND baseline@768`` decision while avoiding unnecessary 768
inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Iterable


EVENT_PATTERN = re.compile(r"(GW\d{6}_\d{6})")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def list_images(source: Path) -> list[Path]:
    images = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"source contains no supported images: {source}")
    return images


def short_circuit_decision(
    baseline_640: float,
    attention_640: float | None,
    baseline_512: float | None,
    baseline_768: float | None,
    *,
    baseline_threshold: float,
    attention_threshold: float,
    threshold_512: float,
    threshold_768: float,
) -> tuple[bool, str]:
    if baseline_640 >= baseline_threshold:
        return True, "baseline_640"
    if attention_640 is not None and attention_640 >= attention_threshold:
        return True, "attention_640"
    if (
        baseline_512 is not None
        and baseline_512 >= threshold_512
        and baseline_768 is not None
        and baseline_768 >= threshold_768
    ):
        return True, "scale_consensus"
    return False, "miss"


def read_truth(labels_dir: Path, images: Iterable[Path]) -> dict[str, bool]:
    truth: dict[str, bool] = {}
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        has_chirp = False
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields and int(float(fields[0])) == 0:
                    has_chirp = True
                    break
        truth[image.stem] = has_chirp
    return truth


def read_eligible_events(path: Path) -> set[str]:
    events: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["纳入召回统计"].strip().lower() == "true":
                events.add(row["事件"])
    if not events:
        raise ValueError(f"catalogue contains no eligible events: {path}")
    return events


def run_stage(
    model: object,
    images: Iterable[Path],
    *,
    imgsz: int,
    conf_floor: float,
    device: str,
    half: bool,
) -> tuple[dict[str, float], float, float]:
    images = list(images)
    scores: dict[str, float] = {}
    inference_ms = 0.0
    started = time.perf_counter()
    for image in images:
        result = model.predict(
            source=str(image),
            imgsz=imgsz,
            conf=conf_floor,
            classes=[0],
            batch=1,
            device=device,
            half=half,
            save=False,
            verbose=False,
        )[0]
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
        scores[image.stem] = score
        inference_ms += float(result.speed.get("inference", 0.0))
    return scores, time.perf_counter() - started, inference_ms


def classification_metrics(
    decisions: dict[str, bool], truth: dict[str, bool]
) -> dict[str, int | float]:
    tp = sum(truth[name] and decisions[name] for name in truth)
    fp = sum(not truth[name] and decisions[name] for name in truth)
    fn = sum(truth[name] and not decisions[name] for name in truth)
    tn = sum(not truth[name] and not decisions[name] for name in truth)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def event_metrics(
    decisions: dict[str, bool], eligible_events: set[str]
) -> dict[str, int | float | list[str]]:
    recalled: set[str] = set()
    hit_images = 0
    for image, decision in decisions.items():
        if not decision:
            continue
        hit_images += 1
        match = EVENT_PATTERN.search(image)
        if match and match.group(1) in eligible_events:
            recalled.add(match.group(1))
    return {
        "eligible_events": len(eligible_events),
        "recalled_events": len(recalled),
        "event_recall": len(recalled) / len(eligible_events),
        "missed_events": sorted(eligible_events - recalled),
        "detector_images": len(decisions),
        "detector_image_hits": hit_images,
        "detector_image_hit_rate": hit_images / len(decisions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--baseline-threshold", type=float, default=0.14)
    parser.add_argument("--attention-threshold", type=float, default=0.36)
    parser.add_argument("--threshold-512", type=float, default=0.15)
    parser.add_argument("--threshold-768", type=float, default=0.07)
    parser.add_argument("--conf-floor", type=float, default=0.01)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.baseline.is_file() or not args.attention.is_file():
        raise FileNotFoundError("baseline and attention weights must exist")
    images = list_images(args.source)

    from ultralytics import YOLO

    baseline_model = YOLO(args.baseline)
    attention_model = YOLO(args.attention)
    baseline_640, wall_640, ms_640 = run_stage(
        baseline_model,
        images,
        imgsz=640,
        conf_floor=args.conf_floor,
        device=args.device,
        half=args.half,
    )
    attention_images = [
        image
        for image in images
        if baseline_640[image.stem] < args.baseline_threshold
    ]
    attention_640, wall_attention, ms_attention = run_stage(
        attention_model,
        attention_images,
        imgsz=640,
        conf_floor=args.conf_floor,
        device=args.device,
        half=args.half,
    )
    scale_512_images = [
        image
        for image in attention_images
        if attention_640[image.stem] < args.attention_threshold
    ]
    baseline_512, wall_512, ms_512 = run_stage(
        baseline_model,
        scale_512_images,
        imgsz=512,
        conf_floor=args.conf_floor,
        device=args.device,
        half=args.half,
    )
    scale_768_images = [
        image
        for image in scale_512_images
        if baseline_512[image.stem] >= args.threshold_512
    ]
    baseline_768, wall_768, ms_768 = run_stage(
        baseline_model,
        scale_768_images,
        imgsz=768,
        conf_floor=args.conf_floor,
        device=args.device,
        half=args.half,
    )

    decisions: dict[str, bool] = {}
    stages: dict[str, str] = {}
    for image in images:
        name = image.stem
        decision, stage = short_circuit_decision(
            baseline_640[name],
            attention_640.get(name),
            baseline_512.get(name),
            baseline_768.get(name),
            baseline_threshold=args.baseline_threshold,
            attention_threshold=args.attention_threshold,
            threshold_512=args.threshold_512,
            threshold_768=args.threshold_768,
        )
        decisions[name] = decision
        stages[name] = stage

    metrics: dict[str, object] = {}
    if args.labels_dir:
        metrics["validation"] = classification_metrics(
            decisions, read_truth(args.labels_dir, images)
        )
    if args.catalogue:
        metrics["gw5"] = event_metrics(
            decisions, read_eligible_events(args.catalogue)
        )
    timings = {
        "baseline_640_wall_seconds": wall_640,
        "attention_640_wall_seconds": wall_attention,
        "baseline_512_wall_seconds": wall_512,
        "baseline_768_wall_seconds": wall_768,
        "total_wall_seconds": wall_640 + wall_attention + wall_512 + wall_768,
        "summed_model_inference_ms": ms_640 + ms_attention + ms_512 + ms_768,
        "attention_invocations": len(attention_images),
        "baseline_512_invocations": len(scale_512_images),
        "baseline_768_invocations": len(scale_768_images),
        "baseline_768_short_circuit_fraction": (
            1.0 - len(scale_768_images) / len(scale_512_images)
            if scale_512_images
            else 1.0
        ),
    }
    stage_counts = {
        stage: sum(value == stage for value in stages.values())
        for stage in ("baseline_640", "attention_640", "scale_consensus", "miss")
    }
    payload = {
        "metadata": {
            "baseline": str(args.baseline),
            "attention": str(args.attention),
            "source": str(args.source),
            "images": len(images),
            "conf_floor": args.conf_floor,
            "half": args.half,
            "device": args.device,
            "policy": {
                "baseline_threshold": args.baseline_threshold,
                "attention_threshold": args.attention_threshold,
                "threshold_512": args.threshold_512,
                "threshold_768": args.threshold_768,
            },
        },
        "timings": timings,
        "stage_counts": stage_counts,
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for image in images:
        name = image.stem
        rows.append(
            {
                "image": name,
                "baseline_640": baseline_640[name],
                "attention_640": attention_640.get(name, ""),
                "baseline_512": baseline_512.get(name, ""),
                "baseline_768": baseline_768.get(name, ""),
                "decision": decisions[name],
                "decision_stage": stages[name],
            }
        )
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
