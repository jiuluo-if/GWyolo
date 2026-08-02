"""使用同一个固定数据划分验证多个分割检查点。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from benchmark_gw5 import ModelSpec, parse_model_spec


def validate_model(
    spec: ModelSpec,
    data: Path,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
) -> dict[str, object]:
    from ultralytics import YOLO

    model = YOLO(spec.weights)
    started = time.perf_counter()
    metrics = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        plots=False,
        save_json=False,
        project=str(project),
        name=spec.name,
        exist_ok=True,
        verbose=False,
    )
    row: dict[str, object] = {
        "model": spec.name,
        "weights": str(spec.weights),
        "imgsz": imgsz,
        "batch": batch,
        "wall_seconds": time.perf_counter() - started,
        "preprocess_ms": float(metrics.speed.get("preprocess", 0.0)),
        "inference_ms": float(metrics.speed.get("inference", 0.0)),
        "postprocess_ms": float(metrics.speed.get("postprocess", 0.0)),
    }
    row.update({key: float(value) for key, value in metrics.results_dict.items()})
    for class_id, class_name in metrics.names.items():
        box = metrics.box.class_result(class_id)
        mask = metrics.seg.class_result(class_id)
        for prefix, values in (("box", box), ("mask", mask)):
            row[f"{class_name}_{prefix}_precision"] = float(values[0])
            row[f"{class_name}_{prefix}_recall"] = float(values[1])
            row[f"{class_name}_{prefix}_map50"] = float(values[2])
            row[f"{class_name}_{prefix}_map50_95"] = float(values[3])
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model_spec, required=True)
    parser.add_argument("--data", type=Path, default=Path("gw_data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        validate_model(
            spec=spec,
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=Path("runs/optimization_validation"),
        )
        for spec in args.model
    ]
    with (args.output_dir / "validation_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
