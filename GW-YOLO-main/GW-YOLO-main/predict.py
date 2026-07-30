"""逐图执行 chirp 推理，并生成可用于 GW5 审计的完整标签集。

不接受路径列表作为一个 Ultralytics 批次；模型实例会复用，但每次只预测一张图像，
以避免 Attention Residual 的显存峰值和已知的 micro-batch 置信度塌缩问题。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_paths(source: Path) -> list[Path]:
    """返回单张图像或目录中的全部支持图像，并稳定排序。"""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图像格式：{source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    paths = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"未找到支持的图像：{source}")
    return paths


def prepare_output(output: Path, overwrite: bool) -> tuple[Path, Path]:
    """创建空的输出目录；默认拒绝覆盖已有审计结果。"""
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，拒绝覆盖：{output}；如确认覆盖请使用 --overwrite")
        shutil.rmtree(output)
    labels_dir = output / "labels"
    images_dir = output / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir()
    return labels_dir, images_dir


def write_labels(label_path: Path, result) -> int:
    """写入全部类别的 YOLO 检测框，并为无检出图像创建空标签文件。"""
    rows: list[str] = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        class_ids = result.boxes.cls.detach().cpu().tolist()
        height, width = result.orig_shape
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(float, box)
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            rows.append(
                f"{int(class_id)} {x_center:.6f} {y_center:.6f} "
                f"{box_width:.6f} {box_height:.6f} {float(confidence):.6f}"
            )
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="模型权重路径")
    parser.add_argument("--source", type=Path, required=True, help="单张图像或图像目录")
    parser.add_argument("--output", type=Path, required=True, help="新的审计输出目录")
    parser.add_argument("--conf", type=float, default=0.20, help="推理置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="输入边长")
    parser.add_argument("--device", default="", help="Ultralytics 设备参数")
    parser.add_argument("--half", action="store_true", help="启用 FP16；仅限已验证的单图模式")
    parser.add_argument("--overwrite", action="store_true", help="删除已有输出目录后重新生成")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf 必须在 [0, 1] 内")

    from ultralytics import YOLO

    images = image_paths(args.source)
    labels_dir, images_dir = prepare_output(args.output, args.overwrite)
    model = YOLO(args.weights)
    detection_count = 0
    for image in images:
        result = model.predict(
            source=str(image), imgsz=args.imgsz, conf=args.conf, device=args.device,
            half=args.half, batch=1, save=False, verbose=False,
        )[0]
        detection_count += write_labels(labels_dir / f"{image.stem}.txt", result)
        rendered = result.plot()
        result_image = images_dir / image.name
        import cv2
        cv2.imwrite(str(result_image), rendered)

    manifest = {
        "权重": str(args.weights.resolve()), "输入": str(args.source.resolve()),
        "图像数": len(images), "检测框数": detection_count, "置信度阈值": args.conf,
        "图像尺寸": args.imgsz, "FP16": args.half, "推理模式": "逐图，复用同一模型实例",
    }
    (args.output / "prediction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
