"""根据 PDF 事件目录计算 GW5.0 的事件级 chirp 召回率。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

EVENT_PATTERN = re.compile(
    r"GW\s*(?P<event>\d{6}_\d{6})\s+(?P<snr>[\d.]+)\s+(?P<mass1>--|[\d.]+)\s+(?P<mass2>--|[\d.]+)"
)
IMAGE_EVENT_PATTERN = re.compile(r"(GW\d{6}_\d{6})")
DETECTOR_PATTERN = re.compile(r"(?<![A-Z0-9])(H1|L1|V1)(?![A-Z0-9])", re.IGNORECASE)


def read_catalogue(pdf_path: Path) -> dict[str, dict[str, str | bool]]:
    """按事件名读取 PDF，并记录双星质量信息是否完整。"""
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "缺少 pypdf；请在包含 pypdf 的环境中运行事件审计。"
        ) from exc
    text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)
    events: dict[str, dict[str, str | bool]] = {}
    for match in EVENT_PATTERN.finditer(text):
        event_id = f"GW{match.group('event')}"
        mass1, mass2 = match.group("mass1"), match.group("mass2")
        events[event_id] = {
            "网络 SNR": match.group("snr"),
            "质量 1 (太阳质量)": mass1,
            "质量 2 (太阳质量)": mass2,
            "伴星质量完整": mass1 != "--" and mass2 != "--",
        }
    return events


def chirp_detections(labels_dir: Path) -> dict[str, list[dict[str, str | float]]]:
    """从 YOLO 标签中按事件汇总 chirp（类别 0）检出。"""
    detections: dict[str, list[dict[str, str | float]]] = defaultdict(list)
    for label_path in labels_dir.glob("*.txt"):
        match = IMAGE_EVENT_PATTERN.search(label_path.stem)
        if not match:
            continue
        for row in label_path.read_text(encoding="utf-8").splitlines():
            fields = row.split()
            if len(fields) >= 6 and fields[0] == "0":
                detections[match.group(1)].append(
                    {"图像": label_path.stem, "置信度": float(fields[5])}
                )
    return detections


def audit_input_coverage(images_dir: Path | None, labels_dir: Path) -> dict[str, object]:
    """审计图像、标签与 H1/L1/V1 覆盖，防止缺图悄然改变召回分母。"""
    if images_dir is None:
        return {"已检查": False, "原因": "未提供 --images，无法核对图像覆盖"}
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    )
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in labels_dir.glob("*.txt")}
    detector_coverage: dict[str, set[str]] = defaultdict(set)
    unparseable_images: list[str] = []
    for image in image_paths:
        event_match = IMAGE_EVENT_PATTERN.search(image.stem)
        detector_match = DETECTOR_PATTERN.search(image.stem)
        if event_match and detector_match:
            detector_coverage[event_match.group(1)].add(detector_match.group(1).upper())
        elif event_match:
            unparseable_images.append(image.name)
    expected = {"H1", "L1", "V1"}
    incomplete_events = {
        event: sorted(expected - detectors)
        for event, detectors in sorted(detector_coverage.items())
        if detectors != expected
    }
    return {
        "已检查": True,
        "输入图像数": len(image_paths),
        "标签文件数": len(label_stems),
        "缺少标签的图像数": len(image_stems - label_stems),
        "无对应输入图像的标签数": len(label_stems - image_stems),
        "探测器覆盖不完整事件": incomplete_events,
        "无法解析探测器名的事件图像": unparseable_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("事件数据.pdf"))
    parser.add_argument("--labels", type=Path, default=Path("runs/filtered/predict_5.0/labels"))
    parser.add_argument("--images", type=Path, help="本次推理的输入图像目录，用于覆盖审计")
    parser.add_argument("--output", type=Path, default=Path("docs/gw5_recall_details.csv"))
    parser.add_argument("--summary", type=Path, default=Path("docs/gw5_recall_summary.json"))
    args = parser.parse_args()

    catalogue = read_catalogue(args.pdf)
    if not args.labels.is_dir():
        raise FileNotFoundError(args.labels)
    detections = chirp_detections(args.labels)
    coverage = audit_input_coverage(args.images, args.labels)
    rows = []
    for event_id, record in sorted(catalogue.items()):
        eligible = bool(record["伴星质量完整"])
        event_detections = detections.get(event_id, [])
        rows.append(
            {
                "事件": event_id,
                **record,
                "纳入召回统计": eligible,
                "检出 chirp 的探测器图像": ";".join(item["图像"] for item in event_detections),
                "最高 chirp 置信度": max((item["置信度"] for item in event_detections), default=""),
                "任一探测器检出 chirp": bool(event_detections),
            }
        )

    eligible_rows = [row for row in rows if row["纳入召回统计"]]
    recalled_rows = [row for row in eligible_rows if row["任一探测器检出 chirp"]]
    summary = {
        "目录事件数": len(rows),
        "剔除的无伴星质量事件数": len(rows) - len(eligible_rows),
        "纳入统计事件数": len(eligible_rows),
        "召回事件数": len(recalled_rows),
        "召回率": len(recalled_rows) / len(eligible_rows) if eligible_rows else None,
        "统计口径": "质量 1 和质量 2 均非 -- 的事件纳入统计；H1、L1、V1 任一图像检出类别 0 chirp 即召回。",
        "输入覆盖审计": coverage,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
