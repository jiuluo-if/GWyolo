"""按成对 H1/L1 网络窗口审计时间隔离纯负集的 chirp 误报。

只有同时满足 GWTC 否决、可审计的开发时间隔离和双人独立批准的 H1/L1
四秒窗口才进入生产 FAR 分母。任一探测器命中即产生网络候选，连续命中窗口
合并为一次告警。审计集不得用于重新选择模型或阈值。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


WINDOW_SECONDS = 4.0
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
REQUIRED_MANIFEST_COLUMNS = {
    "window_id",
    "image",
    "image_hash",
    "start_utc",
    "end_utc",
    "detector",
    "gwtc_veto_passed",
    "time_isolated_verified",
    "human_review_status",
    "human_reviewers",
    "human_reviewed_at",
    "human_review_reason",
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def parse_utc(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include UTC offset: {value!r}")
    return parsed


def reviewer_names(value: str) -> set[str]:
    return {
        item.strip()
        for item in value.replace(",", ";").split(";")
        if item.strip()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_eligibility(row: dict[str, object]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(row["gwtc_veto_passed"]):
        reasons.append("gwtc_veto_not_passed")
    if not bool(row["time_isolated_verified"]):
        reasons.append("development_isolation_not_verified")
    if str(row["human_review_status"]).strip().lower() != "approved":
        reasons.append("human_review_not_approved")
    if len(reviewer_names(str(row["human_reviewers"]))) < 2:
        reasons.append("fewer_than_two_reviewers")
    if not str(row["human_reviewed_at"]).strip():
        reasons.append("missing_review_timestamp")
    if not str(row["human_review_reason"]).strip():
        reasons.append("missing_review_reason")
    return not reasons, reasons


def load_predictions(path: Path) -> dict[str, bool]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "decision"} <= set(reader.fieldnames):
            raise ValueError("predictions CSV must contain image and decision columns")
        result: dict[str, bool] = {}
        for row in reader:
            image = row["image"].strip()
            if not image or image in result:
                raise ValueError(f"prediction image is missing or duplicated: {image!r}")
            result[image] = parse_bool(row["decision"])
    if not result:
        raise ValueError("predictions CSV has no rows")
    return result


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_COLUMNS - fields
        if missing:
            raise ValueError(f"manifest missing required columns: {sorted(missing)}")
        rows: list[dict[str, object]] = []
        seen_windows: set[str] = set()
        seen_images: set[str] = set()
        for raw in reader:
            window_id = raw["window_id"].strip()
            image = raw["image"].strip()
            detector = raw["detector"].strip().upper()
            if not window_id or window_id in seen_windows:
                raise ValueError(f"window_id is missing or duplicated: {window_id!r}")
            if not image or image in seen_images:
                raise ValueError(f"image is missing or duplicated: {image!r}")
            if detector not in {"H1", "L1"}:
                raise ValueError(f"unsupported detector for paired audit: {detector!r}")
            image_hash = raw["image_hash"].strip()
            if not SHA256_PATTERN.fullmatch(image_hash):
                raise ValueError(
                    f"window {window_id} has no valid SHA-256 image hash"
                )
            start, end = parse_utc(raw["start_utc"]), parse_utc(raw["end_utc"])
            exposure_seconds = (end - start).total_seconds()
            if not math.isclose(exposure_seconds, WINDOW_SECONDS):
                raise ValueError(
                    f"window {window_id} must span {WINDOW_SECONDS:g} seconds, "
                    f"got {exposure_seconds:g}"
                )
            row: dict[str, object] = {
                "window_id": window_id,
                "image": image,
                "image_hash": image_hash.lower(),
                "detector": detector,
                "start_utc": raw["start_utc"].strip(),
                "end_utc": raw["end_utc"].strip(),
                "start": start,
                "end": end,
                "gwtc_veto_passed": parse_bool(raw["gwtc_veto_passed"]),
                "time_isolated_verified": parse_bool(
                    raw["time_isolated_verified"]
                ),
                "human_review_status": raw["human_review_status"].strip(),
                "human_reviewers": raw["human_reviewers"].strip(),
                "human_reviewed_at": raw["human_reviewed_at"].strip(),
                "human_review_reason": raw["human_review_reason"].strip(),
            }
            if row["human_reviewed_at"]:
                parse_utc(str(row["human_reviewed_at"]))
            rows.append(row)
            seen_windows.add(window_id)
            seen_images.add(image)
    if not rows:
        raise ValueError("manifest has no rows")
    return rows


def poisson_cdf(count: int, mean: float) -> float:
    term = math.exp(-mean)
    total = term
    for index in range(1, count + 1):
        term *= mean / index
        total += term
    return total


def poisson_upper_mean(count: int, confidence: float) -> float:
    """不依赖 SciPy 求 Poisson 单侧精确上限。"""
    target = 1.0 - confidence
    lower, upper = 0.0, max(1.0, count + 1.0)
    while poisson_cdf(count, upper) > target:
        upper *= 2.0
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if poisson_cdf(count, middle) > target:
            lower = middle
        else:
            upper = middle
    return upper


def paired_windows(
    manifest: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[
        tuple[datetime, datetime], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    for row in manifest:
        key = (row["start"], row["end"])
        detector = str(row["detector"])
        if detector in grouped[key]:
            raise ValueError(
                f"duplicate {detector} row for {row['start_utc']}–{row['end_utc']}"
            )
        grouped[key][detector] = row

    eligible: list[dict[str, object]] = []
    ineligible: list[dict[str, object]] = []
    for (start, end), detectors in sorted(grouped.items()):
        reasons: list[str] = []
        if set(detectors) != {"H1", "L1"}:
            reasons.append("missing_paired_h1_l1")
        for detector in ("H1", "L1"):
            if detector in detectors:
                accepted, detector_reasons = row_eligibility(detectors[detector])
                reasons.extend(
                    f"{detector.lower()}_{reason}" for reason in detector_reasons
                )
        target = {
            "start": start,
            "end": end,
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
            "detectors": detectors,
        }
        if reasons:
            ineligible.append({**target, "reason": ";".join(sorted(set(reasons)))})
        else:
            eligible.append(target)

    for previous, current in zip(eligible, eligible[1:]):
        if current["start"] < previous["end"]:
            raise ValueError(
                "eligible network windows overlap: "
                f"{previous['start_utc']} and {current['start_utc']}"
            )
    return eligible, ineligible


def audit(
    manifest: list[dict[str, object]],
    predictions: dict[str, bool],
    confidence: float,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    manifest_images = {str(row["image"]) for row in manifest}
    missing = sorted(manifest_images - set(predictions))
    extra = sorted(set(predictions) - manifest_images)
    if missing or extra:
        raise ValueError(
            f"manifest/prediction mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    windows, ineligible = paired_windows(manifest)
    if not windows:
        raise ValueError("no eligible paired H1/L1 network windows")

    rows: list[dict[str, object]] = []
    alerts = 0
    previous_hit = False
    previous_end: datetime | None = None
    exposure_seconds = 0.0
    for window in windows:
        detectors = window["detectors"]
        h1_hit = predictions[str(detectors["H1"]["image"])]
        l1_hit = predictions[str(detectors["L1"]["image"])]
        network_hit = h1_hit or l1_hit
        is_consecutive_hit = (
            network_hit
            and previous_hit
            and previous_end is not None
            and window["start"] == previous_end
        )
        if network_hit and not is_consecutive_hit:
            alerts += 1
        exposure_seconds += (window["end"] - window["start"]).total_seconds()
        rows.append(
            {
                "start_utc": window["start_utc"],
                "end_utc": window["end_utc"],
                "h1_window_id": detectors["H1"]["window_id"],
                "l1_window_id": detectors["L1"]["window_id"],
                "h1_hit": h1_hit,
                "l1_hit": l1_hit,
                "network_hit": network_hit,
                "starts_new_alert": network_hit and not is_consecutive_hit,
            }
        )
        previous_hit = network_hit
        previous_end = window["end"]

    exposure_days = exposure_seconds / 86400.0
    rate_per_day = alerts / exposure_days
    upper_per_day = (
        poisson_upper_mean(alerts, confidence) / exposure_days
    )
    return {
        "schema_version": 2,
        "scope": "production_far_eligible_windows_only",
        "network_windows": len(rows),
        "ineligible_network_windows": len(ineligible),
        "network_seconds": exposure_seconds,
        "network_days": exposure_days,
        "false_alarms": alerts,
        "false_alarms_per_day": rate_per_day,
        "confidence": confidence,
        "one_sided_upper_false_alarms_per_day": upper_per_day,
        "zero_alert_days_needed_for_0_1_per_day_upper": (
            -math.log(1.0 - confidence) / 0.1
        ),
        "eligibility_rule": (
            "paired H1/L1; both GWTC-vetoed, development-time-isolated, "
            "and approved by at least two independent reviewers"
        ),
        "alert_rule": (
            "any H1/L1 chirp decision is a network hit; consecutive "
            "four-second hit windows merge into one false alarm"
        ),
        "audit_set_must_not_be_used_for_threshold_tuning": True,
    }, rows, ineligible


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    summary, rows, ineligible = audit(
        load_manifest(args.manifest),
        load_predictions(args.predictions),
        args.confidence,
    )
    summary["manifest_sha256"] = sha256_file(args.manifest)
    summary["predictions_sha256"] = sha256_file(args.predictions)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "network_windows.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "ineligible_windows.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = ["start_utc", "end_utc", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in fieldnames} for row in ineligible
        )
    (args.output_dir / "negative_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
