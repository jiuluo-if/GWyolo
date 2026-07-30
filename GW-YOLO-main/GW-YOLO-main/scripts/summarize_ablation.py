"""汇总锁定变量消融的逐种子验证、GW5 与负集审计结果。

输入 CSV 必须包含 ``model`` 列；每行代表一个种子。脚本按 ``variant-seedN``
中的 variant 汇总数值列的样本数、均值和标准差，拒绝缺少任一预期种子的结构，避免
以不完整结果作结构结论。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


MODEL_PATTERN = re.compile(r"^(?P<variant>baseline|p4|p3p4)-seed(?P<seed>\d+)$")


def read_rows(path: Path) -> list[dict[str, str]]:
    """读取带 model 列的审计 CSV。"""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "model" not in rows[0]:
        raise ValueError(f"审计 CSV 必须包含至少一行和 model 列：{path}")
    return rows


def summarize(rows: list[dict[str, str]], expected_seeds: set[int]) -> dict[str, object]:
    """按结构汇总数值指标，并检查每种结构的随机种子是否完整。"""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seeds: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        match = MODEL_PATTERN.fullmatch(row["model"])
        if not match:
            raise ValueError(f"model 必须形如 baseline-seed0：{row['model']}")
        variant = match.group("variant")
        grouped[variant].append(row)
        seeds[variant].add(int(match.group("seed")))

    summary: dict[str, object] = {}
    for variant, variant_rows in sorted(grouped.items()):
        if seeds[variant] != expected_seeds:
            raise ValueError(
                f"{variant} 的种子不完整：已有 {sorted(seeds[variant])}，"
                f"期望 {sorted(expected_seeds)}"
            )
        metrics: dict[str, object] = {"种子": sorted(seeds[variant]), "原始行": variant_rows}
        for key in variant_rows[0]:
            if key == "model":
                continue
            try:
                values = [float(row[key]) for row in variant_rows]
            except (ValueError, TypeError):
                continue
            metrics[key] = {
                "样本数": len(values), "均值": fmean(values),
                "标准差": stdev(values) if len(values) > 1 else 0.0,
            }
        summary[variant] = metrics
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="合并后的逐种子审计 CSV")
    parser.add_argument("--output", type=Path, required=True, help="汇总 JSON 输出路径")
    parser.add_argument("--seeds", default="0,1,2", help="要求完成的种子，例如 0,1,2")
    args = parser.parse_args()
    expected_seeds = {int(item) for item in args.seeds.split(",") if item.strip()}
    if not expected_seeds:
        raise ValueError("--seeds 不能为空")
    summary = summarize(read_rows(args.input), expected_seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
