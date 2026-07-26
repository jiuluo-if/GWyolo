from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_gw5 import (
    Prediction,
    fuse_predictions,
    parse_model_spec,
    read_catalogue,
    summarize_predictions,
)


class BenchmarkGw5Tests(unittest.TestCase):
    def test_parse_model_spec_preserves_path(self) -> None:
        spec = parse_model_spec("baseline=weights/best.pt")
        self.assertEqual(spec.name, "baseline")
        self.assertEqual(spec.weights, Path("weights/best.pt"))

    def test_catalogue_and_threshold_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalogue.csv"
            path.write_text(
                "事件,网络 SNR,质量 1 (太阳质量),质量 2 (太阳质量),"
                "纳入召回统计\n"
                "GW240101_000000,8.0,2.0,1.0,True\n"
                "GW240102_000000,9.0,--,--,False\n",
                encoding="utf-8",
            )
            catalogue = read_catalogue(path)

        predictions = [
            Prediction(
                experiment="baseline@640",
                model="baseline",
                imgsz=640,
                image="GW240101_000000-v1-H1-qscan",
                event="GW240101_000000",
                confidences=(0.4, 0.2),
                inference_ms=5.0,
            ),
            Prediction(
                experiment="baseline@640",
                model="baseline",
                imgsz=640,
                image="GW240102_000000-v1-L1-qscan",
                event="GW240102_000000",
                confidences=(0.3,),
                inference_ms=7.0,
            ),
        ]
        rows = summarize_predictions(
            predictions,
            catalogue,
            thresholds=(0.25, 0.5),
            wall_seconds=1.0,
            kind="single",
            members="baseline@640",
        )
        self.assertEqual(rows[0]["recalled_events"], 1)
        self.assertEqual(rows[0]["excluded_event_hits"], 1)
        self.assertEqual(rows[0]["detector_image_hits"], 2)
        self.assertEqual(rows[1]["recalled_events"], 0)

    def test_fusion_uses_max_confidence_per_image(self) -> None:
        first = Prediction(
            "first@640", "first", 640, "image", "GW240101_000000", (0.2,), 5.0
        )
        second = Prediction(
            "second@640", "second", 640, "image", "GW240101_000000", (0.7,), 6.0
        )
        fused, wall = fuse_predictions(
            "ensemble@640", [([first], 1.0), ([second], 2.0)]
        )
        self.assertEqual(fused[0].max_confidence, 0.7)
        self.assertEqual(fused[0].inference_ms, 11.0)
        self.assertEqual(wall, 3.0)


if __name__ == "__main__":
    unittest.main()
