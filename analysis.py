"""Inspect chirp detections and map their width to the selected physical axes."""

import argparse
from pathlib import Path

import cv2


def map_xp2axis(axis_lengths, box_min, box_max, axis_min, axis_max):
    """Map a detected box from pixel coordinates to physical-axis lengths."""
    axis_size = [axis_max[i] - axis_min[i] for i in (0, 1)]
    box_size = [box_min[0] - axis_min[0], axis_max[1] - box_max[1]]
    return [0 if axis_size[i] == 0 else box_size[i] / axis_size[i] * axis_lengths[i] for i in (0, 1)]


def inspect_image(weights: Path, image: Path, duration: float, bandwidth: float) -> None:
    """Run inference and display a vertical marker for each chirp detection."""
    from ultralytics import YOLO
    from line_find import detect_line_segments

    model = YOLO(weights)
    for result in model(str(image)):
        path_x, path_y = detect_line_segments(result.path)
        axis_min, axis_max = [path_x[0], path_y[0]], [path_x[1], path_y[-1]]
        boxes = result.boxes
        for index, (x1, y1, x2, y2) in enumerate(boxes.xyxy.cpu().numpy()):
            label = result.names[int(boxes.cls[index])]
            if label.lower() != "chirp":
                continue
            detected = map_xp2axis([duration, bandwidth], [x1, y1], [x2, y2], axis_min, axis_max)
            confidence = float(boxes.conf[index])
            print(f"Detected {label}: confidence={confidence:.2f}, time={detected[0]:.2f}s")
            rendered = cv2.imread(result.path, cv2.IMREAD_COLOR)
            cv2.line(rendered, (int(x2), int(path_y[0])), (int(x2), int(path_y[-1])), (0, 0, 255), 2)
            cv2.imshow("Image Window", rendered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect chirp detections in one image.")
    parser.add_argument("weights", type=Path, help="Path to model weights")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--duration", type=float, default=3.0, help="Image duration in seconds")
    parser.add_argument("--bandwidth", type=float, default=1000.0, help="Image bandwidth in Hz")
    args = parser.parse_args()
    inspect_image(args.weights, args.image, args.duration, args.bandwidth)
