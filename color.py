"""Display segmentation masks with randomly assigned colors."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def show_masks(weights: Path, image: Path) -> None:
    """Run segmentation for one image and display the annotated result."""
    from ultralytics import YOLO

    model = YOLO(weights)
    img = cv2.imread(str(image))
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {image}")

    results = model(img)
    if results[0].masks is None:
        cv2.imshow("Detection", results[0].plot())
    else:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes
        color_mask = np.zeros_like(img)

        for index, mask in enumerate(masks):
            color = np.random.randint(0, 255, 3).tolist()
            color_mask[(mask * 255).astype(np.uint8) > 128] = color
            x1, y1, x2, y2 = map(int, boxes.xyxy[index])
            cls = int(boxes.cls[index])
            conf = float(boxes.conf[index])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Colorful Masks", cv2.addWeighted(img, 0.5, color_mask, 0.5, 0))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display colorful masks for a segmentation result.")
    parser.add_argument("weights", type=Path, help="Path to model weights")
    parser.add_argument("image", type=Path, help="Path to input image")
    args = parser.parse_args()
    show_masks(args.weights, args.image)
