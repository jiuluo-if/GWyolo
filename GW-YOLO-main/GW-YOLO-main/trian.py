"""Train the chirp-aware segmentation model without touching baseline runs."""

from pathlib import Path

from ultralytics import YOLO


MODEL_CONFIG = Path("configs/yolo26m-chirp-attn-seg.yaml")
PRETRAINED_WEIGHTS = Path("yolo26m-seg.pt")
# All artifacts are nested here. `exist_ok=False` makes Ultralytics create an
# incremented run directory rather than overwrite an earlier experiment.
OUTPUT_PROJECT = Path("segment_new")
RUN_NAME = "chirp_c2psa"


def main():
    model = YOLO(MODEL_CONFIG)
    model.load(PRETRAINED_WEIGHTS)

    results = model.train(
        data="gw_data.yaml",
        epochs=300,
        imgsz=640,
        # Segmentation at 640 plus position-sensitive attention is memory
        # intensive. A small physical batch keeps the high-resolution chirp
        # detail; Ultralytics compensates with gradient accumulation.
        batch=2,
        workers=0,  # Windows multiprocessing safety
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        weight_decay=0.0005,
        cls=1.5,
        box=8.0,
        mask_ratio=2,  # retain detail in thin chirp masks
        overlap_mask=True,
        amp=True,
        # Shift and scale are physically plausible; large composite images and
        # pixel erasing make a faint continuous chirp harder to learn.
        mosaic=0.35,
        close_mosaic=20,
        mixup=0.05,
        translate=0.08,
        scale=0.20,
        erasing=0.15,
        # Do not reverse or rotate the time-frequency sweep direction.
        degrees=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        patience=50,
        save=True,
        project=OUTPUT_PROJECT,
        name=RUN_NAME,
        exist_ok=False,
    )
    print(f"Training complete. Best weights are saved in: {results.save_dir}")


if __name__ == "__main__":
    main()
