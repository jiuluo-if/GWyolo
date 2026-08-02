"""训练 chirp 感知分割模型，且不覆盖既有 baseline 实验。"""

from pathlib import Path

from ultralytics import YOLO


MODEL_CONFIG = Path("configs/yolo26m-chirp-attn-seg.yaml")
PRETRAINED_WEIGHTS = Path("yolo26m-seg.pt")
# 全部产物写入该目录。`exist_ok=False` 会让 Ultralytics 创建递增的运行目录，
# 而非覆盖先前实验。
OUTPUT_PROJECT = Path("segment_new")
RUN_NAME = "chirp_c2psa"


def main():
    model = YOLO(MODEL_CONFIG)
    model.load(PRETRAINED_WEIGHTS)

    results = model.train(
        data="gw_data.yaml",
        epochs=300,
        imgsz=640,
        # 640 分割与位置敏感注意力占用显存较高。较小的物理批次保留高分辨率
        # chirp 细节；Ultralytics 会以梯度累积补偿有效批次大小。
        batch=2,
        workers=0,  # Windows 多进程兼容性
        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        weight_decay=0.0005,
        cls=1.5,
        box=8.0,
        mask_ratio=2,  # 保留细长 chirp 掩膜的细节
        overlap_mask=True,
        amp=True,
        # 平移和缩放符合物理图像变化；过强的拼接和像素擦除会让微弱连续的
        # chirp 更难学习。
        mosaic=0.35,
        close_mosaic=20,
        mixup=0.05,
        translate=0.08,
        scale=0.20,
        erasing=0.15,
        # 不反转或旋转时频扫频方向。
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
    print(f"训练完成，最佳权重保存于：{results.save_dir}")


if __name__ == "__main__":
    main()
