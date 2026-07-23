from pathlib import Path

from ultralytics import YOLO


MODEL_CONFIG = Path("configs/yolo26m-attn-residual-seg.yaml")
PRETRAINED_WEIGHTS = Path("yolo26m-seg.pt")


def main():
    # Build the attention-residual architecture first, then initialize all
    # matching layers from the official YOLO26m segmentation checkpoint.
    model = YOLO(MODEL_CONFIG)
    model.load(PRETRAINED_WEIGHTS)

    results = model.train(
        data="gw_data.yaml",
        epochs=300,
        imgsz=640,
        batch=8,
        workers=0,               # Windows 单GPU 必须设为 0
        cls=1.5,                 # 提高分类损失权重
        # 数据增强（匹配 chirp 特性）
        mosaic=0.5,
        mixup=0.2,
        translate=0.1,
        scale=0.3,
        erasing=0.3,
        # 关闭破坏物理特性的增强
        degrees=0.0,
        flipud=0.0,
        fliplr=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        # 训练控制
        patience=50,
        save=True,
        save_period=10,
        exist_ok=True,
    )
    print(f"✓ 训练完成！最佳模型保存在: {results.save_dir}")

if __name__ == "__main__":
    main()
