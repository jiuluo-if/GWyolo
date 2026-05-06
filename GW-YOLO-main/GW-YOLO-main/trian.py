from ultralytics import YOLO


def main():
    # 加载预训练模型
    print("正在加载预训练模型")
    model = YOLO("yolo26n.pt")
    print("✓ 模型加载完成\n")

    # 3. 开始训练
    print("开始训练...")
    print("-" * 50)

    results = model.train(
        data="gw_data.yaml",  # 数据集配置文件路径
        epochs=500,  # 训练轮数
        imgsz=640,  # 输入图像尺寸
        batch=16,  # 批大小（根据 GPU 显存调整）
        workers=4,  # 数据加载线程数
        lr0=0.01,  # 初始学习率
        patience=50,  # 早停轮数（50轮无提升则停止）
        save=True,  # 保存训练检查点
        save_period=10,  # 每10轮保存一次检查点
        val=True,  # 每轮结束后验证
        # project="runs/train",  # 结果保存目录
        # name="yolo26n_train",  # 本次训练的子目录名
        exist_ok=True,  # 允许覆盖已有目录
        verbose=True,  # 输出详细日志
    )

    print("-" * 50)
    print("✓ 训练完成！")


if __name__ == "__main__":
    main()