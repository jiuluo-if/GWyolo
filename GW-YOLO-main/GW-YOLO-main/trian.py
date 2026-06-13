from ultralytics import YOLO


def main():
    print("正在加载预训练模型...")
    # 建议使用 yolo26m 或 yolo26l 增强特征提取能力，若显存有限则保持 yolo26n
    model = YOLO("yolo26n.pt")  # 可改为 "yolo26m.pt"
    print("✓ 模型加载完成\n")

    print("开始训练（针对引力波 chirp 优化）...")
    print("-" * 50)

    results = model.train(
        # ----- 基础配置 -----
        data="gw_data.yaml",  # 数据集配置文件
        epochs=300,  # 训练轮数（足够收敛）
        batch=16,  # 批次大小（根据显存调整，8~32）
        imgsz=640,  # 输入图像尺寸（时频图常用 640x640）
        workers=16,  # 数据加载线程数
        device=0,  # GPU 编号，CPU 可写 'cpu'

        # ----- 优化器与学习率 -----
        optimizer="auto",  # 自动选择（AdamW / SGD）
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率因子 (lr0 * lrf)
        momentum=0.937,  # SGD momentum
        weight_decay=0.0005,  # 权重衰减

        # ----- 损失权重（提升分类损失，抑制噪声误检）-----
        box=7.5,  # 边界框损失权重（默认 7.5）
        cls=1.5,  # 分类损失权重（默认 0.5，提高使模型更关注 chirp vs noise）
        dfl=1.5,  # 分布焦点损失权重（默认 1.5）

        # ----- 类别不平衡处理 -----
        focal_loss=1.0,  # 启用 Focal Loss（默认 0 关闭），聚焦难分样本

        # ----- 数据增强（物理合理，匹配 chirp 特性）-----
        mosaic=0.5,  # 50% 概率使用 mosaic 拼接
        mixup=0.2,  # 20% 概率使用 mixup 混合
        copy_paste=0.0,  # 无需实例分割
        degrees=0.0,  # 禁止旋转（时频图方向固定）
        translate=0.1,  # 平移 ±10%（模拟到达时间漂移）
        scale=0.3,  # 缩放 ±30%（模拟振幅/信噪比变化）
        shear=0.0,  # 禁止剪切
        perspective=0.0,  # 禁止透视
        flipud=0.0,  # 禁止上下翻转（频率轴）
        fliplr=0.0,  # 禁止左右翻转（时间轴）
        hsv_h=0.0,  # 色调不变（色图无信息）
        hsv_s=0.0,
        hsv_v=0.0,
        erasing=0.3,  # 30% 概率随机擦除矩形区域（模拟部分遮挡/强噪声）

        # ----- 训练控制 -----
        patience=50,  # 50 轮无提升则早停
        save=True,  # 保存检查点
        save_period=10,  # 每 10 轮保存一次
        val=True,  # 每轮结束后验证
        verbose=True,  # 输出详细日志
        exist_ok=True,  # 允许覆盖已有目录
        # project="runs/train",     # 默认即可
        # name="",                  # 自动命名
    )

    print("-" * 50)
    print("✓ 训练完成！最佳模型保存在:", results.save_dir)


if __name__ == "__main__":
    main()