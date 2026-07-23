# GW-YOLO

基于 Ultralytics YOLO 的时频图目标分割项目，用于识别两类目标：`chirp` 和 `noise`。

## 快速开始

1. 创建 Python 环境并安装依赖：

   ```powershell
   python -m pip install ultralytics opencv-python numpy
   ```

2. 按 `gw_data.yaml` 准备数据集。训练图像与标签分别位于 `train/images`、`train/labels`；验证集由 `val` 指向。
3. 检查 `trian.py` 中的权重、训练参数和 `gw_data.yaml` 的路径后运行：

   ```powershell
   python trian.py
   ```

4. 在 `predict.py` 顶部修改 `WEIGHTS_PATH`、`SOURCE_PATH`、`OUTPUT_DIR`，再运行：

   ```powershell
   python predict.py
   ```

预测标签将写入 `OUTPUT_DIR/labels`，可视化图像写入 `OUTPUT_DIR/images`。脚本只保留置信度最高的 `chirp`，保留全部 `noise`。

## 仓库约定

- `runs/`、模型权重、缓存和本地数据均为运行产物，不应作为常规源码提交。
- 当前 `trian.py` 是既有训练入口，文件名为历史拼写；后续重构时应迁移为 `scripts/train.py`，并保留兼容入口。
- 详细的项目上下文、风险点和建议结构见 [docs/PROJECT_MEMORY.md](docs/PROJECT_MEMORY.md)。

## 建议目标布局

```text
configs/          # 数据集、训练与推理配置
scripts/          # train.py / predict.py / preprocess.py
src/gw_yolo/      # 可复用业务逻辑
data/             # 本地数据（忽略）
artifacts/        # runs、权重、预测结果（忽略）
docs/             # 项目说明与实验记录
```

当前代码仍保留原始路径，避免破坏正在使用的训练与推理流程。
