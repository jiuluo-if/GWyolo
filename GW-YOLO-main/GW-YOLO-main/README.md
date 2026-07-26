# GW-YOLO

基于 Ultralytics YOLO 的时频图目标分割项目，用于识别两类目标：`chirp` 和 `noise`。

## 快速开始

1. 创建 Python 环境并安装依赖：

   ```powershell
   python -m pip install -r requirements.txt
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

## 可复现实验

固定验证集对比：

```powershell
python scripts/benchmark_validation.py `
  --model baseline=runs/segment/train/weights/best.pt `
  --model c2psa=runs/segment/segment_new/chirp_c2psa-2/weights/best.pt `
  --output-dir docs/experiments/batch_00_validation
```

GW5 多尺度与模型后融合：

```powershell
python scripts/benchmark_gw5.py `
  --model baseline=runs/segment/train/weights/best.pt `
  --model c2psa=runs/segment/segment_new/chirp_c2psa-2/weights/best.pt `
  --imgsz 512 --imgsz 640 --imgsz 768 `
  --output-dir docs/experiments/batch_01_inference
```

成本约束工作点选择：

```powershell
python scripts/select_operating_points.py `
  --summary docs/experiments/batch_01_inference/threshold_summary.csv `
  --profile efficient:0.30:25 `
  --profile balanced:0.40:50 `
  --profile max_recall:0.45:120 `
  --output-dir docs/experiments/batch_02_calibration
```

验证阴性代理与按需级联：

```powershell
python scripts/benchmark_cascade.py `
  --primary runs/segment/train/weights/best.pt `
  --secondary runs/segment/segment_new/chirp_c2psa-2/weights/best.pt `
  --source imgs/gw5.0 `
  --catalogue docs/gw5_recall_details.csv `
  --operating-threshold 0.25 `
  --chunk-size 1 --half `
  --output-dir runs/optimization_cascade_fp16
```

当前结论与研究边界见
[完整优化报告](docs/FINAL_OPTIMIZATION_REPORT.md) 和
[项目研究报告](docs/FINAL_RESEARCH_REPORT.md)。

Attention Residual 非对称阈值校准：

```powershell
python scripts/calibrate_attention_fusion.py `
  --validation-predictions docs/experiments/batch_03_negative_proxy/predictions.csv `
  --gw5-predictions docs/experiments/batch_04_cascade/predictions.csv `
  --catalogue docs/gw5_recall_details.csv `
  --output-dir docs/experiments/batch_06_asymmetric_calibration
```

校准得到 baseline/Attention Residual 阈值 `0.14/0.36`。复现实测级联时分别传入
`--operating-threshold 0.14 --secondary-threshold 0.36`；阈值选择只使用验证集，
GW5 只用于冻结策略后的事件级评估。
