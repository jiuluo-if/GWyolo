# GW-YOLO

基于 Ultralytics YOLO 的时频图目标分割项目，用于识别两类目标：`chirp` 和 `noise`。

当前高召回研究策略为 baseline@640、Attention Residual@640 与 baseline 512/768
跨尺度一致性的短路级联。固定验证为 69/3/0/11，补图后的 GW5 为 102/104；这不是
生产结论。当前 v1 负集只有 1,800 秒网络暴露和两个候选告警，不具备生产 FAR
发布资格；详见 [当前成果总表](docs/CURRENT_ACHIEVEMENTS.md)。

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

4. 显式指定权重、输入和新的输出目录，再运行逐图推理：

   ```powershell
   python predict.py `
     --weights runs/segment/train/weights/best.pt `
     --source imgs/gw5.0 `
     --output runs/filtered/predict_gw5_audit
   ```

预测标签写入 `OUTPUT/labels`，可视化图像写入 `OUTPUT/images`，并生成
`prediction_manifest.json`。无检出图像也会留下空标签；默认拒绝覆盖已有输出。

## 仓库约定

- `runs/`、模型权重、缓存和本地数据均为运行产物，不应作为常规源码提交。
- 当前 `trian.py` 是既有训练入口，文件名为历史拼写；后续重构时应迁移为 `scripts/train.py`，并保留兼容入口。
- 当前全部已完成成果、证据边界和待办见 [当前成果总表](docs/CURRENT_ACHIEVEMENTS.md)。
- 文档的当前/历史边界及更新顺序见 [文档导航与维护规则](docs/DOCUMENTATION_GUIDE.md)。
- 批次 00–12 的接受/拒绝决策见 [实验批次索引](docs/experiments/README.md)。

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

跨尺度一致性校准：

```powershell
python scripts/calibrate_scale_consensus.py `
  --validation-640 docs/experiments/batch_06_asymmetric_calibration/validation_runtime/predictions.csv `
  --validation-512 docs/experiments/batch_08_multiscale_rescue/validation_512/predictions.csv `
  --validation-768 docs/experiments/batch_08_multiscale_rescue/validation_768/predictions.csv `
  --gw5-predictions docs/experiments/batch_08_multiscale_rescue/gw5_scores/predictions.csv `
  --output-dir docs/experiments/batch_09_scale_consensus
```

当前短路策略实测：

```powershell
conda run -n yolo python scripts/benchmark_scale_consensus_cascade.py `
  --baseline runs/segment/train/weights/best.pt `
  --attention runs/segment/segment_new/chirp_c2psa-2/weights/best.pt `
  --source imgs/gw5.0 `
  --catalogue docs/gw5_recall_details.csv `
  --output-dir docs/experiments/batch_10_short_circuit_runtime/gw5
```

脚本默认阈值为 baseline 0.14、Attention 0.36、baseline@512 0.15、
baseline@768 0.07。仅在前三级失败且 512 达到 0.15 时才运行 768。

## Attention Residual 多种子训练

先审计锁定变量的 3×3 实验矩阵：

```powershell
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py --dry-run
```

正式训练：

```powershell
$env:PYTHONUTF8 = "1"
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py
```

脚本统一比较 baseline、仅 P4 Attention Residual、P3+P4 Attention Residual，
默认种子为 `0,1,2`，会冻结数据、模型配置和预训练权重的 SHA-256，把 `project`
规范化为绝对路径，并拒绝覆盖已有输出目录。长跑中断后使用相同参数加 `--resume`，
脚本会核对实验清单并从对应 `weights/last.pt` 恢复。
1-epoch 三结构烟雾测试已通过，但不能替代正式 300-epoch 多种子实验。完整说明见
[Attention Residual 锁定变量训练指南](docs/TRAINING_ABLATION_GUIDE.md)。
