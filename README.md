# GW-YOLO

> 基于 **Ultralytics YOLO** 的引力波时频图目标分割研究项目。
>
> 识别类别：`chirp`（啁啾信号）与 `noise`（噪声）。

## 项目概览

GW-YOLO 面向引力波 Q-scan 时频图的目标分割与可审计推理。项目当前保留两条互补的研究路线：

| 路线 | 作用 | 当前定位 |
| --- | --- | --- |
| `baseline@640` | 主模型、稳定基线 | 冻结的研究与负样本审计基线 |
| Attention Residual@640 | 互补模型 | 等待完整多种子训练证据 |
| baseline@512 / @768 | 条件式多尺度救援 | 仅用于冻结后的跨尺度短路级联 |

> [!IMPORTANT]
> 当前高召回策略在固定验证集达到 **69/3/0/11**，在补齐图像后的 GW5 事件集达到 **102/104**。这些是研究评估结果，**不是生产性能或 FAR 声明**。
>
> 现有 v1 负样本审计仅覆盖 1,800 秒网络暴露，并出现 2 个候选告警；它不具备生产 FAR 发布资格。

**快速入口：** [当前成果总表](docs/CURRENT_ACHIEVEMENTS.md) · [文档导航与维护规则](docs/DOCUMENTATION_GUIDE.md) · [实验批次索引](docs/experiments/README.md)

---

## 快速开始

### 1. 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 2. 准备数据

依据 [gw_data.yaml](gw_data.yaml) 配置数据集路径：

```text
train/
├── images/     # 训练图像
└── labels/     # YOLO 分割标签

val/            # 验证集（由 gw_data.yaml 指向）
```

### 3. 训练

检查 [trian.py](trian.py) 中的权重、训练参数和数据集路径后执行：

```powershell
python trian.py
```

> `trian.py` 是保留兼容性的历史文件名。新的受控多种子消融训练请使用下文的专用脚本。

### 4. 可审计推理

推理必须显式指定权重、输入与**新的**输出目录：

```powershell
python predict.py `
  --weights runs/segment/train/weights/best.pt `
  --source imgs/gw5.0 `
  --output runs/filtered/predict_gw5_audit
```

输出目录中包含：

```text
predict_gw5_audit/
├── images/                    # 标注后的可视化图像
├── labels/                    # 每张输入图像对应的标签（无检出时为空文件）
└── prediction_manifest.json   # 权重、输入与推理参数清单
```

模型实例会复用，但始终逐图推理；默认拒绝覆盖既有输出，确保审计结果可追溯。

## 仓库约定

| 项目 | 约定 |
| --- | --- |
| 运行产物 | `runs/`、模型权重、缓存与本地数据不作为常规源码提交 |
| 训练入口 | `trian.py` 是保留兼容性的历史文件名；后续重构应迁移为 `scripts/train.py` |
| 当前结论 | 以 [当前成果总表](docs/CURRENT_ACHIEVEMENTS.md) 为准 |
| 文档维护 | 参阅 [文档导航与维护规则](docs/DOCUMENTATION_GUIDE.md) |
| 实验决策 | 批次 00–12 的接受/拒绝记录见 [实验批次索引](docs/experiments/README.md) |

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

---

## 当前研究状态

| 项目 | 已有证据 | 结论边界 |
| --- | --- | --- |
| GW5 事件级评估 | 102/104；H1/L1/V1 任一图像检出 class 0 `chirp` 即召回，排除质量字段 `--` | 不是独立纯负集，不能推导 FPR/FAR |
| 跨尺度短路级联 | baseline640 → Attention640 → baseline512 → 条件 baseline768；GW5 实测 33.55 秒 | 参数由固定验证集选择，GW5 只作策略冻结后的评估 |
| Attention Residual | 已完成 1-epoch、三结构烟雾测试 | 尚未完成 3 结构 × 3 种子 × 300 epoch 正式训练 |
| v1 时间隔离负集 | 450 个 H1/L1 同步四秒窗口，2 个候选告警 | 候选率证据，不是可发布的生产 FAR |

完整证据、已拒绝结论与待办事项请阅读 [当前成果总表](docs/CURRENT_ACHIEVEMENTS.md)。

## 可复现实验

> 阈值只能依据固定验证集校准；GW5 用于策略冻结后的事件级评估。每个实验批次的完整输入与产物见 [实验批次索引](docs/experiments/README.md)。

### 固定验证集对比

```powershell
python scripts/benchmark_validation.py `
  --model baseline=runs/segment/train/weights/best.pt `
  --model c2psa=runs/segment/segment_new/chirp_c2psa-2/weights/best.pt `
  --output-dir docs/experiments/batch_00_validation
```

### GW5 多尺度与模型后融合

```powershell
python scripts/benchmark_gw5.py `
  --model baseline=runs/segment/train/weights/best.pt `
  --model c2psa=runs/segment/segment_new/chirp_c2psa-2/weights/best.pt `
  --imgsz 512 --imgsz 640 --imgsz 768 `
  --output-dir docs/experiments/batch_01_inference
```

### 成本约束工作点选择

```powershell
python scripts/select_operating_points.py `
  --summary docs/experiments/batch_01_inference/threshold_summary.csv `
  --profile efficient:0.30:25 `
  --profile balanced:0.40:50 `
  --profile max_recall:0.45:120 `
  --output-dir docs/experiments/batch_02_calibration
```

### 验证阴性代理与按需级联

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

当前结论与研究边界以 [当前成果总表](docs/CURRENT_ACHIEVEMENTS.md) 为准；批次原始产物请从 [实验批次索引](docs/experiments/README.md) 进入。

### Attention Residual 非对称阈值校准

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

### 跨尺度一致性校准

```powershell
python scripts/calibrate_scale_consensus.py `
  --validation-640 docs/experiments/batch_06_asymmetric_calibration/validation_runtime/predictions.csv `
  --validation-512 docs/experiments/batch_08_multiscale_rescue/validation_512/predictions.csv `
  --validation-768 docs/experiments/batch_08_multiscale_rescue/validation_768/predictions.csv `
  --gw5-predictions docs/experiments/batch_08_multiscale_rescue/gw5_scores/predictions.csv `
  --output-dir docs/experiments/batch_09_scale_consensus
```

### 当前短路策略实测

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

先检查锁定变量的 3×3 实验矩阵：

```powershell
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py --dry-run
```

确认后启动正式训练：

```powershell
$env:PYTHONUTF8 = "1"
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py
```

脚本统一比较 baseline、仅 P4 Attention Residual、P3+P4 Attention Residual，默认种子为 `0,1,2`。它会冻结数据、模型配置和预训练权重的 SHA-256，把 `project` 规范化为绝对路径，并拒绝覆盖已有输出目录。长跑中断后使用相同参数加 `--resume`，脚本会核对实验清单并从对应 `weights/last.pt` 恢复。

> [!NOTE]
> 1-epoch 三结构烟雾测试已通过，但不能替代正式 300-epoch 多种子实验。完整说明见 [Attention Residual 锁定变量训练指南](docs/TRAINING_ABLATION_GUIDE.md)。

---

## 文档地图

| 想了解什么 | 阅读入口 |
| --- | --- |
| 当前可对外陈述的结果、证据与限制 | [CURRENT_ACHIEVEMENTS.md](docs/CURRENT_ACHIEVEMENTS.md) |
| 文档层级、历史材料与更新规则 | [DOCUMENTATION_GUIDE.md](docs/DOCUMENTATION_GUIDE.md) |
| 批次实验的输入、输出与接受决策 | [experiments/README.md](docs/experiments/README.md) |
| 正式多种子消融训练 | [TRAINING_ABLATION_GUIDE.md](docs/TRAINING_ABLATION_GUIDE.md) |
| 时间隔离负样本与 FAR 口径 | [TIME_ISOLATED_NEGATIVE_AUDIT.md](docs/TIME_ISOLATED_NEGATIVE_AUDIT.md) |

---

## 研究与发布原则

- 不将 GW5 图像命中解释为 FPR 或 FAR。
- 不在负样本审计集上重新选择阈值。
- 不以单个 seed 或 1-epoch 烟雾测试作为结构收益结论。
- 推理保留逐图、空标签和清单输出，确保输入覆盖与结果可审计。
