# Attention Residual 锁定变量训练指南

## 目的

最新实验表明，现有 P3+P4 Attention Residual 单模型不能替换 baseline，但在事件级
召回上具有互补价值。由于旧权重的训练参数不完全一致且每种结构只有一个随机种子，
当前差异不能严格归因于 Attention Residual 结构。

`scripts/train_attention_residual_ablation.py` 用完全一致的训练变量生成三种结构、
三个随机种子的受控实验，为后续均值、标准差和结构归因提供权重。正式版还会冻结
数据、模型配置和预训练权重的内容指纹，并支持中断后安全续训。

## 实验矩阵

| 变体 | 配置 | 唯一结构变量 |
| --- | --- | --- |
| `baseline` | `configs/yolo26m-chirp-baseline-seg.yaml` | 不增加 P3/P4 注意力细化 |
| `p4` | `configs/yolo26m-chirp-p4-attn-seg.yaml` | 仅 P4 增加 2 个 C2PSA |
| `p3p4` | `configs/yolo26m-chirp-attn-seg.yaml` | P3 增加 1 个、P4 增加 2 个 C2PSA |

默认种子为 `0,1,2`，共 9 个训练任务，并按种子交错执行三种结构以减小长跑时间顺序
偏置。三种配置都保留 YOLO26m 原始 P5 C2PSA；
本消融比较的是额外 P3/P4 Attention Residual。

## 锁定训练变量

| 参数 | 固定值 |
| --- | --- |
| 初始化 | `yolo26m-seg.pt` 的可匹配预训练权重 |
| 数据 | `gw_data.yaml`，现有 331/83 划分 |
| 输入/epoch/batch | 640 / 300 / 2 |
| 优化器 | AdamW |
| `lr0` / `lrf` | 0.002 / 0.01 |
| warm-up / weight decay | 5 / 0.0005 |
| `cls` / `box` / `mask_ratio` | 1.5 / 8.0 / 2 |
| `mosaic` / `mixup` / `erasing` | 0.35 / 0.05 / 0.15 |
| `translate` / `scale` | 0.08 / 0.20 |
| 随机性 | 显式 `seed`，`deterministic=True` |

旋转、上下翻转、左右翻转和 HSV 扰动保持关闭，避免破坏 chirp 的时频方向。

自定义注意力层会使最终 `Segment26` 的外层编号后移。脚本先执行常规整模型权重加载，
再按分割头内部的局部键名和张量形状显式迁移兼容权重；类别数不匹配的输出张量和新增
注意力层保持随机初始化。当前三种配置均能迁移 362/376 个分割头张量。这样不会让
候选结构额外承担“整个分割头随机初始化”的混杂。

## 先审计计划

```powershell
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py --dry-run
```

`--dry-run` 不导入 Ultralytics、不启动 GPU，也不写入训练目录，会打印全部 9 个任务、
锁定参数以及以下 SHA-256：

- `gw_data.yaml` 与其引用的训练/验证图像、标签；
- 三种模型 YAML；
- `yolo26m-seg.pt`。

正式训练前应保存终端输出并确认图像数为 414（331 训练 + 83 验证）、标签数为 414、
权重路径和输出盘空间正确。`labels.cache` 不进入数据指纹，避免缓存刷新破坏续训。

## 正式训练

```powershell
$env:PYTHONUTF8 = "1"
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py
```

训练入口会先把 `--project` 规范化为绝对路径，避免当前 Ultralytics 对相对 project
再次附加默认 `runs/segment`。因此实验清单、目录冲突检查和真实 `save_dir` 必须位于
同一项目根目录；脚本会显式核对真实 `save_dir`，不一致时立即停止。直接调用 `yolo`
环境解释器并启用 UTF-8，也可避开当前机器上 `conda run` 回显中文日志时的 GBK 问题。

默认输出：

```text
runs/segment/attention_residual_ablation_formal_v1/
  experiment_manifest.json
  training_status.json
  baseline-seed0/
  baseline-seed1/
  baseline-seed2/
  p4-seed0/
  p4-seed1/
  p4-seed2/
  p3p4-seed0/
  p3p4-seed1/
  p3p4-seed2/
```

每个成功任务还会写入 `training_complete.json`。脚本默认拒绝复用任何已存在的目标
目录，防止覆盖造成实验口径混合；实验意外中断时，不删除目录，使用：

```powershell
$env:PYTHONUTF8 = "1"
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py --resume
```

续训会重新计算全部内容指纹并要求其与 `experiment_manifest.json` 完全一致，跳过同时
具有完成标记和 `best.pt` 的任务，从其余任务的 `weights/last.pt` 恢复。已有目录若既
没有完成标记也没有 `last.pt`，脚本会停止并要求人工审计，不会猜测或覆盖。

## 分批训练

显存或时间受限时可逐任务执行：

```powershell
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py `
  --variants p4 `
  --seeds 0
```

也可一次训练一个结构：

```powershell
& C:\miniconda3\envs\yolo\python.exe scripts/train_attention_residual_ablation.py `
  --variants baseline `
  --seeds 0,1,2
```

分批运行时应为不同批次指定不同 `--project`。同一批次中断后必须用同样的
`--variants`、`--seeds`、`--epochs`、`--batch` 和 `--project` 加 `--resume`；不得
删除旧结果后复用同名目录。

## 训练后评估要求

每个种子必须使用相同流程完成：

1. 固定验证集 box/mask 总体与 chirp 逐类指标。
2. GW5 事件级召回：排除质量不完整事件，H1/L1/V1 任一图像命中即召回。
3. 在验证集上独立校准阈值；旧模型的 0.14/0.36/0.15/0.07 只能作为历史参考，
   新权重必须重新校准，GW5 不参与选参。
4. 记录训练耗时、最佳 epoch、参数量、推理延迟和峰值显存。
5. 每个结构报告三种子的均值、标准差和全部原始值。

将每个种子的固定验证、GW5 盲评与负集审计合并为带 `model` 列的 CSV 后，使用以下命令
拒绝不完整种子并生成结构汇总：

```powershell
python scripts/summarize_ablation.py `
  --input docs/experiments/ablation/all_seed_audits.csv `
  --output docs/experiments/ablation/summary.json
```

在上述结果完成前，不得用单次最好权重宣称 Attention Residual 带来结构改进。训练后
还必须在时间隔离纯负集上报告 FP 与 `false alarms/day`。生产默认仍保持旧
baseline@0.20；现有尺度共识级联仅作为旧权重的高召回研究策略，不能直接把其阈值
套到新权重。
