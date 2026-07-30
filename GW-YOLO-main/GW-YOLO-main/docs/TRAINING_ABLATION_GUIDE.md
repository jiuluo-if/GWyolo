# Attention Residual 锁定变量训练指南

## 目的

最新实验表明，现有 P3+P4 Attention Residual 单模型不能替换 baseline，但在事件级
召回上具有互补价值。由于旧权重的训练参数不完全一致且每种结构只有一个随机种子，
当前差异不能严格归因于 Attention Residual 结构。

`scripts/train_attention_residual_ablation.py` 用完全一致的训练变量生成三种结构、
三个随机种子的受控实验，为后续均值、标准差和结构归因提供权重。

## 实验矩阵

| 变体 | 配置 | 唯一结构变量 |
| --- | --- | --- |
| `baseline` | `configs/yolo26m-chirp-baseline-seg.yaml` | 不增加 P3/P4 注意力细化 |
| `p4` | `configs/yolo26m-chirp-p4-attn-seg.yaml` | 仅 P4 增加 2 个 C2PSA |
| `p3p4` | `configs/yolo26m-chirp-attn-seg.yaml` | P3 增加 1 个、P4 增加 2 个 C2PSA |

默认种子为 `0,1,2`，共 9 个训练任务。三种配置都保留 YOLO26m 原始 P5 C2PSA；
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
python scripts/train_attention_residual_ablation.py --dry-run
```

`--dry-run` 不导入 Ultralytics、不启动 GPU，也不写入训练目录，会打印全部 9 个任务和
锁定参数。正式训练前应保存终端输出并确认数据、权重和输出盘空间。

## 正式训练

```powershell
conda run -n yolo python scripts/train_attention_residual_ablation.py
```

训练入口会先把 `--project` 规范化为绝对路径，避免当前 Ultralytics 对相对 project
再次附加默认 `runs/segment`。因此实验清单、目录冲突检查和真实 `save_dir` 必须位于
同一项目根目录；如三者不一致，应停止任务并检查运行环境。

默认输出：

```text
runs/segment/attention_residual_ablation/
  experiment_manifest.json
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

脚本拒绝复用任何已存在的目标目录，防止续跑或覆盖造成实验口径混合。

## 分批训练

显存或时间受限时可逐任务执行：

```powershell
conda run -n yolo python scripts/train_attention_residual_ablation.py `
  --variants p4 `
  --seeds 0
```

也可一次训练一个结构：

```powershell
conda run -n yolo python scripts/train_attention_residual_ablation.py `
  --variants baseline `
  --seeds 0,1,2
```

分批运行时应为不同批次指定不同 `--project`。脚本同时拒绝覆盖已有任务目录和
`experiment_manifest.json`；不得删除旧结果后复用同名目录。

## 训练后评估要求

每个种子必须使用相同流程完成：

1. 固定验证集 box/mask 总体与 chirp 逐类指标。
2. GW5 事件级召回：排除质量不完整事件，H1/L1/V1 任一图像命中即召回。
3. 在验证集上独立校准阈值，GW5 不参与选参。
4. 记录训练耗时、最佳 epoch、参数量、推理延迟和峰值显存。
5. 每个结构报告三种子的均值、标准差和全部原始值。

将每个种子的固定验证、GW5 盲评与负集审计合并为带 `model` 列的 CSV 后，使用以下命令
拒绝不完整种子并生成结构汇总：

```powershell
python scripts/summarize_ablation.py `
  --input docs/experiments/ablation/all_seed_audits.csv `
  --output docs/experiments/ablation/summary.json
```

在上述结果完成前，不得用单次最好权重宣称 Attention Residual 带来结构改进。生产默认
仍保持 baseline@0.20；现有非对称级联仅作为高召回研究策略。
