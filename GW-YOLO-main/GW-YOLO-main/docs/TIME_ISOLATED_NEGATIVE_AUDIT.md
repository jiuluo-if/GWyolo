# 时间隔离纯负集网络审计

这是生产可用性门槛，不是阈值选参集。清单以同步的四秒 H1/L1 网络窗口为基本单位；
每个时段必须各有一行 H1 和 L1，且两行都通过 GWTC 目录否决、可审计的开发时间隔离
和双人独立人工批准。

## 数据清单

以 `docs/time_isolated_negative_manifest_template.csv` 为模板：

- `image` 必须与推理输出中的图片 stem 完全一致，`image_hash` 保存不可变内容指纹。
- H1/L1 的 `start_utc` 和 `end_utc` 必须完全相同，且窗口严格为四秒。
- `gwtc_veto_passed` 只表示目录否决，不能代替人工复核。
- `time_isolated_verified` 必须有可追溯的开发期排除证据。
- `human_review_status=approved` 时，`human_reviewers` 至少包含两个用分号分隔的独立身份，
  并填写复核时间和理由。
- 不要把训练集中的 `noise` 标注或 14 张验证负图填入清单；它们只能用于开发期代理。

只有同时合格的成对 H1/L1 窗口进入 FAR 分母。缺少探测器、任一行未批准或确认不完整的
窗口会写入 `ineligible_windows.csv`，但不进入生产统计。

## 运行

先用冻结模型、阈值和图片生成规则对全部清单图片逐图推理。当前研究短路级联可生成
`image,decision` 预测表：

```powershell
& C:\miniconda3\envs\yolo\python.exe scripts/benchmark_scale_consensus_cascade.py `
  --baseline runs/segment/train/weights/best.pt `
  --attention runs/segment/segment_new/chirp_c2psa-2/weights/best.pt `
  --source <时间隔离负集图片目录> `
  --output-dir runs/time_isolated_negative_predictions
```

生产默认应冻结为 baseline@640、阈值 0.20；上面的多成员级联只用于比较候选策略，
不能把旧权重的研究阈值直接发布为生产阈值。

再按网络窗口审计：

```powershell
python scripts/audit_time_isolated_negatives.py `
  --manifest docs/time_isolated_negative_manifest.csv `
  --predictions runs/time_isolated_negative_predictions/predictions.csv `
  --output-dir docs/experiments/batch_12_time_isolated_negatives
```

任一 H1/L1 命中即为网络命中；连续四秒命中窗口合并为一次告警。输出包括：

- `network_windows.csv`：合格网络窗口及 H1/L1 决策；
- `ineligible_windows.csv`：未进入分母的窗口与原因；
- `negative_audit_summary.json`：有效网络天数、告警数、`false alarms/day` 点估计和
  Poisson 单侧置信上界。

即使零告警也必须报告上界。单侧 95% 上界要不超过 0.1 次/天，零告警时至少需要
29.96 个、工程上取 30 个有效网络日。

## 当前 v1 证据边界

`production_audit_v1` 有 900 张 H1/L1 图像，对应 450 个同步四秒网络窗口，
总网络暴露仅 1,800 秒。冻结 baseline@640、阈值 0.20 后有两个 H1 候选、L1 为零，
连续窗口合并后仍为两个候选告警；点估计为 96.0 次/天，单侧 95% 上界为
302.20 次/天。

该批次的 `verified_no_chirp` 只是 GWTC 目录否决，且没有可审计的开发时间隔离和双人
复核，因此只能称为“当前负集候选审计”，不能发布为生产 FAR，也不得用于调阈值。

## 发布门槛

冻结训练数据、模型权重 SHA-256、阈值、图片生成参数和告警合并规则后才收集正式审计
数据。发布时联合报告固定 GW5 事件召回、时间隔离负集 FAR 上界、延迟和显存；架构结论
还需完成 baseline/P4/P3+P4 × seed 0/1/2 × 300 epoch 的受控训练。
