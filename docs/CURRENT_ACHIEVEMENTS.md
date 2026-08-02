# GW-YOLO 当前成果总表

> 截止日期：2026-07-30
> 证据范围：项目记忆、Git 提交与 PR、仓库文档/测试，
> 以及 `F:\python\Prod_negative_audit` 中冻结的 v1 候选审计产物。

## 一、已经完成并有证据支持的成果

| 方向 | 当前成果 | 主要证据 |
| --- | --- | --- |
| GW5 事件审计 | 建立事件级规则：排除质量字段 `--`；H1/L1/V1 任一图像检出 class 0 `chirp` 即召回 | `evaluate_gw5_recall.py`、`docs/gw5_recall_details.csv` |
| 历史 C2PSA 复现 | 旧 C2PSA 权重在 429 张图像上为 88/104（84.62%）；该值仅保留为历史快照 | `docs/GW5_MODEL_REPORT.md` |
| 单模型对比 | 固定验证集上 baseline 优于完整 C2PSA；Attention Residual 不替换 baseline，只保留为互补研究成员 | `docs/FINAL_RESEARCH_REPORT.md` |
| 非对称级联 | 仅用固定验证集选择 baseline/Attention 阈值 0.14/0.36；验证 68/3/1/11，旧 429 图 GW5 为 100/104 | `docs/experiments/batch_06_asymmetric_calibration/` |
| GW5 覆盖修复 | 补齐 `GW240922_142106` 的 H1/L1/V1，图像从 429 增至 432；冻结策略变为 101/104，提升来自数据完整性而非模型 | `docs/experiments/batch_07_completed_images/` |
| 跨尺度一致性 | 接受 `baseline640>=0.14 OR attention640>=0.36 OR (baseline512>=0.15 AND baseline768>=0.07)`；验证 69/3/0/11，GW5 102/104，177/432 图像命中 | `docs/experiments/batch_09_scale_consensus/` |
| 短路运行时 | 顺序执行 baseline640 → Attention640 → baseline512 → 条件 baseline768；GW5 仅 8/259 个多尺度候选调用 768，总耗时 33.55 秒，相对四成员全量约降低 47.07% | `docs/experiments/batch_10_short_circuit_runtime/` |
| 可审计推理 | `predict.py` 要求显式权重、输入和新输出目录；逐图预测并复用模型；无检出也写空标签；输出 `prediction_manifest.json` | `predict.py` |
| GW5 覆盖审计 | 评估器支持 `--images`，记录输入数、缺失标签、孤立标签和 H1/L1/V1 覆盖差异 | `evaluate_gw5_recall.py` |
| 消融汇总 | `scripts/summarize_ablation.py` 强制 baseline/P4/P3+P4 各具备预期 seed 0/1/2，再输出均值、标准差和原始行 | `scripts/summarize_ablation.py` |
| 受控训练入口 | 3 结构 × 3 种子共享数据、初始化、优化器、增强、尺寸、epoch 和 batch；兼容分割头迁移 362/376 张量 | `scripts/train_attention_residual_ablation.py` |
| 长跑可靠性 | 训练入口增加数据/配置/权重 SHA-256、任务状态、完成标记、真实 `save_dir` 校验和 `--resume` 安全续训 | `docs/TRAINING_ABLATION_GUIDE.md` |
| 端到端烟雾测试 | baseline/P4/P3+P4 各完成 seed0、1 epoch；耗时约 55.51/53.28/73.80 秒，峰值显存约 2.52/2.50/5.44 GiB | `docs/experiments/batch_11_training_smoke/` |
| 负集审计门槛 | 建立成对 H1/L1、GWTC 否决、开发时间隔离、双人复核、连续四秒告警合并和 Poisson 上界口径 | `docs/TIME_ISOLATED_NEGATIVE_AUDIT.md` |

## 二、当前生产负样本候选审计

冻结条件：

- 权重：`runs/segment/train/weights/best.pt`
- 权重 SHA-256：`e5e0b03ed0106cbac9b7e4c904a5abe2a17a65ca19e007884fd96cc773fba884`
- 推理：baseline、`imgsz=640`、class 0、阈值 0.20、逐图 batch 1
- 输入：`production_audit_v1`，H1 450 + L1 450，共 450 个同步四秒网络窗口
- 未修改训练集、验证集、负样本图像和图片生成规则，也未使用审计结果调阈值

结果：

| 指标 | 数值 |
| --- | ---: |
| H1 超阈值图像 | 2/450 |
| L1 超阈值图像 | 0/450 |
| 网络命中窗口 | 2/450 |
| 连续窗口合并后候选告警 | 2 |
| 有效网络暴露 | 1,800 秒，即 0.020833 天 |
| 候选告警点估计 | 96.0 次/天 |
| Poisson 单侧 95% 上界 | 302.20 次/天 |
| 900 张墙钟耗时 | 30.82 秒 |

这是候选率证据，不是生产 FAR。v1 的 `verified_no_chirp` 只是 GWTC 目录否决，
缺少双人独立复核和可审计的开发时间隔离。零告警条件下若要把单侧 95% FAR 上界压到
0.1 次/天，至少需要 29.96 个、工程上取 30 个有效网络日。

## 三、已拒绝或明确不能宣称的结论

- 不把 GW5 探测器图像命中数解释为 FPR/FAR；GW5 没有独立纯负观测。
- 不把补齐三张图像造成的 100/104 → 101/104 解释为模型提升。
- 不接受只在 GW5 增益、但固定验证仍有 FN 的最大分数多尺度救援。
- 不用 1-epoch 烟雾指标或单个最佳 seed 排名结构。
- 不把 v1 的 96.0 候选/天解释为稳定生产误报率。
- 不在审计负集上重新选择阈值；生产默认仍冻结为 baseline@640、0.20。
- 不向 Ultralytics 传路径列表微批量；保持一图一次调用并复用模型。

## 四、尚未完成的关键工作

1. 完成 baseline、P4、P3+P4 × seed 0/1/2 × 300 epoch 的九个正式训练任务。
2. 对每个新权重只在固定验证集重新校准阈值；GW5 继续作为冻结后的盲评。
3. 汇总三种子均值/标准差、GW5 事件召回、时间隔离负集 FP/FAR、延迟与显存。
4. 建立至少 30 个有效网络日的成对 H1/L1 v2 负集，完成 GWTC 否决、开发时间隔离和双人复核。
5. 只有上述证据齐备后，才讨论 Attention Residual 的结构收益或生产工作点变更。

## 五、发布历史

- PR #2：Attention Residual 校准与受控训练入口，已合并。
- PR #3：跨尺度一致性、短路运行时和烟雾训练证据，已合并。
- 提交 `9eb916a`：可审计推理、GW5 覆盖审计和多种子汇总流程，已推送到
  `codex/iterative-optimization`。

本表是对外发布的当前入口；更细的历史结论见 `docs/archive/`，可复现依据以
`docs/experiments/` 的原始产物和对应脚本为准。
