# 实验批次索引

本目录按“读取 → 优化 → 测试 → 报告 → 留档”保存实验。每个批次只回答一个主要问题，
原始 CSV/JSON 与中文 `REPORT.md` 同目录存放。验证集负责选参与接受/拒绝决策，GW5
只在策略冻结后做事件级评估。

| 批次 | 主题 | 结论 |
| --- | --- | --- |
| 00 | 固定验证集基线复核 | Attention Residual 单模型不替换 baseline |
| 01 | 双模型与多尺度后融合 | 证明事件级互补，但成本和 FP 约束不足 |
| 02 | Pareto 工作点 | 建立研究用成本档位 |
| 03 | 14 张阴性代理 | 简单 OR 融合增加 FP |
| 04 | baseline→Attention 级联 | 接受按需调用 |
| 05 | FP16 与 micro-batch | 接受 FP16 batch1；拒绝路径 batch2 |
| 06 | 非对称阈值 | 接受 `0.14/0.36` 研究工作点 |
| 07 | 补齐缺图盲测 | 冻结策略从 100/104 提升至 101/104 |
| 08 | 最大值多尺度救援 | 验证无增益，拒绝 |
| 09 | 跨尺度一致性救援 | 验证 69/3/0/11，GW5 102/104，接受 |
| 10 | 短路一致性级联 | 保持决策，跳过 96.91% 的 768 调用 |
| 11 | 三结构 1-epoch 烟雾训练 | 三拓扑均可运行；不用于精度排名 |
| 12 | 时间隔离纯负集审计 | 待真实负集清单；输出 false alarms/day 及其置信上界 |

当前研究策略、边界与下一步见：

- `docs/CURRENT_ACHIEVEMENTS.md`
- `docs/archive/FINAL_OPTIMIZATION_REPORT.md`
- `docs/archive/FINAL_RESEARCH_REPORT.md`

生产结论仍需要按观测时间隔离的纯负样本与 `false alarms/day`。不得把 GW5 的探测器
图像命中率解释成 FPR。

时间隔离负集的清单格式、运行命令与发布门槛见
`docs/TIME_ISOLATED_NEGATIVE_AUDIT.md`。
