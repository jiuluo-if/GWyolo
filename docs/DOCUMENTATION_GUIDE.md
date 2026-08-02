# GW-YOLO 文档导航与维护规则

> 最后整理：2026-08-02。本文只定义文档入口和证据层级，不替代实验原始产物。

## 先读什么

1. [当前成果总表](CURRENT_ACHIEVEMENTS.md)：当前可主张的结果、证据边界与未完成事项；需要项目状态时以它为准。
2. [时间隔离负集网络审计](TIME_ISOLATED_NEGATIVE_AUDIT.md)：生产 FAR 的资格条件和执行口径。
3. [受控变量训练指南](TRAINING_ABLATION_GUIDE.md)：baseline/P4/P3+P4 的正式 3×3 消融入口与报告条件。
4. [实验批次索引](experiments/README.md)：各批次的原始结果和接受/拒绝决定。

## 文档分类

| 类别 | 文档 | 使用规则 |
| --- | --- | --- |
| 当前状态 | `CURRENT_ACHIEVEMENTS.md` | 作为成果、生产边界和下一步工作的唯一摘要入口。 |
| 操作规范 | `TIME_ISOLATED_NEGATIVE_AUDIT.md`、`TRAINING_ABLATION_GUIDE.md` | 作为审计和训练的执行说明；不要以口头结论替代其中的资格条件。 |
| 机器可读审计 | `gw5_recall_details.csv`、`gw5_recall_summary.json`、`time_isolated_negative_manifest_template.csv` | 原始或模板数据，不以人工改写替代脚本再生成。 |
| 历史快照 | `archive/` | 仅用于复现和追溯；其中的 429 图像、88/104、99/104、100/104 等旧工作点不得被表述为当前状态。 |

## 更新规则

- 新结果先写入对应的实验产物，再更新 `CURRENT_ACHIEVEMENTS.md`；需要保留来龙去脉时，在相应实验目录新增带日期的说明。
- 结构优劣必须基于固定训练集、统一评价规则和完整 3 结构 × 3 种子结果；烟雾训练或单个 seed 不能作为结构结论。
- GW5 的事件级规则固定为：排除质量字段 `--`，H1/L1/V1 任一图像检出 class 0 `chirp` 即为召回。
- `production_audit_v1` 是候选率证据，而不是生产 FAR；不得以其结果反向调参。
- 论文生成物、渲染页和同一图片的多格式导出属于 `output/` 的派生产物，不应再作为 `docs/` 的独立证据源。
