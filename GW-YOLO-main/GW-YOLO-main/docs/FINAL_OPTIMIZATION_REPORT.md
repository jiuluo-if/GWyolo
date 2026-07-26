# GW-YOLO 完整迭代优化报告

> 日期：2026-07-26
> 分支：`codex/iterative-optimization`
> 前提：保留 Attention Residual（P3/P4 C2PSA）研究成员，不改变既有训练集。

## 1. 最终结论

本轮把项目整理为 `configs/`、`scripts/`、`tests/`、`docs/experiments/` 四层实验结构，
建立了固定验证、GW5 事件审计、多尺度/多模型融合、成本校准、按需级联、FP16 消融和
非对称阈值校准的可复现链路。

最终结果：

- 单模型默认仍保留 baseline；当前 Attention Residual 单模型不能替换 baseline。
- 高召回研究策略更新为 baseline=0.14、Attention Residual=0.36 的非对称级联。
- 固定验证集为 TP=68、FP=3、FN=1、TN=11；没有超过 baseline@0.20 的 FP 代理。
- GW5 事件召回为 **100/104（96.15%）**，命中 171/429 张探测器图像。
- FP32 实测耗时 27.21 秒，比完整双模型 34.94 秒节省 22.11%。
- 仅继续降低现有两模型阈值已经到达 100/104 的瓶颈。

## 2. 固定评估口径

| 项目 | 固定值 |
| --- | --- |
| 训练/验证图像 | 331 / 83 |
| GW5 探测器图像 | 429 |
| PDF 目录事件 | 161 |
| 排除质量不完整事件 | 57 |
| 纳入事件 | 104 |
| 事件召回 | H1/L1/V1 任一图像检出 class 0 chirp |
| 验证阴性代理 | 14 张无 chirp 标注图像 |
| 当前 FP 上限 | 3/14 |

GW5 是正事件目录，不含独立纯负观测区间。报告中的“探测器图像命中数”只能表示复核
工作量代理，不能解释为真实误报率。

## 3. 批次结果

| 批次 | 新技术 | 关键结果 | 决策 |
| --- | --- | --- | --- |
| 00 | 固定验证复核 | C2PSA chirp mask mAP50-95 0.41077，低于 baseline 0.44584 | 否决单模型替换 |
| 01 | 多尺度与模型后融合 | baseline 多尺度 96/104；双模型 640 为 99/104；全融合 100/104 | 保留研究模式 |
| 02 | Pareto 成本校准 | 按命中代理和耗时选择工作点 | 建立可解释档位 |
| 03 | 验证阴性代理 | 共阈值融合 FP 从 3 增至 5 | 默认回退 baseline@0.20 |
| 04 | baseline→Attention Residual 级联 | 保持 99/104，FP32 耗时降低 18.87% | 接受计算优化 |
| 05 | FP16 与 micro-batch | FP16 batch1 保持事件召回；batch2 置信度塌缩 | 接受 FP16，拒绝 batch2 |
| 06 | 非对称阈值级联 | 验证 68/3/1/11；GW5 100/104；27.21 秒 | 接受高召回策略 |

每批原始 CSV、JSON 和中文报告均位于 `docs/experiments/batch_00_*` 至
`docs/experiments/batch_06_*`。

## 4. Attention Residual 结论

Attention Residual 配置在 P3/P4 路径加入 C2PSA。它包含残差自注意力与前馈连接，
用于补充细弱轨迹的全局连续性。现有单种子权重在固定验证集上的平均分割质量和速度都
弱于 baseline，因此不能证明结构本身更优。

它的有效价值来自错误互补：baseline 漏检的部分 GW5 事件能被 Attention Residual
召回。非对称校准避免让两个置信度分布不同的模型共用阈值，并把第二模型限制在
baseline 的低置信度区域。

## 5. 最终非对称级联

决策规则：

```text
baseline_score >= 0.14
OR
(baseline_score < 0.14 AND attention_residual_score >= 0.36)
```

阈值只在固定验证集上从 0.01 至 0.60、步长 0.01 的 3600 个组合中选择。排序先最大化
TP，再约束 FP<=3，最后选择仍可行的最低阈值；GW5 不参与选参。

| 指标 | baseline@0.20 | 非对称级联 |
| --- | ---: | ---: |
| 验证 TP/FP/FN/TN | 68/3/1/11 | 68/3/1/11 |
| GW5 事件召回 | 90/104 | **100/104** |
| GW5 图像命中 | 144/429 | 171/429 |
| Attention Residual 调用 | 0 | 273/429 |
| FP32 总耗时 | 约 14–15 秒 | 27.21 秒 |

## 6. 瓶颈判定

在 3600 个组合中，1175 个满足 FP<=3；这些组合的验证 TP 上限为 68，GW5 召回上限
为 100/104。

剩余事件：

- `GW240601_231004`：Attention Residual 最高分 0.02939。
- `GW240922_142106`：`imgs/gw5.0` 中没有对应探测器图像。
- `GW241009_022835`：两模型最高分 0.03192 / 0.03827。
- `GW241114_235258`：两模型最高分 0.01400 / 0.02221。

继续在同一批分数上降低阈值会先破坏验证 FP 约束，无法可靠越过当前上限。

## 7. 工程产物

- `scripts/benchmark_validation.py`：固定验证集模型复核。
- `scripts/benchmark_gw5.py`：低阈值一次推理、阈值扫描与融合。
- `scripts/select_operating_points.py`：Pareto 工作点选择。
- `scripts/benchmark_cascade.py`：支持独立 `--secondary-threshold` 的按需级联。
- `scripts/calibrate_attention_fusion.py`：验证集独立非对称阈值校准。
- `tests/`：13 个纯逻辑单元测试。
- `docs/PROJECT_MEMORY.md`：项目口径、失败记录、当前瓶颈和后续约束。

## 8. 验证

```powershell
python -m unittest discover -s tests -v
```

结果：13/13 通过。批次 06 的离线重放与真实 FP32 推理完全一致：

- 验证：68/3/1/11。
- GW5：100/104，171/429。
- 任一探测器命中规则和 57 个质量不完整事件排除规则保持不变。

## 9. 发布建议

可发布本轮代码、测试和研究报告。生产默认仍应使用 baseline@0.20，直到获得按观测
时间隔离的纯负样本并报告 false alarms/day。高召回研究模式可使用非对称级联；
FP16 只允许 `chunk_size=1`。

下一轮最小充分实验是补齐缺失图像和纯负样本，冻结数据、优化器、增强和 epoch，
对 baseline、仅 P4 Attention Residual、P3+P4 Attention Residual 各运行至少三个
随机种子，再联合报告均值、标准差、GW5 召回、负集误报、延迟和显存。
