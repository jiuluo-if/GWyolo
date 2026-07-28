# GW-YOLO 完整迭代优化报告

> 日期：2026-07-28
> 分支：`codex/iterative-optimization`
> 前提：保留 Attention Residual（P3/P4 C2PSA）研究成员，不改变既有训练集。

## 1. 最终结论

本轮把项目整理为 `configs/`、`scripts/`、`tests/`、`docs/experiments/` 四层实验结构，
建立了固定验证、GW5 事件审计、多尺度/多模型融合、成本校准、按需级联、FP16 消融和
非对称阈值校准的可复现链路。

在补齐图像并继续完成批次 07–11 后，最终结果更新为：

- 单模型默认仍保留 baseline；当前 Attention Residual 单模型不能替换 baseline。
- 高召回研究策略为 `baseline640>=0.14 OR attention640>=0.36 OR
  (baseline512>=0.15 AND baseline768>=0.07)`。
- 固定验证集为 **TP=69、FP=3、FN=0、TN=11**，在当前 FP 代理不变时召回 100%。
- 补图后的 GW5 事件召回为 **102/104（98.08%）**，命中 177/432 张探测器图像。
- 短路 FP32 级联实测 33.55 秒；相对四成员全量推理约 63.38 秒降低 47.07%。
- 验证已无 FN，剩余两事件现有六路最高分仅 0.09440/0.04752；现有权重的推理层
  优化到达瓶颈。

## 2. 固定评估口径

| 项目 | 固定值 |
| --- | --- |
| 训练/验证图像 | 331 / 83 |
| GW5 探测器图像 | 432 |
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
| 07 | 补图后冻结盲测 | 缺失事件由 baseline 找回；GW5 101/104 | 接受数据修复 |
| 08 | 最大值多尺度救援 | 验证无提升，GW5 的偶然 +1 不作为选参依据 | 拒绝 |
| 09 | 跨尺度一致性 | 验证 69/3/0/11；GW5 102/104 | 接受 |
| 10 | 短路调度 | 保持 102/104，仅 8/259 候选调用 768 | 接受计算优化 |
| 11 | 三结构烟雾训练 | 三结构 1 epoch 均完成，发现并修复 project 路径嵌套 | 接受运行性 |

每批原始 CSV、JSON 和中文报告均位于 `docs/experiments/batch_00_*` 至
`docs/experiments/batch_11_*`，总索引见 `docs/experiments/README.md`。

## 4. Attention Residual 结论

Attention Residual 配置在 P3/P4 路径加入 C2PSA。它包含残差自注意力与前馈连接，
用于补充细弱轨迹的全局连续性。现有单种子权重在固定验证集上的平均分割质量和速度都
弱于 baseline，因此不能证明结构本身更优。

它的有效价值来自错误互补：baseline 漏检的部分 GW5 事件能被 Attention Residual
召回。非对称校准避免让两个置信度分布不同的模型共用阈值，并把第二模型限制在
baseline 的低置信度区域。

## 5. 最终短路一致性级联

决策规则：

```text
baseline_score >= 0.14
OR
(baseline_score < 0.14 AND attention_residual_score >= 0.36)
OR
(前两级未命中 AND baseline512 >= 0.15 AND baseline768 >= 0.07)
```

前两级阈值来自批次 06；新增 512/768 阈值只在固定验证集上从 0.01 至 0.60、
步长 0.01 的 3600 个组合中选择。排序先最大化 TP，再约束 FP<=3，同指标选择更高
组合阈值；GW5 不参与选参。

| 指标 | baseline@0.20 | 非对称级联 |
| --- | ---: | ---: |
| 验证 TP/FP/FN/TN | 68/3/1/11 | **69/3/0/11** |
| GW5 事件召回 | 91/104 | **102/104** |
| GW5 图像命中 | 146/432 | 177/432 |
| Attention Residual 调用 | 0 | 274/432 |
| 768 调用 | 0 | 8/432 |
| FP32 总耗时 | 约 14 秒 | 33.55 秒 |

## 6. 瓶颈判定

补图后数据覆盖硬上限已消失。跨尺度一致性已找回唯一验证 FN，验证阳性达到 69/69。
当前剩余事件：

- `GW240601_231004`：双模型三尺度最高分 0.09440（Attention@768）。
- `GW241114_235258`：双模型三尺度最高分 0.04752（Attention@768）。

验证集已无可用于接受新救援规则的 FN；继续利用 GW5 选择低阈值将产生测试集泄漏。
因此现有权重的推理优化到达方法学瓶颈，下一步必须依赖新权重或新的独立校准/负集。

## 7. 工程产物

- `scripts/benchmark_validation.py`：固定验证集模型复核。
- `scripts/benchmark_gw5.py`：低阈值一次推理、阈值扫描与融合。
- `scripts/select_operating_points.py`：Pareto 工作点选择。
- `scripts/benchmark_cascade.py`：支持独立 `--secondary-threshold` 的按需级联。
- `scripts/calibrate_attention_fusion.py`：验证集独立非对称阈值校准。
- `scripts/calibrate_multiscale_rescue.py`：最大值救援负向消融。
- `scripts/calibrate_scale_consensus.py`：跨尺度 AND 一致性校准。
- `scripts/benchmark_scale_consensus_cascade.py`：短路实测级联。
- `tests/`：覆盖事件口径、校准、短路和训练目录契约的纯逻辑测试。
- `docs/PROJECT_MEMORY.md`：项目口径、失败记录、当前瓶颈和后续约束。

## 8. 验证

```powershell
python -m unittest discover -s tests -v
```

结果：31/31 通过；新增测试覆盖跨尺度接受规则、GW5 事件聚合、短路等价性和绝对
训练输出路径契约。

批次 09/10 的离线重放与真实 FP32 推理完全一致：

- 验证：69/3/0/11。
- GW5：102/104，177/432。
- 任一探测器命中规则和 57 个质量不完整事件排除规则保持不变。

## 9. 发布建议

可发布本轮代码、测试和研究报告。生产默认仍应使用 baseline@0.20，直到获得按观测
时间隔离的纯负样本并报告 false alarms/day。高召回研究模式可使用短路一致性级联；
FP16 只允许 `chunk_size=1`。

下一轮最小充分实验是补齐缺失图像和纯负样本，冻结数据、优化器、增强和 epoch，
对 baseline、仅 P4 Attention Residual、P3+P4 Attention Residual 各运行至少三个
随机种子，再联合报告均值、标准差、GW5 召回、负集误报、延迟和显存。
