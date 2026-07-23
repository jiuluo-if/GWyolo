# GW-YOLO 项目记忆

> 最后梳理：2026-07-23。本文记录当前仓库事实与维护约定，供后续开发、训练和交接使用。

## 1. 项目目标

项目使用 Ultralytics YOLO 分割模型处理时频图，检测两类目标：

| 类别 ID | 名称 | 处理约定 |
| --- | --- | --- |
| 0 | `chirp` | 推理导出时每张图仅保留置信度最高的一个 |
| 1 | `noise` | 推理导出时保留所有检测框 |

## 2. 当前工作流

```text
原始图像 -> img2640.py 预处理（可选，缩放为 640×640）
         -> train/images + train/labels -> trian.py -> runs/segment/.../weights/best.pt
待预测图像 -> predict.py -> runs/filtered/<任务名>/{images,labels}
单图时频分析 -> analysis.py + line_find.py -> 根据坐标轴映射 chirp 时间
```

## 3. 关键文件

| 文件 | 作用 | 注意事项 |
| --- | --- | --- |
| `gw_data.yaml` | 数据集路径、类别数和类别名 | 当前为 Windows 相对路径；训练/验证目录必须存在 |
| `trian.py` | YOLO 分割训练入口 | 名称拼写为历史遗留；使用 `yolo26m-seg.pt`、300 epochs、640 图像尺寸、batch=8 |
| `predict.py` | 批量推理与导出 | 顶部为硬编码绝对路径；导出的是检测框 YOLO 标签（含置信度），不是分割多边形 |
| `analysis.py` | 单图 chirp 检测和时间/频率坐标映射 | 路径硬编码，依赖 GUI `cv2.imshow` |
| `line_find.py` | Hough 直线检测，用于定位图表坐标轴 | 可能在未检测到足够水平/垂直线时返回空列表 |
| `color.py` | 分割掩码可视化实验脚本 | 路径硬编码，依赖 GUI |
| `img2640.py` | 批量缩放预处理工具 | 当前启用 `resize_fill`，会改变原始长宽比；如需保真应切换 letterbox |

## 4. 数据与产物现状

- 训练集约 663 个文件，标签在 `train/labels`，图像在 `train/images`。
- 验证目录包含 `val_3.0`、`val_4.0_1`、`val_4.0_2`；当前配置只使用 `val/val_3.0`。
- `runs/` 同时包含 detect、segment、filtered 的训练和预测产物；不要把它作为稳定的源码依赖路径。
- 根目录有多个 `.pt` 权重文件；应在实验记录中注明每次使用的权重来源、数据版本和指标。

## 5. 已知风险与维护规则

1. **路径可移植性**：多个脚本使用 `F:\\...` 绝对路径。迁移机器、共享代码或容器运行前必须改为配置项/相对路径。
2. **任务一致性**：训练基座是 `*-seg.pt`，但 `predict.py` 仅导出 boxes。若业务需要轮廓，应明确导出 masks 或多边形标签。
3. **数据缩放**：`resize_fill` 会拉伸时频图，可能改变目标形状和时频比例；训练与推理必须使用同一种预处理策略。
4. **标签格式**：推理输出每行有第六列 confidence；标准 YOLO 训练检测标签通常只接受前五列，不能直接回灌训练集。
5. **坐标轴检测**：`analysis.py` 假定 `line_find.py` 找到足够的边框线，使用前需对空结果做校验。
6. **Windows 训练**：训练脚本设置 `workers=0`，这是 Windows 下避免多进程 DataLoader 问题的保守选择。

## 6. 渐进式重构计划

不直接移动现有文件与数据，以免中断正在使用的绝对路径。下一轮功能改造建议按以下顺序进行：

1. 新建 `configs/`，将权重、数据路径、阈值和输出目录移出 Python 源码。
2. 提取 `src/gw_yolo/`：数据路径解析、预测过滤、标签写入、坐标轴映射分别模块化。
3. 新建 `scripts/train.py`、`scripts/predict.py`，保留 `trian.py` 一段过渡时间作为兼容入口。
4. 将数据映射为 `data/{train,val}/{images,labels}`，将运行结果归档到 `artifacts/`；完成全局路径验证后再迁移实体文件。
5. 增加 `requirements.txt`/锁文件、最小 smoke test，以及每次实验的参数与指标记录。

## 7. 运行前检查清单

- `gw_data.yaml` 的 `train`、`val` 路径与文件实际位置一致。
- 图像和同名标签一一对应，类别 ID 仅为 0 或 1。
- 训练、验证、推理采用相同的尺寸和预处理策略。
- `predict.py` 指向与任务一致的 `best.pt`，且输出目录不是需要保留的旧实验目录。
- 使用 `analysis.py` 前确认图表的横轴时长、纵轴频率范围及坐标线参数。
