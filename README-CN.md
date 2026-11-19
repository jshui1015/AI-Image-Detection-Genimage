# 基于 GenImage 的 AI 图像检测

[English](README.md) · 简体中文

本项目研究 AI 生成图像检测，比较 ImageNet 预训练 ResNet-50、冻结 CLIP ViT-B/32 编码器加线性分类头，以及使用鲁棒增强训练的 ResNet-50（V2）。实验覆盖跨生成器检测、图像退化、概率校准和跨数据集失败分析。

项目状态更新于 **2026 年 11 月 19 日**。标签约定为 **真实图像 = 0，AI 生成图像 = 1**。检测器输出 AI 得分，其含义受评估数据分布与校准方法影响。

## 已完成的实验

| 工作 | 已完成范围 | 结果与证据 |
|---|---|---|
| 数据准备 | GenImage/ImageNet 盘点、类别映射核实、重复与来源组审计、固定训练/验证/测试清单，以及额外 6,000 张图片的 holdout | [固定清单](manifests/)与[数据审计](inventory/README.md) |
| 基线模型 | ImageNet 预训练 ResNet-50，以及冻结 CLIP ViT-B/32 加线性分类头 | [训练记录](BASELINE.md) |
| 鲁棒训练 | 初版增强实验，以及混合干净图与退化图训练的 V2 | [方法与开发报告](FINAL_REPORT.md) |
| 三种子评估 | ResNet、V2、CLIP 各使用 20261031、20261101、20261102 三个种子 | [主实验报告](handoff/2026-11-08/FINAL_REPORT.md) |
| 跨生成器检测 | 已见 BigGAN/SD1.4/ADM，以及分别评估的未见 GLIDE/VQDM | [逐种子结果](handoff/2026-11-08/tables/per_seed_results.csv) |
| 图像扰动鲁棒性 | 固定 6,000 张 holdout 图片的七种条件，包含 JPEG、缩放和模糊 | [Holdout 结果](handoff/2026-11-08/tables/holdout_results.csv) |
| 概率校准 | 九个模型在验证集拟合温度缩放，并评估 Brier、NLL、ECE | [校准结果](handoff/2026-11-08/tables/calibration_results.csv) |
| 外部数据评估 | CIFAKE 官方测试集全部 20,000 张，以及 COCOXGEN 全部 5,305 张 | [外部结果与局限](handoff/2026-11-08/FINAL_REPORT.md) |
| 统计分析 | 种子样本标准差、配对成组 bootstrap 区间、推理一致性检查 | [主实验证据包](handoff/2026-11-08/evidence/) |
| 分数诊断 | 按真假标签统计分位数、ECDF 图、排序指标和固定阈值混淆计数 | [分数分析](handoff/2026-11-16/score_diagnosis/REPORT.md) |
| 分辨率控制 | 一个 ResNet 与三个 V2，三种测试集 × original/64/32 条件，共 36 组评估 | [分辨率实验输出](handoff/2026-11-16/resolution_control/) |
| 阈值适配 | 按 COCO 原图组划分开发/保留集，每个模型在开发集选择一个阈值 | [阈值适配结果](handoff/2026-11-16/threshold_transfer/summary.csv) |
| 生成配置分析 | 在保留集比较 Fooocus/SDXL × 长/短提示词四种配置 | [最新诊断报告](handoff/2026-11-16/README.md) |

## 主要实验结果

下表为三个种子的平均 AUROC。

| 模型 | 已见生成器 | GLIDE | VQDM | CIFAKE | COCOXGEN |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 0.99533 | 0.99210 | 0.97419 | 0.34534 | 0.73601 |
| 鲁棒 V2 | 0.99360 | 0.98501 | 0.97690 | 0.43781 | 0.69718 |
| 冻结 CLIP + 线性分类头 | 0.97196 | 0.91804 | 0.92852 | 0.45244 | 0.82968 |

COCOXGEN 对四个平衡的“真图与生成配置”单元等权取宏平均。CIFAKE 包含 SD1.4，GenImage 训练来源也包含该生成器。这些外部评估衡量跨数据来源与生成流程的迁移能力。AUROC 描述分数排序，Accuracy 和 Recall 还受到分类阈值影响。

V2 改善了指定退化条件下的鲁棒性。额外 holdout 上，缩小一半再恢复的 AUROC 从 ResNet 的 **0.85638** 提升至 V2 的 **0.97682**；模糊半径 1.0 时从 **0.92977** 提升至 **0.97993**。干净图片表现存在小幅取舍。[鲁棒性分析](handoff/2026-11-08/FINAL_REPORT.md)

## 最新诊断发现

- **低分辨率处理造成明显退化。** 已见测试集上，ResNet AUROC 从 0.9951 降至 low64 的 0.5311 和 low32 的 0.3555。V2 三种子均值依次为 0.9936、0.7852、0.4277。处理保持长宽比、限制最长边，再执行原有 `cap128_jpeg95`。该实验为 CIFAKE 的 32×32 输入失效提供部分解释。[分辨率研究](handoff/2026-11-16/README.md)
- **阈值迁移解释了 COCOXGEN 失效的一部分。** 诊断保留集上，阈值适配使 ResNet/V2/CLIP 平衡准确率提升至 66.11%/63.71%/74.76%，同时真图误报率达到 34.50%/31.99%/21.09%。真假分数重叠仍限制检测表现。[阈值研究](handoff/2026-11-16/threshold_transfer/summary.csv)
- **不同生成配置的检测难度存在差异。** 所评估模型在长提示词版本上的平均 AUROC 更低。这些描述性比较同时涉及生成内容和处理流程变化。[生成配置研究](handoff/2026-11-16/threshold_transfer/CELL_REPORT.md)

最新诊断是在外部结果已经查看后开展。阈值适配使用 318 个 COCO 原图组开发、743 个原图组保留评估。完整数据此前已有查看记录，报告应将其表述为事后目标域分析。分辨率、内容、来源和压缩历史各自的因果贡献仍有待进一步研究。

## 报告与归档材料

- [实验交接说明](HANDOFF.md)连接原项目与后续补充工作。
- [11 月 8 日主报告](handoff/2026-11-08/FINAL_REPORT.md)包含九模型评估、校准、外部测试和置信区间。
- [11 月 16 日诊断报告与数据](handoff/2026-11-16/README.md)包含最新失败分析与 36 组分辨率评估。
- [原始方法与开发报告](FINAL_REPORT.md)保留最初方法及开发过程，并添加当前状态说明。
- [图表](handoff/2026-11-08/figures/)、[数据表](handoff/2026-11-08/tables/)与[写作指南](handoff/2026-11-08/WRITING_GUIDE.md)用于论文和展示。
- [Release 目录](release/GenImage_Final_2026-11-03/)包含原始代码、固定清单和 Git LFS 模型检查点。新增九模型检查点的可用位置见[资产索引](handoff/2026-11-08/ASSETS_AND_REPRODUCTION.md)。

原始图像需要单独下载或取得共享集群存储访问权限。归档 CLIP `.pt` 保存分类头，还需要对应 CLIP 编码器。完全哈希审计覆盖已记录的检查，近重复与预训练数据接触仍属于局限。

## 复现

环境配置、Hugging Face 数据下载、训练、评估与远程演示访问步骤见[复现说明](REPRODUCIBILITY.md)。详细说明当前使用中文。

在学校集群先加载模块，再激活已有环境。环境名和项目目录可按各自配置替换。

```bash
module load anaconda3/2023.09-0-biybti3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv
cd "/scratch/$USER/GenImage"
git lfs pull
```

使用 CPU 复算已保存预测的诊断。

```bash
python scripts/diagnose_score_shift.py
python scripts/diagnose_threshold_transfer.py
python scripts/analyze_cocoxgen_cells.py
```

分辨率实验提交至学校 AMD CPU 分区。这一步需要原始图片和 release 中的四份完整检查点。

```bash
mkdir -p logs
sbatch --partition=amd scripts/run_resolution_cpu.sbatch
squeue -u "$USER"
```

已完成的参考作业编号为 **2311461**，四个任务均成功退出，耗时约 26–28 分钟。新作业结果写入 `outputs/resolution_control/<数组作业编号>/`。参考逐图预测、输入样例、尺寸、指标、日志及校验清单见[诊断归档](handoff/2026-11-16/)。

## 项目目录

```text
GenImage/
├── README.md                         英文项目概览
├── README-CN.md                      中文项目概览
├── HANDOFF.md                        实验交接与当前状态
├── FINAL_REPORT.md                   原始方法与开发过程
├── REPRODUCIBILITY.md                环境配置与复现指南
├── BASELINE.md                       训练和评估记录
├── train.py                          ResNet-50 训练
├── train_robust_v2.py                 鲁棒 V2 训练
├── clip_detector.py                  冻结 CLIP 检测器
├── evaluate.py                       固定清单评估
├── evaluate_robustness_holdout.py     Holdout 扰动评估
├── scripts/                          数据准备与诊断脚本
├── manifests/                        固定划分与审计配置
├── inventory/                        数据盘点与审计说明
├── handoff/
│   ├── 2026-11-08/                   主报告、图表、数据表和预测
│   └── 2026-11-16/                   诊断报告、数据与样例
├── demo.py                           Streamlit 上传演示
├── demo_examples/                    示例输入
├── outputs/                          本地生成结果，Git 忽略
└── release/GenImage_Final_2026-11-03/ 原始复现归档
```
