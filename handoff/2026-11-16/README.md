# 外部泛化失败诊断补充报告

本报告整理分数分布诊断、分辨率控制实验、COCOXGEN 阈值迁移与生成配置分析。基础预测来自 2026-11-08 交接包。新增分辨率实验为 Slurm 作业 2311461，四个数组任务均 COMPLETED，退出码均为 0:0。

## 1 研究问题与实验范围

已有实验发现 GenImage 内部检测表现较高，CIFAKE 与 COCOXGEN 外部表现明显下降。本轮检查分数排序、输入分辨率和分类阈值对失败现象的影响。

这些诊断在外部测试结果已经查看之后开展，属于事后探索。实验阈值适配使用固定开发部分，最终报告仍须披露完整外部数据此前已被查看。当前结果描述已有模型在指定数据上的行为，原因解释受实验控制范围限制。

## 2 分数分布诊断

分析 ResNet、V2、CLIP 各三个种子的已有逐图 logits，沿用各自 GenImage 验证阈值。按标签分别统计得分均值、分位数、召回率和误报率。COCOXGEN 根据正式清单提取 5,305 张图片，排除预测文件附带的旧域一致性抽查行。

| 模型 | CIFAKE AUROC | CIFAKE 假图召回率 | COCOXGEN AUROC | COCOXGEN 假图召回率 |
|---|---:|---:|---:|---:|
| ResNet | 0.3453 | 5.14% | 0.7360 | 0.94% |
| V2 | 0.4378 | 0.08% | 0.6972 | 0.63% |
| CLIP | 0.4524 | 0.85% | 0.8297 | 20.96% |

数值为三个种子均值。CIFAKE 上三类模型 AUROC 均低于 0.5，表明排序失效。COCOXGEN 上仍保留一定排序能力，但假图得分大幅降低，原阈值下大量漏检。COCOXGEN pooled AP 和普通 Accuracy 受 80% 假图比例影响，跨数据比较优先使用平衡单元宏平均。

详见 [得分报告](score_diagnosis/REPORT.md)、[逐模型指标](score_diagnosis/metrics.csv)、[分位数](score_diagnosis/score_quantiles.csv)。分数属于未经校准的模型得分。

## 3 分辨率控制实验

使用本地 release 中一个 ResNet 基线和三个 V2 检查点，逐一核对 SHA256 与锁定模型记录、验证阈值配对。覆盖已见测试 10,000 张、GLIDE 4,000 张、VQDM 4,000 张。同一图片的三个处理版本保持对应关系。

- original 使用现有 cap128_jpeg95 确定性预处理。
- low64、low32 先用 Lanczos 将最长边限制为 64 或 32 像素，保持比例且避免上采样，再执行 cap128_jpeg95。
- 后续统一使用短边 256 的双线性缩放、224 中心裁剪与 ImageNet 归一化。
- 三种条件均在 CPU FP32 环境重新推理，保持各检查点原验证阈值。

这里的 32/64 表示最长边上限。非正方形图片的短边会更小，具体尺寸已逐图记录。因此 low32 与 CIFAKE 原生 32×32 的图像形成过程存在差异。

| 测试集 | 模型 | original AUROC | low64 AUROC | low32 AUROC |
|---|---|---:|---:|---:|
| 已见 | ResNet | 0.9951 | 0.5311 | 0.3555 |
| 已见 | V2 三种子均值 | 0.9936 | 0.7852 | 0.4277 |
| GLIDE | ResNet | 0.9892 | 0.6550 | 0.3842 |
| GLIDE | V2 三种子均值 | 0.9850 | 0.8329 | 0.4824 |
| VQDM | ResNet | 0.9723 | 0.5539 | 0.5109 |
| VQDM | V2 三种子均值 | 0.9769 | 0.7736 | 0.5766 |

V2 已见测试的 low64 AUROC 为 0.7852 ± 0.1367，low32 为 0.4277 ± 0.0891，± 为种子样本标准差。对应假图召回率从原始条件 98.51% 降至 6.42% 与 0.51%。模型对这种分辨率处理敏感，V2 在 64 像素下保留更多排序能力，32 像素条件仍严重退化。

该实验支持低分辨率处理是影响因素。它尚无法量化分辨率对 CIFAKE 总体性能差距的独立贡献，也未覆盖 CLIP。不同分辨率下同源图片构成配对样本，派生版本不能计为新增独立图片。CPU 与历史 GPU 混合精度结果可能有轻微数值差异，本表以本轮 CPU original 为参照。

原始结果位于 [resolution_control](resolution_control/)，每个模型目录含 9 行 metrics.csv、9 个逐图 NPZ、实际输入示例 PNG 和 protocol.json。NPZ dimensions 列依次为原始宽、高、降采样宽、高。

## 4 COCOXGEN 阈值迁移诊断

按 COCO 原图组，以 seed 20261116 的 SHA256 排序固定划分 318 组开发数据与 743 组保留数据。同一组包含一张真图和四张生成图。开发集为每个模型选择一个使四个平衡单元宏平均平衡准确率最高的阈值，所有生成配置共用。并列时选择距离原 logit 阈值最近者。评估复用已有预测。

| 模型 | 阈值 | 平衡准确率 | 假图召回率 | 真图误报率 |
|---|---|---:|---:|---:|
| ResNet | 原始 | 50.45% | 0.99% | 0.09% |
| ResNet | 适配 | 66.11% | 66.72% | 34.50% |
| V2 | 原始 | 50.30% | 0.61% | 0.00% |
| V2 | 适配 | 63.71% | 59.40% | 31.99% |
| CLIP | 原始 | 59.57% | 20.70% | 1.57% |
| CLIP | 适配 | 74.76% | 70.61% | 21.09% |

均为保留集四单元宏平均，再对三个种子取均值。阈值调整后 AUROC 保持一致。减少漏检伴随大量真图误报，说明阈值偏移与分数重叠共同限制检测效果。适配结果应独立标注为目标域适配。

详见 [划分清单](threshold_transfer/split_manifest.csv)、[阈值](threshold_transfer/thresholds.csv)、[逐模型保留指标](threshold_transfer/holdout_metrics.csv)、[实验协议](threshold_transfer/protocol.json)。宏平均混淆计数为单元平均，共用真图不能跨单元累加为独立样本。

## 5 生成配置分析

同一保留集、同一模型与同一阈值下比较四种配置。每个单元含 743 张真图及 743 张假图。

| 模型 | Fooocus 短提示词 | Fooocus 长提示词 | SDXL 短提示词 | SDXL 长提示词 |
|---|---:|---:|---:|---:|
| ResNet | 0.7676 | 0.6672 | 0.7929 | 0.6828 |
| V2 | 0.7472 | 0.5868 | 0.7859 | 0.6283 |
| CLIP | 0.8485 | 0.8117 | 0.8549 | 0.7800 |

表中为三个种子平均 AUROC。长提示词配置在所有模型上均更低。生成器比较还伴随生成流程和文件处理变化，提示词对比也可能包含生成内容变化，因此当前仅报告相关差异。尚未进行配对显著性检验。

详见 [配置报告](threshold_transfer/CELL_REPORT.md)、[均值与种子标准差](threshold_transfer/cell_summary.csv)、[长短提示词差值](threshold_transfer/prompt_contrasts.csv)。

## 6 可支持的结论与限制

1. 指定的低分辨率处理能使同一批图片的检测性能大幅退化，为 CIFAKE 失败提供部分解释。
2. COCOXGEN 存在得分偏移，目标域阈值适配能改善平衡准确率，同时引入较高真图误报。
3. 不同生成配置检测难度存在差异，长提示词配置在当前评估中更难识别。
4. 分辨率实验仅含一个 ResNet 基线和三个 V2；其他分析覆盖三个模型家族各三种子。种子和数据范围须随表格明确说明。
5. 本轮尚未隔离内容来源、生成器与压缩历史的具体因果贡献。开发与保留划分是在完整测试已查看后建立，新增独立测试需要新的未查看来源。

## 7 目录与复现

```text
handoff/2026-11-16/
  README.md
  SHA256SUMS
  score_diagnosis/       得分指标、分位数、ECDF 图
  resolution_control/   四模型逐图预测、样例、配置与 Slurm 日志
  threshold_transfer/   来源组划分、阈值、分组指标与汇总
```

在项目根目录激活已有 myenv 后运行 CPU 分析。

```bash
python scripts/diagnose_score_shift.py
python scripts/diagnose_threshold_transfer.py
python scripts/analyze_cocoxgen_cells.py
```

脚本默认更新 outputs 下的结果。此处保存本轮输出快照。基础输入依赖 handoff/2026-11-08/evidence，环境使用已有 NumPy、Pillow、Matplotlib 和 PyTorch。

分辨率实验通过 Slurm 提交，四模型分别申请 8 CPU 与 16 GB 内存。先创建日志目录，再提交。

```bash
mkdir -p logs
sbatch --partition=amd scripts/run_resolution_cpu.sbatch
```

该作业依赖 release/GenImage_Final_2026-11-03/models 的四份完整权重和原始 GenImage 图片。结果默认位于 outputs/resolution_control/数组作业编号。四个任务均以 0:0 完成，本轮耗时约 26–28 分钟。
