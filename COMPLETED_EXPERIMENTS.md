# 已完成实验 / Completed Experiments

更新日期 / Updated: 2026-11-19

## 中文版

| 工作 | 实验目的 | 已完成范围 | 结果与证据 |
|---|---|---|---|
| 数据准备 | 核实标签、类别与来源，减少跨集合重复和数据泄漏，保证模型比较公平 | GenImage/ImageNet 盘点、类别映射核实、重复与来源组审计、固定训练/验证/测试清单，以及额外 6,000 张图片的 holdout | [固定清单](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/manifests) · [数据审计](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/inventory/README.md) |
| 基线模型 | 建立参考性能，比较 CNN 学习到的检测特征与预训练 CLIP 特征 | ImageNet 预训练 ResNet-50，以及冻结 CLIP ViT-B/32 加线性分类头 | [训练记录](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/BASELINE.md) |
| 鲁棒训练 | 检查图像增强能否提高抗退化能力，并衡量干净图片性能的取舍 | 初版增强实验，以及混合干净图与退化图训练的 V2 | [方法与开发报告](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/FINAL_REPORT.md) |
| 三种子评估 | 衡量训练随机性，检查结果和改进是否稳定 | ResNet、V2、CLIP 各使用 20261031、20261101、20261102 三个种子 | [主实验报告](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/FINAL_REPORT.md) |
| 跨生成器检测 | 检查独立图片上的基本检测能力，以及对训练中未见生成器的迁移能力 | 已见 BigGAN/SD1.4/ADM，以及分别评估的未见 GLIDE/VQDM | [逐种子结果](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/per_seed_results.csv) |
| 图像扰动鲁棒性 | 测量压缩、缩放和模糊后的性能下降，检验鲁棒训练的实际收益 | 固定 6,000 张 holdout 图片的七种条件，包含干净图片、JPEG、缩放和模糊 | [Holdout 结果](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/holdout_results.csv) |
| 概率校准 | 检查得分与实际真假比例的对应关系，以及校准效果的跨域迁移 | 九个模型在验证集拟合温度缩放，并评估 Brier、NLL、ECE | [校准结果](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/calibration_results.csv) |
| 外部数据评估 | 检查图片来源、内容和生成流程变化后的泛化表现 | CIFAKE 官方测试集全部 20,000 张，以及 COCOXGEN 全部 5,305 张 | [外部结果与局限](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/FINAL_REPORT.md) |
| 统计分析 | 衡量训练波动与测试抽样不确定性，核对评估实现的一致性 | 种子样本标准差、配对成组 bootstrap 区间、推理一致性检查 | [主实验证据包](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/handoff/2026-11-08/evidence) |
| 分数诊断 | 区分排序能力下降、分数分布偏移和固定阈值失效等现象 | 按真假标签统计分位数、ECDF 图、排序指标和固定阈值混淆计数 | [分数分析](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/score_diagnosis/REPORT.md) |
| 分辨率控制 | 固定图片、模型和阈值，检查指定低分辨率处理造成的退化，为 CIFAKE 失败寻找证据 | 一个 ResNet 与三个 V2，三种测试集 × original/64/32，共 36 组评估；64/32 为保持长宽比的最长边上限 | [分辨率实验输出](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/handoff/2026-11-16/resolution_control) |
| 阈值适配 | 量化目标域阈值调整能缓解多少漏检，以及带来的真图误报代价 | 按 COCO 原图组划分 318 组开发集、743 组保留集，每个模型在开发集选择一个共用阈值 | [阈值适配结果](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/threshold_transfer/summary.csv) |
| 生成配置分析 | 检查检测难度是否随生成流程和提示词配置变化 | 在保留集比较 Fooocus/SDXL × 长/短提示词四种配置 | [生成配置分析](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/threshold_transfer/CELL_REPORT.md) |

分数诊断、分辨率控制、阈值适配和生成配置分析均在外部结果已查看后开展，属于事后诊断。生成配置差异目前为描述性结果，具体因果贡献仍需进一步控制实验。

## English version

| Work | Purpose | Completed scope | Results and evidence |
|---|---|---|---|
| Data preparation | Verify labels, classes, and sources, reduce duplicates and leakage across splits, and support fair model comparisons | GenImage/ImageNet inventory, class mapping verification, duplicate and source-group audits, fixed training/validation/test manifests, and an additional 6,000-image holdout | [Fixed manifests](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/manifests) · [Data audit](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/inventory/README.md) |
| Baseline models | Establish reference performance and compare learned CNN detection features with pretrained CLIP features | ImageNet-pretrained ResNet-50 and a frozen CLIP ViT-B/32 encoder with a linear classification head | [Training record](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/BASELINE.md) |
| Robust training | Assess whether augmentation improves resistance to image degradation and measure the tradeoff in clean-image performance | Initial augmentation experiments and V2 training with a mixture of clean and degraded images | [Methods and development report](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/FINAL_REPORT.md) |
| Three-seed evaluation | Measure training variability and assess the stability of results and improvements | ResNet, V2, and CLIP evaluated with seeds 20261031, 20261101, and 20261102 | [Main experimental report](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/FINAL_REPORT.md) |
| Cross-generator detection | Assess detection on independent images and transfer to generators absent from training | Seen BigGAN/SD1.4/ADM and separately evaluated unseen GLIDE/VQDM | [Per-seed results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/per_seed_results.csv) |
| Image perturbation robustness | Measure performance degradation under compression, resizing, and blur, and evaluate the benefits of robust training | Seven conditions on a fixed 6,000-image holdout, including clean inputs, JPEG, resizing, and blur | [Holdout results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/holdout_results.csv) |
| Probability calibration | Assess correspondence between scores and observed class frequencies, including calibration transfer across domains | Validation-fitted temperature scaling for nine models, evaluated using Brier score, NLL, and ECE | [Calibration results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/tables/calibration_results.csv) |
| External dataset evaluation | Assess generalization under changes in image sources, content, and generation pipelines | All 20,000 official CIFAKE test images and all 5,305 COCOXGEN images | [External results and limitations](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-08/FINAL_REPORT.md) |
| Statistical analysis | Quantify training variability and test-sampling uncertainty, and check inference consistency | Seed sample standard deviations, paired cluster bootstrap intervals, and inference consistency checks | [Main experiment evidence](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/handoff/2026-11-08/evidence) |
| Score diagnosis | Identify ranking degradation, score-distribution shifts, and fixed-threshold failures | Classwise score quantiles, ECDF plots, ranking metrics, and fixed-threshold confusion counts | [Score analysis](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/score_diagnosis/REPORT.md) |
| Resolution control | Hold images, models, and thresholds fixed to assess degradation caused by the specified low-resolution processing and investigate CIFAKE failure | One ResNet and three V2 checkpoints; three test sets × original/64/32, totaling 36 evaluations; 64/32 denote aspect-ratio-preserving longest-side caps | [Resolution experiment outputs](https://github.com/jshui1015/AI-Image-Detection-Genimage/tree/master/handoff/2026-11-16/resolution_control) |
| Threshold adaptation | Quantify how much target-domain threshold adjustment reduces missed detections and increases real-image false positives | COCO-source-group split into 318 development groups and 743 holdout groups; one shared threshold selected per model on development data | [Threshold adaptation results](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/threshold_transfer/summary.csv) |
| Generation configuration analysis | Assess whether detection difficulty varies with generation pipelines and prompt configurations | Holdout comparisons of four configurations combining Fooocus/SDXL with short/long prompts | [Generation configuration analysis](https://github.com/jshui1015/AI-Image-Detection-Genimage/blob/master/handoff/2026-11-16/threshold_transfer/CELL_REPORT.md) |

Score diagnosis, resolution control, threshold adaptation, and generation configuration analysis were conducted after the external results had been inspected and constitute post-hoc diagnostics. Configuration differences are currently descriptive; their causal contributions require further controlled experiments.
