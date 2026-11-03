# GenImage AI 图像检测项目

本项目使用固定 GenImage/ImageNet 清单，完成 ResNet-50 基线、冻结 CLIP 特征线性分类器、鲁棒增强实验、生成器与图像扰动评估，以及 Streamlit 演示。

## 复现指南

详细环境、数据前提和命令见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 交接说明

- 只看结论：发送 [`FINAL_REPORT.md`](FINAL_REPORT.md)。
- 审阅完整产物：发送 [`release/GenImage_Final_2026-11-03.tar`](release/GenImage_Final_2026-11-03.tar)，约405 MB；包内目录和文件说明见归档版 [`README.md`](release/GenImage_Final_2026-11-03/README.md)。
- 复现实验：除归档包外，还需要原始图像目录和兼容的 GPU/Conda 环境。原始图像没有包含在归档中；清单路径相对于项目根目录。共享集群上请使用队友自己有权限访问的项目根目录。CLIP 包内只有线性分类头，CLIP ViT-B/32 编码器权重需单独获取或使用已有缓存。

不要把最终测试结果用于继续挑选模型、阈值或增强配置。V2 的新 holdout 扰动测试覆盖 BigGAN、SD 1.4、ADM；另已在固定 GLIDE/VQDM 未见生成器集上完成一次性测试，汇总见 `FINAL_REPORT.md` 第4.4节。

## 项目目录速览

```text
GenImage/
├── README.md                         # 本文件：项目与交接入口
├── FINAL_REPORT.md                   # 汇总实验方法、结果和限制
├── REPRODUCIBILITY.md                # 环境、数据、训练与评估复现步骤
├── BASELINE.md                       # 训练与评估过程记录
├── train.py                          # ResNet-50 基线训练
├── train_robust_v2.py                # V2 鲁棒训练
├── clip_detector.py                  # 冻结 CLIP 特征检测器
├── evaluate.py                       # 固定清单常规评估
├── evaluate_validation_robustness.py # 仅验证集扰动评估
├── evaluate_robustness_holdout.py    # 新 holdout 最终评估
├── scripts/                          # 构建划分、未见生成器和新 holdout 清单
├── manifests/                        # 固定 CSV 划分、审计配置和新 holdout
├── inventory/                        # 候选清单与数据审计材料
├── outputs/                          # 权重、验证报告、测试报告和错例分析
├── demo.py                           # Streamlit 图片检测演示
└── release/
    └── GenImage_Final_2026-11-03/    # 可交接归档目录
```

## 关键结果文件

- 最终汇总：[`FINAL_REPORT.md`](FINAL_REPORT.md)
- 数据清单和划分配置：`manifests/`
- ResNet 基线：`outputs/resnet50-cap128-jpeg95/`
- CLIP 基线：`outputs/clip-vit-b32-cap128-jpeg95/`
- 鲁棒 V2（三个 seed）：`outputs/resnet50-robust-v2-*` 与 `outputs/resnet50-robust-v2-complete/`
- 新 holdout 对照：`outputs/robustness_new_holdout/`
- 归档校验清单：`release/GenImage_Final_2026-11-03/PACKAGE_MANIFEST.json`
