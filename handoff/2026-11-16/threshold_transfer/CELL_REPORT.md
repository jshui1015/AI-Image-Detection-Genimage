> 文档状态更新于 2026-11-19。分数诊断、36 组分辨率评估、COCOXGEN 阈值适配与生成配置分析已完成，见[最新实验补充报告](../README.md)。九模型主实验结果见 [三种子主报告](../../2026-11-08/FINAL_REPORT.md)。历史章节中的待办以当前交接说明为准。

# COCOXGEN configuration analysis

Holdout only: 743 COCO groups, 743 real and 743 fake images per cell. Thresholds were selected on separate development groups. All data had previously been inspected. Values average three fitted seeds. Shared reals across cells remain paired. Differences are descriptive, without significance testing.

| Model | Cell | AUROC | Original recall | Adapted recall | Adapted real FPR |
|---|---|---|---|---|---|
| resnet | fooocus-short | 0.7676 | 1.66% | 71.87% | 34.50% |
| resnet | fooocus-long | 0.6672 | 0.45% | 57.83% | 34.50% |
| resnet | sdxl-short | 0.7929 | 1.21% | 76.81% | 34.50% |
| resnet | sdxl-long | 0.6828 | 0.63% | 60.39% | 34.50% |
| v2 | fooocus-short | 0.7472 | 1.03% | 68.60% | 31.99% |
| v2 | fooocus-long | 0.5868 | 0.09% | 44.82% | 31.99% |
| v2 | sdxl-short | 0.7859 | 1.17% | 74.20% | 31.99% |
| v2 | sdxl-long | 0.6283 | 0.13% | 49.98% | 31.99% |
| clip | fooocus-short | 0.8485 | 19.96% | 74.29% | 21.09% |
| clip | fooocus-long | 0.8117 | 17.45% | 68.69% | 21.09% |
| clip | sdxl-short | 0.8549 | 28.62% | 76.31% | 21.09% |
| clip | sdxl-long | 0.7800 | 16.78% | 63.17% | 21.09% |

Generator comparisons also change generation and file-processing pipelines. Prompt comparisons describe paired generated variants and do not isolate semantic content or a causal prompt-length effect. A shared threshold per model is reused for all cells.
