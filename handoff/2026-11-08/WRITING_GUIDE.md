# 最终论文与PPT写作指南

## 先确定论文主线

建议标题：**AI生成图像检测的鲁棒性与跨数据来源泛化：ResNet、增强ResNet与冻结CLIP的比较**。

核心问题：同来源高分是否能转化为退化输入和独立来源上的可靠检测？证据显示指定增强有帮助，但不能保证跨来源泛化；排序指标和固定阈值决策必须分开讨论。

## 论文材料映射

| 章节 | 要写什么 | 已有依据 |
|---|---|---|
| Introduction | 检测需求、数据偏差、跨域难点、研究问题与贡献 | 项目原报告与本交接主线；不要虚构部署效果 |
| Related work | CNN检测、CLIP迁移、GenImage、图像编码偏差、校准 | 原报告参考线索及下方引用清单；阅读正式来源后补齐引用 |
| Data | 固定GenImage划分与审计，6000 holdout，两个外部来源 | 仓库manifests、原报告数据章节、新报告§2及§6.2 |
| Methods | 三类模型、cap128_jpeg95、V1/V2、冻结编码器、阈值与温度 | 原报告§3–7、train.py/train_robust_v2.py/clip_detector.py、新报告§1、§5 |
| Protocol | 三个固定seed、七条件扰动、指标、bootstrap聚类单位 | 最新报告§2、§4.1、§6.2；evidence/followup/protocol.json |
| Results | 内部三域、鲁棒性、校准、CIFAKE、COCOXGEN | 下方八图及CSV |
| Discussion | V2 trade-off、跨来源失败、阈值迁移、校准局限 | 最新报告§4–9；不把分辨率与域变化混为单一因果 |
| Conclusion | 限定到本实验的发现、未来开发与独立测试设计 | 最新报告摘要与§9 |
| Appendix | 每seed结果、完整阈值、环境、哈希、重复敏感性、作业记录 | tables、evidence、MODEL_INDEX.csv、SOURCE_VERIFICATION.json |

## 图表与数值来源

| 图 | PPT/论文用途 | 数值依据 |
|---|---|---|
| [01_generalization](figures/01_generalization.png) | seen/GLIDE/VQDM与CIFAKE；三种子均值和SD | tables/main_results.csv、per_seed_results.csv |
| [02_robustness](figures/02_robustness.png) | 七条件鲁棒性对照 | tables/holdout_results.csv |
| [03_calibration](figures/03_calibration.png) | 校准迁移的收益与失败 | tables/calibration_results.csv |
| [04_external_confusion](figures/04_external_confusion.png) | CIFAKE大量假图漏检 | tables/cifake_decisions.csv |
| [05_external_scores](figures/05_external_scores.png) | CIFAKE分数分布与阈值 | evidence/cifake/*/logits.npz及锁定阈值 |
| [06_error_cases](figures/06_error_cases.png) | 真实错误案例展示 | cases中的图片与案例CSV；固定seed极端错误，不是随机样本 |
| [07_cocoxgen](figures/07_cocoxgen.png) | 新外部来源、AUROC与阈值分离 | tables/followup_external_summary.csv、tables/followup_external_confidence.csv |
| [08_paired_robustness](figures/08_paired_robustness.png) | V2相对ResNet配对差值与区间 | tables/followup_paired_differences.csv |

每张图另有同名PDF。图号是资产编号，论文按出现顺序重新编号。论文建议主文保留01、02、07、08及一组误判示例，其余按篇幅放附录。不要改动数值或删除负结果。图06图注须说明32×32仅为显示放大，V2该seed没有真图误报的空格不代表漏图。

## 建议10–12页PPT（按实际时长调整）

1. 标题、团队、研究问题。
2. 任务与数据：固定划分、未见生成器与外部来源的区别。
3. 方法：ResNet、V2、冻结CLIP；统一输入与验证阈值。
4. 实验协议：3seed、6000图×7条件、指标及不确定性。
5. 内部表现：图01的GenImage部分；避免把CIFAKE差成绩藏掉。
6. 鲁棒性：图02，解释V2训练增强。
7. 配对证据：图08，说明clean的小代价与扰动收益。
8. CIFAKE失败：图04/05/06，解释Accuracy约50%不代表有效。
9. COCOXGEN：图07，强调AUROC0.82968和召回20.96%并存。
10. 校准：图03，验证集校准不保证跨域。
11. 局限与结论：来源有限、预训练/近重复不完全可知、固定阈值失效。
12. 分工、复现入口、Q&A；完整结果和术语放备份页。

## 统计与表述规则

- 正类AI=1，真图=0；Recall指假图召回，AUROC/AP阈值无关。
- “±”用三seed样本SD(ddof=1)。原历史报告部分用总体SD，不直接混用。
- bootstrap CI描述固定九模型下测试样本抽样不确定性，不等同于新训练种子不确定性。
- holdout按1000语义类别成组；COCOXGEN按1061原图编号成组，同组真图和4合成变体一起抽。
- COCOXGEN主指标先对4个平衡单元等权平均，再对3seed平均；全量5305张80%假图的普通Accuracy不是主对照。
- 图08每指标跨3条件做Bonferroni校正，不能宣传为所有分析的全局错误率控制。
- 后补分析是在看过部分旧结果后进行，不能写成预注册或完全未触碰的确认性研究。
- CIFAKE是跨数据来源，SD1.4本身不是未见生成器。COCOXGEN原图分辨率较高，但仍执行原cap128输入控制，不是高分辨率模型训练。
- 数据哈希未重叠不等于排除近重复或预训练污染。原数据语义映射也只有有限人工抽查。
- CI不交零支持指定比较，不推出V2所有指标/所有来源最好；blur下V2假图召回略低。

## 常见答辩问题

**为什么内部很好，外部失败？** 来源、内容、编码和生成流程同时变化；实验显示泛化差，但没有单因素消融证明原因。

**为什么不改阈值或反转CIFAKE分数？** 这样会利用已查看的测试标签；本项目保留原验证阈值以测真实迁移。未来可用新开发集调参并用另一个未查看测试集验收。

**为什么CLIP三个seed差别小？** 冻结且复用相同特征，只训练线性头，训练自由度较少；不能视作三个独立数据集。

**演示验收了吗？** 队友已有ResNet demo代码和样例；本轮未运行验收，也未扩展到九模型。

## 引用与学术诚信

最新报告末尾保留GenImage、CIFAKE、COCOXGEN和COCO数据来源。方法应补正式引用：He et al. (2016) ResNet；Radford et al. (2021) CLIP；Guo et al. (2017) temperature scaling；原报告提及UniversalFakeDetect及Fake or JPEG?作为相关工作线索。作者应核对正式论文条目与学校引用格式，不把本指南当作已校对的BibTeX。

正式作者名单、每人贡献比例、课程模板和AI辅助披露规则尚未提供，由团队确认。不得虚构个人贡献、实验或评分保证。
