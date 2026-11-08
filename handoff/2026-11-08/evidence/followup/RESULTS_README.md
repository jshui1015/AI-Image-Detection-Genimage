# 补充实验结果

先看 consistency.csv：36项（九模型×四旧测试域）必须 passed=True。
最大概率误差上限预先固定为0.002；同时报告所有决策差异，阈值以外超过0.002的翻转即停止。
这验证的是样本上的数值一致性，不声称完整数据或浮点计算逐位相同。

external_summary.csv 是外部主要结果：四个平衡的真图/生成器-提示词测试单元等权平均，再对三个固定种子取均值。
external_metrics.csv 中 pooled_imbalanced 为次要全量结果，假图比例80%，不能把其普通准确率与CIFAKE平衡集直接比较。
外部置信区间按1061个COCO原图编号成组重采样，所有变体、所有模型与种子使用相同重采样。
holdout_confidence.csv 与 paired_differences.csv 按1000个语义类别成组配对重采样。
置信区间描述测试抽样不确定性，条件于已训练九模型；不代表新随机种子的训练不确定性。
外部图片从未用于训练、温度拟合、阈值优化或挑选种子。
数据来源： https://huggingface.co/datasets/heikeadel/cocoxgen
真图来源：官方COCO val2017存储桶；版本/清单/哈希见protocol.json与external_manifest.csv。
检测了旧数据与新增数据的完全字节/解码RGB重叠；未声称排除近重复或基础模型预训练接触。
本次实验是CIFAKE之后新增的独立来源验证，不声称仅隔离了分辨率因素。
