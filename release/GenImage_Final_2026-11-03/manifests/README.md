> 文档状态更新于 2026-11-19。分数诊断、36 组分辨率评估、COCOXGEN 阈值适配与生成配置分析已完成，见[最新实验补充报告](../../../handoff/2026-11-16/README.md)。九模型主实验结果见 [三种子主报告](../../../handoff/2026-11-08/FINAL_REPORT.md)。历史章节中的待办以当前交接说明为准。

> 本目录为 2026-11-03 归档快照。原始方法、单种子指标与复现代码保留对应日期的含义，当前进展由以上链接提供。

# 固定实验划分

随机种子：20261031。路径相对于项目根目录；label=0为真图，1为生成图。原始图像未移动或复制。

| 清单 | 真图 | 假图 | 总量 |
|---|---:|---:|---:|
| train.csv | 20,000 | 20,000 | 40,000 |
| val.csv | 4,000 | 4,000 | 8,000 |
| test_seen.csv | 5,000 | 5,000 | 10,000 |
| reserved_real_glide.csv | 2,000 | 0 | 2,000 |
| reserved_real_vqdm.csv | 2,000 | 0 | 2,000 |

三个完整实验集合共58,000张，另预留4,000张独立真图，共62,000张。补入GLIDE与VQDM各2,000张独立假图后，总计66,000张。两个reserved文件目前不构成完整未见测试集。

训练/验证取自官方train，测试及预留真图取自官方val。每个完整集合覆盖1000类，真假每类数量分别为训练20/20、验证4/4、已见测试5/5；每个预留真图集每类2张。每类生成器配额相差不超过1。

已知跨官方train/val的同源组全部排除。五个清单之间及内部不重复使用image_id、path、source_group；每来源组最多一张。此保证限于当前审计已识别的同源关系，不代表穷尽裁剪与旋转等未知同源版本。

所有模型和增强实验复用这些清单，不重新抽样。GLIDE/VQDM及预留真图只供最终未见测试，不用于模型选择、阈值或概率校准。

复现命令（输出目录必须尚不存在）：
```bash
python scripts/build_splits.py --seed 20261031 --output /tmp/genimage-splits-reproduction
```

配额、来源清单哈希和输出哈希见split_config.json；逐类分布见class_distribution.csv。

## GLIDE/VQDM 最终未见生成器测试清单

数据就绪后生成：`test_glide_unseen.csv` 与 `test_vqdm_unseen.csv`，各4,000张（2,000真+2,000假），1000类每类真假各2张。假图来自对应生成器的官方 `val/ai`，每类从6张中用固定种子20261102和SHA-256稳定排序选2张；真图使用同名 `reserved_real_*.csv` 中每类2张。GLIDE/VQDM不进入训练、验证、模型选择或阈值校准。生成配置见 `unseen_test_config.json`，审计见 `../inventory/unseen_generator_audit.json` 和 `../inventory/unseen_generator_label_spotcheck.json`。
