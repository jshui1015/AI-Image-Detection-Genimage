> 文档状态更新于 2026-11-19。分数诊断、36 组分辨率评估、COCOXGEN 阈值适配与生成配置分析已完成，见[最新实验补充报告](../2026-11-16/README.md)。九模型主实验结果见 [三种子主报告](FINAL_REPORT.md)。历史章节中的待办以当前交接说明为准。

# AI-Image-Detection 实验交付包

先打开 `FINAL_REPORT.html` 阅读带图报告：图片已内嵌，可离线打开。
编辑文字使用 `FINAL_REPORT.md`；剩余交付事项见 `PROJECT_CHECKLIST.md`。

## 内容
- figures：八张最终图，每张包含 PNG 与 PDF，方便插入报告或幻灯片。
- tables：汇总与逐种子统计、校准和去重敏感性结果。
- cases：CIFAKE 与 GenImage 错误案例索引；11张经哈希核验的 CIFAKE 图片与拼图。内部 GenImage 图片未打包。
- evidence：四批实验的分析证据，包括指标、原始预测、协议及运行脚本；不包含完整数据集或模型权重。
- audit.json：本次 CIFAKE 复算核验摘要。
- scripts：重建统计、图表和报告的代码。

## 重建本地分析
在本目录使用包含 Python 3、NumPy、Matplotlib、Pillow 的环境，依次运行：

```bash
python scripts/build_results.py
python scripts/make_cases.py
python scripts/add_followup.py
python scripts/write_report.py
```

这些步骤读取打包证据，不需要 GPU，不重新训练模型。生成文件写入本包对应目录。
`scripts/export_genimage_cases.py` 是可选服务器导出工具；需要服务器共享原图，不属于上述本地流程。

## 结果解释与保存
CIFAKE 的低 AUROC 是已经核验的外部泛化失败，不能通过反转标签、挑种子或在测试集调参改写结论。
误判图为固定种子的极端错误案例，不代表随机样本。图上的分数是模型输出，不是已经验证可靠的真实概率。
本包是实验章节及证据材料；课程最终提交还需团队背景/文献、作者信息、老师要求；本轮demo验收已由用户取消。
原服务器结果未修改。本包作为仓库handoff目录随本次交接提交。

`SHA256SUMS.txt` 记录打包时所有其他文件的哈希，可检查解压内容是否改变。重新生成文件后，校验记录可能需要更新。

## 本次更新
新增COCOXGEN指标及成组置信区间、原holdout配对差值图、一致性核验记录和COCOXGEN错误案例索引。新增图片未下载进包，案例索引不等同于图片拼图。
本地复核摘要见 followup_audit.json；服务器作业证据见 evidence/followup/。保留原始CIFAKE失败结果。
