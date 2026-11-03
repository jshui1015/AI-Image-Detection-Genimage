"""Write the human-readable summary of the content audit."""

import json
from pathlib import Path

O = Path("/scratch/jshui/GenImage/inventory")
d = json.loads((O / "dedup_report.json").read_text())
c = json.loads((O / "class_verification.json").read_text())
source = json.loads((O / "metadata/sources.json").read_text())
lines = [
    "# 内容去重与类别核实报告",
    "",
    "原始图片未修改、未删除。本次输出清单供后续分层抽样及来源分组划分。",
    "",
    "## 全量内容检查",
    "",
    f"- 输入文件：{d['input_files']:,}",
    f"- 解码成功：{d['decoded_successfully']:,}；无效文件：{d['invalid_files']:,}（unreadable_files.csv）",
    f"- 字节重复组：{d['byte_duplicate_groups']:,}；多余文件副本：{d['byte_redundant_files']:,}",
    f"- 像素重复组：{d['pixel_duplicate_groups']:,}；多余文件副本：{d['pixel_redundant_files']:,}（与字节重复重叠，不能相加）",
    f"- 跨官方train/val的像素重复组：{d['pixel_groups_spanning_official_splits']:,}",
    f"- 内容类别冲突的精确重复组：{d['conflicting_class_groups']:,}；真假标签冲突：{d['conflicting_label_groups']:,}组（class_conflict_members.csv）",
    f"- 可用清单保留：{d['status_counts'].get('keep',0):,} 张",
    "",
    "## 近重复检查",
    "",
    f"在同一ImageNet内容类别内，对精确去重后的图片进行dHash候选检索，汉明距离≤3，再检查32×32 RGB缩略图RMSE≤3。保守候选对：{d['near_conservative_pairs']:,}；其中跨真假标签待复核：{d['near_cross_label_pairs']:,}。",
    "",
    "近似图片不自动删除；同标签的候选按连通关系共享source_group。后续须按整组划分，初始独立图片子集每个source_group最多选一张。视觉复核确认7对真图为相同场景的不同版本；另1对BigGAN图片过暗，无法确认共同来源，保守分组。详见near_duplicate_visual_review.csv。结果见near_duplicate_candidates.csv。该扫描不穷尽裁剪、旋转或跨类别标注的同源图，低信息量dHash也不参与近重复匹配。",
    "",
    "## 类别映射",
    "",
    f"- 全部{c['mapping_records']:,}条文件记录完成类别解析；未解析：{len(c['unresolved'])}。",
    "- 训练真图使用文件名synset；验证真图按ILSVRC文件序号查询TensorFlow Datasets官方维护标签。",
    "- 标准1000类JSON与GenImage仓库内timm标准synset顺序完全一致。",
    "- SD附带index_synset.yaml有72处不一致，已拒用并保留为核查证据。",
    "- 生成图文件名前缀均覆盖0～999；18张样本视觉核对一致。原始导出命名脚本未找到，因此映射证据明确标记为推定，未声称逐张语义验证。",
    "- ADM训练真图缺15类，且部分其他类别数量不足；其他分组类别统计见verified_class_counts.csv。",
    "",
    "## 文件用途",
    "",
    "- deduplicated_manifest.csv：排除无效、精确重复副本及精确标签/类别冲突后的候选池，尚未划分训练/验证/测试。",
    "- content_manifest.csv：全部文件及其状态、canonical_path、source_group。",
    "- exact_duplicate_members.csv / byte_duplicate_members.csv：重复组所有成员。",
    "- class_mapping.csv：1000类编号、synset与名称。",
    "- class_mapping_manifest.csv：全部文件的类别与映射证据。",
    "- metadata/sources.json：资料URL与下载文件SHA-256，用于追溯。",
    "",
    "## 后续划分约束",
    "",
    "先处理near_duplicate_candidates.csv中跨标签待复核记录（如有），再按source_group整体分配；不得只按单张图片随机划分。清单中的official_split是原始官方划分，不是实验split。若同源组跨官方train/val，必须整体排除或保留在同一实验集合，不能直接沿用边界。",
    "",
    "本次未进行图片来源真实性认证，也未生成最终实验划分。",
]
(O / "CONTENT_REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
