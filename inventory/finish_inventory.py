"""Finalize image inventory manifests after scanning is complete."""

import collections
import csv
import json
import random
from pathlib import Path

from PIL import Image

R = Path("/scratch/jshui/GenImage")
O = R / "inventory"
report = json.loads((O / "report.json").read_text())
rng = random.Random(20261031)
groups = {}
zeros = []
classcounts = collections.defaultdict(collections.Counter)
for row in csv.DictReader((O / "files.csv").open()):
    key = (row["generator"], row["official_split"], row["kind"])
    d = groups.setdefault(key, {"n": 0, "sample": []})
    d["n"] += 1
    name = Path(row["path"]).name
    if row["bytes"] == "0":
        zeros.append(row)
    if row["kind"] == "ai":
        c = name.split("_")[0]
        c = c if c.isdigit() else "unparsed"
    else:
        c = row["filename_class_candidate"] or "needs_val_ground_truth"
    classcounts[key][c] += 1
    if len(d["sample"]) < 100:
        d["sample"].append(row["path"])
    else:
        j = rng.randrange(d["n"])
        if j < 100:
            d["sample"][j] = row["path"]
for g in report["groups"]:
    key = (g["generator"], g["official_split"], g["kind"])
    errors = []
    dims = collections.Counter()
    modes = collections.Counter()
    for p in groups[key]["sample"]:
        try:
            with Image.open(R / p) as im:
                dims[str(im.size)] += 1
                modes[im.mode] += 1
                im.load()
        except Exception as e:
            errors.append({"path": p, "error": str(e)})
    g.update(
        sample_checked=len(groups[key]["sample"]),
        sample_errors=errors,
        sample_dimensions=dict(dims),
        sample_modes=dict(modes),
        filename_class_token_count=len(classcounts[key]),
    )
    print(
        key,
        "checked",
        g["sample_checked"],
        "errors",
        errors,
        "dimensions",
        dict(dims),
        flush=True,
    )
report["pillow_available"] = True
report["decode_environment"] = "/home/jshui/.conda/envs/myenv/bin/python"
report["limitations"][
    1
] = "Decoded 100 random files in each of 12 groups; not a full corruption scan."
(O / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
with (O / "zero_byte_files.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=list(zeros[0]))
    w.writeheader()
    w.writerows(zeros)
with (O / "class_counts.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["generator", "official_split", "kind", "filename_class_token", "files"])
    for key, counts in classcounts.items():
        for c, n in sorted(counts.items()):
            w.writerow([*key, c, n])
lines = [
    "# GenImage 数据盘点",
    "",
    "统计对象：三个已解压目录中的 train/val × nature/ai；原始文件未修改。",
    "",
    "| 数据集 | train 真 | train 假 | val 真 | val 假 | 文件大小 GiB |",
    "|---|---:|---:|---:|---:|---:|",
]
for gen in ["ADM", "BigGAN", "stable_diffusion_v_1_4"]:
    gs = [g for g in report["groups"] if g["generator"] == gen]
    ns = {(g["official_split"], g["kind"]): g["files"] for g in gs}
    lines.append(
        f"| {gen} | {ns['train','nature']:,} | {ns['train','ai']:,} | {ns['val','nature']:,} | {ns['val','ai']:,} | {sum(g['bytes'] for g in gs)/2**30:.2f} |"
    )
lines += [
    "",
    f"总文件数：{sum(g['files'] for g in report['groups']):,}；文件逻辑大小：{sum(g['bytes'] for g in report['groups'])/2**30:.2f} GiB（不等同于磁盘实际分配空间）。",
    "",
    f"零字节文件：{len(zeros)}，见 zero_byte_files.csv，应排除。",
    "",
    "每组随机解码100张，共1200张；错误数："
    + str(sum(len(g["sample_errors"]) for g in report["groups"]))
    + "。随机种子20261031。",
    "",
    "真实图扩展名为 JPEG，生成图扩展名为 PNG，存在格式偏差，后续实验需控制。",
    "",
    "ADM训练真图文件名含985种synset；BigGAN和SD1.4各1000种。假图文件名数字类别标识见class_counts.csv；数字标识与synset映射尚未核实。验证真图需ImageNet验证集标注才能确定类别。",
    "",
    "12个分组间未发现同名文件交集；这不等于内容无重复。尚未进行文件内容哈希、像素哈希或感知去重，来源也尚未外部核验。",
    "",
    "当前未见GLIDE和VQDM目录。",
]
(O / "README.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
