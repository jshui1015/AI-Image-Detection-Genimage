"""Inspect and summarize outputs from the content-level audit."""

import collections
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

O = Path("/scratch/jshui/GenImage/inventory")
R = O.parent
d = json.loads((O / "dedup_report.json").read_text())
groups = collections.defaultdict(list)
classes = {
    r["class_index"]: r["synset"]
    for r in csv.DictReader((O / "class_mapping.csv").open())
}
for r in csv.DictReader((O / "exact_duplicate_members.csv").open()):
    groups[r["hash"]].append(r)
d["conflicting_label_groups"] = sum(
    len({r["label"] for r in rs}) > 1 for rs in groups.values()
)
d["conflicting_class_groups"] = sum(
    len({r["class_index"] for r in rs}) > 1 for rs in groups.values()
)
with (O / "class_conflict_members.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "pixel_sha256",
            "path",
            "label",
            "class_index",
            "content_class",
            "official_split",
        ]
    )
    for sha, rs in groups.items():
        if len({r["label"] for r in rs}) > 1 or len({r["class_index"] for r in rs}) > 1:
            for r in rs:
                w.writerow(
                    [
                        sha,
                        r["path"],
                        r["label"],
                        r["class_index"],
                        classes[r["class_index"]],
                        r["official_split"],
                    ]
                )
(O / "dedup_report.json").write_text(json.dumps(d, ensure_ascii=False, indent=2))
pairs = [
    (r["path_a"], r["path_b"])
    for r in csv.DictReader((O / "near_duplicate_candidates.csv").open())
]
if pairs:
    canvas = Image.new("RGB", (660, len(pairs) * 235), "white")
    draw = ImageDraw.Draw(canvas)
    for y, pair in enumerate(pairs):
        for x, p in enumerate(pair):
            with Image.open(R / p) as im:
                canvas.paste(
                    ImageOps.contain(im.convert("RGB"), (320, 205)), (x * 330, y * 235)
                )
            draw.text((x * 330, y * 235 + 207), Path(p).name, fill="black")
    canvas.save(O / "near_duplicate_review.jpg")
print(
    "label_conflicts",
    d["conflicting_label_groups"],
    "class_conflicts",
    d["conflicting_class_groups"],
)
print("retained", d["status_counts"])
print("near_pairs", len(pairs))
