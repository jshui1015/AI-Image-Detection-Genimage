"""Validate generated content-audit output files and their counts."""

import collections
import csv
import json
from pathlib import Path

O = Path("/scratch/jshui/GenImage/inventory")
d = json.loads((O / "dedup_report.json").read_text())
counts = collections.Counter()
canonical = {}
ids = set()
source_for = {}
for r in csv.DictReader((O / "deduplicated_manifest.csv").open()):
    assert (
        r["status"] == "keep"
        and r["source_group"]
        and r["image_id"]
        and r["content_class"]
    )
    assert r["image_id"] not in ids
    ids.add(r["image_id"])
    canonical[r["path"]] = r["source_group"]
assert len(ids) == d["status_counts"]["keep"]
for r in csv.DictReader((O / "content_manifest.csv").open()):
    counts[r["status"]] += 1
    if r["status"] == "duplicate":
        assert (
            r["canonical_path"] in canonical
            and canonical[r["canonical_path"]] == r["source_group"]
        )
assert sum(counts.values()) == d["input_files"] and dict(counts) == d["status_counts"]
for r in csv.DictReader((O / "near_duplicate_candidates.csv").open()):
    if (
        r["label_a"] == r["label_b"]
        and r["path_a"] in canonical
        and r["path_b"] in canonical
    ):
        assert canonical[r["path_a"]] == canonical[r["path_b"]]
for filename in ["files.csv", "content_hashes.csv", "class_mapping_manifest.csv"]:
    with (O / filename).open() as f:
        assert sum(1 for _ in csv.DictReader(f)) == d["input_files"], filename
result = {
    "passed": True,
    "input_files": d["input_files"],
    "retained_files": len(ids),
    "status_counts": dict(counts),
    "checks": [
        "full manifest and hash coverage",
        "unique retained image IDs",
        "duplicate canonical references",
        "near-pair source-group consistency",
        "status arithmetic",
    ],
}
(O / "validation_results.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
p = O / "README.md"
s = p.read_text()
note = "> 后续全量内容去重与类别核实已完成，最新结果见 [CONTENT_REPORT.md](CONTENT_REPORT.md)。下文为初始盘点记录。\n\n"
if not s.startswith("> 后续"):
    p.write_text(note + s)
