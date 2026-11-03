"""Inventory candidate image files and record metadata for later audits."""

import collections
import csv
import json
import os
import random
import time
from pathlib import Path

ROOT = Path("/scratch/jshui/GenImage")
OUT = ROOT / "inventory"
try:
    from PIL import Image
except ImportError:
    Image = None
rng = random.Random(20261031)
summary = []
names = {}
samples = []
extra = []
with (OUT / "files.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "generator",
            "official_split",
            "kind",
            "path",
            "bytes",
            "extension",
            "filename_class_candidate",
        ]
    )
    for gen in ["ADM", "BigGAN", "stable_diffusion_v_1_4"]:
        for base, dirs, files in os.walk(ROOT / gen):
            for name in files:
                p = Path(base) / name
                rel = p.relative_to(ROOT)
                parts = rel.parts
                split = next((x for x in parts if x in ("train", "val")), None)
                kind = next((x for x in parts if x in ("nature", "ai")), None)
                if not split or not kind:
                    extra.append(str(rel))
                    continue
                key = (gen, split, kind)
                if key not in names:
                    names[key] = {
                        "n": 0,
                        "bytes": 0,
                        "ext": collections.Counter(),
                        "zero": 0,
                        "samples": [],
                        "classes": collections.Counter(),
                        "examples": [],
                        "basenames": set(),
                    }
                d = names[key]
                size = p.stat().st_size
                ext = p.suffix.lower()
                candidate = (
                    name.split("_")[0]
                    if name.startswith("n") and len(name.split("_")[0]) == 9
                    else ""
                )
                d["n"] += 1
                d["bytes"] += size
                d["ext"][ext] += 1
                d["zero"] += size == 0
                d["classes"][candidate] += 1
                d["basenames"].add(name)
                if len(d["examples"]) < 5:
                    d["examples"].append(name)
                if len(d["samples"]) < 100:
                    d["samples"].append(str(p))
                else:
                    j = rng.randrange(d["n"])
                    if j < 100:
                        d["samples"][j] = str(p)
                w.writerow([gen, split, kind, str(rel), size, ext, candidate])
            if files:
                print("scanned", Path(base).relative_to(ROOT), len(files), flush=True)
for key, d in names.items():
    errors = []
    dims = collections.Counter()
    modes = collections.Counter()
    for p in d["samples"]:
        if Image:
            try:
                with Image.open(p) as im:
                    dims[str(im.size)] += 1
                    modes[im.mode] += 1
                    im.load()
            except Exception as e:
                errors.append({"path": p, "error": str(e)})
    summary.append(
        dict(
            generator=key[0],
            official_split=key[1],
            kind=key[2],
            files=d["n"],
            bytes=d["bytes"],
            extensions=dict(d["ext"]),
            zero_byte_files=d["zero"],
            filename_class_candidates=len([c for c in d["classes"] if c]),
            unparsed_class_files=d["classes"][""],
            examples=d["examples"],
            sample_checked=len(d["samples"]) if Image else 0,
            sample_errors=errors,
            sample_dimensions=dict(dims),
            sample_modes=dict(modes),
        )
    )
    print("SUMMARY", summary[-1], flush=True)
overlaps = []
keys = list(names)
for i, a in enumerate(keys):
    for b in keys[i + 1 :]:
        n = len(names[a]["basenames"] & names[b]["basenames"])
        if n:
            overlaps.append({"a": a, "b": b, "shared_basenames": n})
report = {
    "seed": 20261031,
    "pillow_available": Image is not None,
    "groups": summary,
    "shared_basenames_not_content_duplicates": overlaps,
    "other_files": extra,
    "limitations": [
        "No full content hashing or perceptual deduplication performed.",
        "Image decoding checked on up to 100 random files per group only.",
        "Filename class candidates require verification against metadata.",
    ],
}
(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("DONE", sum(x["files"] for x in summary), flush=True)
