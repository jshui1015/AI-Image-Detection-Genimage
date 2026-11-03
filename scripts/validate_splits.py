"""Independently validate exported splits and selected image paths."""

import collections
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path(__file__).resolve().parents[1]
M = R / "manifests"
EXPECTED = {
    "train": (20000, {"BigGAN": 6667, "stable_diffusion_v_1_4": 6667, "ADM": 6666}),
    "val": (4000, {"BigGAN": 1333, "stable_diffusion_v_1_4": 1333, "ADM": 1334}),
    "test_seen": (5000, {"BigGAN": 1667, "stable_diffusion_v_1_4": 1667, "ADM": 1666}),
    "reserved_real_glide": (2000, {}),
    "reserved_real_vqdm": (2000, {}),
}


def main():
    """Check split counts, source-group isolation, and class balance."""
    selected = {}
    identities = set()
    sources = set()
    per_split = {}
    errors = []
    for name, (real, fakes) in EXPECTED.items():
        rows = list(csv.DictReader((M / (name + ".csv")).open()))
        counts = collections.Counter((r["label"], r["generator"]) for r in rows)
        classes = collections.Counter((r["label"], int(r["class_index"])) for r in rows)
        assert counts == collections.Counter(
            {("0", "real"): real, **{("1", g): n for g, n in fakes.items()}}
        ), name
        for cls in range(1000):
            assert classes["0", cls] == real // 1000
            assert classes["1", cls] == sum(fakes.values()) // 1000
            gc = [
                sum(
                    r["label"] == "1"
                    and r["generator"] == g
                    and int(r["class_index"]) == cls
                    for r in rows
                )
                for g in fakes
            ]
            assert not gc or max(gc) - min(gc) <= 1
        for r in rows:
            assert r["split"] == name and r["official_split"] == (
                "train" if name in ("train", "val") else "val"
            )
            assert (
                r["path"] not in selected
                and r["image_id"] not in identities
                and r["source_group"] not in sources
            )
            selected[r["path"]] = r
            identities.add(r["image_id"])
            sources.add(r["source_group"])
        per_split[name] = len(rows)
    # Verify every selected record against the audited keep manifest.
    found = set()
    for r in csv.DictReader((R / "inventory/deduplicated_manifest.csv").open()):
        if r["path"] in selected:
            assert r["status"] == "keep"
            expected = selected[r["path"]]
            for k in [
                "image_id",
                "label",
                "generator",
                "class_index",
                "content_class",
                "source_group",
                "official_split",
                "mapping_evidence",
            ]:
                assert r[k] == expected[k], (r["path"], k)
            found.add(r["path"])
    assert found == set(selected)
    for r in csv.DictReader((R / "inventory/content_manifest.csv").open()):
        if r["source_group"] in sources:
            expected = next(iter([r["official_split"]]))
            # Every retained source must have one official boundary even among aliases.
            old = per_source.setdefault(r["source_group"], expected)
            assert old == expected

    def exists(path):
        """Check that a manifest image exists below the project root."""
        return (R / path).is_file()

    with ThreadPoolExecutor(max_workers=16) as pool:
        missing = [p for p, ok in zip(selected, pool.map(exists, selected)) if not ok]
    assert not missing, missing
    combined = list(csv.DictReader((M / "all_splits.csv").open()))
    assert len(combined) == 62000 and {r["path"] for r in combined} == set(selected)
    cfg = json.loads((M / "split_config.json").read_text())
    for name, digest in cfg["output_sha256"].items():
        assert hashlib.sha256((M / name).read_bytes()).hexdigest() == digest
    result = {
        "passed": True,
        "total_selected": len(selected),
        "split_sizes": per_split,
        "all_selected_paths_exist": True,
        "all_records_match_audited_candidates": True,
        "pairwise_image_id_path_source_group_overlap": 0,
        "known_official_boundary_crossing_selected_groups": 0,
        "per_class_quotas_and_generator_balance_passed": True,
    }
    (M / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


per_source = {}
if __name__ == "__main__":
    main()
