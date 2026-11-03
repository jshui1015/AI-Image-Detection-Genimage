#!/usr/bin/env python3
"""Freeze a new image-disjoint robustness holdout from unused GenImage val rows."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inventory/deduplicated_manifest.csv"
OUT = ROOT / "manifests/test_robustness_new.csv"
CONFIG = ROOT / "manifests/test_robustness_new_config.json"
SEED = 20261103
GENERATORS = ("BigGAN", "stable_diffusion_v_1_4", "ADM")


def rank_key(row):
    """Return the deterministic ordering key used for candidate selection."""
    value = f"{SEED}|{row['generator']}|{row['content_class']}|{row['image_id']}"
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    """Build a source-group-isolated holdout from remaining validation images."""
    if OUT.exists() or CONFIG.exists():
        raise SystemExit(f"Refusing to overwrite frozen holdout: {OUT} or {CONFIG}")

    used_ids, used_paths, used_groups = set(), set(), set()
    prior_manifests = [
        ROOT / "manifests/all_splits.csv",
        ROOT / "manifests/test_glide_unseen.csv",
        ROOT / "manifests/test_vqdm_unseen.csv",
    ]
    for path in prior_manifests:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                used_ids.add(row["image_id"])
                used_paths.add(row["path"])
                used_groups.add(row["source_group"])

    candidates = []
    with SOURCE.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "keep" or row["official_split"] != "val":
                continue
            if row["generator"] not in ("real", *GENERATORS):
                continue
            if row["image_id"] in used_ids or row["path"] in used_paths:
                continue
            if row["source_group"] in used_groups:
                continue
            candidates.append(row)

    buckets = {}
    for row in candidates:
        buckets.setdefault((row["content_class"], row["generator"]), []).append(row)
    for rows in buckets.values():
        rows.sort(key=rank_key)

    classes = sorted({r["content_class"] for r in candidates})
    selected, selected_ids, selected_paths, selected_groups = [], set(), set(), set()
    for cls in classes:
        reals = buckets.get((cls, "real"), [])
        if len(reals) < len(GENERATORS):
            raise RuntimeError(
                f"Insufficient unused real images for {cls}: {len(reals)}"
            )
        # Allocate three distinct real controls, one to each generator comparison.
        for gen, real in zip(GENERATORS, reals[: len(GENERATORS)]):
            fake_rows = buckets.get((cls, gen), [])
            if not fake_rows:
                raise RuntimeError(f"No unused fake candidate for {cls}/{gen}")
            fake = fake_rows[0]
            for row, matched_gen in ((real, gen), (fake, gen)):
                if (
                    row["image_id"] in selected_ids
                    or row["path"] in selected_paths
                    or row["source_group"] in selected_groups
                ):
                    raise RuntimeError(
                        f"Duplicate image/path/source group at {cls}/{gen}"
                    )
                selected_ids.add(row["image_id"])
                selected_paths.add(row["path"])
                selected_groups.add(row["source_group"])
                selected.append(
                    {
                        **row,
                        "split": "test_robust_new",
                        "matched_generator": matched_gen,
                    }
                )

    fields = list(selected[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    counts = Counter((r["matched_generator"], r["label"]) for r in selected)
    config = {
        "seed": SEED,
        "manifest": str(OUT.relative_to(ROOT)),
        "source": "unused rows from the existing official GenImage/ImageNet validation inventory",
        "selection": "stable SHA-256 ordering; one real and one fake per class per seen generator",
        "generators": list(GENERATORS),
        "images": len(selected),
        "classes": len(classes),
        "unique_image_ids": len(selected_ids),
        "unique_paths": len(selected_paths),
        "unique_source_groups": len(selected_groups),
        "overlap_with_previous_manifests": 0,
        "counts_by_matched_generator_and_label": {
            f"{g}/label_{label}": n for (g, label), n in sorted(counts.items())
        },
        "independence_scope": "image/source-group-disjoint from prior manifests; same underlying GenImage/ImageNet source",
        "policy": "Evaluate once after model/configuration freeze; do not use for model or threshold selection.",
    }
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
