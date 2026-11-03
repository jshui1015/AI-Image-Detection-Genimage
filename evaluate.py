#!/usr/bin/env python3
"""One-shot evaluation of a locked checkpoint on a held-out test manifest."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from train import ROOT, Images, ResNet50, evaluate, metrics


def main():
    """Evaluate a locked checkpoint on one fixed test manifest."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "outputs/resnet50-rerun/best.pt"))
    p.add_argument("--manifest", default="")
    p.add_argument(
        "--split",
        choices=["test_seen", "test_glide_unseen", "test_vqdm_unseen"],
        default="test_seen",
    )
    p.add_argument("--output-dir", default="")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--control",
        choices=["auto", "raw", "jpeg95", "cap128", "cap128_jpeg95"],
        default="auto",
    )
    a = p.parse_args()
    ckpt_path = Path(a.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    report_path = ckpt_path.parent / "validation_report.json"
    val = json.loads(report_path.read_text())
    threshold = float(val["validation_threshold_max_f1"])
    # The checkpoint records its preprocessing control; keep it paired with the model.
    control = ckpt.get("input_control", "raw")
    manifest = (
        Path(a.manifest)
        if a.manifest
        else (
            ROOT / "manifests/all_splits.csv"
            if a.split == "test_seen"
            else ROOT / f"manifests/{a.split}.csv"
        )
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run test evaluation in the allocated GPU job."
        )
    if a.control != "auto" and a.control != control:
        raise ValueError(
            f"Requested control {a.control!r} does not match checkpoint control {control!r}; use --control auto or omit --control."
        )
    device = torch.device("cuda")
    ds = Images(manifest, a.split, input_control=control)
    loader = DataLoader(
        ds,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=True,
    )
    if not len(ds):
        raise ValueError(f"No rows found for split {a.split!r} in {manifest}")
    model = ResNet50()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    ys = []
    ps = []
    gens = []
    paths = []
    with torch.inference_mode():
        for x, y, g, path in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model(x.to(device, non_blocking=True))
            ys.extend(y.int().tolist())
            ps.extend(z.float().sigmoid().cpu().tolist())
            gens.extend(g)
            paths.extend(path)
    out = Path(a.output_dir) if a.output_dir else ckpt_path.parent / a.split
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Report both the complete balanced set and class-matched real/fake comparisons.
    groups = {"all": list(range(len(ys)))}
    real_by_class = defaultdict(list)
    fake_by_generator_class = defaultdict(list)
    for i, row in enumerate(ds.rows):
        c = int(row["class_index"])
        if row["label"] == "0":
            real_by_class[c].append(i)
        else:
            fake_by_generator_class[(row["generator"], c)].append(i)
    for gen in sorted(set(gens) - {"real"}):
        matched = []
        for c in range(1000):
            fake = fake_by_generator_class[(gen, c)]
            matched.extend(fake)
            matched.extend(real_by_class[c][: len(fake)])
        groups[f"{gen}_vs_class_matched_real"] = matched
    for name, idx in groups.items():
        for threshold_name, t in [
            ("fixed_validation_f1", threshold),
            ("default_0_5", 0.5),
        ]:
            rows.append(
                {
                    "group": name,
                    "threshold_name": threshold_name,
                    "threshold": t,
                    "count": len(idx),
                    **metrics([ys[i] for i in idx], [ps[i] for i in idx], t),
                }
            )
    with (out / "metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (out / "per_image.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "label",
                "generator",
                "score_ai",
                "prediction_fixed_validation_f1",
                "prediction_0_5",
            ]
        )
        w.writerows(
            (
                paths[i],
                ys[i],
                gens[i],
                ps[i],
                int(ps[i] >= threshold),
                int(ps[i] >= 0.5),
            )
            for i in range(len(ys))
        )
    classrows = []
    byclass = defaultdict(list)
    for i, row in enumerate(ds.rows):
        byclass[int(row["class_index"])].append(i)
    for c, idx in sorted(byclass.items()):
        classrows.append(
            {
                "class_index": c,
                "count": len(idx),
                **metrics([ys[i] for i in idx], [ps[i] for i in idx], threshold),
            }
        )
    with (out / "per_class.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(classrows[0]))
        w.writeheader()
        w.writerows(classrows)
    baseline_path = ROOT / "outputs/resnet50-rerun/test_seen/report.json"
    baseline = (
        json.loads(baseline_path.read_text())
        if a.split == "test_seen" and control != "raw" and baseline_path.exists()
        else None
    )
    baseline_metrics = (
        {(r["group"], r["threshold_name"]): r for r in baseline["metrics"]}
        if baseline
        else {}
    )
    comparison = []
    for r in rows:
        b = baseline_metrics.get((r["group"], r["threshold_name"]))
        if b:
            comparison.append(
                {
                    "group": r["group"],
                    "threshold_name": r["threshold_name"],
                    "delta_accuracy": r["accuracy"] - b["accuracy"],
                    "delta_roc_auc": r["roc_auc"] - b["roc_auc"],
                    "delta_average_precision": r["average_precision"]
                    - b["average_precision"],
                    "delta_f1": r["f1"] - b["f1"],
                }
            )
    result = {
        "checkpoint": str(ckpt_path),
        "best_epoch": ckpt["epoch"],
        "seed": ckpt["seed"],
        "device": torch.cuda.get_device_name(0),
        "split": a.split,
        "manifest": str(manifest),
        "input_control": control,
        "control_definition": {
            "jpeg95": "Re-encode every decoded RGB image as JPEG quality 95 at its original dimensions.",
            "cap128": "Downscale images so the longest edge is at most 128 pixels; preserve aspect ratio.",
            "cap128_jpeg95": "Apply cap128, then JPEG quality 95.",
            "raw": "Use original decoded images.",
        }[control],
        "threshold": threshold,
        "threshold_source": "validation_report.json, selected before test evaluation",
        "metrics": rows,
        "delta_vs_raw": comparison,
        "files": len(ys),
        "test_used_for_selection_or_calibration": False,
        "per_generator_policy": "Each generator is compared with the same-class real images, matched to its fake count per class. The fixed validation threshold is reused without retuning.",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
