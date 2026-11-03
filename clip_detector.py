#!/usr/bin/env python3
"""Frozen CLIP image embeddings with a trainable linear AI-image detector."""
import argparse
import csv
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor, CLIPModel

from train import ROOT, metrics

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_OUT = ROOT / "outputs/clip-vit-b32-cap128-jpeg95"
SPLITS = {
    "train": ("all_splits.csv", "train"),
    "val": ("all_splits.csv", "val"),
    "test_seen": ("all_splits.csv", "test_seen"),
    "test_glide_unseen": ("test_glide_unseen.csv", "test_glide_unseen"),
    "test_vqdm_unseen": ("test_vqdm_unseen.csv", "test_vqdm_unseen"),
}


class ManifestImages(Dataset):
    """Read image rows from a fixed split and apply the CLIP image processor."""

    def __init__(self, manifest, split, processor, control="cap128_jpeg95", limit=0):
        """Select the manifest rows and store the CLIP preprocessing settings."""
        with Path(manifest).open() as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if limit:
            self.rows = self.rows[:limit]
        self.processor = processor
        self.control = control

    def __len__(self):
        """Return the number of selected manifest rows."""
        return len(self.rows)

    def __getitem__(self, i):
        """Return CLIP-ready pixels, label, class, generator, and path."""
        r = self.rows[i]
        with Image.open(ROOT / r["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        if self.control in ("cap128", "cap128_jpeg95"):
            w, h = im.size
            scale = min(1.0, 128 / max(w, h))
            if scale < 1:
                im = im.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.Resampling.LANCZOS,
                )
        if self.control in ("jpeg95", "cap128_jpeg95"):
            buffer = io.BytesIO()
            im.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            with Image.open(buffer) as enc:
                im = enc.convert("RGB")
        x = self.processor(images=im, return_tensors="pt")["pixel_values"][0]
        return (
            x,
            torch.tensor(float(r["label"])),
            int(r["class_index"]),
            r["generator"],
            r["path"],
        )


def model_and_processor(model_id, cache_dir):
    """Load CLIP and freeze its image encoder for feature extraction."""
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
    processor = CLIPImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, processor


def feature_file(out, split):
    """Return the cache path used for one split's frozen image features."""
    return out / "features" / f"{split}.pt"


def extract(
    model,
    processor,
    manifest,
    split,
    out,
    model_id,
    control,
    device,
    batch_size,
    workers,
    limit=0,
    force=False,
):
    """Extract and cache deterministic frozen CLIP image embeddings."""
    target = feature_file(out, split)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force and not limit:
        saved = torch.load(target, map_location="cpu", weights_only=False)
        if (
            saved.get("model_id") == model_id
            and saved.get("input_control") == control
            and saved.get("manifest") == str(manifest)
        ):
            print(f"Using cached {split} CLIP features:", target, flush=True)
            return saved
    ds = ManifestImages(manifest, split, processor, control, limit)
    if not len(ds):
        raise ValueError(f"No rows for {split} in {manifest}")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    feats = []
    labels = []
    classes = []
    gens = []
    paths = []
    start = time.time()
    model.eval()
    with torch.inference_mode():
        for x, y, c, g, p in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                result = model.get_image_features(pixel_values=x)
            f = torch.nn.functional.normalize(result.pooler_output.float(), dim=1)
            feats.append(f.cpu())
            labels.append(y)
            classes.extend(c.tolist())
            gens.extend(g)
            paths.extend(p)
    data = {
        "features": torch.cat(feats),
        "labels": torch.cat(labels),
        "classes": torch.tensor(classes),
        "generators": gens,
        "paths": paths,
        "model_id": model_id,
        "input_control": control,
        "manifest": str(manifest),
        "split": split,
        "count": len(ds),
    }
    torch.save(data, target)
    print(
        f"Extracted {split}: {len(ds)} images in {time.time()-start:.1f}s -> {target}",
        flush=True,
    )
    return data


def fit_head(
    train, val, device, out, seed, max_epochs, patience, lr, weight_decay, smoke
):
    """Train a linear head and choose its classification threshold on validation."""
    torch.manual_seed(seed)
    x = train["features"].to(device)
    y = train["labels"].to(device)
    vx = val["features"].to(device)
    vy = val["labels"].numpy().astype(int)
    head = nn.Linear(x.shape[1], 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossfn = nn.BCEWithLogitsLoss()
    best = -1.0
    bad = 0
    best_state = None
    history = []
    epochs = 1 if smoke else max_epochs
    for epoch in range(epochs):
        epoch_start = time.time()
        head.train()
        opt.zero_grad(set_to_none=True)
        loss = lossfn(head(x).squeeze(1), y)
        loss.backward()
        opt.step()
        head.eval()
        with torch.inference_mode():
            ps = head(vx).squeeze(1).sigmoid().cpu().numpy().tolist()
        vm = metrics(vy, ps)
        rec = {
            "epoch": epoch + 1,
            "train_loss": float(loss.detach()),
            "val_roc_auc": vm["roc_auc"],
            "val_average_precision": vm["average_precision"],
            "seconds": round(time.time() - epoch_start, 2),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        improved = vm["roc_auc"] > best
        if improved:
            best = vm["roc_auc"]
            bad = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in head.state_dict().items()
            }
        else:
            bad += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or bad >= patience:
            print("PROGRESS", json.dumps(rec), flush=True)
        if not smoke and bad >= patience:
            print(
                f"Early stopping after {bad} epochs without validation AUROC improvement.",
                flush=True,
            )
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        ps = head(vx).squeeze(1).sigmoid().cpu().numpy().tolist()
    candidates = sorted(set(ps))
    threshold = max(candidates, key=lambda t: (metrics(vy, ps, t)["f1"], -abs(t - 0.5)))
    report = {
        "model_id": train["model_id"],
        "input_control": train["input_control"],
        "feature_dim": int(x.shape[1]),
        "seed": seed,
        "device": str(device),
        "best_val_roc_auc": best,
        "validation_threshold_max_f1": threshold,
        "validation_at_selected_threshold": metrics(vy, ps, threshold),
        "validation_at_0_5": metrics(vy, ps, 0.5),
        "best_epoch": next(
            (r["epoch"] for r in history if r["val_roc_auc"] == best), None
        ),
        "history": history,
        "test_used_for_selection_or_calibration": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": train["model_id"],
            "input_control": train["input_control"],
            "feature_dim": int(x.shape[1]),
            "state_dict": best_state,
            "seed": seed,
            "best_epoch": report["best_epoch"],
        },
        out / "best.pt",
    )
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    print("BEST_CHECKPOINT", out / "best.pt", flush=True)
    print("VALIDATION_REPORT", out / "validation_report.json", flush=True)
    return report


def evaluate_head(data, ckpt, val_report, device, out):
    """Evaluate a trained head with its saved validation threshold."""
    head = nn.Linear(ckpt["feature_dim"], 1)
    head.load_state_dict(ckpt["state_dict"])
    head.to(device).eval()
    with torch.inference_mode():
        scores = (
            head(data["features"].to(device))
            .squeeze(1)
            .sigmoid()
            .cpu()
            .numpy()
            .tolist()
        )
    ys = data["labels"].numpy().astype(int).tolist()
    threshold = float(val_report["validation_threshold_max_f1"])
    groups = {"all": list(range(len(ys)))}
    real_by_class = defaultdict(list)
    fake_by_generator_class = defaultdict(list)
    for i, (c, y, gen) in enumerate(
        zip(data["classes"].tolist(), ys, data["generators"])
    ):
        if y == 0:
            real_by_class[c].append(i)
        else:
            fake_by_generator_class[(gen, c)].append(i)
    for gen in sorted(set(data["generators"]) - {"real"}):
        idx = []
        for c in range(1000):
            fake = fake_by_generator_class[(gen, c)]
            idx.extend(fake)
            idx.extend(real_by_class[c][: len(fake)])
        groups[f"{gen}_vs_class_matched_real"] = idx
    metrics_rows = []
    for name, idx in groups.items():
        for tn, t in [("fixed_validation_f1", threshold), ("default_0_5", 0.5)]:
            metrics_rows.append(
                {
                    "group": name,
                    "threshold_name": tn,
                    "threshold": t,
                    "count": len(idx),
                    **metrics([ys[i] for i in idx], [scores[i] for i in idx], t),
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0]))
        w.writeheader()
        w.writerows(metrics_rows)
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
                data["paths"][i],
                ys[i],
                data["generators"][i],
                scores[i],
                int(scores[i] >= threshold),
                int(scores[i] >= 0.5),
            )
            for i in range(len(ys))
        )
    result = {
        "checkpoint": str(out.parent / "best.pt"),
        "model_id": ckpt["model_id"],
        "input_control": ckpt["input_control"],
        "split": data["split"],
        "manifest": data["manifest"],
        "files": len(ys),
        "threshold": threshold,
        "threshold_source": "CLIP validation_report.json; selected before this test evaluation",
        "metrics": metrics_rows,
        "test_used_for_selection_or_calibration": False,
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    """Run CLIP feature training or evaluate a locked checkpoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["train", "evaluate"], default="train")
    p.add_argument("--split", choices=list(SPLITS), default="test_seen")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--cache-dir", default="")
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--manifest", default="")
    p.add_argument(
        "--input-control",
        choices=["raw", "jpeg95", "cap128", "cap128_jpeg95"],
        default="cap128_jpeg95",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20261031)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--force-features", action="store_true")
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device(
        "cuda"
        if a.device == "auto" and torch.cuda.is_available()
        else ("cpu" if a.device == "auto" else a.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if a.device == "auto" and device.type == "cpu" and not a.smoke_test:
        raise RuntimeError(
            "No GPU visible; run full CLIP extraction inside the allocated GPU job."
        )
    out = Path(a.output_dir)
    model_id = a.model_id
    cache = a.cache_dir or None
    model, processor = model_and_processor(model_id, cache)
    model.to(device)
    manifest = (
        Path(a.manifest) if a.manifest else ROOT / f"manifests/{SPLITS[a.split][0]}"
    )
    if a.stage == "train":
        tr = extract(
            model,
            processor,
            ROOT / "manifests/all_splits.csv",
            "train",
            out,
            model_id,
            a.input_control,
            device,
            a.batch_size,
            a.workers,
            16 if a.smoke_test else 0,
            a.force_features,
        )
        va = extract(
            model,
            processor,
            ROOT / "manifests/all_splits.csv",
            "val",
            out,
            model_id,
            a.input_control,
            device,
            a.batch_size,
            a.workers,
            16 if a.smoke_test else 0,
            a.force_features,
        )
        fit_head(
            tr,
            va,
            device,
            out,
            a.seed,
            a.epochs,
            a.patience,
            a.lr,
            a.weight_decay,
            a.smoke_test,
        )
    else:
        ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        val = json.loads((out / "validation_report.json").read_text())
        actual_manifest = manifest
        data = extract(
            model,
            processor,
            actual_manifest,
            SPLITS[a.split][1],
            out,
            ckpt["model_id"],
            ckpt["input_control"],
            device,
            a.batch_size,
            a.workers,
            force=a.force_features,
        )
        evaluate_head(data, ckpt, val, device, out / a.split)


if __name__ == "__main__":
    main()
