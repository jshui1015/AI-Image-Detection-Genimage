import csv
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader

from train import ROOT, ResNet50, metrics

MANIFEST = ROOT / "manifests/all_splits.csv"
MODEL_DIR = ROOT / "outputs/resnet50-cap128-jpeg95"
SPLITS = ["test_seen", "test_glide_unseen", "test_vqdm_unseen"]
CONDITIONS = [
    ("clean", None),
    ("jpeg_q75", ("jpeg", 75)),
    ("jpeg_q50", ("jpeg", 50)),
    ("resize_075", ("resize", 0.75)),
    ("resize_050", ("resize", 0.50)),
    ("blur_r05", ("blur", 0.5)),
    ("blur_r10", ("blur", 1.0)),
]

class PerturbedTestSet(Dataset):
    def __init__(self, rows, perturbation):
        self.rows = rows
        self.perturbation = perturbation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(ROOT / r["path"]) as src:
            im = ImageOps.exif_transpose(src).convert("RGB")

        # Match the baseline model's deterministic cap128 + JPEG95 input control.
        w, h = im.size
        scale = min(1.0, 128 / max(w, h))
        if scale < 1:
            im = im.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        with Image.open(buf) as encoded:
            im = encoded.convert("RGB")

        if self.perturbation:
            kind, value = self.perturbation
            if kind == "jpeg":
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=int(value))
                buf.seek(0)
                with Image.open(buf) as encoded:
                    im = encoded.convert("RGB")
            elif kind == "resize":
                w, h = im.size
                small = (max(1, round(w * value)), max(1, round(h * value)))
                im = im.resize(small, Image.Resampling.BILINEAR)
                im = im.resize((w, h), Image.Resampling.BILINEAR)
            elif kind == "blur":
                im = im.filter(ImageFilter.GaussianBlur(radius=float(value)))

        # Same deterministic validation/test resize and center crop as train.Images.
        w, h = im.size
        scale = 256 / min(w, h)
        nw, nh = round(w * scale), round(h * scale)
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        left, top = (nw - 224) // 2, (nh - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))

        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy())
        x = x.permute(2, 0, 1).div_(255)
        mean = x.new_tensor([.485, .456, .406])[:, None, None]
        std = x.new_tensor([.229, .224, .225])[:, None, None]
        x = (x - mean) / std
        return x, int(r["label"])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("请在已分配 GPU 的节点运行此评估脚本。")

    report = json.loads((MODEL_DIR / "validation_report.json").read_text())
    threshold = float(report["validation_threshold_max_f1"])
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=False)
    model = ResNet50().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with MANIFEST.open(newline="") as f:
        all_rows = list(csv.DictReader(f))

    rows_by_split = {
        "test_seen": [r for r in all_rows if r["split"] == "test_seen"]
    }
    for split in ("test_glide_unseen", "test_vqdm_unseen"):
        manifest_path = ROOT / "manifests" / f"{split}.csv"
        with manifest_path.open(newline="") as f:
            rows_by_split[split] = list(csv.DictReader(f))

    output = ROOT / "outputs/perturbation_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    results = []

    for split in SPLITS:
        rows = rows_by_split[split]
        if not rows:
            raise RuntimeError(f"清单中没有 split={split}")

        for condition, perturbation in CONDITIONS:
            dataset = PerturbedTestSet(rows, perturbation)
            loader = DataLoader(
                dataset, batch_size=64, shuffle=False,
                num_workers=4, pin_memory=True,
            )
            ys, ps = [], []
            with torch.inference_mode():
                for x, y in loader:
                    logits = model(x.to(device, non_blocking=True))
                    ys.extend(y.tolist())
                    ps.extend(logits.sigmoid().cpu().tolist())

            m = metrics(ys, ps, threshold)
            result = {
                "split": split,
                "condition": condition,
                "count": len(ys),
                "threshold": threshold,
                **m,
            }
            results.append(result)
            print(json.dumps(result), flush=True)

    payload = {
        "checkpoint": str(MODEL_DIR / "best.pt"),
        "threshold": threshold,
        "threshold_source": "fixed validation_report.json",
        "device": torch.cuda.get_device_name(0),
        "conditions": [name for name, _ in CONDITIONS],
        "派生图片按原测试图片计数": True,
        "results": results,
    }
    out_file = output / "resnet50_perturbations.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print("REPORT", out_file)

if __name__ == "__main__":
    main()
