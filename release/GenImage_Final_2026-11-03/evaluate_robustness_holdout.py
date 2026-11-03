import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset, DataLoader

from train import ROOT, ResNet50, metrics

MANIFEST = ROOT / "manifests/test_robustness_new.csv"
MODEL_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "outputs/resnet50-robust-v2-seed20261102"
CONDITIONS = [
    ("clean", None), ("jpeg_q75", ("jpeg", 75)), ("jpeg_q50", ("jpeg", 50)),
    ("resize_075", ("resize", 0.75)), ("resize_050", ("resize", 0.50)),
    ("blur_r05", ("blur", 0.5)), ("blur_r10", ("blur", 1.0)),
]


class HoldoutSet(Dataset):
    def __init__(self, rows, perturbation):
        self.rows, self.perturbation = rows, perturbation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(ROOT / r["path"]) as src:
            im = ImageOps.exif_transpose(src).convert("RGB")
        w, h = im.size
        scale = min(1.0, 128 / max(w, h))
        if scale < 1:
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        with Image.open(buf) as enc:
            im = enc.convert("RGB")

        if self.perturbation:
            kind, value = self.perturbation
            if kind == "jpeg":
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=int(value))
                buf.seek(0)
                with Image.open(buf) as enc:
                    im = enc.convert("RGB")
            elif kind == "resize":
                w, h = im.size
                small = (max(1, round(w * value)), max(1, round(h * value)))
                im = im.resize(small, Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)
            elif kind == "blur":
                im = im.filter(ImageFilter.GaussianBlur(radius=float(value)))

        w, h = im.size
        scale = 256 / min(w, h)
        nw, nh = round(w * scale), round(h * scale)
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        left, top = (nw - 224) // 2, (nh - 224) // 2
        im = im.crop((left, top, left + 224, top + 224))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1).div_(255)
        mean = x.new_tensor([.485, .456, .406])[:, None, None]
        std = x.new_tensor([.229, .224, .225])[:, None, None]
        return x.sub(mean).div(std), int(r["label"]), r["matched_generator"]


def main():
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing frozen holdout manifest: {MANIFEST}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Run this one-time final evaluation on an allocated GPU node.")
    report = json.loads((MODEL_DIR / "validation_report.json").read_text())
    threshold = float(report["validation_threshold_max_f1"])
    ckpt = torch.load(MODEL_DIR / "best.pt", map_location="cpu", weights_only=False)
    model = ResNet50().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with MANIFEST.open(newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for condition, perturbation in CONDITIONS:
        loader = DataLoader(HoldoutSet(rows, perturbation), batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)
        ys, ps, groups = [], [], []
        with torch.inference_mode():
            for x, y, g in loader:
                z = model(x.to(device, non_blocking=True))
                ys.extend(y.tolist())
                ps.extend(z.sigmoid().cpu().tolist())
                groups.extend(g)
        result = {"split": "test_robust_new", "condition": condition,
                  "group": "all", "count": len(ys), "threshold": threshold,
                  **metrics(ys, ps, threshold)}
        results.append(result)
        print(json.dumps(result), flush=True)
        for group in sorted(set(groups)):
            idx = [i for i, g in enumerate(groups) if g == group]
            result = {"split": "test_robust_new", "condition": condition,
                      "group": group, "count": len(idx), "threshold": threshold,
                      **metrics([ys[i] for i in idx], [ps[i] for i in idx], threshold)}
            results.append(result)
            print(json.dumps(result), flush=True)

    out_dir = ROOT / "outputs/robustness_new_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{MODEL_DIR.name}.json"
    payload = {"checkpoint": str(MODEL_DIR / "best.pt"), "threshold": threshold,
               "threshold_source": "fixed validation_report.json", "device": torch.cuda.get_device_name(0),
               "holdout_manifest": str(MANIFEST), "conditions": [n for n, _ in CONDITIONS],
               "derived_images_counted_as_new_samples": False, "results": results}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("REPORT", out)


if __name__ == "__main__":
    main()
