import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import Images, ResNet50

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifests/all_splits.csv"

MODELS = {
    "baseline": ROOT / "outputs/resnet50-cap128-jpeg95",
    "robust": ROOT / "outputs/resnet50-robust-aug",
}

def predict(model_dir, device):
    report = json.loads((model_dir / "validation_report.json").read_text())
    threshold = float(report["validation_threshold_max_f1"])
    checkpoint = torch.load(
        model_dir / "best.pt", map_location="cpu", weights_only=False
    )

    model = ResNet50()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    dataset = Images(
        MANIFEST, "val", train=False,
        input_control="cap128_jpeg95", robust_augment=False
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda")
    )

    rows = []
    with torch.inference_mode():
        for x, y, generator, paths in loader:
            logits = model(x.to(device, non_blocking=True))
            probs = logits.sigmoid().cpu().tolist()
            for label, gen, path, prob in zip(y.tolist(), generator, paths, probs):
                rows.append({
                    "path": path,
                    "label": int(label),
                    "generator": gen,
                    "probability": float(prob),
                    "threshold": threshold,
                    "prediction": int(prob >= threshold),
                })
    return rows, report

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name, model_dir in MODELS.items():
        rows, report = predict(model_dir, device)
        results[name] = {r["path"]: r for r in rows}
        print(
            f'{name}: device={device}, n={len(rows)}, '
            f'best_epoch={report["best_epoch"]}, '
            f'val_AUROC={report["best_val_roc_auc"]:.6f}, '
            f'threshold={rows[0]["threshold"]:.6f}'
        )

    assert results["baseline"].keys() == results["robust"].keys(), \
        "两个模型的验证图片不一致"
    out = ROOT / "outputs/validation_error_analysis"
    out.mkdir(parents=True, exist_ok=True)

    manifest_rows = {
        r["path"]: r for r in csv.DictReader(MANIFEST.open())
        if r["split"] == "val"
    }
    fields = [
        "path", "content_class", "label", "generator",
        "baseline_probability", "baseline_prediction", "baseline_error",
        "robust_probability", "robust_prediction", "robust_error",
        "change",
    ]
    comparisons = []
    for path, b in results["baseline"].items():
        r = results["robust"][path]
        meta = manifest_rows[path]
        be = int(b["prediction"] != b["label"])
        re = int(r["prediction"] != r["label"])
        if be and re:
            change = "both_wrong"
        elif be and not re:
            change = "robust_fixed"
        elif not be and re:
            change = "robust_regressed"
        else:
            change = "both_correct"
        comparisons.append({
            "path": path,
            "content_class": meta.get("content_class", ""),
            "label": b["label"],
            "generator": b["generator"],
            "baseline_probability": b["probability"],
            "baseline_prediction": b["prediction"],
            "baseline_error": be,
            "robust_probability": r["probability"],
            "robust_prediction": r["prediction"],
            "robust_error": re,
            "change": change,
        })

    with (out / "val_all_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    errors = [r for r in comparisons if r["change"] != "both_correct"]
    with (out / "val_error_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    from collections import Counter
    counts = Counter(r["change"] for r in comparisons)
    print("validation comparison:", dict(counts))
    print("all predictions:", out / "val_all_predictions.csv")
    print("error cases:", out / "val_error_cases.csv")

if __name__ == "__main__":
    main()
