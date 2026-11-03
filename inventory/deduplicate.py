"""Read-only image audit. Append-only, resumable hash cache; never deletes images."""

import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path("/scratch/jshui/GenImage")
OUT = ROOT / "inventory"
CACHE = OUT / "content_hashes.csv"
FIELDS = [
    "path",
    "bytes",
    "mtime_ns",
    "file_sha256",
    "pixel_sha256",
    "dhash64",
    "thumbnail_sha256",
    "width",
    "height",
    "mode",
    "error",
]


def audit(row):
    """Compute image fingerprints and metadata for one candidate row."""
    p = ROOT / row["path"]
    d = dict.fromkeys(FIELDS, "")
    d["path"] = row["path"]
    try:
        st = p.stat()
        d.update(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        if not st.st_size:
            raise ValueError("zero-byte file")
        raw = p.read_bytes()
        d["file_sha256"] = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            d.update(width=im.width, height=im.height, mode=im.mode)
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            d["pixel_sha256"] = hashlib.sha256(
                f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()
            ).hexdigest()
            thumb = rgb.resize((32, 32), Image.Resampling.LANCZOS)
            d["thumbnail_sha256"] = hashlib.sha256(thumb.tobytes()).hexdigest()
            a = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
            bits = (a[:, 1:] > a[:, :-1]).ravel()
            d["dhash64"] = f'{int.from_bytes(np.packbits(bits).tobytes(),"big"):016x}'
    except Exception as e:
        d["error"] = f"{type(e).__name__}: {e}"
    return d


def main():
    """Detect exact and near duplicates and write the audit manifests."""
    done = set()
    if CACHE.exists():
        with CACHE.open() as f:
            for r in csv.DictReader(f):
                if r.get("path") and r.get("error") is not None:
                    done.add(r["path"])
    start = time.monotonic()
    count = 0
    errors = 0
    with (OUT / "files.csv").open() as f, CACHE.open(
        "a", buffering=1024 * 1024
    ) as out, concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        if out.tell() == 0:
            writer.writeheader()
        batch = []
        for row in csv.DictReader(f):
            if row["path"] in done:
                continue
            batch.append(row)
            if len(batch) < 512:
                continue
            for r in pool.map(audit, batch):
                writer.writerow(r)
                count += 1
                errors += bool(r["error"])
            batch = []
            out.flush()
            if count % 5120 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(done) + count,
                            "new_errors": errors,
                            "seconds": round(time.monotonic() - start),
                            "rate": round(count / (time.monotonic() - start), 1),
                        }
                    ),
                    flush=True,
                )
        for r in pool.map(audit, batch):
            writer.writerow(r)
            count += 1
            errors += bool(r["error"])
    print("HASHING_COMPLETE", len(done) + count, "errors_this_run", errors, flush=True)


if __name__ == "__main__":
    main()
