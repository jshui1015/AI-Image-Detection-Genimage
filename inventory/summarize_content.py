"""Finalize exact dedup and conservative near-duplicate source grouping.
Near matches are retained, grouped for leakage prevention, never auto-deleted.
"""

import collections
import csv
import hashlib
import io
import itertools
import json
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

R = Path("/scratch/jshui/GenImage")
O = R / "inventory"


def log(*a):
    """Print a progress message and append it to the audit log."""
    print(*a, flush=True)


def main():
    """Build content-level counts and review summaries from image metadata."""
    dbpath = Path(tempfile.mkdtemp(prefix="genimage-content-")) / "audit.sqlite"
    db = sqlite3.connect(dbpath)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute(
        "CREATE TABLE images(id INTEGER PRIMARY KEY,path TEXT,label INTEGER,generator TEXT,official_split TEXT,class_index INTEGER,content_class TEXT,evidence TEXT,file_sha TEXT,pixel_sha TEXT,thumbnail_sha TEXT,dhash TEXT,width INTEGER,height INTEGER,error TEXT)"
    )
    count = 0
    with (O / "class_mapping_manifest.csv").open() as mf, (
        O / "content_hashes.csv"
    ).open() as hf:
        batch = []
        for m, h in itertools.zip_longest(csv.DictReader(mf), csv.DictReader(hf)):
            assert m and h and m["path"] == h["path"], "Hash coverage/order mismatch"
            count += 1
            batch.append(
                (
                    count,
                    m["path"],
                    int(m["label"]),
                    m["generator"],
                    m["official_split"],
                    int(m["class_index"]),
                    m["content_class"],
                    m["mapping_evidence"],
                    h["file_sha256"],
                    h["pixel_sha256"],
                    h["thumbnail_sha256"],
                    h["dhash64"],
                    int(h["width"] or 0),
                    int(h["height"] or 0),
                    h["error"],
                )
            )
            if len(batch) >= 10000:
                db.executemany(
                    "INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
                )
                batch = []
        db.executemany(
            "INSERT INTO images VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
        )
        db.commit()
    log("loaded", count)
    db.executescript(
        "CREATE INDEX pixel_idx ON images(pixel_sha); CREATE INDEX file_idx ON images(file_sha); CREATE INDEX class_idx ON images(class_index);"
    )
    groups = list(
        db.execute(
            "SELECT pixel_sha,MIN(id),COUNT(*),COUNT(DISTINCT label),COUNT(DISTINCT content_class),COUNT(DISTINCT official_split) FROM images WHERE error='' GROUP BY pixel_sha"
        )
    )
    exact = {
        sha: (n, labels, classes, splits, first)
        for sha, first, n, labels, classes, splits in groups
    }
    canonical = {sha: first for sha, first, n, labels, classes, splits in groups}
    conflict = {
        sha
        for sha, nlabels in ((s, v[1:3]) for s, v in exact.items())
        if max(nlabels) > 1
    }
    log("pixel unique", len(groups), "conflict groups", len(conflict))
    for field, name in [
        ("pixel_sha", "exact_duplicate_members.csv"),
        ("file_sha", "byte_duplicate_members.csv"),
    ]:
        with (O / name).open("w") as f:
            w = csv.writer(f)
            w.writerow(
                ["hash", "path", "label", "generator", "official_split", "class_index"]
            )
            for row in db.execute(
                f"SELECT {field},path,label,generator,official_split,class_index FROM images WHERE error='' AND {field} IN (SELECT {field} FROM images WHERE error='' GROUP BY {field} HAVING COUNT(*)>1) ORDER BY {field},id"
            ):
                w.writerow(row)
    with (O / "unreadable_files.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["path", "error"])
        w.writerows(db.execute("SELECT path,error FROM images WHERE error<>''"))
    db.executescript(
        "CREATE TABLE representatives AS SELECT * FROM images WHERE error='' AND id IN (SELECT MIN(id) FROM images WHERE error='' GROUP BY pixel_sha); CREATE INDEX rep_class_idx ON representatives(class_index);"
    )
    # Representatives only; low-information hashes bypass perceptual matching.
    parent = {}
    rank = {}

    def find(a):
        """Return the disjoint-set representative for one source group."""
        parent.setdefault(a, a)
        if parent[a] != a:
            parent[a] = find(parent[a])
        return parent[a]

    def union(a, b):
        """Merge two source groups in the disjoint-set structure."""
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    near_count = 0
    candidate_count = 0
    crosslabel = 0

    def thumb(path):
        """Return a thumbnail for one image path, or a placeholder on failure."""
        with Image.open(R / path) as im:
            return np.asarray(
                ImageOps.exif_transpose(im)
                .convert("RGB")
                .resize((32, 32), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )

    with (O / "near_duplicate_candidates.csv").open("w") as f, ThreadPoolExecutor(
        max_workers=8
    ) as pool:
        w = csv.writer(f)
        w.writerow(
            [
                "path_a",
                "path_b",
                "pixel_sha_a",
                "pixel_sha_b",
                "class_index",
                "label_a",
                "label_b",
                "dhash_distance",
                "rgb_rmse",
                "action",
            ]
        )
        for c in range(1000):
            rows = list(
                db.execute(
                    "SELECT id,path,label,pixel_sha,dhash FROM representatives WHERE class_index=?",
                    (c,),
                )
            )
            rows = [r for r in rows if 4 < int(r[4], 16).bit_count() < 60]
            if len(rows) < 2:
                continue
            hs = np.array([int(r[4], 16) for r in rows], dtype=np.uint64)
            dist = np.bitwise_count(hs[:, None] ^ hs[None, :])
            ii, jj = np.where(np.triu(dist <= 3, k=1))
            candidate_count += len(ii)
            if len(ii):
                needed = sorted(set(ii.tolist() + jj.tolist()))
                thumbs = dict(
                    zip(needed, pool.map(thumb, [rows[k][1] for k in needed]))
                )
                for i, j in zip(ii, jj):
                    a, b = rows[i], rows[j]
                    rmse = float(np.sqrt(np.mean((thumbs[i] - thumbs[j]) ** 2)))
                    if rmse > 3.0:
                        continue
                    action = (
                        "retain_both_group_same_split"
                        if a[2] == b[2]
                        else "cross_label_review"
                    )
                    w.writerow(
                        [
                            a[1],
                            b[1],
                            a[3],
                            b[3],
                            c,
                            a[2],
                            b[2],
                            int(dist[i, j]),
                            round(rmse, 6),
                            action,
                        ]
                    )
                    near_count += 1
                    if a[2] == b[2]:
                        union(a[3], b[3])
                    else:
                        crosslabel += 1
            if c % 100 == 0:
                log(
                    "near class",
                    c,
                    "hash candidates",
                    candidate_count,
                    "conservative matches",
                    near_count,
                )
    # Exact groups spanning official boundaries stay grouped; subsequent splitting must honor source_group.
    fields = [
        "image_id",
        "path",
        "label",
        "generator",
        "content_class",
        "class_index",
        "source_group",
        "official_split",
        "mapping_evidence",
        "status",
        "canonical_path",
    ]
    first_paths = {
        sha: path
        for sha, path in db.execute(
            "SELECT pixel_sha,path FROM images WHERE id IN (SELECT MIN(id) FROM images WHERE error='' GROUP BY pixel_sha)"
        )
    }
    kept = collections.Counter()
    statuses = collections.Counter()
    dims = collections.Counter()
    with (O / "content_manifest.csv").open("w") as f, (
        O / "deduplicated_manifest.csv"
    ).open("w") as uf:
        w = csv.writer(f)
        uw = csv.writer(uf)
        w.writerow(fields)
        uw.writerow(fields)
        for (
            iid,
            path,
            label,
            gen,
            split,
            idx,
            syn,
            evidence,
            filehash,
            pixel,
            th,
            dh,
            width,
            height,
            error,
        ) in db.execute("SELECT * FROM images ORDER BY id"):
            status = (
                "invalid"
                if error
                else (
                    "mapping_or_label_conflict"
                    if pixel in conflict
                    else "duplicate" if iid != canonical[pixel] else "keep"
                )
            )
            source = "src_" + find(pixel) if pixel else ""
            row = [
                filehash,
                path,
                label,
                gen,
                syn,
                idx,
                source,
                split,
                evidence,
                status,
                first_paths.get(pixel, ""),
            ]
            w.writerow(row)
            statuses[status] += 1
            dims[(gen, label, width, height)] += 1
            if status == "keep":
                uw.writerow(row)
                kept[(gen, label, split)] += 1
    byte_stats = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM images WHERE error='' GROUP BY file_sha HAVING COUNT(*)>1)"
    ).fetchone()
    error_count = db.execute("SELECT COUNT(*) FROM images WHERE error<>''").fetchone()[
        0
    ]
    res = {
        "input_files": count,
        "decoded_successfully": count - error_count,
        "invalid_files": error_count,
        "byte_duplicate_groups": byte_stats[0],
        "byte_redundant_files": byte_stats[1],
        "pixel_duplicate_groups": sum(n > 1 for n, _, _, _, _ in exact.values()),
        "pixel_redundant_files": sum(n - 1 for n, _, _, _, _ in exact.values()),
        "pixel_groups_spanning_official_splits": sum(
            v[0] > 1 and v[3] > 1 for v in exact.values()
        ),
        "conflicting_pixel_groups": len(conflict),
        "status_counts": dict(statuses),
        "kept_groups": [
            {"generator": k[0], "label": k[1], "official_split": k[2], "files": v}
            for k, v in sorted(kept.items())
        ],
        "near_hash_candidates": candidate_count,
        "near_conservative_pairs": near_count,
        "near_cross_label_pairs": crosslabel,
        "near_method": {
            "hash": "64-bit dHash",
            "maximum_hamming_distance": 3,
            "maximum_RGB_32x32_RMSE": 3.0,
            "scope": "within same ImageNet class, exact pixel representatives only",
            "action": "retain near images, union same-label source groups for future split; no automatic near-image deletion",
        },
        "limitations": [
            "No exhaustive crop/rotation matching.",
            "Perceptual matching within class only; low-information hashes skipped.",
            "Fake prefix mapping supported by canonical indexes, 0..999 coverage and 18 visual examples; original export naming script unavailable.",
            "Image source authenticity not independently certified.",
        ],
    }
    (O / "dedup_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    log(json.dumps(res, ensure_ascii=False, indent=2))
    db.close()
    log("temporary_index", str(dbpath))


if __name__ == "__main__":
    main()
