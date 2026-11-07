#!/usr/bin/env python3
"""Read-only Asano ZIP duplicate, near-duplicate, and group-overlap audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_RE = re.compile(r"^data/(train|test)/([ABC])/([^/]+[.]tif)$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", dest="zip_path", type=Path, required=True)
    p.add_argument("--fold-root", type=Path, default=None,
                   help="Root containing seed_*/folds.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--strict-phash", type=int, default=4)
    p.add_argument("--review-phash", type=int, default=8)
    p.add_argument("--strict-correlation", type=float, default=0.995)
    p.add_argument("--review-correlation", type=float, default=0.980)
    return p.parse_args()


def group_from_filename(filename: str) -> str:
    """Heuristic: h27.12 -> h27; m111.2.31 -> m111.2."""
    stem = Path(filename).stem
    return stem.rsplit(".", 1)[0] if "." in stem else stem


def dct_matrix(n: int) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)[:, None]
    c = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    c[0] *= math.sqrt(1 / n)
    c[1:] *= math.sqrt(2 / n)
    return c


DCT32 = dct_matrix(32)


def bits_to_int(bits) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | int(bool(bit))
    return out


def image_fingerprints(raw: bytes):
    with Image.open(BytesIO(raw)) as im:
        im.load()
        width, height = im.size
        mode = im.mode
        pixel_hash = hashlib.sha256(
            f"{mode}|{width}|{height}|".encode() + im.tobytes()
        ).hexdigest()
        gray32 = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        gray64 = np.asarray(im.convert("L").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32)
        gray9x8 = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float32)

    coeff = DCT32 @ gray32 @ DCT32.T
    low = coeff[:8, :8].ravel()
    median = float(np.median(low[1:]))
    phash = bits_to_int(low > median)
    diff = gray9x8[:, 1:] > gray9x8[:, :-1]
    dhash = bits_to_int(diff.ravel())
    vec = gray64.ravel()
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return pixel_hash, phash, dhash, vec.astype(np.float32), width, height, mode


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def duplicate_groups(inventory, key, kind):
    grouped = defaultdict(list)
    for r in inventory:
        grouped[r[key]].append(r)
    rows = []
    for digest, members in grouped.items():
        if len(members) < 2:
            continue
        splits = sorted({m["split"] for m in members})
        classes = sorted({m["class"] for m in members})
        rows.append({
            "duplicate_type": kind,
            "hash": digest,
            "count": len(members),
            "cross_split": int(len(splits) > 1),
            "cross_class": int(len(classes) > 1),
            "splits": "|".join(splits),
            "classes": "|".join(classes),
            "samples": "|".join(sorted(m["sample"] for m in members)),
        })
    return rows


def inspect_near_duplicates(inventory, vectors, args):
    train_idx = [i for i, r in enumerate(inventory) if r["split"] == "train"]
    test_idx = [i for i, r in enumerate(inventory) if r["split"] == "test"]
    train_vectors = np.stack([vectors[i] for i in train_idx])
    test_vectors = np.stack([vectors[i] for i in test_idx])
    correlation_matrix = train_vectors @ test_vectors.T
    candidates = []
    for train_pos, i in enumerate(train_idx):
        a = inventory[i]
        for test_pos, j in enumerate(test_idx):
            b = inventory[j]
            pd = (int(a["phash_hex"], 16) ^ int(b["phash_hex"], 16)).bit_count()
            if pd > args.review_phash:
                continue
            dd = (int(a["dhash_hex"], 16) ^ int(b["dhash_hex"], 16)).bit_count()
            corr = float(correlation_matrix[train_pos, test_pos])
            strict = pd <= args.strict_phash and dd <= args.strict_phash and corr >= args.strict_correlation
            review = pd <= args.review_phash and dd <= args.review_phash and corr >= args.review_correlation
            if strict or review:
                candidates.append({
                    "train_sample": a["sample"], "train_class": a["class"],
                    "test_sample": b["sample"], "test_class": b["class"],
                    "same_class": int(a["class"] == b["class"]),
                    "phash_distance": pd, "dhash_distance": dd,
                    "normalized_64px_correlation": corr,
                    "tier": "strict" if strict else "review",
                })
        if (train_pos + 1) % 200 == 0:
            print(f"near-duplicate comparison: {train_pos + 1}/{len(train_idx)} train images", flush=True)
    candidates.sort(key=lambda r: (
        0 if r["tier"] == "strict" else 1,
        r["phash_distance"] + r["dhash_distance"],
        -r["normalized_64px_correlation"],
    ))
    top_n = min(50, correlation_matrix.size)
    flat = correlation_matrix.ravel()
    top_flat = np.argpartition(flat, -top_n)[-top_n:]
    top_flat = top_flat[np.argsort(flat[top_flat])[::-1]]
    closest = []
    for rank, flat_idx in enumerate(top_flat, 1):
        train_pos, test_pos = np.unravel_index(int(flat_idx), correlation_matrix.shape)
        i, j = train_idx[train_pos], test_idx[test_pos]
        a, b = inventory[i], inventory[j]
        pd = (int(a["phash_hex"], 16) ^ int(b["phash_hex"], 16)).bit_count()
        dd = (int(a["dhash_hex"], 16) ^ int(b["dhash_hex"], 16)).bit_count()
        closest.append({
            "rank": rank, "train_sample": a["sample"], "train_class": a["class"],
            "test_sample": b["sample"], "test_class": b["class"],
            "same_class": int(a["class"] == b["class"]),
            "phash_distance": pd, "dhash_distance": dd,
            "normalized_64px_correlation": float(flat[flat_idx]),
        })
    return candidates, closest


def audit_groups(inventory):
    groups = defaultdict(list)
    for r in inventory:
        groups[r["filename_group"]].append(r)
    rows = []
    for group, members in sorted(groups.items()):
        train = [m for m in members if m["split"] == "train"]
        test = [m for m in members if m["split"] == "test"]
        rows.append({
            "filename_group": group,
            "train_images": len(train), "test_images": len(test),
            "shared_across_split": int(bool(train) and bool(test)),
            "train_classes": "|".join(sorted({m["class"] for m in train})),
            "test_classes": "|".join(sorted({m["class"] for m in test})),
            "all_samples": "|".join(sorted(m["sample"] for m in members)),
        })
    return rows


def audit_folds(fold_root: Path | None, group_lookup):
    rows = []
    if not fold_root or not fold_root.exists():
        return rows
    for seed_dir in sorted(fold_root.glob("seed_*")):
        path = seed_dir / "folds.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        ordered = data["ordered_samples"]
        idx_to_sample = {int(r["index"]): r["sample"] for r in ordered}
        for fold in data["folds"]:
            train_samples = [idx_to_sample[int(i)] for i in fold["train_indices"]]
            val_samples = [idx_to_sample[int(i)] for i in fold["validation_indices"]]
            train_groups = {group_lookup[s] for s in train_samples}
            val_groups = {group_lookup[s] for s in val_samples}
            shared = train_groups & val_groups
            val_in_shared = sum(group_lookup[s] in train_groups for s in val_samples)
            rows.append({
                "seed": int(data.get("seed", seed_dir.name.split("_", 1)[1])),
                "fold": int(fold["fold"]),
                "train_images": len(train_samples), "validation_images": len(val_samples),
                "train_groups": len(train_groups), "validation_groups": len(val_groups),
                "shared_groups": len(shared),
                "validation_images_with_train_group": val_in_shared,
                "validation_images_with_train_group_percent": 100 * val_in_shared / len(val_samples),
            })
    return rows


def create_figure(inventory, exact_rows, near_rows, group_rows, fold_rows, out_path):
    split_counts = Counter(r["split"] for r in inventory)
    exact_cross_raw = sum(r["cross_split"] for r in exact_rows if r["duplicate_type"] == "raw_bytes")
    exact_cross_px = sum(r["cross_split"] for r in exact_rows if r["duplicate_type"] == "decoded_pixels")
    strict = sum(r["tier"] == "strict" for r in near_rows)
    review = sum(r["tier"] == "review" for r in near_rows)
    shared_groups = sum(r["shared_across_split"] for r in group_rows)
    total_groups = len(group_rows)
    fold_pct = [r["validation_images_with_train_group_percent"] for r in fold_rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), constrained_layout=True)
    axes[0].bar(["Train", "Test"], [split_counts["train"], split_counts["test"]], color=["#355C9A", "#E47B5B"])
    axes[0].set_title("Dataset images")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=.2)
    for i, v in enumerate([split_counts["train"], split_counts["test"]]):
        axes[0].text(i, v + 15, str(v), ha="center")

    labels = ["Raw exact", "Pixel exact", "Strict near", "Review near"]
    vals = [exact_cross_raw, exact_cross_px, strict, review]
    axes[1].bar(labels, vals, color=["#7B5EA7", "#9D79BC", "#159D88", "#E5B454"])
    axes[1].set_title("Train–test duplicate signals")
    axes[1].tick_params(axis="x", rotation=24)
    axes[1].grid(axis="y", alpha=.2)
    if max(vals) == 0:
        axes[1].set_ylim(0, 1)
        axes[1].text(.5, .55, "None detected", transform=axes[1].transAxes,
                     ha="center", va="center", fontsize=12, weight="bold", color="#087F6B")
    for i, v in enumerate(vals):
        axes[1].text(i, v + max(0.2, max(vals or [1]) * .025), str(v), ha="center", fontsize=9)

    axes[2].bar(["Shared filename\ngroups", "All filename\ngroups"], [shared_groups, total_groups], color=["#E47B5B", "#355C9A"])
    axes[2].set_title("Filename-derived group overlap")
    subtitle = f"OOF validation images sharing train group: {statistics.mean(fold_pct):.1f}%" if fold_pct else "OOF fold files unavailable"
    axes[2].text(.5, .96, subtitle, transform=axes[2].transAxes, ha="center", va="top", fontsize=8)
    axes[2].grid(axis="y", alpha=.2)
    for i, v in enumerate([shared_groups, total_groups]):
        axes[2].text(i, v + max(0.2, total_groups * .025), str(v), ha="center")
    fig.suptitle("Asano public dataset integrity audit (read-only)", weight="bold")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory, vectors, errors = [], [], []
    with zipfile.ZipFile(args.zip_path) as zf:
        members = []
        for info in zf.infolist():
            m = IMAGE_RE.match(info.filename)
            if m:
                members.append((info, m))
        members.sort(key=lambda x: x[0].filename)
        print(f"found {len(members)} primary TIFF images", flush=True)
        for idx, (info, match) in enumerate(members, 1):
            split, cls, filename = match.groups()
            raw = zf.read(info)
            try:
                pixel_hash, phash, dhash, vec, width, height, mode = image_fingerprints(raw)
            except Exception as exc:
                errors.append({"sample": info.filename, "error": repr(exc)})
                continue
            inventory.append({
                "sample": info.filename, "split": split.lower(), "class": cls.upper(),
                "filename": filename, "filename_group": group_from_filename(filename),
                "file_size": info.file_size, "zip_crc32": f"{info.CRC:08x}",
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "pixel_sha256": pixel_hash,
                "phash_hex": f"{phash:016x}", "dhash_hex": f"{dhash:016x}",
                "width": width, "height": height, "mode": mode,
            })
            vectors.append(vec)
            if idx % 100 == 0:
                print(f"fingerprints: {idx}/{len(members)}", flush=True)
    if errors:
        write_csv(args.output / "image_read_errors.csv", errors, ["sample", "error"])
    if len(inventory) != len(members):
        raise RuntimeError(f"Could not fingerprint all images: {len(inventory)}/{len(members)}")

    exact_rows = duplicate_groups(inventory, "raw_sha256", "raw_bytes")
    exact_rows += duplicate_groups(inventory, "pixel_sha256", "decoded_pixels")
    near_rows, closest_rows = inspect_near_duplicates(inventory, vectors, args)
    group_rows = audit_groups(inventory)
    group_lookup = {r["sample"]: r["filename_group"] for r in inventory if r["split"] == "train"}
    fold_rows = audit_folds(args.fold_root, group_lookup)

    basename_groups = defaultdict(list)
    for r in inventory:
        basename_groups[r["filename"]].append(r)
    basename_rows = []
    for name, members in sorted(basename_groups.items()):
        if {m["split"] for m in members} == {"train", "test"}:
            basename_rows.append({
                "filename": name,
                "train_samples": "|".join(m["sample"] for m in members if m["split"] == "train"),
                "test_samples": "|".join(m["sample"] for m in members if m["split"] == "test"),
                "same_class": int(len({m["class"] for m in members}) == 1),
            })

    inventory_fields = [
        "sample", "split", "class", "filename", "filename_group", "file_size", "zip_crc32",
        "raw_sha256", "pixel_sha256", "phash_hex", "dhash_hex", "width", "height", "mode",
    ]
    write_csv(args.output / "image_inventory.csv", inventory, inventory_fields)
    write_csv(args.output / "exact_duplicate_groups.csv", exact_rows,
              ["duplicate_type", "hash", "count", "cross_split", "cross_class", "splits", "classes", "samples"])
    write_csv(args.output / "near_duplicate_candidates.csv", near_rows,
              ["train_sample", "train_class", "test_sample", "test_class", "same_class", "phash_distance",
               "dhash_distance", "normalized_64px_correlation", "tier"])
    write_csv(args.output / "closest_visual_pairs.csv", closest_rows,
              ["rank", "train_sample", "train_class", "test_sample", "test_class", "same_class",
               "phash_distance", "dhash_distance", "normalized_64px_correlation"])
    write_csv(args.output / "filename_group_overlap.csv", group_rows,
              ["filename_group", "train_images", "test_images", "shared_across_split", "train_classes", "test_classes", "all_samples"])
    write_csv(args.output / "same_basename_across_split.csv", basename_rows,
              ["filename", "train_samples", "test_samples", "same_class"])
    write_csv(args.output / "oof_fold_group_overlap.csv", fold_rows,
              ["seed", "fold", "train_images", "validation_images", "train_groups", "validation_groups", "shared_groups",
               "validation_images_with_train_group", "validation_images_with_train_group_percent"])
    create_figure(inventory, exact_rows, near_rows, group_rows, fold_rows, args.output / "DATASET_INTEGRITY_AUDIT.png")

    counts = Counter((r["split"], r["class"]) for r in inventory)
    cross_raw = [r for r in exact_rows if r["duplicate_type"] == "raw_bytes" and r["cross_split"]]
    cross_pixel = [r for r in exact_rows if r["duplicate_type"] == "decoded_pixels" and r["cross_split"]]
    strict = [r for r in near_rows if r["tier"] == "strict"]
    review = [r for r in near_rows if r["tier"] == "review"]
    shared_group_names = {r["filename_group"] for r in group_rows if r["shared_across_split"]}
    test_shared = sum(r["split"] == "test" and r["filename_group"] in shared_group_names for r in inventory)
    test_total = sum(r["split"] == "test" for r in inventory)
    fold_pct = [r["validation_images_with_train_group_percent"] for r in fold_rows]
    max_corr = closest_rows[0]["normalized_64px_correlation"] if closest_rows else float("nan")

    report = [
        "# Asano 공개 데이터 중복·유사 이미지·그룹 중첩 점검", "",
        "## 점검 범위", "",
        "- 원본 ZIP을 수정하거나 풀지 않고 1,500개 primary TIFF를 읽기 전용으로 검사했다.",
        "- 모델을 학습하거나 성능을 다시 계산하지 않았다.",
        "- train/test 이미지는 데이터 무결성 점검에만 읽었으며, 이 결과로 hyperparameter를 선택하지 않는다.",
        "", "## 데이터 수", "",
        "| Split | A | B | C | 전체 |", "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "test"):
        vals = [counts[(split, c)] for c in "ABC"]
        report.append(f"| {split} | {vals[0]} | {vals[1]} | {vals[2]} | {sum(vals)} |")
    report += [
        "", "## 핵심 결과", "",
        "| 점검 항목 | 결과 | 의미 |", "|---|---:|---|",
        f"| train–test 완전 동일 파일(byte SHA-256) 그룹 | {len(cross_raw)} | 0이 아니면 직접 중복 가능성 |",
        f"| train–test 동일 decoded pixel 그룹 | {len(cross_pixel)} | 메타데이터가 달라도 픽셀이 같은 이미지 |",
        f"| train–test strict 근접중복 후보 | {len(strict)} | pHash/dHash≤{args.strict_phash}, corr≥{args.strict_correlation:.3f}; 육안 확인 필요 |",
        f"| train–test review 근접중복 후보 | {len(review)} | 더 넓은 기준의 검토 후보; 중복 확정 아님 |",
        f"| 가장 높은 저해상도 정규화 상관계수 | {max_corr:.4f} | `closest_visual_pairs.csv`에 상위 50쌍 저장 |",
        f"| train–test 동일 전체 파일명 | {len(basename_rows)} | 0이 아니면 분할 중복 여부 확인 필요 |",
        f"| 양 split에 나타난 파일명 유래 그룹 | {len(shared_group_names)}/{len(group_rows)} | 생물학적 그룹 ID인지 메타데이터 확인 필요 |",
        f"| test 이미지 중 train과 파일명 그룹 공유 | {test_shared}/{test_total} ({100*test_shared/test_total:.1f}%) | 그룹이 배양 batch라면 독립성에 영향 가능 |",
    ]
    if fold_pct:
        report.append(
            f"| OOF validation 이미지 중 해당 fold train과 그룹 공유 | 평균 {statistics.mean(fold_pct):.1f}% "
            f"(범위 {min(fold_pct):.1f}–{max(fold_pct):.1f}%) | 그룹이 독립 실험단위라면 random image-fold가 낙관적일 수 있음 |"
        )
    report += [
        "", "## 해석 시 주의", "",
        "1. byte/pixel hash가 같은 train–test 쌍은 직접 중복에 대한 강한 증거다.",
        "2. perceptual-hash 후보는 형태가 비슷한 서로 다른 organoid도 잡을 수 있으므로 `near_duplicate_candidates.csv`와 원본을 육안 확인해야 한다.",
        "3. 파일명의 마지막 점 뒤 번호를 제거한 값을 임시 그룹으로 사용했다(예: `h27.12→h27`, `m111.2.31→m111.2`).",
        "4. 이 접두사가 donor, well, culture batch 또는 촬영 session을 뜻한다는 공식 메타데이터는 현재 없으므로, 그룹 중첩만으로 data leakage라고 단정할 수 없다.",
        "5. 접두사가 실제 독립 실험단위로 확인되면 group-aware cross-validation을 별도 민감도 분석으로 수행하는 것이 적절하다.",
        "", "## 생성 파일", "",
        "- `image_inventory.csv`: 모든 이미지의 split/class/hash/크기/그룹",
        "- `exact_duplicate_groups.csv`: 완전 동일 파일 또는 pixel 그룹",
        "- `near_duplicate_candidates.csv`: train–test 근접중복 검토 후보",
        "- `closest_visual_pairs.csv`: 저해상도 정규화 상관계수가 높은 상위 50쌍(중복 확정 아님)",
        "- `filename_group_overlap.csv`: 파일명 유래 그룹의 split 중첩",
        "- `same_basename_across_split.csv`: 전체 파일명이 같은 train–test 항목",
        "- `oof_fold_group_overlap.csv`: 5-seed×5-fold 내부 그룹 중첩",
        "- `DATASET_INTEGRITY_AUDIT.png`: 요약 그림",
        "", "**MODEL TEST EVALUATION: NOT PERFORMED**", "",
    ]
    (args.output / "DATASET_AUDIT_RESULTS_KO.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "analysis": "read-only dataset integrity audit",
        "zip": str(args.zip_path),
        "images": len(inventory),
        "model_training": False,
        "model_test_evaluation": False,
        "test_images_read_for_integrity_audit": True,
        "near_duplicate_thresholds": {
            "strict_phash_and_dhash_max": args.strict_phash,
            "strict_correlation_min": args.strict_correlation,
            "review_phash_and_dhash_max": args.review_phash,
            "review_correlation_min": args.review_correlation,
        },
    }
    (args.output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Dataset integrity audit complete")
    print(args.output / "DATASET_AUDIT_RESULTS_KO.md")


if __name__ == "__main__":
    main()
