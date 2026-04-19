#!/usr/bin/env python3
"""Prepare CCPD images and labels in YOLO format.

This script samples a subset of CCPD images, parses filename-encoded bounding
boxes, and writes a train/val YOLO dataset structure.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import struct
from pathlib import Path
from typing import Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BBOX_PATTERN = re.compile(r"^(\d+)&(\d+)_(\d+)&(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CCPD filename annotations to YOLO labels."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/CCPD2020/ccpd_green"),
        help="Path to CCPD subset root (contains train/val/test folders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/ccpd"),
        help="Destination root for YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of random images to sample.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (e.g., 0.8 => 80%% train, 20%% val).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def find_images(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]


def parse_bbox_from_filename(image_path: Path) -> tuple[int, int, int, int]:
    parts = image_path.stem.split("-")
    if len(parts) < 3:
        raise ValueError("Unexpected CCPD filename format.")

    match = BBOX_PATTERN.match(parts[2])
    if not match:
        raise ValueError("Could not parse bbox segment from filename.")

    x1, y1, x2, y2 = map(int, match.groups())
    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    return x_min, y_min, x_max, y_max


def read_png_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG header.")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def read_jpeg_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError("Invalid JPEG header.")

        while True:
            marker_prefix = f.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue

            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)

            if not marker:
                break

            marker_val = marker[0]
            if marker_val in {0xD8, 0xD9}:
                continue

            segment_len_raw = f.read(2)
            if len(segment_len_raw) != 2:
                break
            segment_len = struct.unpack(">H", segment_len_raw)[0]
            if segment_len < 2:
                raise ValueError("Invalid JPEG segment length.")

            is_sof = 0xC0 <= marker_val <= 0xCF and marker_val not in {0xC4, 0xC8, 0xCC}
            if is_sof:
                segment_data = f.read(segment_len - 2)
                if len(segment_data) < 5:
                    raise ValueError("Invalid SOF segment.")
                height, width = struct.unpack(">HH", segment_data[1:5])
                return width, height

            f.seek(segment_len - 2, 1)

    raise ValueError("Could not locate JPEG dimensions.")


def read_image_size(image_path: Path) -> tuple[int, int]:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return read_png_size(image_path)
    if suffix in {".jpg", ".jpeg"}:
        return read_jpeg_size(image_path)
    raise ValueError(f"Unsupported image type: {suffix}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def to_yolo_line(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    class_id: int = 0,
) -> str:
    x_min, y_min, x_max, y_max = bbox

    x_min = clamp(float(x_min), 0.0, float(width - 1))
    x_max = clamp(float(x_max), 0.0, float(width - 1))
    y_min = clamp(float(y_min), 0.0, float(height - 1))
    y_max = clamp(float(y_max), 0.0, float(height - 1))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Degenerate bounding box after clamping.")

    x_center = ((x_min + x_max) / 2.0) / width
    y_center = ((y_min + y_max) / 2.0) / height
    box_w = (x_max - x_min) / width
    box_h = (y_max - y_min) / height

    x_center = clamp(x_center, 0.0, 1.0)
    y_center = clamp(y_center, 0.0, 1.0)
    box_w = clamp(box_w, 0.0, 1.0)
    box_h = clamp(box_h, 0.0, 1.0)

    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_unique_name(source: Path, used_names: set[str]) -> str:
    split_name = source.parent.name
    candidate = f"{split_name}_{source.name}"

    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    stem = f"{split_name}_{source.stem}"
    suffix = source.suffix
    index = 1
    while True:
        alt = f"{stem}_{index}{suffix}"
        if alt not in used_names:
            used_names.add(alt)
            return alt
        index += 1


def process_split(
    split_name: str,
    files: Iterable[Path],
    output_root: Path,
    used_names: set[str],
) -> tuple[int, int]:
    image_dir = output_root / "images" / split_name
    label_dir = output_root / "labels" / split_name
    ensure_dir(image_dir)
    ensure_dir(label_dir)

    success = 0
    skipped = 0

    for src in files:
        try:
            bbox = parse_bbox_from_filename(src)
            width, height = read_image_size(src)
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions.")

            yolo_line = to_yolo_line(bbox, width, height, class_id=0)
            image_name = build_unique_name(src, used_names)

            dst_image = image_dir / image_name
            dst_label = label_dir / f"{Path(image_name).stem}.txt"

            shutil.copy2(src, dst_image)
            dst_label.write_text(yolo_line + "\n", encoding="utf-8")
            success += 1
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Skipping {src}: {exc}")

    return success, skipped


def main() -> int:
    args = parse_args()

    if not args.source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {args.source_dir}")
        return 1

    if args.sample_size <= 0:
        print("[ERROR] --sample-size must be > 0")
        return 1

    if not (0.0 < args.train_ratio < 1.0):
        print("[ERROR] --train-ratio must be between 0 and 1")
        return 1

    all_images = find_images(args.source_dir)
    if not all_images:
        print(f"[ERROR] No images found in {args.source_dir}")
        return 1

    rng = random.Random(args.seed)
    sample_count = min(args.sample_size, len(all_images))
    sampled_images = rng.sample(all_images, sample_count)
    rng.shuffle(sampled_images)

    split_index = int(sample_count * args.train_ratio)
    train_files = sampled_images[:split_index]
    val_files = sampled_images[split_index:]

    used_names: set[str] = set()

    train_ok, train_skipped = process_split(
        split_name="train",
        files=train_files,
        output_root=args.output_dir,
        used_names=used_names,
    )
    val_ok, val_skipped = process_split(
        split_name="val",
        files=val_files,
        output_root=args.output_dir,
        used_names=used_names,
    )

    total_ok = train_ok + val_ok
    total_skipped = train_skipped + val_skipped

    print("\n[INFO] CCPD to YOLO conversion finished")
    print(f"[INFO] Source directory   : {args.source_dir}")
    print(f"[INFO] Output directory   : {args.output_dir}")
    print(f"[INFO] Requested sample   : {args.sample_size}")
    print(f"[INFO] Sampled images     : {sample_count}")
    print(f"[INFO] Train/Val requested: {args.train_ratio:.2f}/{1.0 - args.train_ratio:.2f}")
    print(f"[INFO] Train processed    : {train_ok} (skipped {train_skipped})")
    print(f"[INFO] Val processed      : {val_ok} (skipped {val_skipped})")
    print(f"[INFO] Total processed    : {total_ok} (skipped {total_skipped})")

    return 0 if total_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
