#!/usr/bin/env python3
"""Train YOLO on CCPD with Apple MPS acceleration.

This script intentionally enforces MPS usage and does not allow CPU fallback.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on CCPD dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/ccpd.yaml"),
        help="Path to YOLO dataset yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO checkpoint (e.g., yolov8n.pt, yolov11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Data loader workers.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/train"),
        help="Output directory for training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="ccpd_yolo_mps",
        help="Run name.",
    )
    return parser.parse_args()


def ensure_mps_is_ready() -> None:
    if not torch.backends.mps.is_built():
        raise RuntimeError(
            "Current PyTorch build does not include MPS support. "
            "Install a PyTorch build with MPS enabled."
        )

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available on this machine/session. "
            "Training is configured to use device='mps' only."
        )

    # Disable unsupported op fallback to CPU to keep runs strictly on MPS.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

    ensure_mps_is_ready()

    model = YOLO(args.model)

    results = model.train(
        data=str(args.data),
        device="mps",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
    )

    print("[INFO] Training finished.")
    print(f"[INFO] Results object: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
