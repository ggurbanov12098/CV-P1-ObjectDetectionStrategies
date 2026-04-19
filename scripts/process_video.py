#!/usr/bin/env python3
"""Run side-by-side classical and YOLO plate detection on video."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, cast

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from classical_detector import detect_license_plate_classical


RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
MAX_FRAMES = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a video with Classical CV and YOLO side-by-side."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("chongqing_4k.mp4"),
        help="Input video path.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/runs/train/ccpd_yolo_mps/weights/best.pt"),
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison_output.mp4"),
        help="Output comparison video path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=MAX_FRAMES,
        help="Maximum number of frames to process.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=TARGET_WIDTH,
        help="Processing width.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=TARGET_HEIGHT,
        help="Processing height.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Inference device for YOLO.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable live cv2.imshow preview (recommended on Colab/headless environments).",
    )
    return parser.parse_args()


def put_branch_label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(frame, (10, 10), (390, 52), color, thickness=-1)
    cv2.putText(
        frame,
        text,
        (18, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2,
        cv2.LINE_AA,
    )


def ensure_device(device: str) -> None:
    if device == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError(
                "Current PyTorch build does not include MPS support; cannot run on device='mps'."
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in this environment.")
        return

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this environment.")
        return

    if device == "cpu":
        return

    raise ValueError(f"Unsupported device: {device}")


def put_metrics(
    frame: np.ndarray,
    detections: int,
    latency_ms: float,
    text_color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        f"Detections: {detections}",
        (18, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Latency: {latency_ms:.1f} ms",
        (18, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        text_color,
        2,
        cv2.LINE_AA,
    )


def put_fps_overlay(stacked: np.ndarray, fps_value: float) -> None:
    text = f"FPS: {fps_value:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    pad_x = 14
    pad_y = 10
    frame_h, frame_w = stacked.shape[:2]

    box_w = text_w + (2 * pad_x)
    box_h = text_h + baseline + (2 * pad_y)

    box_x1 = max(0, (frame_w - box_w) // 2)
    box_y1 = max(0, frame_h - box_h - 18)
    box_x2 = min(frame_w - 1, box_x1 + box_w)
    box_y2 = min(frame_h - 1, box_y1 + box_h)

    cv2.rectangle(stacked, (box_x1, box_y1), (box_x2, box_y2), (32, 32, 32), thickness=-1)

    text_x = box_x1 + pad_x
    text_y = box_y2 - pad_y - baseline
    cv2.putText(
        stacked,
        text,
        (text_x, text_y),
        font,
        font_scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_yolo_detections(
    frame: np.ndarray,
    model: YOLO,
    conf: float,
    device: str,
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    output = frame.copy()
    detection_count = 0
    results = model.predict(source=output, device=device, conf=conf, verbose=False)

    if not results:
        return output, detection_count

    boxes = results[0].boxes
    if boxes is None:
        return output, detection_count

    for box in boxes:
        coords = box.xyxy[0].tolist()
        if len(coords) != 4:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in coords]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(output, (x1, y1), (x2, y2), GREEN, 2)
        detection_count += 1

    return output, detection_count


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    if not args.weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")

    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be > 0")

    ensure_device(args.device)

    model = YOLO(str(args.weights))
    model.to(args.device)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    output_width = args.target_width * 2
    output_height = args.target_height
    fourcc_fn_obj = getattr(cv2, "VideoWriter_fourcc", None)
    fourcc_fn: Callable[[str, str, str, str], int]
    if callable(fourcc_fn_obj):
        fourcc_fn = cast(Callable[[str, str, str, str], int], fourcc_fn_obj)
    else:
        fourcc_fn = cv2.VideoWriter.fourcc
    fourcc = fourcc_fn(*"mp4v")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        str(args.output),
        fourcc,
        fps,
        (output_width, output_height),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for: {args.output}")

    window_name = "Classical vs YOLO Comparison"
    processed = 0
    ema_fps: float | None = None

    try:
        while processed < args.max_frames:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("[INFO] End of video reached.")
                break

            resized = cv2.resize(
                frame,
                (args.target_width, args.target_height),
                interpolation=cv2.INTER_AREA,
            )

            classical_start = time.perf_counter()
            classical_result = detect_license_plate_classical(
                resized.copy(),
                box_color=RED,
                box_thickness=2,
                return_count=True,
            )
            classical_frame, classical_count = classical_result
            classical_latency_ms = (time.perf_counter() - classical_start) * 1000.0
            put_branch_label(classical_frame, "Classical (OpenCV)", RED)
            put_metrics(classical_frame, classical_count, classical_latency_ms, RED)

            yolo_start = time.perf_counter()
            ml_frame, ml_count = draw_yolo_detections(
                resized.copy(),
                model,
                conf=args.conf,
                device=args.device,
                width=args.target_width,
                height=args.target_height,
            )
            yolo_latency_ms = (time.perf_counter() - yolo_start) * 1000.0
            put_branch_label(ml_frame, f"ML (YOLO {args.device})", GREEN)
            put_metrics(ml_frame, ml_count, yolo_latency_ms, GREEN)

            stacked = np.hstack((classical_frame, ml_frame))

            frame_latency_ms = (time.perf_counter() - frame_start) * 1000.0
            inst_fps = 1000.0 / frame_latency_ms if frame_latency_ms > 0 else 0.0
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)
            put_fps_overlay(stacked, ema_fps)

            out.write(stacked)

            if not args.headless:
                cv2.imshow(window_name, stacked)

            processed += 1
            if not args.headless:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User requested stop (q pressed).")
                    break

            if processed % 50 == 0:
                print(f"[INFO] Processed {processed} frames...")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Saved side-by-side output to: {args.output}")
    print(f"[INFO] Total frames processed: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
