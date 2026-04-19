#!/usr/bin/env python3
"""Compare classical CV against multiple YOLO models on one video."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from classical_detector import detect_license_plate_classical


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CLASSICAL_COLOR = (0, 0, 255)
MODEL_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),
    (255, 200, 0),
    (255, 128, 0),
    (0, 255, 255),
    (255, 0, 255),
    (203, 192, 255),
)

TARGET_WIDTH = 640
TARGET_HEIGHT = 360


@dataclass
class BranchStats:
    frames: int = 0
    total_detections: int = 0
    total_latency_ms: float = 0.0

    def update(self, detections: int, latency_ms: float) -> None:
        self.frames += 1
        self.total_detections += detections
        self.total_latency_ms += latency_ms

    def avg_detections(self) -> float:
        if self.frames == 0:
            return 0.0
        return self.total_detections / self.frames

    def avg_latency_ms(self) -> float:
        if self.frames == 0:
            return 0.0
        return self.total_latency_ms / self.frames


@dataclass
class ModelEntry:
    name: str
    path: Path
    model: YOLO
    color: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple YOLO weights against classical CV on one video."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("chongqing_4k_cropped.mov"),
        help="Input video path.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("scripts/models"),
        help="Directory that contains YOLO .pt weights.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=None,
        help=(
            "Optional subset of model weights (filenames or paths). "
            "If omitted, every .pt file in --models-dir is used."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multi_model_comparison_output.mp4"),
        help="Output comparison video path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Maximum number of frames to process. "
            "If omitted, the script processes the full video length (or until EOF when "
            "frame count metadata is unavailable)."
        ),
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=TARGET_WIDTH,
        help="Width of each comparison panel.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=TARGET_HEIGHT,
        help="Height of each comparison panel.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=0,
        help="Number of columns in output grid (0 = auto).",
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
        help="Inference device for YOLO models.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable live cv2.imshow preview.",
    )
    return parser.parse_args()


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


def resolve_weight_paths(models_dir: Path, requested_weights: list[str] | None) -> list[Path]:
    if not models_dir.exists() or not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    if requested_weights is None:
        weights = sorted(models_dir.glob("*.pt"))
        if not weights:
            raise FileNotFoundError(f"No .pt files found in: {models_dir}")
        return weights

    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_weight in requested_weights:
        raw_path = Path(raw_weight)
        candidates = (raw_path, models_dir / raw_weight)
        found = next((candidate for candidate in candidates if candidate.exists()), None)

        if found is None:
            raise FileNotFoundError(
                f"Weight not found: {raw_weight} (checked as path and inside {models_dir})"
            )

        if found.suffix.lower() != ".pt":
            raise ValueError(f"Expected a .pt model file, got: {found}")

        canonical = found.resolve()
        if canonical in seen:
            continue

        seen.add(canonical)
        resolved.append(found)

    if not resolved:
        raise FileNotFoundError("No valid model weights selected.")

    return resolved


def build_grid_shape(panel_count: int, requested_cols: int) -> tuple[int, int]:
    if panel_count <= 0:
        raise ValueError("Panel count must be > 0")

    if requested_cols > 0:
        cols = requested_cols
    else:
        cols = max(2, math.ceil(math.sqrt(panel_count)))

    rows = math.ceil(panel_count / cols)
    return rows, cols


def put_panel_label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    pad_x = 10
    pad_y = 8
    x1 = 10
    y1 = 10
    x2 = min(frame.shape[1] - 1, x1 + text_w + (2 * pad_x))
    y2 = min(frame.shape[0] - 1, y1 + text_h + baseline + (2 * pad_y))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
    text_x = x1 + pad_x
    text_y = y2 - pad_y - baseline
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, WHITE, thickness, cv2.LINE_AA)


def put_metrics(
    frame: np.ndarray,
    detections: int,
    latency_ms: float,
    text_color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        f"Detections: {detections}",
        (14, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Latency: {latency_ms:.1f} ms",
        (14, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        text_color,
        2,
        cv2.LINE_AA,
    )


def put_runtime_overlay(canvas: np.ndarray, fps_value: float, frame_idx: int) -> None:
    text = f"FPS: {fps_value:.2f} | Frame: {frame_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    pad_x = 12
    pad_y = 8
    box_w = text_w + (2 * pad_x)
    box_h = text_h + baseline + (2 * pad_y)

    x1 = max(0, canvas.shape[1] - box_w - 14)
    y1 = 14
    x2 = min(canvas.shape[1] - 1, x1 + box_w)
    y2 = min(canvas.shape[0] - 1, y1 + box_h)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), BLACK, thickness=-1)

    text_x = x1 + pad_x
    text_y = y2 - pad_y - baseline
    cv2.putText(
        canvas,
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
    box_color: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    output = frame.copy()
    detection_count = 0
    results = model.predict(source=output, device=device, conf=conf, verbose=False)

    if not results:
        return output, detection_count

    boxes = results[0].boxes
    if boxes is None:
        return output, detection_count

    height, width = output.shape[:2]
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

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
        detection_count += 1

    return output, detection_count


def compose_grid(
    panels: list[np.ndarray],
    rows: int,
    cols: int,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    canvas = np.zeros((rows * panel_height, cols * panel_width, 3), dtype=np.uint8)

    for idx, panel in enumerate(panels):
        row_idx = idx // cols
        col_idx = idx % cols
        y1 = row_idx * panel_height
        y2 = y1 + panel_height
        x1 = col_idx * panel_width
        x2 = x1 + panel_width
        canvas[y1:y2, x1:x2] = panel

    return canvas


def print_summary(stats: dict[str, BranchStats], processed_frames: int, output_path: Path) -> None:
    print(f"[INFO] Saved comparison output to: {output_path}")
    print(f"[INFO] Total frames processed: {processed_frames}")
    print("[SUMMARY] Branch performance averages:")

    for branch_name, branch_stats in stats.items():
        print(
            "[SUMMARY] "
            f"{branch_name}: "
            f"avg_detections/frame={branch_stats.avg_detections():.3f}, "
            f"avg_latency={branch_stats.avg_latency_ms():.2f} ms"
        )


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be > 0")

    if args.grid_cols < 0:
        raise ValueError("--grid-cols must be >= 0")

    ensure_device(args.device)

    weight_paths = resolve_weight_paths(args.models_dir, args.weights)
    model_entries: list[ModelEntry] = []

    print(f"[INFO] Loading {len(weight_paths)} model(s) on device={args.device}...")
    for idx, weight_path in enumerate(weight_paths):
        model = YOLO(str(weight_path))
        model.to(args.device)

        model_entry = ModelEntry(
            name=weight_path.stem,
            path=weight_path,
            model=model,
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
        )
        model_entries.append(model_entry)
        print(f"[INFO] Loaded model: {weight_path}")

    panel_count = 1 + len(model_entries)
    rows, cols = build_grid_shape(panel_count, args.grid_cols)
    output_width = cols * args.target_width
    output_height = rows * args.target_height

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input}")

    effective_max_frames: int | None = args.max_frames
    if effective_max_frames is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            effective_max_frames = total_frames
            print(
                "[INFO] --max-frames not set; "
                f"processing full video length ({effective_max_frames} frames)."
            )
        else:
            effective_max_frames = None
            print(
                "[INFO] --max-frames not set and frame count metadata unavailable, "
                "processing until EOF."
            )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

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

    stats: dict[str, BranchStats] = {"Classical (OpenCV)": BranchStats()}
    for model_entry in model_entries:
        stats[f"YOLO ({model_entry.name})"] = BranchStats()

    window_name = "Classical vs Multi-YOLO Comparison"
    processed = 0
    ema_fps: float | None = None

    try:
        while effective_max_frames is None or processed < effective_max_frames:
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

            panels: list[np.ndarray] = []

            classical_start = time.perf_counter()
            classical_result = detect_license_plate_classical(
                resized.copy(),
                box_color=CLASSICAL_COLOR,
                box_thickness=2,
                return_count=True,
            )
            classical_frame, classical_count = classical_result
            classical_latency_ms = (time.perf_counter() - classical_start) * 1000.0
            put_panel_label(classical_frame, "Classical (OpenCV)", CLASSICAL_COLOR)
            put_metrics(classical_frame, classical_count, classical_latency_ms, CLASSICAL_COLOR)
            panels.append(classical_frame)
            stats["Classical (OpenCV)"].update(classical_count, classical_latency_ms)

            for model_entry in model_entries:
                yolo_start = time.perf_counter()
                model_frame, model_count = draw_yolo_detections(
                    resized.copy(),
                    model_entry.model,
                    conf=args.conf,
                    device=args.device,
                    box_color=model_entry.color,
                )
                yolo_latency_ms = (time.perf_counter() - yolo_start) * 1000.0

                label = f"YOLO ({model_entry.name})"
                put_panel_label(model_frame, label, model_entry.color)
                put_metrics(model_frame, model_count, yolo_latency_ms, model_entry.color)
                panels.append(model_frame)
                stats[label].update(model_count, yolo_latency_ms)

            canvas = compose_grid(
                panels,
                rows=rows,
                cols=cols,
                panel_width=args.target_width,
                panel_height=args.target_height,
            )

            frame_latency_ms = (time.perf_counter() - frame_start) * 1000.0
            inst_fps = 1000.0 / frame_latency_ms if frame_latency_ms > 0 else 0.0
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)
            put_runtime_overlay(canvas, ema_fps, processed + 1)

            out.write(canvas)

            if not args.headless:
                cv2.imshow(window_name, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User requested stop (q pressed).")
                    break

            processed += 1
            if processed % 25 == 0:
                print(f"[INFO] Processed {processed} frames...")

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print_summary(stats, processed, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())