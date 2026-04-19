#!/usr/bin/env python3
"""Classical computer-vision baseline for license plate detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def detect_license_plate_classical(
    frame: np.ndarray,
    box_color: tuple[int, int, int] = (0, 255, 0),
    box_thickness: int = 2,
    return_count: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Detect license-plate candidates in one image/frame using classical CV.

    Pipeline:
    1) grayscale
    2) bilateral filter
    3) Canny edges
    4) contour extraction
    5) polygon approximation and 4-point filtering
    6) aspect-ratio filtering in [2.0, 5.0]
    7) draw rectangles on candidates
    """
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")

    output = frame.copy()

    # 1) Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Noise reduction while preserving edges
    denoised = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    # 3) Edge detection
    edges = cv2.Canny(denoised, threshold1=80, threshold2=200)

    # 4) Contour extraction
    contours_info = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    detection_count = 0

    for contour in contours:
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            continue

        # 5) Geometric filtering: keep closed quadrilateral-like contours
        approx = cv2.approxPolyDP(contour, epsilon=0.02 * perimeter, closed=True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        # 6) Aspect-ratio filtering for likely license plate boxes
        x, y, w, h = cv2.boundingRect(approx)
        if h <= 0:
            continue

        aspect_ratio = w / float(h)
        if 2.0 <= aspect_ratio <= 5.0:
            # 7) Visualization
            cv2.rectangle(output, (x, y), (x + w, y + h), box_color, box_thickness)
            detection_count += 1

    if return_count:
        return output, detection_count

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical plate detector on one image.")
    parser.add_argument("--input", type=Path, required=True, help="Input image path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/classical_detection.jpg"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    frame = cv2.imread(str(args.input))
    if frame is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    detected = detect_license_plate_classical(frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output), detected)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.output}")

    print(f"[INFO] Saved result to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
