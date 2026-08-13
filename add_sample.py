import os
import sys
import json
import argparse
import cv2
import numpy as np
import config
from src.detector import filter_by_color, extract_primary_contour, get_contour_metrics


def main():
    parser = argparse.ArgumentParser(description="Extract object metrics from an image and add to calibration data.")
    parser.add_argument("image", type=str, help="Path to the calibration image file")
    parser.add_argument("x", type=float, help="Real X offset in mm (positive = Right, negative = Left)")
    parser.add_argument("y", type=float, help="Real Y offset in mm (positive = Down, negative = Up)")
    parser.add_argument("z", type=float, help="Real Z distance from camera lens in mm")
    parser.add_argument("--json", type=str, default="calibration_data.json", help="Path to target JSON dataset file")
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        sys.exit(1)

    # 1. Load image
    raw_frame = cv2.imread(args.image)
    if raw_frame is None:
        print(f"[ERROR] Could not read image file: {args.image}")
        sys.exit(1)

    # Resize to match standard configuration frame dimensions if defined
    frame_width = getattr(config, 'FRAME_WIDTH', raw_frame.shape[1])
    frame_height = getattr(config, 'FRAME_HEIGHT', raw_frame.shape[0])
    frame = cv2.resize(raw_frame, (frame_width, frame_height))

    # 2. Retrieve HSV thresholds from config
    lower_hsv = getattr(config, 'LOWER_HSV', np.array([17, 123, 40]))
    upper_hsv = getattr(config, 'UPPER_HSV', np.array([55, 255, 255]))

    # 3. Perform object detection
    binary_mask = filter_by_color(frame, lower_hsv, upper_hsv, use_morphology=True)
    contour = extract_primary_contour(binary_mask, min_area=400.0)

    if contour is None:
        print("[ERROR] No valid object contour detected using current HSV configuration in config.py!")
        sys.exit(1)

    metrics = get_contour_metrics(contour)
    if metrics is None:
        print("[ERROR] Failed to calculate contour metrics.")
        sys.exit(1)

    cx, cy = metrics["center"]
    area = metrics["area_px"]

    print(f"[DETECTION SUCCESS]")
    print(f"  -> Pixel Center: X={cx}, Y={cy}")
    print(f"  -> Pixel Area:   {area} px^2")

    # 4. Load or initialize JSON database
    data = []
    if os.path.exists(args.json):
        try:
            with open(args.json, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("[WARNING] Existing calibration JSON file was malformed. Initializing new list.")
            data = []

    # 5. Determine auto-incremented ID
    new_id = 1 if not data else max(item.get("id", 0) for item in data) + 1

    # 6. Append new data entry matching requested schema
    new_sample = {
        "id": new_id,
        "cx_px": float(cx),
        "cy_px": float(cy),
        "area_px": float(area),
        "real_x_mm": float(args.x),
        "real_y_mm": float(args.y),
        "real_z_mm": float(args.z)
    }

    data.append(new_sample)

    # 7. Save back to json file
    with open(args.json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[SAVED] Successfully added Sample ID {new_id} to '{args.json}'")
    print(f"  -> Real Target: X={args.x} mm, Y={args.y} mm, Z={args.z} mm")


if __name__ == "__main__":
    main()