import os
import sys
import json
import argparse
import cv2
import numpy as np
import config
from objectDetector import filter_by_color, extract_primary_contour, get_contour_metrics

"""
Script for adding data points from command line.
Example usage: python add_sample.py --json data.json      image.png    0 0 0
                                        (json file path) (image path)  x y z
"""

def add_sample():
    # Code for parsing command line string
    parser = argparse.ArgumentParser(description="Extract object metrics from an image and add to calibration data.")
    parser.add_argument("image", type=str, help="Path to the calibration image file")
    parser.add_argument("x", type=float, help="Real X offset (positive = Right, negative = Left)")
    parser.add_argument("y", type=float, help="Real Y offset (positive = Down, negative = Up)")
    parser.add_argument("z", type=float, help="Real Z distance from camera lens")
    parser.add_argument("--json", type=str, default="calibration_data.json", help="Path to target JSON dataset file")
    
    args = parser.parse_args()

    # Check if the given image exists
    if not os.path.exists(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        sys.exit(1)

    # Load image
    image = cv2.imread(args.image)

    # Resize to match standard configuration frame dimensions if defined
    frame_width = getattr(config, 'FRAME_WIDTH', image.shape[1])
    frame_height = getattr(config, 'FRAME_HEIGHT', image.shape[0])
    image = cv2.resize(image, (frame_width, frame_height))

    # 2. Retrieve HSV thresholds from config
    lower_hsv = getattr(config, 'LOWER_HSV', np.array([17, 123, 40]))
    upper_hsv = getattr(config, 'UPPER_HSV', np.array([55, 255, 255]))

    # 3. Perform object detection
    binary_mask = filter_by_color(image, lower_hsv, upper_hsv, use_morphology=True)
    contour = extract_primary_contour(binary_mask, min_area=400.0)
    metrics = get_contour_metrics(contour)

    cx, cy = metrics["center"]
    area = metrics["area_px"]

    print(f"  -> Pixel Center: X={cx}, Y={cy}")
    print(f"  -> Pixel Area:   {area} px^2")

    # 4. Load or initialize JSON database
    data = []
    if os.path.exists(args.json):
        try:
            with open(args.json, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("[WARNING] Existing calibration JSON file was malformed.")
            sys.exit(1)

    # 5. Determine auto-incremented ID
    new_id = 1 if not data else max(item.get("id", 0) for item in data) + 1

    # 6. Append new data entry matching requested schema
    new_sample = {
        "id": new_id,
        "cx_px": float(cx),
        "cy_px": float(cy),
        "area_px": float(area),
        "real_x": float(args.x),
        "real_y": float(args.y),
        "real_z": float(args.z)
    }

    data.append(new_sample)

    # 7. Save back to json file
    with open(args.json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"[SAVED] Successfully added Sample {new_id} to '{args.json}'")


if __name__ == "__main__":
    add_sample()