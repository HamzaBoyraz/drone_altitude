import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT, LOWER_HSV, UPPER_HSV
from objectDetector import filter_by_color, extract_primary_contour, get_contour_metrics

"""
Writes into the json file that the data for calibration held. Runs from the command line.
"""

image_name = "Untitled"

def generate_calibration_dataset(
    images_dir: str = "training_images",
    json_path: str = "training_images/real_coordinates.json",
    output_path: str = "calibration_data.json"
) -> None:
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find real coordinates file at: {json_path}")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Could not find training images directory at: {images_dir}")

    # Load real world coordinates
    with open(json_path, "r") as f:
        real_coords_list = json.load(f)

    calibration_samples = []

    for item in real_coords_list:
        sample_id = item["id"]
        img_filename = f"{image_name}{sample_id}.png"
        img_path = os.path.join(images_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"  [WARNING] Image not found: {img_path}. Skipping.")
            continue

        image = cv2.imread(img_path)

        # Resize image if it doesn't match config dimensions
        if image.shape[1] != FRAME_WIDTH or image.shape[0] != FRAME_HEIGHT:
            image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))

        # 1. Apply color filter using config HSV thresholds
        binary_mask = filter_by_color(image, LOWER_HSV, UPPER_HSV)
        contour = extract_primary_contour(binary_mask)
        if contour is None:
            print(f"  [WARNING] No valid contour found in {img_filename}. Skipping.")
            continue
        metrics = get_contour_metrics(contour)

        cx, cy = metrics["center"]
        area_px = metrics["area_px"]

        # Build formatted sample entry
        sample = {
            "id": sample_id,
            "cx_px": float(cx),
            "cy_px": float(cy),
            "area_px": float(area_px),
            "real_x": float(item["real_x"]),
            "real_y": float(item["real_y"]),
            "real_z": float(item["real_z"])
        }
        calibration_samples.append(sample)
        print(f"  [SUCCESS] Sample #{sample_id} -> Center: ({cx}, {cy}), Area: {area_px:.0f} px")

    # Save compiled dataset to output JSON file
    with open(output_path, "w") as f:
        json.dump(calibration_samples, f, indent=4)

    print(f"\n[COMPLETE] Successfully generated '{output_path}' with {len(calibration_samples)} valid samples.\n")


if __name__ == "__main__":
    generate_calibration_dataset()