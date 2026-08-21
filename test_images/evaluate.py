import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT, LOWER_HSV, UPPER_HSV
from objectDetector import filter_by_color, extract_primary_contour, get_contour_metrics
from dataPointsManager import DataPointsManager
from src.distance_estimator import SpatialEstimator

"""
Script for evaluating the fitted function in given dataset.
Runs from the command line.

"""

def evaluate_model(
    test_images_dir: str = "test_images",
    test_json_path: str = "test_images/real_coordinates.json",
    calibration_path: str = "calibration_data.json"
) -> None:
    if not os.path.exists(test_json_path):
        raise FileNotFoundError(f"Could not find test coordinates JSON at '{test_json_path}'")

    if not os.path.exists(test_images_dir):
        raise FileNotFoundError(f"Could not find test images directory at: {test_images_dir}")

    # 1. Initialize Calibration Manager and fit SpatialEstimator with config
    calib_mgr = DataPointsManager(filepath=calibration_path)
    estimator = SpatialEstimator(calibration_mgr=calib_mgr, frame_size=(FRAME_WIDTH, FRAME_HEIGHT))

    # 2. Load test ground truth coordinates
    with open(test_json_path, "r") as f:
        test_coords_list = json.load(f)

    print(f"\n================ EVALUATION REPORT ================")
    print(f"Directory: '{test_images_dir}' | Total Test Samples: {len(test_coords_list)}\n")

    errors_x = []
    errors_y = []
    errors_z = []
    euclidean_errors = []
    successful_evals = 0

    print(f"{'ID':<4} | {'Real (X, Y, Z)':<22} | {'Predicted (X, Y, Z)':<22} | {'Error (X, Y, Z)':<22} | {'Dist Err':<8}")
    print("-" * 96)

    for item in test_coords_list:
        sample_id = item["id"]
        img_filename = f"Untitled{sample_id}.png"
        img_path = os.path.join(test_images_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"#{sample_id:<3} | [WARNING] Image not found: {img_path}")
            continue

        image = cv2.imread(img_path)

        if image.shape[1] != FRAME_WIDTH or image.shape[0] != FRAME_HEIGHT:
            image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
            print("Image is resized")

        # 3. Run detection pipeline (compatible with src/detector.py signature)
        binary_mask = filter_by_color(image, LOWER_HSV, UPPER_HSV)
        contour = extract_primary_contour(binary_mask)
        if contour is None:
            print(f"#{sample_id:<3} | [WARNING] No valid contour detected.")
            continue
        metrics = get_contour_metrics(contour)

        cx, cy = metrics["center"]
        area_px = metrics["area_px"]

        # 4. Predict coordinates using SpatialEstimator.predict()
        pred = estimator.predict(float(cx), float(cy), float(area_px))
        if pred is None:
            print(f"#{sample_id:<3} | [WARNING] Prediction failed. Skipping test item.")
            continue

        real_x = float(item["real_x"])
        real_y = float(item["real_y"])
        real_z = float(item["real_z"])

        pred_x, pred_y, pred_z = pred["X"], pred["Y"], pred["Z"]

        # Calculate error metrics
        err_x = abs(pred_x - real_x)
        err_y = abs(pred_y - real_y)
        err_z = abs(pred_z - real_z)
        dist_err = np.sqrt((pred_x - real_x)**2 + (pred_y - real_y)**2 + (pred_z - real_z)**2)

        errors_x.append(err_x)
        errors_y.append(err_y)
        errors_z.append(err_z)
        euclidean_errors.append(dist_err)
        successful_evals += 1

        real_str = f"({real_x:.2f}, {real_y:.2f}, {real_z:.2f})"
        pred_str = f"({pred_x:.2f}, {pred_y:.2f}, {pred_z:.2f})"
        err_str = f"({err_x:.2f}, {err_y:.2f}, {err_z:.2f})"

        print(f"#{sample_id:<3} | {real_str:<22} | {pred_str:<22} | {err_str:<22} | {dist_err:.2f}m")

    print("-" * 30)

    # 5. Generate Performance Summary
    if successful_evals > 0:
        mae_x = np.mean(errors_x)
        mae_y = np.mean(errors_y)
        mae_z = np.mean(errors_z)
        rmse_x = np.sqrt(np.mean(np.array(errors_x)**2))
        rmse_y = np.sqrt(np.mean(np.array(errors_y)**2))
        rmse_z = np.sqrt(np.mean(np.array(errors_z)**2))
        mean_dist_err = np.mean(euclidean_errors)

        print(f"\n### PERFORMANCE SUMMARY ({successful_evals}/{len(test_coords_list)} Samples Evaluated)")
        print(f"* **Mean Absolute Error (MAE):** X = {mae_x:.4f}m | Y = {mae_y:.4f}m | Z = {mae_z:.4f}m")
        print(f"* **Root Mean Squared Error (RMSE):** X = {rmse_x:.4f}m | Y = {rmse_y:.4f}m | Z = {rmse_z:.4f}m")
        print(f"* **Average 3D Euclidean Distance Error:** {mean_dist_err:.4f} meters")
    else:
        print("\n[ERROR] No test samples were successfully evaluated.")
    
    print("===================================================\n")


if __name__ == "__main__":
    evaluate_model()