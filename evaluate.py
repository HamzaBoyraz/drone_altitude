import numpy as np
from src.calibration import CalibrationManager
from src.distance_estimator import SpatialEstimator


def evaluate_calibration(calibration_file: str = "calibration_data.json"):
    """
    Loads saved calibration data, fits the SpatialEstimator model,
    and computes prediction errors (MAE & RMSE) across all calibration points.
    """
    calib_mgr = CalibrationManager(calibration_file)
    if len(calib_mgr.samples) < 2:
        print("\n[ERROR] At least 2 calibration samples are required to evaluate the model.")
        return

    estimator = SpatialEstimator(calib_mgr)
    if not estimator.is_fitted:
        print("\n[ERROR] Failed to fit SpatialEstimator model.")
        return

    errors_x, errors_y, errors_z = [], [], []

    print("\n======================= CALIBRATION ACCURACY EVALUATION =======================")
    print(f"Mode: {'NON-LINEAR (Distortion Compensated)' if estimator.use_distortion_terms else 'LINEAR'}")
    print(f"Total Calibration Points: {len(calib_mgr.samples)}\n")
    print(f"{'ID':<4} | {'Target (X, Y, Z) mm':<24} | {'Predicted (X, Y, Z) mm':<24} | {'Absolute Error (mm)'}")
    print("-" * 78)

    for sample in calib_mgr.samples:
        cx, cy, area = sample["cx_px"], sample["cy_px"], sample["area_px"]
        real_x, real_y, real_z = sample["real_x_mm"], sample["real_y_mm"], sample["real_z_mm"]

        pred = estimator.predict(cx, cy, area)
        if pred is None:
            continue

        err_x = abs(pred["X_mm"] - real_x)
        err_y = abs(pred["Y_mm"] - real_y)
        err_z = abs(pred["Z_mm"] - real_z)

        errors_x.append(err_x)
        errors_y.append(err_y)
        errors_z.append(err_z)

        target_str = f"({real_x:.1f}, {real_y:.1f}, {real_z:.1f})"
        pred_str = f"({pred['X_mm']:.1f}, {pred['Y_mm']:.1f}, {pred['Z_mm']:.1f})"
        err_str = f"({err_x:.1f}, {err_y:.1f}, {err_z:.1f})"

        print(f"{sample['id']:<4} | {target_str:<24} | {pred_str:<24} | {err_str}")

    # Compute Statistical Metrics
    mae_x, mae_y, mae_z = np.mean(errors_x), np.mean(errors_y), np.mean(errors_z)
    rmse_x = np.sqrt(np.mean(np.array(errors_x) ** 2))
    rmse_y = np.sqrt(np.mean(np.array(errors_y) ** 2))
    rmse_z = np.sqrt(np.mean(np.array(errors_z) ** 2))

    print("-" * 78)
    print("SUMMARY ERROR METRICS:")
    print(f"  • Mean Absolute Error (MAE):  X = {mae_x:.2f} mm | Y = {mae_y:.2f} mm | Z = {mae_z:.2f} mm")
    print(f"  • Root Mean Sq Error (RMSE): X = {rmse_x:.2f} mm | Y = {rmse_y:.2f} mm | Z = {rmse_z:.2f} mm")
    print("===============================================================================\n")


if __name__ == "__main__":
    evaluate_calibration()