import cv2
import numpy as np
import config
from src.detector import filter_by_color, extract_primary_contour, get_contour_metrics
from src.calibration import CalibrationManager
from src.distance_estimator import SpatialEstimator
from src.kalman_filter import SpatialKalmanFilter


def empty_callback(x):
    pass


def setup_trackbars(window_name: str):
    """Creates a dedicated window with 6 HSV tuning trackbars."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 280)

    cv2.createTrackbar("H Min", window_name, int(config.LOWER_HSV[0]), 179, empty_callback)
    cv2.createTrackbar("H Max", window_name, int(config.UPPER_HSV[0]), 179, empty_callback)
    cv2.createTrackbar("S Min", window_name, int(config.LOWER_HSV[1]), 255, empty_callback)
    cv2.createTrackbar("S Max", window_name, int(config.UPPER_HSV[1]), 255, empty_callback)
    cv2.createTrackbar("V Min", window_name, int(config.LOWER_HSV[2]), 255, empty_callback)
    cv2.createTrackbar("V Max", window_name, int(config.UPPER_HSV[2]), 255, empty_callback)


def get_hsv_trackbar_values(window_name: str) -> tuple:
    """Reads current positions from the slider bars."""
    h_min = cv2.getTrackbarPos("H Min", window_name)
    h_max = cv2.getTrackbarPos("H Max", window_name)
    s_min = cv2.getTrackbarPos("S Min", window_name)
    s_max = cv2.getTrackbarPos("S Max", window_name)
    v_min = cv2.getTrackbarPos("V Min", window_name)
    v_max = cv2.getTrackbarPos("V Max", window_name)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    return lower, upper


def draw_axis_legend(frame: np.ndarray):
    """Draws a visual 3D coordinate system reference overlay on top-left of the image."""
    origin = (50, 60)
    
    # +X Axis (Right - Red)
    cv2.arrowedLine(frame, origin, (origin[0] + 50, origin[1]), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(frame, "+X (Right)", (origin[0] + 55, origin[1] + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # +Y Axis (Down - Green)
    cv2.arrowedLine(frame, origin, (origin[0], origin[1] + 50), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(frame, "+Y (Down)", (origin[0] - 15, origin[1] + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    # +Z Axis (Depth - Cyan)
    cv2.circle(frame, origin, 8, (255, 255, 0), 2)
    cv2.circle(frame, origin, 2, (255, 255, 0), -1)
    cv2.putText(frame, "+Z (Depth)", (origin[0] + 15, origin[1] - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)


def prompt_user_coordinates() -> tuple:
    """Prompts user in terminal for physical coordinates X, Y, Z."""
    print("\n------------------- RECORD CALIBRATION SAMPLE -------------------")
    print("Camera Coordinate Reference:")
    print("  X = Horizontal offset [positive = Right, negative = Left]")
    print("  Y = Vertical offset   [positive = Down,  negative = Up]")
    print("  Z = Distance to camera lens[positive = Depth forward]")
    print("-----------------------------------------------------------------")
    
    try:
        x = float(input("Enter measured Real X offset: "))
        y = float(input("Enter measured Real Y offset: "))
        z = float(input("Enter measured Real Z distance: "))
        return x, y, z
    except ValueError:
        print("[ERROR] Invalid numeric input! Sample discarded.")
        return None


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    # 1. Initialize UI Sliders Window
    tuner_window = "HSV Tuner Controls"
    setup_trackbars(tuner_window)

    # 2. Initialize Backend Managers & Filters
    calib_mgr = CalibrationManager("calibration_data.json")
    estimator = SpatialEstimator(calib_mgr, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    kalman = SpatialKalmanFilter(dt=0.033)

    print("\n[READY] Video feed started.")
    print("Controls:")
    print("  • Adjust sliders in 'HSV Tuner Controls' to pick object color.")
    print("  • [C] Capture sample point for calibration.")
    print("  • [L] Reload calibration dataset from file.")
    print("  • [E] Evaluate MAE accuracy metrics in terminal.")
    print("  • [S] Save current HSV sliders to config.py.")
    print("  • [Q] Quit application.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Read live slider positions from UI
        lower_hsv, upper_hsv = get_hsv_trackbar_values(tuner_window)

        # Apply smooth non-squarish color filtering (uses MORPH_ELLIPSE)
        binary_mask = filter_by_color(frame, lower_hsv, upper_hsv, use_morphology=True)
        
        # Extract smooth raw contour (uses CHAIN_APPROX_NONE)
        contour = extract_primary_contour(binary_mask, min_area=400.0)

        current_metrics = None
        raw_pos_3d = None

        if contour is not None:
            metrics = get_contour_metrics(contour)
            if metrics:
                current_metrics = metrics
                cx, cy = metrics["center"]
                area = metrics["area_px"]

                # Draw smooth contour outline & centroid
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Compute raw 3D spatial estimate
                raw_pos_3d = estimator.predict(cx, cy, area)

                # Overlay Pixel Coordinates and Area
                cv2.putText(frame, f"Pixel: ({cx}, {cy}) | Area: {int(area)} px", (cx + 10, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Pass raw estimate through Kalman Filter for velocity-based trajectory smoothing
        smoothed_3d = kalman.update(raw_pos_3d)

        # Overlay 3D Position Metrics
        if contour is not None and current_metrics is not None:
            cx, cy = current_metrics["center"]
            if smoothed_3d:
                # Filtered 3D position
                filtered_text = f"Filtered: X={smoothed_3d['X']:.0f} Y={smoothed_3d['Y']:.0f} Z={smoothed_3d['Z']:.0f}"
                cv2.putText(frame, filtered_text, (cx + 10, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Raw 3D position (subtle)
                if raw_pos_3d:
                    raw_text = f"Raw: X={raw_pos_3d['X']:.0f} Y={raw_pos_3d['Y']:.0f} Z={raw_pos_3d['Z']:.0f}"
                    cv2.putText(frame, raw_text, (cx + 10, cy + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "3D Pos: Need Calib [C] >= 2 points", (cx + 10, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw 3D Axis Legend HUD
        draw_axis_legend(frame)

        # Status & Control Hotkey HUD Overlays
        mode_str = "NON-LINEAR" if estimator.use_distortion_terms else ("LINEAR" if estimator.is_fitted else "UNFITTED")
        cv2.putText(frame, f"Calib Status: {mode_str} ({len(calib_mgr.samples)} pts)", 
                    (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "[C] Sample | [L] Load | [E] MAE Eval | [S] Save HSV | [Q] Quit", 
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Render Active Video Displays
        cv2.imshow("Object Tracker", frame)
        cv2.imshow("Threshold Mask", binary_mask)

        key = cv2.waitKey(1) & 0xFF

        # --- HOTKEY ACTIONS ---

        # CAPTURE CALIBRATION POINT
        if key == ord('c'):
            if current_metrics is None:
                print("\n[WARNING] No valid object contour detected to sample! Position object and try again.")
            else:
                # Freeze frame overlay
                freeze_frame = frame.copy()
                cv2.putText(freeze_frame, "=== FRAME FROZEN: ENTER COORDINATES IN TERMINAL ===", 
                            (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Object Tracker", freeze_frame)
                cv2.waitKey(1)

                coords = prompt_user_coordinates()
                if coords:
                    rx, ry, rz = coords
                    cx, cy = current_metrics["center"]
                    area = current_metrics["area_px"]
                    
                    calib_mgr.add_sample(cx, cy, area, rx, ry, rz)
                    calib_mgr.save_data()
                    estimator.fit()   # Re-fit multi-variable regression model
                    kalman.reset()    # Reset filter state

        # RELOAD CALIBRATION DATA
        elif key == ord('l'):
            calib_mgr.load_data()
            calib_mgr.print_summary()
            estimator.fit()
            kalman.reset()

        # EVALUATE MAE / RMSE ACCURACY
        elif key == ord('e'):
            try:
                from evaluate import evaluate_calibration
                evaluate_calibration("calibration_data.json")
            except ImportError:
                print("[ERROR] 'evaluate.py' module not found in root directory!")

        # SAVE HSV CONFIG TO FILE
        elif key == ord('s'):
            config.save_config(lower_hsv, upper_hsv)
            print(f"[CONFIG SAVED] Lower: {lower_hsv}, Upper: {upper_hsv}")

        # QUIT APPLICATION
        elif key == ord('q'):
            config.save_config(lower_hsv, upper_hsv)
            print(f"[AUTO-SAVED ON EXIT] Lower: {lower_hsv}, Upper: {upper_hsv}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()