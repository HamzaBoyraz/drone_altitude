import cv2
import numpy as np
import tkinter as tk
import config
from src.detector import filter_by_color, extract_primary_contour, get_contour_metrics
from src.calibration import CalibrationManager
from src.distance_estimator import SpatialEstimator
from src.kalman_filter import SpatialKalmanFilter


# ==================== USER CONFIGURABLE PARAMETERS ====================
VIDEO_PATH = "videos/sari_kare.mp4"

# 1. Main Window Dimensions (Tkinter UI)
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540

# 2. Displayed Video Resolution (Right Panel sizing)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# 3. Captured & Processed Video Resolution (Lower = Faster performance)
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 480

# 4. Playback Speed Multiplier (1.0 = Normal FPS, < 1.0 = Faster, > 1.0 = Slower)
SPEED_MULTIPLIER = 1
# ======================================================================


def save_config(lower_hsv: np.ndarray, upper_hsv: np.ndarray, file_path: str="config.py"):
    """Overwrites config.py with the newly selected lower and upper HSV array values"""
    content = f"""import numpy as np
FRAME_WIDTH = {config.FRAME_WIDTH}
FRAME_HEIGHT = {config.FRAME_HEIGHT}

LOWER_HSV = np.array([{lower_hsv[0]}, {lower_hsv[1]}, {lower_hsv[2]}])
UPPER_HSV = np.array([{upper_hsv[0]}, {upper_hsv[1]}, {upper_hsv[2]}])
    """
    with open(file_path, "w") as f:
        f.write(content)


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
    print("  Y = Vertical offset [positive = Down,  negative = Up]")
    print("  Z = Distance to camera lens [positive = Depth forward]")
    print("-----------------------------------------------------------------")
    
    try:
        x = float(input("Enter measured Real X offset: "))
        y = float(input("Enter measured Real Y offset: "))
        z = float(input("Enter measured Real Z distance: "))
        return x, y, z
    except ValueError:
        print("[ERROR] Invalid numeric input! Sample discarded.")
        return None


class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Tracker & Calibration Suite")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.cap = cv2.VideoCapture(VIDEO_PATH)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        base_delay = int(1000 / fps) if fps > 0 else 30
        self.delay = max(1, int(base_delay * SPEED_MULTIPLIER))

        # Backend Managers & Filters
        self.calib_mgr = CalibrationManager("calibration_data.json")
        self.estimator = SpatialEstimator(self.calib_mgr, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        self.kalman = SpatialKalmanFilter(dt=0.033)

        self.current_metrics = None
        self.photo = None

        # --- SCROLLABLE LEFT PANEL SETUP ---
        self.left_container = tk.Frame(root, width=380, bg="#2b2b2b")
        self.left_container.pack(side=tk.LEFT, fill=tk.Y)
        self.left_container.pack_propagate(False)

        self.canvas = tk.Canvas(self.left_container, bg="#2b2b2b", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.left_container, orient="vertical", command=self.canvas.yview)
        
        self.left_frame = tk.Frame(self.canvas, bg="#2b2b2b")
        self.left_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.left_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda event: self.canvas.itemconfig(self.canvas_window, width=event.width))

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.create_widgets()
        self.update_video()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def create_widgets(self):
        tk.Label(self.left_frame, text="HSV Tuner & Controls", fg="white", bg="#2b2b2b", font=("Arial", 14, "bold")).pack(pady=15)

        # --- COLOR VISUALIZATION SWATCHES ---
        preview_frame = tk.Frame(self.left_frame, bg="#2b2b2b")
        preview_frame.pack(fill=tk.X, padx=20, pady=5)

        tk.Label(preview_frame, text="Min HSV Color:", fg="white", bg="#2b2b2b", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        self.min_color_canvas = tk.Canvas(preview_frame, width=50, height=22, bg="black", highlightthickness=1, highlightbackground="white")
        self.min_color_canvas.pack(side=tk.LEFT, padx=5)

        tk.Label(preview_frame, text="Max HSV Color:", fg="white", bg="#2b2b2b", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        self.max_color_canvas = tk.Canvas(preview_frame, width=50, height=22, bg="black", highlightthickness=1, highlightbackground="white")
        self.max_color_canvas.pack(side=tk.LEFT, padx=5)

        # --- SLIDERS ---
        self.sliders = {}
        params = [
            ("H Min", 0, 179, int(config.LOWER_HSV[0])),
            ("H Max", 0, 179, int(config.UPPER_HSV[0])),
            ("S Min", 0, 255, int(config.LOWER_HSV[1])),
            ("S Max", 0, 255, int(config.UPPER_HSV[1])),
            ("V Min", 0, 255, int(config.LOWER_HSV[2])),
            ("V Max", 0, 255, int(config.UPPER_HSV[2])),
        ]

        for name, mn, mx, val in params:
            frame = tk.Frame(self.left_frame, bg="#2b2b2b")
            frame.pack(fill=tk.X, padx=20, pady=4)
            tk.Label(frame, text=name, fg="white", bg="#2b2b2b", font=("Arial", 10)).pack(anchor="w")
            scale = tk.Scale(frame, from_=mn, to=mx, orient=tk.HORIZONTAL, bg="#3c3f41", fg="white", highlightbackground="#2b2b2b")
            scale.set(val)
            scale.pack(fill=tk.X)
            self.sliders[name] = scale

        # --- BUTTONS ---
        btn_frame = tk.Frame(self.left_frame, bg="#2b2b2b")
        btn_frame.pack(fill=tk.X, padx=20, pady=15)

        tk.Button(btn_frame, text="[C] Capture Sample", bg="#2e8b57", fg="white", font=("Arial", 10, "bold"), command=self.capture_sample).pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text="[L] Reload Dataset", bg="#4682b4", fg="white", font=("Arial", 10, "bold"), command=self.reload_data).pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text="[E] Evaluate MAE Accuracy", bg="#708090", fg="white", font=("Arial", 10, "bold"), command=self.evaluate_data).pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text="[S] Save HSV to config.py", bg="#cd5c5c", fg="white", font=("Arial", 10, "bold"), command=self.save_hsv_config).pack(fill=tk.X, pady=4)
        tk.Button(btn_frame, text="[Q] Quit Application", bg="#b22222", fg="white", font=("Arial", 10, "bold"), command=self.quit_app).pack(fill=tk.X, pady=4)

    def get_hsv_values(self):
        lower = np.array([self.sliders["H Min"].get(), self.sliders["S Min"].get(), self.sliders["V Min"].get()])
        upper = np.array([self.sliders["H Max"].get(), self.sliders["S Max"].get(), self.sliders["V Max"].get()])
        return lower, upper

    def update_color_visualizations(self, lower, upper):
        min_hsv_arr = np.uint8([[lower]])
        max_hsv_arr = np.uint8([[upper]])
        
        min_rgb = cv2.cvtColor(min_hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        max_rgb = cv2.cvtColor(max_hsv_arr, cv2.COLOR_HSV2RGB)[0][0]

        min_hex = f"#{min_rgb[0]:02x}{min_rgb[1]:02x}{min_rgb[2]:02x}"
        max_hex = f"#{max_rgb[0]:02x}{max_rgb[1]:02x}{max_rgb[2]:02x}"

        self.min_color_canvas.config(bg=min_hex)
        self.max_color_canvas.config(bg=max_hex)

    def capture_sample(self):
        if self.current_metrics is not None:
            coords = prompt_user_coordinates()
            if coords:
                rx, ry, rz = coords
                cx, cy = self.current_metrics["center"]
                area = self.current_metrics["area_px"]
                self.calib_mgr.add_sample(cx, cy, area, rx, ry, rz)
                self.calib_mgr.save_data()
                self.estimator.fit()
                self.kalman.reset()
        else:
            print("\n[WARNING] No valid object contour detected to sample! Position object and try again.")

    def reload_data(self):
        self.calib_mgr.load_data()
        self.calib_mgr.print_summary()
        self.estimator.fit()
        self.kalman.reset()

    def evaluate_data(self):
        try:
            from evaluate import evaluate_calibration
            evaluate_calibration("calibration_data.json")
        except ImportError:
            print("[ERROR] 'evaluate.py' module not found in root directory!")

    def save_hsv_config(self):
        lower, upper = self.get_hsv_values()
        save_config(lower, upper)
        print(f"[CONFIG SAVED] Lower: {lower}, Upper: {upper}")

    def quit_app(self):
        lower, upper = self.get_hsv_values()
        save_config(lower, upper)
        print(f"[AUTO-SAVED ON EXIT] Lower: {lower}, Upper: {upper}")
        self.cap.release()
        self.root.destroy()

    def update_video(self):
        ret, raw_frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.kalman.reset()
            self.root.after(self.delay, self.update_video)
            return

        frame = cv2.resize(raw_frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

        lower_hsv, upper_hsv = self.get_hsv_values()
        self.update_color_visualizations(lower_hsv, upper_hsv)

        binary_mask = filter_by_color(frame, lower_hsv, upper_hsv, use_morphology=False)
        contour = extract_primary_contour(binary_mask, min_area=400.0)

        self.current_metrics = None
        raw_pos_3d = None

        scale_x = config.FRAME_WIDTH / PROCESS_WIDTH
        scale_y = config.FRAME_HEIGHT / PROCESS_HEIGHT

        if contour is not None:
            metrics = get_contour_metrics(contour)
            if metrics:
                self.current_metrics = metrics
                cx, cy = metrics["center"]
                area = metrics["area_px"]

                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                scaled_cx = cx * scale_x
                scaled_cy = cy * scale_y
                scaled_area = area * (scale_x * scale_y)

                raw_pos_3d = self.estimator.predict(scaled_cx, scaled_cy, scaled_area)
                
                # Center-relative pixel coordinates calculation (Origin at video center)
                center_x = PROCESS_WIDTH // 2
                center_y = PROCESS_HEIGHT // 2
                rel_x = cx - center_x
                rel_y = cy - center_y

                cv2.putText(frame, f"Pixel: ({rel_x}, {rel_y}) | Area: {int(area)} px", (cx + 10, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        smoothed_3d = self.kalman.update(raw_pos_3d)

        if contour is not None and self.current_metrics is not None:
            cx, cy = self.current_metrics["center"]
            if smoothed_3d:
                filtered_text = f"Filtered: X={smoothed_3d['X']:.0f} Y={smoothed_3d['Y']:.0f} Z={smoothed_3d['Z']:.0f}"
                cv2.putText(frame, filtered_text, (cx + 10, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                if raw_pos_3d:
                    raw_text = f"Raw: X={raw_pos_3d['X']:.0f} Y={raw_pos_3d['Y']:.0f} Z={raw_pos_3d['Z']:.0f}"
                    cv2.putText(frame, raw_text, (cx + 10, cy + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "3D Pos: Need Calib [C] >= 2 points", (cx + 10, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        draw_axis_legend(frame)

        mode_str = "NON-LINEAR" if self.estimator.use_distortion_terms else ("LINEAR" if self.estimator.is_fitted else "UNFITTED")
        cv2.putText(frame, f"Calib Status: {mode_str} ({len(self.calib_mgr.samples)} pts)", 
                    (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        frame_resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        _, encoded_image = cv2.imencode('.ppm', frame_resized)
        self.photo = tk.PhotoImage(data=encoded_image.tobytes())
        self.video_label.config(image=self.photo)

        self.root.after(self.delay, self.update_video)


def main():
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()