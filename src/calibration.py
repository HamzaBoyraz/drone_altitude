import json
import os
from typing import List, Dict, Optional


class CalibrationManager:
    """Manages recording, loading, and saving calibration sample pairs (Pixel Metrics -> Real 3D Coordinates)."""

    def __init__(self, filepath: str = "calibration_data.json"):
        self.filepath = filepath
        self.samples: List[Dict[str, float]] = []
        self.load_data()

    def add_sample(
        self, 
        cx_px: int, 
        cy_px: int, 
        area_px: float, 
        real_x: float, 
        real_y: float, 
        real_z: float
    ) -> Dict[str, float]:
        """Appends a new calibration data point."""
        sample = {
            "id": len(self.samples) + 1,
            "cx_px": float(cx_px),
            "cy_px": float(cy_px),
            "area_px": float(area_px),
            "real_x_mm": float(real_x),
            "real_y_mm": float(real_y),
            "real_z_mm": float(real_z)
        }
        self.samples.append(sample)
        print(f"\n[CALIBRATION] Captured Sample #{sample['id']}: {sample}")
        return sample

    def save_data(self) -> None:
        """Saves samples list to JSON file."""
        with open(self.filepath, "w") as f:
            json.dump(self.samples, f, indent=4)
        print(f"[CALIBRATION] Successfully saved {len(self.samples)} sample(s) to '{self.filepath}'.")

    def load_data(self) -> None:
        """Loads samples from JSON file if present."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.samples = json.load(f)
                print(f"[CALIBRATION] Loaded {len(self.samples)} sample(s) from '{self.filepath}'.")
            except Exception as e:
                print(f"[CALIBRATION ERROR] Could not read '{self.filepath}': {e}")
                self.samples = []
        else:
            self.samples = []
            print(f"[CALIBRATION] No existing file found at '{self.filepath}'. Starting fresh.")

    def print_summary(self) -> None:
        """Displays currently loaded dataset summary in terminal."""
        print(f"\n================ CALIBRATION DATASET ({len(self.samples)} Samples) ================")
        if not self.samples:
            print("  (No samples loaded)")
        for s in self.samples:
            print(f"  #{s['id']} | Pixel Center: ({s['cx_px']:.0f}, {s['cy_px']:.0f}) | Area: {s['area_px']:.0f} px "
                  f"--> Real 3D (X, Y, Z): ({s['real_x_mm']:.1f}, {s['real_y_mm']:.1f}, {s['real_z_mm']:.1f}) mm")
        print("===========================================================================\n")