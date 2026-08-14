import numpy as np
from typing import Optional, Dict, Tuple
from src.calibration import CalibrationManager


class SpatialEstimator:
    """
    Estimates 3D coordinates (X, Y, Z) using multi-variable least squares regression.
    Includes non-linear radial distortion terms (r^2) to compensate for lens distortion
    near image boundaries.
    """

    def __init__(self, calibration_mgr: CalibrationManager, frame_size: Tuple[int, int] = (640, 480)):
        self.calib_mgr = calibration_mgr
        self.u0 = frame_size[0] / 2.0  # Optical center X (px)
        self.v0 = frame_size[1] / 2.0  # Optical center Y (px)

        self.is_fitted = False
        
        # Coeff vectors: [beta_1, beta_2, ..., bias]
        self.z_coeffs = None
        self.x_coeffs = None
        self.y_coeffs = None
        self.use_distortion_terms = False

    def _build_features(
        self, cx: np.ndarray, cy: np.ndarray, area: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs non-linear feature matrices for X, Y, and Z fitting.
        
        Returns:
            (A_z, A_x, A_y) feature design matrices.
        """
        dx = cx - self.u0
        dy = cy - self.v0
        r_sq = (dx ** 2) + (dy ** 2)
        sqrt_a = np.sqrt(area)
        inv_sqrt_a = 1.0 / sqrt_a

        # Base perspective terms
        norm_dx = dx * inv_sqrt_a
        norm_dy = dy * inv_sqrt_a

        if self.use_distortion_terms:
            # 1. Non-linear terms including radial distortion cross-products (r^2)
            radial_dx = norm_dx * r_sq
            radial_dy = norm_dy * r_sq
            radial_z = r_sq * inv_sqrt_a
            inv_a = 1.0 / area  # Higher order area term (S^2)

            # Design Matrix Z: [1/sqrt(A), 1/A, r^2/sqrt(A), 1]
            A_z = np.vstack([inv_sqrt_a, inv_a, radial_z, np.ones_like(cx)]).T
            
            # Design Matrix X: [dx/sqrt(A), (dx*r^2)/sqrt(A), 1]
            A_x = np.vstack([norm_dx, radial_dx, np.ones_like(cx)]).T
            
            # Design Matrix Y: [dy/sqrt(A), (dy*r^2)/sqrt(A), 1]
            A_y = np.vstack([norm_dy, radial_dy, np.ones_like(cy)]).T
        else:
            # Fallback linear design matrices for small datasets (< 4 samples)
            A_z = np.vstack([inv_sqrt_a, np.ones_like(cx)]).T
            A_x = np.vstack([norm_dx, np.ones_like(cx)]).T
            A_y = np.vstack([norm_dy, np.ones_like(cy)]).T

        return A_z, A_x, A_y

    def fit(self) -> bool:
        """Fits non-linear polynomial model using collected calibration points."""
        samples = self.calib_mgr.samples
        if len(samples) < 2:
            self.is_fitted = False
            return False

        # Enable non-linear distortion terms if dataset is sufficiently large
        self.use_distortion_terms = len(samples) >= 4

        areas = np.array([s["area_px"] for s in samples], dtype=float)
        cx_vals = np.array([s["cx_px"] for s in samples], dtype=float)
        cy_vals = np.array([s["cy_px"] for s in samples], dtype=float)
        
        real_x = np.array([s["real_x"] for s in samples], dtype=float)
        real_y = np.array([s["real_y"] for s in samples], dtype=float)
        real_z = np.array([s["real_z"] for s in samples], dtype=float)

        # Build design matrices
        A_z, A_x, A_y = self._build_features(cx_vals, cy_vals, areas)

        # Solve for coefficients using Ordinary Least Squares (OLS)
        self.z_coeffs, _, _, _ = np.linalg.lstsq(A_z, real_z, rcond=None)
        self.x_coeffs, _, _, _ = np.linalg.lstsq(A_x, real_x, rcond=None)
        self.y_coeffs, _, _, _ = np.linalg.lstsq(A_y, real_y, rcond=None)

        self.is_fitted = True
        mode_str = "NON-LINEAR (Distortion Compensated)" if self.use_distortion_terms else "LINEAR"
        print(f"\n[ESTIMATOR FITTED] Mode: {mode_str} | Samples: {len(samples)}")
        return True

    def predict(self, cx: float, cy: float, area: float) -> Optional[Dict[str, float]]:
        """Predicts real 3D position (X, Y, Z) for a detected contour."""
        if not self.is_fitted or area <= 0:
            return None

        # Build 1-row feature matrices
        cx_arr, cy_arr, area_arr = np.array([cx]), np.array([cy]), np.array([area])
        A_z, A_x, A_y = self._build_features(cx_arr, cy_arr, area_arr)

        # Matrix dot products
        z = float(np.dot(A_z[0], self.z_coeffs))
        x = float(np.dot(A_x[0], self.x_coeffs))
        y = float(np.dot(A_y[0], self.y_coeffs))

        return {"X": x, "Y": y, "Z": z}