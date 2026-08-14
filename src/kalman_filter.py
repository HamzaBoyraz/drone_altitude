import cv2
import numpy as np
from typing import Optional, Dict


class SpatialKalmanFilter:
    """
    6D Kalman Filter tracking 3D position (X, Y, Z) and velocities (vx, vy, vz).
    Smoothes out jittery spatial measurements across video frames.
    """

    def __init__(self, dt: float = 0.033):
        # 6 state variables [X, Y, Z, vx, vy, vz], 3 measurements [X, Y, Z]
        self.kf = cv2.KalmanFilter(dynamParams=6, measureParams=3)
        self.is_initialized = False

        # Transition Matrix (F): x_k = x_{k-1} + v * dt
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, dt,  0,  0],
            [0, 1, 0,  0, dt,  0],
            [0, 0, 1,  0,  0, dt],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1,  0],
            [0, 0, 0,  0,  0,  1]
        ], np.float32)

        # Measurement Matrix (H): We measure X, Y, Z directly
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        # Process Noise Covariance (Q): Trust model predictions vs physics noise
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2

        # Measurement Noise Covariance (R): Trust raw detector readings (lower = trust sensor more)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1

        # Post Error Covariance (P)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

    def reset(self):
        """Resets state when target object is lost."""
        self.is_initialized = False

    def update(self, pos_3d: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """
        Runs Kalman Prediction and Correction step.
        Returns smoothed (X, Y, Z) coordinates in.
        """
        # 1. Prediction step
        prediction = self.kf.predict()

        if pos_3d is None:
            # If target is missing, return uncorrected velocity prediction or reset
            if self.is_initialized:
                return {
                    "X": float(prediction[0][0]),
                    "Y": float(prediction[1][0]),
                    "Z": float(prediction[2][0])
                }
            return None

        # Format measurement vector
        measurement = np.array([
            [np.float32(pos_3d["X"])],
            [np.float32(pos_3d["Y"])],
            [np.float32(pos_3d["Z"])]
        ])

        if not self.is_initialized:
            # Initialize state with first measurement
            self.kf.statePost = np.array([
                [measurement[0][0]],
                [measurement[1][0]],
                [measurement[2][0]],
                [0.0], [0.0], [0.0]
            ], np.float32)
            self.is_initialized = True
            return pos_3d

        # 2. Correction step with live camera measurement
        corrected = self.kf.correct(measurement)

        return {
            "X": float(corrected[0][0]),
            "Y": float(corrected[1][0]),
            "Z": float(corrected[2][0])
        }