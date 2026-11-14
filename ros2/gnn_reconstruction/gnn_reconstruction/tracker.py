"""ROS-free Kalman tracker for the GNN emitter estimate (same filter as notebooks/kalman_tracker_sim.ipynb).

Gated constant-velocity linear Kalman filter. State [x, y, vx, vy] in meters and m/s (map frame);
measurement = the GNN's emitter estimate (x, y). Estimates far from the prediction (Mahalanobis
distance^2 above `gate`) are rejected; after `max_rejects` rejections in a row the track restarts.
"""
import numpy as np


class EmitterTracker:
    def __init__(self, accel_std=0.1, meas_std=21.12, gate=9.21, max_rejects=4, vel_std=2.0):
        self.accel_std, self.meas_std = accel_std, meas_std
        self.gate, self.max_rejects, self.vel_std = gate, max_rejects, vel_std
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.R = np.eye(2) * meas_std ** 2
        self.x = None   # state [x, y, vx, vy]
        self.P = None   # covariance
        self.t = None   # time of the last estimate (s)
        self.rejects = 0

    def reset(self):
        self.x = self.P = self.t = None
        self.rejects = 0

    def _start(self, z, t):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([self.meas_std ** 2] * 2 + [self.vel_std ** 2] * 2)
        self.t, self.rejects = t, 0

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.accel_std ** 2

    def update(self, z, t):
        """One GNN estimate z = (x, y) in meters at time t (s). Returns (position, accepted)."""
        z = np.asarray(z, float)
        if self.x is None:
            self._start(z, t)
            return self.x[:2].copy(), True
        dt = max(float(t - self.t), 0.0)
        self.t = t
        self.predict(dt)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        d2 = float(y @ np.linalg.solve(S, y))
        if d2 > self.gate:
            self.rejects += 1
            if self.rejects >= self.max_rejects:
                self._start(z, t)  # consistently somewhere else: re-lock there
                return self.x[:2].copy(), True
            return self.x[:2].copy(), False
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.rejects = 0
        return self.x[:2].copy(), True

    @property
    def velocity(self):
        return None if self.x is None else self.x[2:].copy()

    @property
    def position_std(self):
        """1-sigma position uncertainty (m), the mean of the x and y standard deviations."""
        return None if self.P is None else float(np.sqrt((self.P[0, 0] + self.P[1, 1]) / 2))
