"""Test driver for the GNN node: plays a test-split map as if a rover were driving through it.

Publishes the map's true path loss along a random walk (or at random pixels) on rf/samples,
and scores every map the GNN node publishes against the ground truth (same metrics as the
offline benchmark: RMSE on unobserved, non-building pixels, and emitter error). If the tracker
node runs, it also scores tracker/emitter, so raw GNN and Kalman errors can be compared.

Everything it scores is also written to a CSV log (parameter log_csv, default
~/replay_logs/replay_<date>_<time>.csv, plus a .json with the map and emitter info), which
plot_replay turns into a PNG figure: `ros2 run gnn_reconstruction plot_replay <csv>`.
"""
import csv
import json
import os
import time

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray

from .core import H, W


class ReplayNode(Node):
    def __init__(self):
        super().__init__("rf_replay")
        p = self.declare_parameter
        map_file = p("map_file", "").value
        self.mode = p("mode", "walk").value            # walk | uniform
        self.rate = p("samples_per_sec", 20.0).value
        self.stop_density = p("stop_density_pct", 10.0).value
        self.origin_x, self.origin_y = p("origin_x", 0.0).value, p("origin_y", 0.0).value
        self.res = p("m_per_px", 3.52).value
        self.rng = np.random.default_rng(p("seed", 0).value)

        row = pd.read_parquet(map_file, engine="pyarrow").iloc[0]
        self.env = np.asarray(row["building_mask"], np.float32).reshape(H, W)
        self.gt = np.asarray(row["path_loss"], np.float32).reshape(H, W)
        tx = np.asarray(row["tx_origin"], np.float32).reshape(H, W)
        self.tx_rc = np.unravel_index(int(np.argmax(tx)), tx.shape)

        self.free = np.argwhere(self.env < 0.5)
        self.pos = self.free[self.rng.integers(len(self.free))]
        self.sampled = np.zeros((H, W), bool)
        self.hit_m = p("hit_radius_m", 10.56).value
        self.tx_xy = (self.origin_x + (self.tx_rc[1] + 0.5) * self.res, self.origin_y + (self.tx_rc[0] + 0.5) * self.res)
        self.raw_err, self.kf_err = [], []
        self.coverage_pct = 0.0
        self.t0 = time.monotonic()

        log_csv = p("log_csv", "").value or os.path.expanduser(
            time.strftime("~/replay_logs/replay_%Y%m%d_%H%M%S.csv"))
        if log_csv != "none":
            os.makedirs(os.path.dirname(os.path.abspath(log_csv)), exist_ok=True)
            json.dump({"map_file": os.path.abspath(map_file), "mode": self.mode, "m_per_px": self.res,
                       "origin_x": self.origin_x, "origin_y": self.origin_y, "hit_radius_m": self.hit_m,
                       "tx_xy": self.tx_xy, "tx_rc": [int(v) for v in self.tx_rc]},
                      open(os.path.splitext(log_csv)[0] + ".json", "w"), indent=2)
            self.log_file = open(log_csv, "w", newline="")
            self.log = csv.writer(self.log_file)
            self.log.writerow(["t_s", "kind", "x_m", "y_m", "err_m", "coverage_pct", "rmse_db"])
            self.get_logger().info(f"logging to {log_csv}")
        else:
            self.log = None

        self.pub = self.create_publisher(PointStamped, "rf/samples", 100)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float32MultiArray, "gnn/path_loss_map", self.on_map, latched)
        self.create_subscription(PointStamped, "gnn/emitter_estimate", self.on_estimate, 10)
        self.create_subscription(PointStamped, "tracker/emitter", self.on_tracker, 10)
        self.timer = self.create_timer(1.0 / self.rate, self.step)
        self.get_logger().info(f"replaying {map_file} ({self.mode}), true emitter at px {tuple(map(int, self.tx_rc))}")

    def next_pixel(self):
        if self.mode == "uniform":
            return self.free[self.rng.integers(len(self.free))]
        for _ in range(20):  # random walk that stays out of buildings
            dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][self.rng.integers(4)]
            r, c = self.pos[0] + dr, self.pos[1] + dc
            if 0 <= r < H and 0 <= c < W and self.env[r, c] < 0.5:
                self.pos = np.array([r, c])
                break
        return self.pos

    def step(self):
        if self.sampled.mean() * 100 >= self.stop_density:
            self.timer.cancel()
            self.get_logger().info(f"reached {self.stop_density}% of pixels, stopped publishing")
            return
        r, c = self.next_pixel()
        self.sampled[r, c] = True
        self.coverage_pct = self.sampled.mean() * 100
        self._write("sample", (c + 0.5) * self.res + self.origin_x, (r + 0.5) * self.res + self.origin_y)
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.point.x = self.origin_x + (c + 0.5) * self.res
        msg.point.y = self.origin_y + (r + 0.5) * self.res
        msg.point.z = float(self.gt[r, c])
        self.pub.publish(msg)

    def on_map(self, msg):
        pred = np.asarray(msg.data, np.float32).reshape(H, W)
        score = (~self.sampled) & (self.env < 0.5)
        rmse = float(np.sqrt(np.mean((pred[score] - self.gt[score]) ** 2)))
        est = np.unravel_index(int(np.argmin(np.where(self.env > 0.5, np.inf, pred))), pred.shape)
        err_m = float(np.hypot(est[0] - self.tx_rc[0], est[1] - self.tx_rc[1])) * self.res
        self._write("map", float("nan"), float("nan"), err_m, rmse)
        self.get_logger().info(f"samples {self.sampled.mean() * 100:.2f}%  RMSE {rmse:.2f} dB  "
                               f"emitter error {err_m:.1f} m")

    def _write(self, kind, x, y, err=float("nan"), rmse=float("nan")):
        if self.log:
            self.log.writerow([f"{time.monotonic() - self.t0:.3f}", kind, f"{x:.2f}", f"{y:.2f}",
                               f"{err:.2f}", f"{self.coverage_pct:.3f}", f"{rmse:.3f}"])
            self.log_file.flush()  # keep the log usable if the run is stopped with Ctrl+C

    def _err(self, msg):
        return float(np.hypot(msg.point.x - self.tx_xy[0], msg.point.y - self.tx_xy[1]))

    @staticmethod
    def _summary(errs, hit_m):
        e = np.asarray(errs)
        return f"RMSE {np.sqrt(np.mean(e ** 2)):.1f} m, within {hit_m:.1f} m {100 * np.mean(e <= hit_m + 0.01):.0f}%"

    def on_estimate(self, msg):
        self.raw_err.append(self._err(msg))
        self._write("raw", msg.point.x, msg.point.y, self.raw_err[-1])

    def on_tracker(self, msg):
        self.kf_err.append(self._err(msg))
        self._write("kalman", msg.point.x, msg.point.y, self.kf_err[-1])
        if self.raw_err:
            self.get_logger().info(f"emitter error now: raw GNN {self.raw_err[-1]:.1f} m, Kalman {self.kf_err[-1]:.1f} m | "
                                   f"so far: raw {self._summary(self.raw_err, self.hit_m)}; "
                                   f"Kalman {self._summary(self.kf_err, self.hit_m)}")


def main():
    rclpy.init()
    node = ReplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
