"""ROS 2 node: RF samples in, GNN radio map + emitter estimate out. Topics are listed in INTERFACE.md."""
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Empty, Float32MultiArray, MultiArrayDimension, String

from .core import H, W, MIN_DB, MAX_DB, Reconstructor, SampleGrid, load_building_mask, make_backend


class GnnNode(Node):
    def __init__(self):
        super().__init__("gnn_reconstruction")
        p = self.declare_parameter
        backend = p("backend", "torch").value            # torch | onnx | tensorrt
        model_path = p("model_path", "").value           # .pth / .onnx / .engine
        fp16 = p("fp16", False).value
        mask_file = p("building_mask_file", "").value    # map .parquet or .npy
        self.frame = p("frame_id", "map").value
        self.min_new = p("min_new_samples", 5).value
        self.input_is_rss = p("input_is_rss", False).value
        self.tx_power_dbm = p("tx_power_dbm", 0.0).value
        grid = SampleGrid(p("origin_x", 0.0).value, p("origin_y", 0.0).value, p("m_per_px", 3.52).value)

        env = load_building_mask(mask_file) if mask_file else np.zeros((H, W), np.float32)
        self.recon = Reconstructor(make_backend(backend, model_path, fp16), env, grid)
        self.new_samples = 0
        self.updates = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(Float32MultiArray, "gnn/path_loss_map", latched)
        self.pub_viz = self.create_publisher(OccupancyGrid, "gnn/map_viz", latched)
        self.pub_est = self.create_publisher(PointStamped, "gnn/emitter_estimate", 10)
        self.pub_status = self.create_publisher(String, "gnn/status", 10)
        self.create_subscription(PointStamped, "rf/samples", self.on_sample, 100)
        self.create_subscription(Empty, "gnn/reset", self.on_reset, 1)
        self.create_timer(1.0 / p("update_hz", 2.0).value, self.on_timer)

        self.recon.run()  # warm-up so the first real update isn't slow
        self.get_logger().info(f"backend={self.recon.backend.name} model={model_path} "
                               f"mask={'yes' if mask_file else 'none'} res={grid.res} m/px")

    def on_sample(self, msg):
        value = msg.point.z
        path_loss = self.tx_power_dbm - value if self.input_is_rss else value
        if self.recon.grid.add(msg.point.x, msg.point.y, path_loss):
            self.new_samples += 1
        else:
            self.get_logger().warn("sample outside the map, ignored", throttle_duration_sec=5.0)

    def on_reset(self, _msg):
        self.recon.grid.clear()
        self.new_samples = 0
        self.get_logger().info("samples cleared")

    def on_timer(self):
        if self.new_samples < self.min_new:
            return
        self.new_samples = 0
        pl_db, (row, col), (x, y), ms = self.recon.run()
        self.updates += 1
        stamp = self.get_clock().now().to_msg()
        self.publish_map(pl_db, stamp)

        est = PointStamped()
        est.header.stamp, est.header.frame_id = stamp, self.frame
        est.point.x, est.point.y, est.point.z = float(x), float(y), float(pl_db[row, col])
        self.pub_est.publish(est)

        grid = self.recon.grid
        status = {"update": self.updates, "samples": grid.n_samples,
                  "pixels_sampled": int(grid.mask.sum()), "coverage": round(float(grid.mask.mean()), 5),
                  "inference_ms": round(ms, 2), "backend": self.recon.backend.name,
                  "estimate_px": [int(row), int(col)]}
        self.pub_status.publish(String(data=json.dumps(status)))

    def publish_map(self, pl_db, stamp):
        arr = Float32MultiArray()
        arr.layout.dim = [MultiArrayDimension(label="rows", size=H, stride=H * W),
                          MultiArrayDimension(label="cols", size=W, stride=W)]
        arr.data = pl_db.astype(np.float32).ravel().tolist()
        self.pub_map.publish(arr)

        # RViz view only: 100 = strongest signal (lowest path loss) in this map, 0 = weakest, -1 = building.
        # Stretched to the map's own range so the colors show contrast; the dB values are on gnn/path_loss_map.
        g = self.recon.grid
        viz = OccupancyGrid()
        viz.header.stamp, viz.header.frame_id = stamp, self.frame
        viz.info.resolution = float(g.res)
        viz.info.width, viz.info.height = W, H
        viz.info.origin.position.x, viz.info.origin.position.y = float(g.origin_x), float(g.origin_y)
        viz.info.origin.orientation.w = 1.0
        outside = self.recon.env < 0.5
        lo, hi = np.percentile(pl_db[outside], [1, 99]) if outside.any() else (MIN_DB, MAX_DB)
        strength = np.clip((hi - pl_db) / max(hi - lo, 1e-3) * 97 + 1, 1, 98).astype(np.int8)
        strength[~outside] = -1
        viz.data = strength.ravel().tolist()
        self.pub_viz.publish(viz)


def main():
    rclpy.init()
    node = GnnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
