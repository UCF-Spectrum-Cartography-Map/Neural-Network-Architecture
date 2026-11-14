"""ROS 2 node: GNN emitter estimate in, Kalman-smoothed emitter position and velocity out.
Topics are listed in INTERFACE.md."""
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Vector3Stamped
from std_msgs.msg import Empty, String

from .tracker import EmitterTracker


class TrackerNode(Node):
    def __init__(self):
        super().__init__("emitter_tracker")
        p = self.declare_parameter
        m_per_px = p("m_per_px", 3.52).value
        self.trk = EmitterTracker(accel_std=p("accel_std", 0.1).value,
                                  meas_std=p("meas_std_px", 6.0).value * m_per_px,
                                  gate=p("gate", 9.21).value,
                                  max_rejects=p("max_rejects", 4).value)
        self.frame = p("frame_id", "map").value
        self.n = self.n_rejected = 0

        self.pub_pos = self.create_publisher(PointStamped, "tracker/emitter", 10)
        self.pub_vel = self.create_publisher(Vector3Stamped, "tracker/velocity", 10)
        self.pub_status = self.create_publisher(String, "tracker/status", 10)
        self.create_subscription(PointStamped, "gnn/emitter_estimate", self.on_estimate, 10)
        self.create_subscription(Empty, "gnn/reset", self.on_reset, 1)
        self.get_logger().info(f"accel_std={self.trk.accel_std} m/s^2, meas_std={self.trk.meas_std:.1f} m, "
                               f"gate={self.trk.gate}, m_per_px={m_per_px}")

    def on_estimate(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        (x, y), accepted = self.trk.update((msg.point.x, msg.point.y), t)
        self.n += 1
        self.n_rejected += not accepted

        pos = PointStamped()
        pos.header.stamp, pos.header.frame_id = msg.header.stamp, self.frame
        pos.point.x, pos.point.y = float(x), float(y)
        self.pub_pos.publish(pos)

        vx, vy = self.trk.velocity
        vel = Vector3Stamped()
        vel.header = pos.header
        vel.vector.x, vel.vector.y = float(vx), float(vy)
        self.pub_vel.publish(vel)

        status = {"estimates": self.n, "rejected": self.n_rejected, "accepted": bool(accepted),
                  "position_std_m": round(self.trk.position_std, 2), "speed_mps": round(float((vx ** 2 + vy ** 2) ** 0.5), 3)}
        self.pub_status.publish(String(data=json.dumps(status)))

    def on_reset(self, _msg):
        self.trk.reset()
        self.n = self.n_rejected = 0
        self.get_logger().info("track reset")


def main():
    rclpy.init()
    node = TrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
