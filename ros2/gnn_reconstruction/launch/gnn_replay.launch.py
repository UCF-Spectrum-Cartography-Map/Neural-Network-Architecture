"""GNN node + replay of one test map. Example:
ros2 launch gnn_reconstruction gnn_replay.launch.py \
    model_path:=$HOME/nna/benchmarks/GNN_no_tx.pth map_file:=$MAP rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arg = LaunchConfiguration
    args = [
        DeclareLaunchArgument("backend", default_value="torch", description="torch | onnx | tensorrt"),
        DeclareLaunchArgument("model_path", description=".pth / .onnx / .engine"),
        DeclareLaunchArgument("map_file", description="test-split .parquet map to replay"),
        DeclareLaunchArgument("mode", default_value="walk", description="walk | uniform"),
        DeclareLaunchArgument("m_per_px", default_value="3.52"),
        DeclareLaunchArgument("update_hz", default_value="2.0"),
        DeclareLaunchArgument("fp16", default_value="false",
                              description="half precision (torch backend); FP32 was faster on the Orin 32GB"),
        DeclareLaunchArgument("tracker", default_value="true", description="also run the Kalman tracker node"),
        DeclareLaunchArgument("accel_std", default_value="0.1", description="tracker random acceleration (m/s^2)"),
        DeclareLaunchArgument("log_csv", default_value="",
                              description="replay log path ('' = ~/replay_logs/replay_<date>_<time>.csv, 'none' = off)"),
        DeclareLaunchArgument("rviz", default_value="false", description="also open RViz with the saved view"),
    ]
    gnn = Node(package="gnn_reconstruction", executable="gnn_node", output="screen", parameters=[{
        "backend": arg("backend"), "model_path": arg("model_path"),
        "building_mask_file": arg("map_file"), "m_per_px": arg("m_per_px"), "update_hz": arg("update_hz"),
        "fp16": arg("fp16"),
    }])
    replay = Node(package="gnn_reconstruction", executable="replay_node", output="screen", parameters=[{
        "map_file": arg("map_file"), "mode": arg("mode"), "m_per_px": arg("m_per_px"),
        "log_csv": arg("log_csv"),
    }])
    tracker = Node(package="gnn_reconstruction", executable="tracker_node", output="screen",
                   condition=IfCondition(arg("tracker")), parameters=[{
                       "m_per_px": arg("m_per_px"), "accel_std": arg("accel_std")}])
    # Replay has no robot publishing TF, so give RViz a "map" frame to display in.
    map_frame = Node(package="tf2_ros", executable="static_transform_publisher",
                     arguments=["--frame-id", "world", "--child-frame-id", "map"])
    rviz_cfg = os.path.join(get_package_share_directory("gnn_reconstruction"), "rviz", "gnn.rviz")
    rviz = Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_cfg], condition=IfCondition(arg("rviz")))
    return LaunchDescription(args + [gnn, replay, tracker, map_frame, rviz])
