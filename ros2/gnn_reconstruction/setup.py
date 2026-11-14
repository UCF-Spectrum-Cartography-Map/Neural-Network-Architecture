from glob import glob

from setuptools import setup

package_name = "gnn_reconstruction"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "INTERFACE.md", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="ViG GNN radio-map reconstruction as a ROS 2 node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gnn_node = gnn_reconstruction.gnn_node:main",
            "replay_node = gnn_reconstruction.replay_node:main",
            "tracker_node = gnn_reconstruction.tracker_node:main",
            "plot_replay = gnn_reconstruction.plot_replay:main",
        ],
    },
)
