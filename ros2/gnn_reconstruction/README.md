# gnn_reconstruction

ROS 2 package that runs the team's ViG GNN (`GNN_no_tx.pth`) as a node: RF samples in, a 256 × 256 path-loss map and an emitter estimate out. Topics, units and the model contract are in [INTERFACE.md](INTERFACE.md).

| File | Purpose |
| --- | --- |
| `gnn_reconstruction/gnn_node.py` | the GNN node |
| `gnn_reconstruction/replay_node.py` | test driver: plays a test-split map as a driving rover and scores every map the node publishes |
| `gnn_reconstruction/tracker_node.py`, `tracker.py` | Kalman tracker: smooths `gnn/emitter_estimate` into `tracker/emitter` and `tracker/velocity` |
| `gnn_reconstruction/core.py` | ROS-free part: sample grid, input building, torch / ONNX / TensorRT backends |
| `gnn_reconstruction/model.py` | the GNN architecture (same as `benchmarks/benchmark_reconstruction.py`) |
| `gnn_reconstruction/plot_replay.py` | turns a replay log into a PNG figure (map with raw and Kalman estimates, error over time) |
| `launch/gnn_replay.launch.py` | GNN node + replay of one map (`rviz:=true` also opens RViz) |
| `rviz/gnn.rviz` | saved RViz view of the map and emitter estimate |
| `test/test_core.py` | checks the core against the offline model, no ROS needed |
| `test/test_tracker.py` | checks the tracker against the Kalman notebook's simulation, no ROS needed |

## Build

The package needs ROS 2 (Humble on Ubuntu 22.04, Jazzy on Ubuntu 24.04), numpy, pandas and pyarrow, plus the library for the chosen backend: PyTorch for `.pth`, onnxruntime for `.onnx`, or TensorRT and CUDA PyTorch for `.engine`.

```bash
mkdir -p ~/ros2_ws/src && cp -r gnn_reconstruction ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select gnn_reconstruction --symlink-install
source install/setup.bash
```

## Test with a replayed map

```bash
MAP=$(find ~/nna/data/test_split -name '*.parquet' | head -1)
ros2 launch gnn_reconstruction gnn_replay.launch.py model_path:=$HOME/nna/benchmarks/GNN_no_tx.pth map_file:=$MAP
```

The launch file also starts the Kalman tracker (`tracker:=false` turns it off). The replay node prints the RMSE (on unobserved, non-building pixels) and the emitter error for every map the GNN node publishes. These should match the offline benchmark for the same sample density and sample shape: walk samples (default) give about 20 dB, and `mode:=uniform` gives about 11 dB at 5 %. In a second terminal:

```bash
ros2 topic echo /gnn/status          # inference time, samples, coverage
ros2 topic echo /gnn/emitter_estimate
ros2 topic echo /tracker/status       # Kalman: rejected estimates, uncertainty, speed
```

With the tracker running, the replay node also prints the raw GNN and Kalman emitter errors for every update, plus running RMSE and the share within the 3 px hit radius for both.

Every replay also writes a log to `~/replay_logs/replay_<date>_<time>.csv` (with a `.json` holding the map and emitter info; `log_csv:=<path>` picks another path, `log_csv:=none` turns it off). After the run, a figure for sharing:

```bash
ros2 run gnn_reconstruction plot_replay            # newest log; or pass a .csv path, -o out.png
```

The PNG (saved next to the log) shows the true path-loss map with buildings, the rover's samples, every raw GNN emitter estimate, the Kalman track and the true emitter with the hit radius, plus the emitter error over time for raw GNN and Kalman. It needs matplotlib (`sudo apt install python3-matplotlib`).

Add `rviz:=true` to the launch command to open RViz with the saved view (`rviz/gnn.rviz`: top-down map in the costmap color scheme plus the emitter estimate). To set it up by hand instead, set Fixed Frame to `map` and add a Map display on `/gnn/map_viz` with Durability Policy set to Transient Local. The launch file publishes a static `world` → `map` transform, so RViz has a frame to draw in even without a robot.

The other backends use the same launch file with `backend:=onnx model_path:=.../onnx/GNN.onnx`, or `backend:=tensorrt model_path:=.../engines/GNN.engine`. The ONNX file must come from a `models_radiomap.py` that applies the PatchEmbed LayerNorm. Without that fix, the exported model does not match the trained weights.

## Core tests (no ROS)

```bash
cd gnn_reconstruction
GNN_WEIGHTS=~/nna/benchmarks/GNN_no_tx.pth GNN_MAP=$MAP GNN_ONNX=~/nna/onnx/GNN.onnx python3 -m pytest test/ -q
```
