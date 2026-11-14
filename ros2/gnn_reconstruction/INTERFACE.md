# GNN reconstruction node: interface

This file is the contract between the GNN node and the parts that feed it or use its output (PPO navigator, Sionna moving-emitter data, the upsampled GNN). Any part that keeps these topics and units can be swapped without changing the others.

## Map frame and grid

- All positions are in the `map` frame (parameter `frame_id`), in meters.
- The reconstruction grid is 256 × 256 pixels. Pixel (row, col) covers
  `x ∈ [origin_x + col·m_per_px, +m_per_px)` and `y ∈ [origin_y + row·m_per_px, +m_per_px)`.
  This is the same layout as `nav_msgs/OccupancyGrid`: row 0 is the lowest y, column 0 the lowest x.
- `origin_x`, `origin_y` and `m_per_px` must match the building mask the node is given.

## Inputs

| Topic | Type | Content |
| --- | --- | --- |
| `rf/samples` | `geometry_msgs/PointStamped` | One RF measurement: `point.x`, `point.y` = where it was taken (m, map frame); `point.z` = path loss in dB. If `input_is_rss` is true, `point.z` is received power in dBm instead, and the node uses path loss = `tx_power_dbm` − RSS. |
| `gnn/reset` | `std_msgs/Empty` | Clears all stored samples (new area, or emitter moved far). |

Several samples in the same pixel are averaged. Samples outside the grid are ignored with a warning.

## Outputs

| Topic | Type | Content |
| --- | --- | --- |
| `gnn/path_loss_map` | `std_msgs/Float32MultiArray` | Predicted path loss in dB for every pixel, 256 × 256, row-major (index = row·256 + col). Latched (transient local). |
| `gnn/emitter_estimate` | `geometry_msgs/PointStamped` | Center of the pixel with the lowest predicted path loss outside buildings (m, map frame). `point.z` = its predicted path loss (dB). |
| `gnn/map_viz` | `nav_msgs/OccupancyGrid` | For RViz only: 1–98 = weakest to strongest signal, stretched to this map's own range (so colors show contrast with the costmap color scheme), −1 = building. Latched. |
| `gnn/status` | `std_msgs/String` (JSON) | `update`, `samples`, `pixels_sampled`, `coverage` (fraction of pixels sampled), `inference_ms`, `backend`, `estimate_px`. |

### Tracker node (`tracker_node`)

| Topic | Type | Content |
| --- | --- | --- |
| `tracker/emitter` | `geometry_msgs/PointStamped` | Kalman-smoothed emitter position (m, map frame), one per `gnn/emitter_estimate`, same time stamp. This is the target for PPO. |
| `tracker/velocity` | `geometry_msgs/Vector3Stamped` | Estimated emitter velocity (m/s), x and y. |
| `tracker/status` | `std_msgs/String` (JSON) | `estimates`, `rejected`, `accepted` (last estimate passed the outlier gate), `position_std_m`, `speed_mps`. |

The tracker is a gated constant-velocity Kalman filter (`gnn_reconstruction/tracker.py`). It subscribes to `gnn/emitter_estimate` and resets on `gnn/reset`. Parameters: `accel_std` (0.1 m/s²), `meas_std_px` (6 px), `m_per_px` (3.52), `gate` (9.21), `max_rejects` (4), `frame_id` (`map`).

A new map is published at most `update_hz` times per second, and only after at least `min_new_samples` new samples have arrived.

## Model contract

The model gets a (1, 3, 256, 256) float32 tensor and returns (1, 1, 256, 256):

1. channel 0: building mask (1 = building)
2. channel 1: zeros (the emitter location is never given to the model)
3. channel 2: sampled path loss normalized as (dB − 30) / 110, clipped to [0, 1], and 0 where there is no sample

The output is normalized the same way, and the node converts it back to dB. A replacement model (for example the upsampled GNN) keeps this contract, or adds a backend in `core.py` that converts to and from it.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `backend` | `torch` | `torch` (.pth), `onnx` (.onnx, onnxruntime) or `tensorrt` (.engine built on this Jetson) |
| `model_path` | (required) | weights file for the chosen backend |
| `fp16` | `false` | half precision for the torch backend on GPU (FP32 benchmarked faster on the Orin 32GB at MAXN: 8.9 vs 10.6 ms) |
| `building_mask_file` | empty | map `.parquet` (uses its `building_mask` column) or `.npy`; empty = no buildings |
| `origin_x`, `origin_y` | 0.0 | map-frame position of the grid corner (m) |
| `m_per_px` | 3.52 | pixel size (m); the UCF dataset maps cover 900 m in 256 px |
| `update_hz` | 2.0 | maximum map update rate |
| `min_new_samples` | 5 | new samples needed before the next update |
| `input_is_rss`, `tx_power_dbm` | false, 0.0 | convert received power to path loss |
| `frame_id` | `map` | frame of all outputs |
