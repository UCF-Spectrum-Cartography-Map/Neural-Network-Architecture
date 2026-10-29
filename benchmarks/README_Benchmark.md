# GNN and YOLO26n Benchmarking

This document describes how the team benchmarks the two models that run on the rover: the **GNN** (radio-map reconstruction) and **YOLO26n** (obstacle detection with OAK-D Lite depth).

Each model is benchmarked on three levels:

1. **Accuracy** on held-out data the model never trained on.
2. **On-device cost** on the Jetson: latency, RAM and power.
3. **Mission effect** in the Gazebo sim, and later on the rover.

All final speed numbers are measured on the Jetson, since Colab (T4) and laptop timings do not transfer to it.

## Files

| File | Purpose |
| --- | --- |
| `README_Jetson_Setup.md` | Setting up PyTorch on the Jetson and running the GNN there |
| `benchmark_reconstruction.py` | GNN benchmark: accuracy at fixed sampling densities, latency, RAM, power |
| `GNN_no_tx.pth` | GNN weights trained with the emitter channel zeroed (Oct 21) |
| `pth_to_onnx.py`, `onnx_to_engine.py` | Prior team's ONNX and TensorRT conversion (expects `GNN_best.pth`, see below) |
| `models_radiomap.py` | Prior team's model definitions, used by the conversion scripts |
| `archive/` | Prior team's multi-model benchmark (`ablation_inference_benchmark.py`, `README.md`) and its old results |
| `../data/gnn_inference_test/` | Held-out test maps (`.parquet`) |

## Before every Jetson run

```bash
sudo nvpmodel -q          # records the power mode; the same mode is used for every run
sudo jetson_clocks        # locks clocks so timings are repeatable
cat /etc/nv_tegra_release # records the JetPack / L4T version
```

All runs take place inside the `l4t-pytorch` container (see [README_Jetson_Setup.md](README_Jetson_Setup.md)), with other heavy programs closed except during the concurrency test.

## 1. GNN reconstruction

The model never gets the emitter location (channel 1 is always zeros), the same as on the real hunt.

### Metrics

- **RMSE / MAE in dB**, only on pixels the model had to infer (not sampled, not inside buildings)
- **% of pixels within 3 dB and 6 dB**
- **Emitter-localization error (m)** and **hit rate** (under 5 m): strongest predicted pixel vs the true emitter
- **Latency p50 / p95 / p99**, FPS, peak RAM, power and energy per map

### Run

The benchmark is run from the repository root:

```bash
# main run: emitter channel zeroed, FP16
python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth \
    --test-dir data/gnn_inference_test --out-dir benchmarks/gnn_bench --fp16

# reference runs (they reuse the same saved sample masks)
python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth \
    --test-dir data/gnn_inference_test --out-dir benchmarks/gnn_bench          # FP32
python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth \
    --test-dir data/gnn_inference_test --out-dir benchmarks/gnn_bench --use-tx # with tx
```

The main options are:

| Option | What it does |
| --- | --- |
| `--densities 0.5 1 2 5 10` | % of pixels given as samples (default) |
| `--mask-kind walk` | samples along driving paths instead of random pixels |
| `--avoid-buildings` | never place samples inside buildings |
| `--m-per-px 1.0` | map resolution, for the emitter error in meters |
| `--timed-iters 500` | timed runs for the latency statistics |

The script writes one set of outputs per run tag (for example `fp16_notx_uniform`) to `--out-dir`:

- `summary_<tag>.csv`: per density, the mean and 95 % bootstrap CI
- `per_sample_<tag>.csv`: one row per map and density
- `run_info_<tag>.json`: device, versions, latency, RAM and power
- `tegrastats.log`: raw power and RAM log
- `masks/`: saved sample masks, reused so every run sees the same samples

**Test data:** the full benchmark uses 200 or more maps that were never used in training (the dataset's test split), plus a set of the team's own Sionna campus maps as a generalization check. The 10 maps in `data/gnn_inference_test/` serve only as a smoke test.

### TensorRT

`pth_to_onnx.py` looks for `GNN_best.pth`, so the no-tx weights are first copied under that name into a separate folder:

```bash
mkdir -p checkpoints && cp benchmarks/GNN_no_tx.pth checkpoints/GNN_best.pth
python3 benchmarks/pth_to_onnx.py --weights-dir checkpoints --out-dir onnx --models GNN --static
python3 benchmarks/onnx_to_engine.py --onnx-dir onnx --out-dir engines --models GNN --precision fp16 --static
/usr/src/tensorrt/bin/trtexec --loadEngine=engines/GNN.engine --warmUp=2000 --iterations=500 --percentile=50,95,99
```

The engine is built on the Jetson itself, because TensorRT engines only run on the device they were built on. The GNN's kNN step (`cdist` / `topk`) is the part most likely to fail in TensorRT; if it does, the PyTorch FP16 numbers are reported instead.

### Closed loop (sim)

`rf_explorer` is run with the GNN (code in [Intelligent-Motion-Planning/docker_setup/rf_explorer](https://github.com/UCF-Spectrum-Cartography-Map/Intelligent-Motion-Planning/tree/main/docker_setup/rf_explorer)) on about 5 worlds × 10 emitter placements, with fixed seeds. The reported metrics are success rate, distance driven and time to FOUND.

## 2. YOLO26n obstacle detection

### Stage A: export check on COCO

This stage confirms that the TensorRT export preserves accuracy. The reference is 40.9 mAP50-95 on COCO val2017 ([Ultralytics model card](https://huggingface.co/Ultralytics/YOLO26/blob/main/README.md)).

```bash
yolo export model=yolo26n.pt format=engine half=True imgsz=640             # FP16
yolo export model=yolo26n.pt format=engine int8=True data=coco.yaml imgsz=640  # INT8 (calibrated)
yolo val model=yolo26n.pt     data=coco.yaml imgsz=640 batch=1
yolo val model=yolo26n.engine data=coco.yaml imgsz=640 batch=1
```

FP16 passes if it is within about 0.5 mAP of the `.pt` model, and INT8 if it is within about 1 to 2 mAP.

### Stage B: rover-view test set (the main benchmark)

1. 300 to 500 OAK-D Lite frames (RGB + aligned depth) are recorded from rover height, outdoors, in sun and shade, with varied obstacles. Near-duplicate frames are skipped.
2. The frames are labeled with COCO class names (CVAT or Roboflow). For 50 to 100 frames, the distance to one obstacle is tape-measured.
3. The model is scored with `yolo val` on that dataset, and detections are binned by depth distance.
4. **The set is frozen** and never used for training. It serves as the before-and-after test when the team fine-tunes the model.

The metrics are:

- Recall at IoU 0.5 for 0 to 3 m, 3 to 6 m and beyond 6 m
- Missed-obstacle rate inside the stopping distance
- False positives per minute
- Depth error (median depth inside the box vs the tape measure) at 1, 2, 4 and 6 m

## 3. Jetson protocol (both models)

1. **Fixed clocks:** a fixed power mode plus `jetson_clocks`, both recorded.
2. **Warm-up:** 50 to 100 untimed runs, then at least 500 timed runs.
3. **Percentiles:** p50 / p95 / p99 latency is reported, not just the mean.
4. **Two timings:** engine only (`trtexec`), and end to end through the team's pipeline (pre/post-processing included).
5. **RAM:** the Orin shares memory between CPU and GPU, so peak used RAM comes from `tegrastats`, not GPU memory.
6. **Power:** `VDD_IN` from `tegrastats`, with energy per inference = W × s.
7. **Thermal soak:** a 10-minute continuous run checks that p95 latency does not drift up.
8. **Concurrency:** GNN + YOLO + depth + PPO + ROS run together, and the latency under load and total RAM are reported.

Proposed budgets (pending agreement with the mentor):

| Component | Budget |
| --- | --- |
| YOLO26n end to end | ≥ 15 FPS, p99 ≤ 100 ms |
| GNN map update | p99 ≤ 200 ms |
| Whole stack RAM on the Orin Nano 8 GB | ≤ 6 GB |

## Results to report

| GNN setting | RMSE dB @ 1 % / 5 % (95 % CI) | Emitter error m | p50 / p99 ms | Peak RAM MB | Power W |
| --- | --- | --- | --- | --- | --- |
| PyTorch FP32, no tx | | | | | |
| PyTorch FP16, no tx | | | | | |
| TensorRT FP16, no tx | | | | | |
| FP32, with tx (reference) | | | | | |

| YOLO26n version | COCO mAP50-95 | Rover recall 0–3 / 3–6 / 6+ m | Depth MAE m | p50 / p99 ms | RAM MB | Power W |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch | | | | | | |
| TensorRT FP16 | | | | | | |
| TensorRT INT8 | | | | | | |
