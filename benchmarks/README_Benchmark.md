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
| `run_benchmarks_orin.sh` | Runs the four standard GNN benchmark settings on the Jetson host in one go and prints the result tables |
| `teg_summary.py` | Power and RAM summary from a `tegrastats` log (handles both Jetson power-rail layouts) |
| `GNN_no_tx.pth` | GNN weights trained with the emitter channel zeroed (Oct 21) |
| `../models/GNN/GNN_no_tx_ver.ipynb` | Training notebook for `GNN_no_tx.pth` (emitter channel zeroed); `GNN_patch_and_parquet_ver.ipynb` in the same folder is the prior team's version, which still feeds the emitter location |
| `pth_to_onnx.py`, `onnx_to_engine.py` | Prior team's ONNX and TensorRT conversion (expects `GNN_best.pth`, see below) |
| `models_radiomap.py` | Prior team's model definitions, used by the conversion scripts |
| `archive/` | Prior team's multi-model benchmark (`ablation_inference_benchmark.py`, `README.md`) and its old results |
| `../data/gnn_inference_test/` | Held-out test maps (`.parquet`) |

## Before every Jetson run

Setup (PyTorch, test maps, ROS) is in [README_Jetson_Setup.md](README_Jetson_Setup.md). Every timed run records the power mode (`sudo nvpmodel -q`) and the JetPack version (`cat /etc/nv_tegra_release`), and locks clocks with `sudo jetson_clocks`; `run_benchmarks_orin.sh` does all three. Other heavy programs are closed except during the concurrency test.

## 1. GNN reconstruction

The model never gets the emitter location (channel 1 is always zeros), the same as on the real hunt.

### Metrics

- **RMSE / MAE in dB**, only on pixels the model had to infer (not sampled, not inside buildings)
- **% of pixels within 3 dB and 6 dB**
- **Emitter-localization error (m)** and **hit rate** (within 3 px, 10.56 m): strongest predicted pixel vs the true emitter
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
| `--m-per-px 3.52` | map resolution, for the emitter error in meters (the UCF dataset maps cover 900 m in 256 px) |
| `--hit-radius-m 10.56` | an emitter estimate within this distance counts as a hit (3 px) |
| `--timed-iters 500` | timed runs for the latency statistics |

The script writes one set of outputs per run tag (for example `fp16_notx_uniform`) to `--out-dir`:

- `summary_<tag>.csv`: per density, the mean and 95 % bootstrap CI
- `per_sample_<tag>.csv`: one row per map and density
- `run_info_<tag>.json`: device, versions, latency, RAM and power
- `tegrastats_<tag>.log`: raw power and RAM log covering the timed runs
- `masks/`: saved sample masks, reused so every run sees the same samples

**Test data:** the full benchmark uses 200 or more maps that were never used in training (the dataset's test split), plus a set of the team's own Sionna campus maps as a generalization check. The 10 maps in `data/gnn_inference_test/` serve only as a smoke test. Unzipping and running the full test split is covered in [README_Jetson_Setup.md](README_Jetson_Setup.md).

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
6. **Power:** total input power from `tegrastats` (`VDD_IN` where the board reports it, otherwise the sum of `VDD_GPU_SOC`, `VDD_CPU_CV` and `VIN_SYS_5V0`), with energy per inference = W × s. The two readings are not directly comparable between boards.
7. **Thermal soak:** a 10-minute continuous run checks that p95 latency does not drift up.
8. **Concurrency:** GNN + YOLO + depth + PPO + ROS run together, and the latency under load and total RAM are reported.

Proposed budgets (pending agreement with the mentor):

| Component | Budget |
| --- | --- |
| YOLO26n end to end | ≥ 15 FPS, p99 ≤ 100 ms |
| GNN map update | p99 ≤ 200 ms |
| Whole stack RAM on the Orin Nano 8 GB | ≤ 6 GB (no longer binding on the Orin 32 GB) |

## Results

### GNN accuracy (200 held-out test-split maps)

Setup: `GNN_no_tx.pth`, emitter channel zeroed, uniform random samples. Errors are on unobserved, non-building pixels; CIs are 95 % bootstrap intervals over maps. Maps are 3.52 m per pixel (900 m in 256 px), and an emitter estimate within 3 px (10.56 m) counts as a hit. Accuracy is the same on every device and power mode tested (Orin Nano 8 GB and Orin 32 GB, FP16 and FP32, within 0.1 dB); the table is from the Orin 32 GB FP16 run on 2026-11-13.

| Samples | RMSE dB (95 % CI) | Pixels within 6 dB | Emitter error m | Emitter within 10.56 m |
| --- | --- | --- | --- | --- |
| 0.5 % | 19.25 (19.16–19.34) | 23 % | 35.3 | 10 % |
| 1 % | 17.54 (17.44–17.63) | 34 % | 26.1 | 14 % |
| 2 % | 15.00 (14.89–15.10) | 48 % | 19.9 | 20 % |
| 5 % | 11.45 (11.33–11.58) | 64 % | 16.6 | 31 % |
| 10 % | 10.25 (10.12–10.37) | 68 % | 14.6 | 34 % |

Other settings (same maps):

| GNN setting | RMSE dB @ 1 % / 5 % (95 % CI) | Emitter error m @ 1 % / 5 % | Emitter within 10.56 m @ 10 % |
| --- | --- | --- | --- |
| FP16, no tx, uniform samples (main run above) | 17.54 (17.44–17.63) / 11.45 (11.33–11.58) | 26.1 / 16.6 | 34 % |
| FP32, no tx, uniform samples | 17.54 (17.44–17.63) / 11.45 (11.33–11.58) | 26.3 / 16.8 | 32 % |
| FP16, with tx, uniform samples (reference) | 17.63 (17.54–17.72) / 11.51 (11.39–11.65) | 33.3 / 24.1 | 14 % |
| FP16, no tx, walk-shaped samples | 21.14 (21.04–21.24) / 20.41 (20.30–20.52) | 253.4 / 103.0 | 8 % |

### GNN on-device cost, Jetson Orin 32 GB (2026-11-13)

Setup: JetPack 7.2 (L4T R39.2.1), PyTorch 2.15.0 (CUDA 13.2), host install, `jetson_clocks`, 500 timed runs on real maps at 2 % samples after 50 warm-up runs. Power and RAM come from a `tegrastats` log covering the timed runs; peak RAM is for the whole Jetson, including the desktop.

| GNN setting | Power mode | p50 / p99 ms | Maps/s | Power mean / max W | Energy per map J | Peak RAM GB |
| --- | --- | --- | --- | --- | --- | --- |
| FP16, no tx, uniform | MAXN | 10.6 / 11.8 | 94 | 24.4 / 25.0 | 0.26 | 5.3 |
| FP32, no tx, uniform | MAXN | 8.9 / 11.1 | 110 | 27.3 / 28.4 | 0.25 | 5.2 |
| FP16, no tx, walk | MAXN | 10.5 / 12.1 | 95 | 24.2 / 25.2 | 0.26 | 5.4 |
| FP16, with tx, uniform | MAXN | 11.3 / 12.8 | 89 | 24.4 / 25.1 | 0.27 | 5.5 |
| FP16, no tx, uniform | 30 W | 18.9 / 26.4 | 53 | 15.6 / 16.3 | 0.29 | 7.1 |
| FP32, no tx, uniform | 30 W | 18.4 / 26.0 | 54 | 16.3 / 16.8 | 0.30 | 6.7 |
| FP16, no tx, walk | 30 W | 15.2 / 23.6 | 62 | 14.4 / 15.4 | 0.23 | 7.2 |
| FP16, with tx, uniform | 30 W | 19.0 / 26.2 | 53 | 15.3 / 16.3 | 0.29 | 7.3 |

- **MAXN vs 30 W:** MAXN is about 1.8× faster with a much tighter tail (p99 under 13 ms in every setting). It draws more power, but energy per map is slightly lower because each map finishes sooner. 30 W is the lower-power option for battery operation and is still about 7× under the 200 ms budget at p99.
- **FP32 vs FP16:** at MAXN, FP32 is faster than FP16 (8.9 vs 10.6 ms p50) with identical accuracy. The model is small (about 31 MB of GPU memory), so the cost of autocasting to half precision outweighs any gain. The ROS 2 node therefore runs the GNN in FP32 by default.
- **Compared with the Orin Nano 8 GB** (14.8 ms p50 at 25 W, below): the Orin 32 GB at MAXN is about 1.4× faster in FP16 and 1.6× faster in FP32.

### GNN, Jetson Orin Nano 8 GB (2026-11-06)

Setup: `GNN_no_tx.pth`, PyTorch FP16, emitter channel zeroed, 200 held-out test-split maps, uniform random samples, 25 W power mode with `jetson_clocks`. Errors are on unobserved, non-building pixels; CIs are 95 % bootstrap intervals over maps. **The emitter columns in this section and the next were computed before the map scale was corrected, assuming 1 m per pixel and a 5 m (5 px) hit radius.** Multiplying the errors by 3.52 gives meters; the accuracy section above has the emitter numbers at the correct scale.

| Samples | RMSE dB (95 % CI) | Pixels within 6 dB | Emitter error px | Emitter within 5 px |
| --- | --- | --- | --- | --- |
| 0.5 % | 19.25 (19.16–19.34) | 23 % | 10.0 | 26 % |
| 1 % | 17.54 (17.44–17.63) | 34 % | 7.4 | 31 % |
| 2 % | 15.00 (14.89–15.10) | 48 % | 5.6 | 48 % |
| 5 % | 11.45 (11.33–11.58) | 64 % | 4.7 | 62 % |
| 10 % | 10.25 (10.12–10.37) | 68 % | 4.1 | 71 % |

| On-device | Value |
| --- | --- |
| Latency p50 / p99 | 14.8 / 20.9 ms (65 maps/s) |
| Power, mean / max during the run | 7.9 / 12.0 W (about 0.18 J per map at max power) |
| Peak system RAM (whole Jetson) | 3.6 GB of 8 GB |
| Model GPU memory | 31 MB |

The GNN runs about 10× under the 200 ms budget. Map quality improves steadily with sampling density, and the emitter is located within 5 px in 71 % of maps at 10 % samples.

### GNN, other settings (same 200 maps and hardware, 2026-11-06)

| GNN setting | RMSE dB @ 1 % / 5 % (95 % CI) | Emitter error px @ 1 % / 5 % | Emitter within 5 px @ 10 % | p50 / p99 ms | Power mean / max W | Peak RAM GB |
| --- | --- | --- | --- | --- | --- | --- |
| FP16, no tx, uniform samples (main run above) | 17.54 (17.44–17.63) / 11.45 (11.33–11.58) | 7.4 / 4.7 | 71 % | 14.8 / 20.9 | 7.9 / 12.0 | 3.6 |
| FP32, no tx, uniform samples | 17.54 (17.44–17.63) / 11.45 (11.33–11.58) | 7.5 / 4.8 | 69 % | 14.6 / 17.9 | 9.2 / 13.7 | 3.4 |
| FP16, with tx, uniform samples (reference) | 17.63 (17.54–17.72) / 11.51 (11.39–11.65) | 9.5 / 6.8 | 39 % | 14.7 / 15.1 | 8.8 / 11.5 | 3.5 |
| FP16, no tx, walk-shaped samples | 21.10 (21.00–21.20) / 20.28 (20.17–20.38) | 63.7 / 27.3 | 26 % | 14.3 / 14.9 | 7.9 / 12.4 | 3.6 |
| TensorRT FP16 engine, no tx (`trtexec`, engine only) | not scored | | | 77.5 / 82.4 | | |
| TensorRT FP32 engine, no tx (`trtexec`, engine only) | not scored | | | 80.4 / 84.8 | | |

- **FP32 vs FP16:** same accuracy and same speed. The model is small (31 MB of GPU memory), so half precision gives no speedup in PyTorch; TensorRT is the remaining option for a faster GNN.
- **TensorRT:** the engine builds on TensorRT 10.3 (after fixing the missing LayerNorm in `models_radiomap.py` and calling `trtexec` directly, since `onnx_to_engine.py` passes `--shapes`, which TensorRT 10 rejects for a static model), but it is about 5× slower than PyTorch (77.5 ms vs 14.8 ms p50). A `trtexec` layer profile shows why: the kNN distance computation in each of the 4 ViG blocks takes about 23 ms (96 % of engine time), because TensorRT runs it as an elementwise operation over all 1,024 × 1,024 patch pairs; every other layer together takes under 4 ms. The GNN is therefore deployed with PyTorch.
- **With tx:** map error is unchanged (within 0.1 dB), which confirms the model ignores the emitter channel. Feeding a non-zero emitter pixel makes localization worse, because the model never saw one in training, so the channel stays zeroed at deployment.
- **Walk-shaped samples** (contiguous paths, as a driving rover collects them) are much harder: error stays near 20 dB and improves slowly with more samples. Part of this is inherent, since a path leaves large areas unobserved, and part is that the model was trained only on scattered samples. Fine-tuning with path-shaped sample masks is the main next step for the GNN.
- Power and RAM come from a `tegrastats` log covering each whole run (one log per run). Peak RAM is for the whole Jetson, including the desktop.

### YOLO26n

| YOLO26n version | COCO mAP50-95 | Rover recall 0–3 / 3–6 / 6+ m | Depth MAE m | p50 / p99 ms | RAM MB | Power W |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch | | | | | | |
| TensorRT FP16 | | | | | | |
| TensorRT INT8 | | | | | | |
