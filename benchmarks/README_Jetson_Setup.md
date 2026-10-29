# Jetson Setup for the GNN Benchmark

This guide covers running the GNN reconstruction model on the Jetson (Orin Nano 8 GB now, a 32 GB Jetson later). No training happens on the Jetson; it only runs inference with the already-trained weights (`GNN_no_tx.pth`) on stored radio maps.

## 1. PyTorch with GPU support

`pip install torch` installs a CPU-only build on the Jetson, so the benchmark runs inside NVIDIA's container instead:

```bash
cat /etc/nv_tegra_release          # JetPack / L4T version
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

The container is started with the repository mounted. `autotag` picks the image that matches the installed JetPack:

```bash
git clone https://github.com/UCF-Spectrum-Cartography-Map/Neural-Network-Architecture.git ~/nna
jetson-containers run -v ~/nna:/work $(autotag l4t-pytorch)
```

Inside the container, the GPU check should print `True`, and the remaining packages are installed:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install pandas pyarrow
```

## 2. Test data

The test maps go in `data/gnn_inference_test/` as `.parquet` files, one map per file, with the columns `building_mask`, `tx_origin` and `path_loss`. They must be maps the model never saw in training. Two sources:

- the held-out test split of the Hugging Face dataset `KR-init/Spectrum-Cartography-256x256-UCF-50K`
- the team's own Sionna maps, in the same format

## 3. Running the benchmark

From `/work` inside the container:

```bash
sudo nvpmodel -q && sudo jetson_clocks      # outside the container if sudo is unavailable inside
python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth \
    --test-dir data/gnn_inference_test --out-dir benchmarks/gnn_bench --fp16
```

The script logs `tegrastats` automatically during the timing runs. Running `tegrastats` in a second terminal shows memory use live. See [README_Benchmark.md](README_Benchmark.md) for the full procedure, the TensorRT conversion and the metrics.

## Notes

- The Orin shares its RAM between CPU and GPU, so `tegrastats` RAM is the number that matters, not GPU memory alone.
- `GNN_no_tx.pth` was trained with the emitter channel zeroed, so the benchmark zeroes it too. The `--use-tx` flag exists only as a reference run.
- The PPO and YOLO models are tested the same way: the trained file is copied over and only inference runs on the Jetson.
