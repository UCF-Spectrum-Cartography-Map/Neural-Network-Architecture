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

Inside the container, the GPU check should print `True`, and the remaining packages are installed. The container's default pip mirror (`jetson.webredirect.org`) is not always reachable, so they are installed from PyPI directly. The container is deleted on exit, so this step is repeated each time it starts:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pip install --index-url https://pypi.org/simple pandas pyarrow
```

If PyPI also fails with `Name or service not known`, the container has no network access, and the Jetson's own internet connection needs checking first.

## 2. Test data

The test maps go in `data/gnn_inference_test/` as `.parquet` files, one map per file, with the columns `building_mask`, `tx_origin` and `path_loss`. They must be maps the model never saw in training. Two sources:

- the held-out test split of the Hugging Face dataset `KR-init/Spectrum-Cartography-256x256-UCF-50K`
- the team's own Sionna maps, in the same format

A zipped test split is unzipped on the Jetson host into the mounted repository (`~/nna`), because the container cannot see folders outside it:

```bash
mkdir -p ~/nna/data/test_split
unzip /path/to/test.zip -d ~/nna/data/test_split
```

## 3. Running the benchmark

`sudo` and `tegrastats` are not available inside the container, so clocks are locked and power logging is started on the Jetson host first:

```bash
sudo nvpmodel -q          # records the power mode (for example 25W); the same mode is used for every run
sudo jetson_clocks
sudo tegrastats --interval 100 --logfile ~/nna/benchmarks/teg_host.log --start
```

The benchmark then runs from the repository root inside the container:

```bash
cd /work
python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth \
    --test-dir data/gnn_inference_test --out-dir benchmarks/gnn_bench --fp16
```

For the unzipped test split, the folder that holds the `.parquet` files is found automatically, so no path needs to be typed:

```bash
TEST_DIR=$(dirname "$(find data/test_split -name '*.parquet' | head -1)"); echo "Using $TEST_DIR"
python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth --test-dir "$TEST_DIR" --out-dir benchmarks/gnn_bench_test --fp16 --max-files 200
exit
```

If the first line prints `Using .`, no `.parquet` files were found under `data/test_split`. Dropping `--max-files 200` runs every map in the split.

Back on the host, logging is stopped and the power and RAM figures are read from the log. The log also covers idle time before and after the run, so the peak values are the reliable ones:

```bash
sudo tegrastats --stop
grep -o 'VDD_IN [0-9]*' ~/nna/benchmarks/teg_host.log | awk '{s+=$2; if($2>m)m=$2} END{print "power mean", s/NR/1000, "W, max", m/1000, "W"}'
grep -o 'RAM [0-9]*' ~/nna/benchmarks/teg_host.log | awk '{if($2>m)m=$2} END{print "peak RAM", m, "MB"}'
```

See [README_Benchmark.md](README_Benchmark.md) for the full procedure, the metrics and the results.

## 4. TensorRT engine (optional)

The TensorRT engine was measured as a comparison only: on the Orin Nano it is about 5x slower than PyTorch FP16 (about 78 ms vs 15 ms per map), because the kNN distance layers take 96 % of its time. The deployed node uses PyTorch.

`benchmarks/models_radiomap.py` must include the PatchEmbed LayerNorm fix, otherwise `pth_to_onnx.py` exports a model that does not match the trained weights (outputs around 230 dB) while its own parity check still passes. `pth_to_onnx.py` looks for `GNN_best.pth`, so the no-tx weights are copied under that name first. Inside the container:

```bash
cd /work
mkdir -p checkpoints onnx engines
cp benchmarks/GNN_no_tx.pth checkpoints/GNN_best.pth
python3 benchmarks/pth_to_onnx.py --weights-dir checkpoints --out-dir onnx --models GNN --static
exit
```

Folders created inside the container belong to root, so on the host they are handed back to the user before building the engine. `onnx_to_engine.py` passes `--shapes`, which TensorRT 10 rejects for a static model, so `trtexec` is called directly on the host:

```bash
sudo chown -R $USER:$USER ~/nna/checkpoints ~/nna/onnx ~/nna/engines
cd ~/nna
/usr/src/tensorrt/bin/trtexec --onnx=onnx/GNN.onnx --saveEngine=engines/GNN.engine --fp16 --memPoolSize=workspace:1024M
```

The engine is timed with `benchmarks/benchmark_reconstruction.py --engine engines/GNN.engine`, or with `trtexec --loadEngine=engines/GNN.engine --warmUp=2000 --iterations=500 --percentile=50,95,99`.

## 5. Host setup for the ROS 2 node

The jetson-containers PyTorch image has no ROS, so the ROS 2 node (`ros2/gnn_reconstruction`) runs on the Jetson host, which has ROS 2 Humble. The host gets NVIDIA's PyTorch wheel for JetPack 6.1, plus the cuSPARSELt library that the wheel needs. numpy stays below 1.25 because the Humble packages are built against numpy 1.x; a pip-installed `opencv-contrib-python` 5 conflicts with that pin and is replaced by a version below 4.11 if present.

```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install --user "numpy<1.25" ./torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install --user pandas pyarrow

wget https://developer.download.nvidia.com/compute/cusparselt/0.6.3/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.6.3/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get install -y libcusparselt0 libcusparselt-dev

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line should print `True`. Building and running the node is described in [ros2/gnn_reconstruction/README.md](../ros2/gnn_reconstruction/README.md).

## Notes

- The Orin shares its RAM between CPU and GPU, so `tegrastats` RAM is the number that matters, not GPU memory alone.
- `GNN_no_tx.pth` was trained with the emitter channel zeroed, so the benchmark zeroes it too. The `--use-tx` flag exists only as a reference run.
- Each run gets its own tegrastats log name (for example `teg_fp16.log`, `teg_fp32.log`), so the power figures can be matched to the run.
- The PPO and YOLO models are tested the same way: the trained file is copied over and only inference runs on the Jetson.
