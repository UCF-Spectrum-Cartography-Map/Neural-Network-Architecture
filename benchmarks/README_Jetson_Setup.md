# Jetson Setup

This is the single setup guide for running the team's models on the Jetson: installing PyTorch and ROS 2, getting the test maps, and running the GNN benchmark. No training happens on the Jetson; it only runs inference with already-trained weights (`GNN_no_tx.pth`).

Related documents, each covering one thing:

| Document | Covers |
| --- | --- |
| this file | Setting up a Jetson and running the benchmark |
| [README_Benchmark.md](README_Benchmark.md) | What the benchmark measures, the protocol, and all results |
| [ros2/gnn_reconstruction/README.md](../ros2/gnn_reconstruction/README.md) | Building and running the ROS 2 GNN + Kalman tracker nodes, replay figures |
| [ros2/gnn_reconstruction/INTERFACE.md](../ros2/gnn_reconstruction/INTERFACE.md) | ROS topics and parameters, for teammates connecting to the nodes |

The main board is the **Jetson Orin 32 GB with JetPack 7.2** (sections 1 to 4). The older Orin Nano 8 GB with JetPack 6 needs a different setup, kept in section 6 because some results in README_Benchmark.md were measured on it. `cat /etc/nv_tegra_release` shows which JetPack a board has (R39 = JetPack 7, R36 = JetPack 6).

## 1. Code and test maps

```bash
sudo apt update && sudo apt install -y git unzip python3-pip
git clone https://github.com/UCF-Spectrum-Cartography-Map/Neural-Network-Architecture.git ~/Neural-Network-Architecture
ln -s ~/Neural-Network-Architecture ~/nna          # all commands use ~/nna
```

The benchmark needs maps the model never saw in training, as `.parquet` files (one map per file, columns `building_mask`, `tx_origin`, `path_loss`): the held-out test split of the Hugging Face dataset `KR-init/Spectrum-Cartography-256x256-UCF-50K`, or the team's own Sionna maps in the same format. The 10 maps in `data/gnn_inference_test/` are only a smoke test. A zipped test split is unzipped into the repository:

```bash
mkdir -p ~/nna/data/test_split
unzip -q /path/to/test_static.zip -d ~/nna/data/test_split
find ~/nna/data/test_split -name '*.parquet' | wc -l      # 200 or more
```

## 2. PyTorch (host, no containers)

The jetson-containers images are built for JetPack 5 and 6 and fail on JetPack 7 with a CUDA driver mismatch, so PyTorch is installed directly on the host. PyTorch's CUDA 13.2 build for ARM servers runs on JetPack 7.2 and sees the Orin GPU. Ubuntu 24.04 requires `--break-system-packages` for `pip --user` installs (they only write to `~/.local`). numpy stays below 2 to match ROS 2 Jazzy, and pyarrow stays at 17.0.0 because newer releases require numpy 2.

```bash
pip3 install --user --break-system-packages torch --index-url https://download.pytorch.org/whl/cu132
pip3 install --user --break-system-packages "numpy<2" pandas "pyarrow==17.0.0"
python3 -c "import torch, numpy, pyarrow; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, pyarrow.__version__)"
```

The last line should print something like `2.15.0+cu132 True 1.26.4 17.0.0`. A warning about compute capability 8.7 is harmless. `run_benchmarks_orin.sh` also runs these installs by itself if anything is missing.

## 3. ROS 2 Jazzy

Ubuntu 24.04 uses ROS 2 Jazzy (not Humble). The ROS nodes use the same host PyTorch. NVIDIA's JetPack 7 image may already include the ROS package source, so it is only added if none exists:

```bash
ls /etc/apt/sources.list.d/ | grep -i ros || {
  sudo apt install -y software-properties-common curl
  sudo add-apt-repository -y universe
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list
}
sudo apt update && sudo apt install -y ros-jazzy-desktop ros-jazzy-rosbag2 python3-colcon-common-extensions python3-matplotlib python3-pytest
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && source ~/.bashrc
```

Building and running the nodes continues in [ros2/gnn_reconstruction/README.md](../ros2/gnn_reconstruction/README.md).

## 4. Running the GNN benchmark

`run_benchmarks_orin.sh` runs the four standard settings (FP16, FP32, walk-shaped samples, with tx) on 200 maps. It records the power mode, runs `jetson_clocks`, logs power with tegrastats during each timed run, and prints the result tables at the end:

```bash
sudo nvpmodel -m 0                         # MAXN; any mode works, it is recorded either way
bash ~/nna/benchmarks/run_benchmarks_orin.sh 2>&1 | tee ~/nna/benchmarks/orin32_run.log
```

Results go to `benchmarks/gnn_bench_orin32_<MMDD>/` (`summary_*.csv`, `per_sample_*.csv`, `run_info_*.json`, `tegrastats_*.log`, `power_summary.txt`). A second run on the same day writes to the same folder, so earlier results are renamed first. Individual runs and all options of `benchmark_reconstruction.py` are described in [README_Benchmark.md](README_Benchmark.md).

TensorRT is not set up on this board, since PyTorch is faster for this model (see README_Benchmark.md).

## 5. Troubleshooting

| Problem | Fix |
| --- | --- |
| `error: externally-managed-environment` from pip | Add `--user --break-system-packages` |
| `pyarrow requires NumPy 2.0 or newer` | `pip3 install --user --break-system-packages "pyarrow==17.0.0"` |
| `CUDA available: False` | Reinstall torch from the `cu132` index (section 2) |
| `Conflicting values set for option Signed-By ... packages.ros.org` | The ROS source was added twice: `sudo rm /etc/apt/sources.list.d/ros2.list && sudo apt update` |
| `colcon: command not found` | ROS is not installed yet (section 3) |
| numpy errors from ROS | `pip3 install --user --break-system-packages "numpy<2"` |
| "No tegrastats logs found" | The benchmark runs failed; the lines above the message show why |

## 6. Legacy: Orin Nano 8 GB (JetPack 6)

These steps were used for the Orin Nano results in README_Benchmark.md. On JetPack 6, `pip install torch` gives a CPU-only build, so the benchmark runs inside NVIDIA's `l4t-pytorch` container and the ROS 2 node runs on the host with ROS 2 Humble.

**Benchmark in the container.** `autotag` picks the image matching the installed JetPack. The container is deleted on exit, so pandas and pyarrow are installed each time (from PyPI directly, since the container's default mirror is not always reachable). `sudo` and `tegrastats` are not available inside the container, so clocks and power logging are handled on the host before and after:

```bash
git clone https://github.com/dusty-nv/jetson-containers && bash jetson-containers/install.sh
sudo nvpmodel -q && sudo jetson_clocks
sudo tegrastats --interval 100 --logfile ~/nna/benchmarks/teg_host.log --start
jetson-containers run -v ~/nna:/work $(autotag l4t-pytorch)
# inside the container:
pip install --index-url https://pypi.org/simple pandas pyarrow
cd /work && TEST_DIR=$(dirname "$(find data/test_split -name '*.parquet' | head -1)")
python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth --test-dir "$TEST_DIR" --out-dir benchmarks/gnn_bench_test --fp16 --max-files 200
exit
# back on the host:
sudo tegrastats --stop && python3 ~/nna/benchmarks/teg_summary.py ~/nna/benchmarks/teg_host.log
```

**TensorRT engine (comparison only).** On the Orin Nano the engine is about 5x slower than PyTorch FP16 (about 78 ms vs 15 ms per map), because the kNN distance layers take 96 % of its time. `benchmarks/models_radiomap.py` must include the PatchEmbed LayerNorm fix, otherwise `pth_to_onnx.py` exports a model that does not match the trained weights (outputs around 230 dB) while its own parity check still passes. `onnx_to_engine.py` passes `--shapes`, which TensorRT 10 rejects for a static model, so `trtexec` is called directly:

```bash
# inside the container
cd /work && mkdir -p checkpoints onnx engines && cp benchmarks/GNN_no_tx.pth checkpoints/GNN_best.pth
python3 benchmarks/pth_to_onnx.py --weights-dir checkpoints --out-dir onnx --models GNN --static && exit
# on the host (folders made in the container belong to root)
sudo chown -R $USER:$USER ~/nna/checkpoints ~/nna/onnx ~/nna/engines && cd ~/nna
/usr/src/tensorrt/bin/trtexec --onnx=onnx/GNN.onnx --saveEngine=engines/GNN.engine --fp16 --memPoolSize=workspace:1024M
```

The engine is scored with `benchmark_reconstruction.py --engine engines/GNN.engine`.

**Host setup for the ROS 2 node (Humble).** The host gets NVIDIA's PyTorch wheel for JetPack 6.1 plus the cuSPARSELt library it needs. numpy stays below 1.25 because the Humble packages are built against numpy 1.x:

```bash
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install --user "numpy<1.25" ./torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip3 install --user pandas pyarrow
wget https://developer.download.nvidia.com/compute/cusparselt/0.6.3/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.6.3_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.6.3/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get install -y libcusparselt0 libcusparselt-dev
```

## Notes

- The Orin shares its RAM between CPU and GPU, so `tegrastats` RAM is the number that matters, not GPU memory alone.
- `GNN_no_tx.pth` was trained with the emitter channel zeroed, so the benchmark zeroes it too. `--use-tx` exists only as a reference run.
- The PPO and YOLO models are deployed the same way: the trained file is copied over and only inference runs on the Jetson.
