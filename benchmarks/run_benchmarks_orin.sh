#!/bin/bash
# GNN benchmark on the new Jetson (JetPack 7.2 / Ubuntu 24.04 / CUDA 13). Runs on the Jetson HOST with
# PyTorch installed by pip: no Docker, no jetson-containers (those images do not run on JetPack 7).
# 4 runs: FP16, FP32, FP16 with walk-shaped samples, FP16 with tx. Usage:
#   bash ~/nna/benchmarks/run_benchmarks_orin.sh
# Results go to ~/nna/benchmarks/gnn_bench_orin32_<MMDD>/ (summary_*.csv, per_sample_*.csv,
# run_info_*.json, tegrastats_*.log, power_summary.txt).
set -euo pipefail
NNA=~/nna
OUT=benchmarks/gnn_bench_orin32_$(date +%m%d)
cd "$NNA"
TEST_DIR=$(dirname "$(find data/test_split -name '*.parquet' | head -1)")
if [ "$TEST_DIR" = "." ]; then echo "No .parquet files under ~/nna/data/test_split - unzip test_static.zip there first."; exit 1; fi
mkdir -p "$OUT"

# One-time Python setup (skipped once torch sees the GPU). PyTorch's CUDA 13.2 build for ARM servers
# runs on JetPack 7.2. numpy stays <2 so it matches ROS 2 Jazzy's numpy. --break-system-packages is
# Ubuntu 24.04's required flag for pip --user installs; it only touches ~/.local.
if ! python3 -c "import torch, pandas; pd = pandas; import pyarrow.parquet; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "=== installing PyTorch (CUDA 13.2) + pandas + pyarrow into ~/.local (one time, a few GB) ==="
  pip3 install --user --break-system-packages torch --index-url https://download.pytorch.org/whl/cu132
  pip3 install --user --break-system-packages "numpy<2" pandas "pyarrow==17.0.0"
fi
python3 -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))" 2>&1 | grep -v "compute capability"

echo "Test maps: $TEST_DIR   Output: $NNA/$OUT"
sudo nvpmodel -q | tee "$OUT/power_mode.txt"
sudo jetson_clocks

run() {
  local tag=$1; shift
  echo; echo "=== run: $tag ($*) ==="
  python3 benchmarks/benchmark_reconstruction.py --weights benchmarks/GNN_no_tx.pth \
    --test-dir "$TEST_DIR" --out-dir "$OUT" --max-files 200 \
    --m-per-px 3.52 --hit-radius-m 10.56 "$@" || echo "!!! run $tag failed"
}

run fp16 --fp16
run fp32
run walk --fp16 --mask-kind walk
run tx --fp16 --use-tx

echo; echo "=== power and RAM (during the timed runs) ==="
python3 benchmarks/teg_summary.py "$OUT"/tegrastats_*.log | tee "$OUT/power_summary.txt"
echo; echo "=== summaries ==="
for f in "$OUT"/summary_*.csv; do [ -e "$f" ] && { echo "--- $f"; cat "$f"; }; done
echo; echo "Done. Send the output above (or zip $NNA/$OUT) back for the README."
