# Upsampling decoder for the ViG GNN (SCRUM-93)

The GNN's output head (`PatchUnembed`) is one `ConvTranspose2d` with kernel = stride = 8, so every
8 x 8 block of the 256 x 256 map comes from a single graph node and the map looks blocky.
`UpsampledViG` replaces that last step with a PixelShuffle decoder (32 -> 64 -> 128 -> 256) that also
sees the raw input through skip connections. Input and output are unchanged:
`(1, 3, 256, 256)` in, `(1, 1, 256, 256)` out, normalized as `(dB - 30) / 110` (the contract in
`ros2/gnn_reconstruction/INTERFACE.md`).

| File | What it is |
| --- | --- |
| `vig_upsampler.py` | Model (`UpsampledViG`), loss, optimizer, `blockiness()` metric |
| `train_upsampler.py` | Training, and scoring with `benchmarks/benchmark_reconstruction.py` functions |
| `map_scale.py` | Meters per pixel per dataset, pose <-> pixel conversion, dataset <-> grid row flip |
| `test_upsampler.py` | Tests (run from the repo root) |
| `benchmarks/GNN_upsampler.pth` | Trained checkpoint (1 Nov 2026 run). It holds the decoder AND its own GNN weights. |

## Run

```bash
python -m pytest test_upsampler.py -q                                   # tests
python train_upsampler.py --eval-only --ckpt benchmarks/GNN_upsampler.pth --out /tmp/upsampler_eval          # uniform masks
python train_upsampler.py --eval-only --ckpt benchmarks/GNN_upsampler.pth --out /tmp/upsampler_eval --mask-kind walk
```

Needs `torch`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `pytest`. `trimesh` and `shapely` are
optional (one landmark test skips without them).

## Result (10 maps in `data/gnn_inference_test`, uniform samples, 3.52 m/px, CPU FP32)

| Samples | RMSE GNN (dB) | RMSE decoder (dB) | Blockiness GNN | Blockiness decoder |
| --- | --- | --- | --- | --- |
| 0.5 % | 18.85 | 10.57 | 15.2 | 1.78 |
| 1 % | 17.21 | 10.02 | 19.5 | 1.68 |
| 2 % | 14.83 | 9.82 | 18.9 | 1.49 |
| 5 % | 11.38 | 9.60 | 13.5 | 1.23 |
| 10 % | 10.28 | 9.91 | 11.0 | 1.10 |

Blockiness = mean step across 8 px patch boundaries / mean step inside patches. Ground truth = 1.00.

## Use in the ROS node

The `torch` backend only loads the plain GNN and rejects this checkpoint. Export the whole model to
ONNX and run the node with `backend:=onnx` (or build a TensorRT engine from the ONNX on the Jetson):

```python
import copy, sys, torch
sys.path.insert(0, "benchmarks")
import benchmark_reconstruction as br
from vig_upsampler import UpsampledViG
gnn = br.load_model("benchmarks/GNN_no_tx.pth", torch.device("cpu"))
model = UpsampledViG(copy.deepcopy(gnn), feat_ch=64, vig_in_ch=3).eval()
model.load_state_dict(torch.load("benchmarks/GNN_upsampler.pth", map_location="cpu"), strict=True)
torch.onnx.export(model, torch.zeros(1, 3, 256, 256), "gnn_upsampler.onnx", input_names=["input"],
                  output_names=["output"], opset_version=17, dynamo=False)
```

## Known limits

- The output is smoother than the ground truth: the grid is gone but fine ray streaks are not recovered.
  At 10 % samples the RMSE gain is 0.4 dB and the within-6 dB share is lower than the GNN's.
- Emitter localization is no better than the baseline GNN with its samples pasted back in.
- Trained on `dataset_256_256_diverse` (about 1 m/px), not on UCF maps (3.52 m/px).
- The decoder only works with the GNN weights inside the checkpoint. On `GNN_no_tx.pth` weights it
  scores 13.8 to 19.7 dB.
- Path-shaped (walk) samples are not solved (SCRUM-104).
- Not measured on the Jetson (T4 FP16: about 10 to 12 ms per map against 6 to 8 ms for the GNN).

## Row orientation

The dataset files store row 0 at the highest y (the generator applies `np.flipud`). The ROS grid has
row 0 at the lowest y. The team decision (16 Nov 2026) is to keep the grid as it is, flip a dataset
building mask when the ROS package loads it, and flip rows around the model call so the model always
sees the layout it was trained on. That change lives in `ros2/gnn_reconstruction` (branch
`ros-row-convention`), not in these files. `map_scale.dataset_to_grid()` does the same flip for
offline code, and `test_dataset_rows_are_top_down` checks the convention against the UCF scene meshes.
