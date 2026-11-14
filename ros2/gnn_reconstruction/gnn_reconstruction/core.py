"""ROS-free core of the GNN node: sample grid, input building and inference backends.

Kept separate from the ROS 2 code so it can be tested on any machine with numpy (+ torch or
onnxruntime), and so the upsampled GNN can replace the model by adding a backend here.
"""
import os
import time

import numpy as np

H = W = 256
MIN_DB, MAX_DB = 30.0, 140.0  # must match training normalization


# ---------------------------------------------------------------- building mask
def load_building_mask(path):
    """Building mask (H x W, 1 = building) from a map .parquet (building_mask column) or a .npy file."""
    if path.endswith(".npy"):
        env = np.load(path).astype(np.float32)
    else:
        import pandas as pd
        row = pd.read_parquet(path, engine="pyarrow").iloc[0]
        env = np.asarray(row["building_mask"], dtype=np.float32)
    return env.reshape(H, W)


# ---------------------------------------------------------------- samples -> grid
class SampleGrid:
    """Keeps RF samples on the H x W grid. Pixel (row, col) covers
    x in [origin_x + col*res, +res), y in [origin_y + row*res, +res), the same layout as
    nav_msgs/OccupancyGrid. Several samples in one pixel are averaged."""

    def __init__(self, origin_x=0.0, origin_y=0.0, m_per_px=1.0):
        self.origin_x, self.origin_y, self.res = origin_x, origin_y, m_per_px
        self.sum = np.zeros((H, W), np.float64)
        self.count = np.zeros((H, W), np.int32)
        self.n_samples = 0

    def to_pixel(self, x, y):
        col = int(np.floor((x - self.origin_x) / self.res))
        row = int(np.floor((y - self.origin_y) / self.res))
        if 0 <= row < H and 0 <= col < W:
            return row, col
        return None

    def to_xy(self, row, col):
        return (self.origin_x + (col + 0.5) * self.res,
                self.origin_y + (row + 0.5) * self.res)

    def add(self, x, y, path_loss_db):
        px = self.to_pixel(x, y)
        if px is None:
            return False
        self.sum[px] += path_loss_db
        self.count[px] += 1
        self.n_samples += 1
        return True

    @property
    def mask(self):
        return self.count > 0

    def mean_db(self):
        out = np.zeros((H, W), np.float32)
        m = self.mask
        out[m] = self.sum[m] / self.count[m]
        return out

    def clear(self):
        self.sum[:] = 0
        self.count[:] = 0
        self.n_samples = 0


def build_input(env, grid):
    """(1, 3, H, W) float32: building mask, zeroed tx channel, normalized sparse samples."""
    sparse = np.zeros((H, W), np.float32)
    m = grid.mask
    sparse[m] = np.clip((grid.mean_db()[m] - MIN_DB) / (MAX_DB - MIN_DB), 0, 1)
    return np.stack([env, np.zeros_like(env), sparse])[None].astype(np.float32)


def to_db(pred_norm):
    return pred_norm.reshape(H, W) * (MAX_DB - MIN_DB) + MIN_DB


def estimate_emitter(pl_db, env):
    """Pixel with the lowest predicted path loss outside buildings (row, col)."""
    masked = np.where(env > 0.5, np.inf, pl_db)
    return np.unravel_index(int(np.argmin(masked)), masked.shape)


# ---------------------------------------------------------------- backends
class TorchBackend:
    def __init__(self, weights, fp16=True, device=None):
        import torch
        from .model import load_torch_model
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16 and self.device.type == "cuda"
        self.model = load_torch_model(weights, self.device)
        self.name = f"torch-{self.device.type}{'-fp16' if self.fp16 else ''}"

    def __call__(self, x):
        torch = self.torch
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    y = self.model(t).float()
            else:
                y = self.model(t)
        return y.cpu().numpy()


class OnnxBackend:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input = self.sess.get_inputs()[0].name
        self.name = f"onnx-{self.sess.get_providers()[0]}"

    def __call__(self, x):
        return self.sess.run(None, {self.input: x})[0]


class TensorRTBackend:
    """Runs a .engine built on this Jetson (benchmarks/onnx_to_engine.py). Uses torch CUDA
    tensors as the input/output buffers, so it needs a CUDA build of torch."""

    def __init__(self, engine_path):
        import tensorrt as trt
        import torch
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.inp = torch.empty((1, 3, H, W), dtype=torch.float32, device="cuda")
        self.out = torch.empty((1, 1, H, W), dtype=torch.float32, device="cuda")
        self.stream = torch.cuda.Stream()
        self.v3 = hasattr(self.ctx, "execute_async_v3")  # TensorRT 8.5+ / 10
        if self.v3:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            for n in names:
                buf = self.inp if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.out
                self.ctx.set_tensor_address(n, buf.data_ptr())
        self.name = "tensorrt"

    def __call__(self, x):
        torch = self.torch
        with torch.cuda.stream(self.stream):
            self.inp.copy_(torch.from_numpy(x), non_blocking=True)
            if self.v3:
                self.ctx.execute_async_v3(self.stream.cuda_stream)
            else:
                self.ctx.execute_async_v2([self.inp.data_ptr(), self.out.data_ptr()], self.stream.cuda_stream)
        self.stream.synchronize()
        return self.out.cpu().numpy()


def make_backend(kind, path, fp16=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if kind == "torch":
        return TorchBackend(path, fp16=fp16)
    if kind == "onnx":
        return OnnxBackend(path)
    if kind == "tensorrt":
        return TensorRTBackend(path)
    raise ValueError(f"unknown backend {kind!r} (torch | onnx | tensorrt)")


class Reconstructor:
    """Samples in, map + emitter estimate out. One object per area / building mask."""

    def __init__(self, backend, env, grid):
        self.backend, self.env, self.grid = backend, env, grid

    def run(self):
        t0 = time.perf_counter()
        pl_db = to_db(self.backend(build_input(self.env, self.grid)))
        ms = (time.perf_counter() - t0) * 1000
        row, col = estimate_emitter(pl_db, self.env)
        return pl_db, (row, col), self.grid.to_xy(row, col), ms
