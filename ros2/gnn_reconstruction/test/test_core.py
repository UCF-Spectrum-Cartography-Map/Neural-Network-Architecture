"""Checks the ROS-free core against the offline benchmark's model on a real map (no ROS needed).
Run from the package folder:  GNN_WEIGHTS=... GNN_MAP=... python3 -m pytest test/ -q"""
import os

import numpy as np
import pandas as pd
import pytest

from gnn_reconstruction.core import (H, W, MAX_DB, MIN_DB, Reconstructor, SampleGrid, dataset_to_grid,
                                     load_building_mask, make_backend)

WEIGHTS = os.environ.get("GNN_WEIGHTS", "")
MAP = os.environ.get("GNN_MAP", "")
ONNX = os.environ.get("GNN_ONNX", "")
need = pytest.mark.skipif(not (os.path.exists(WEIGHTS) and os.path.exists(MAP)), reason="set GNN_WEIGHTS and GNN_MAP")


def test_grid_roundtrip():
    g = SampleGrid(origin_x=-10.0, origin_y=5.0, m_per_px=0.5)
    assert g.to_pixel(*g.to_xy(17, 200)) == (17, 200)
    assert g.add(-10.0 + 0.25, 5.0 + 0.25, 80.0) and g.add(-10.0 + 0.3, 5.0 + 0.3, 90.0)
    assert g.mean_db()[0, 0] == pytest.approx(85.0) and g.n_samples == 2
    assert not g.add(-11.0, 5.0, 80.0)


def test_dataset_mask_is_flipped_into_the_grid_layout(tmp_path):
    """A .parquet mask is stored with row 0 at the highest y; the grid has row 0 at the lowest y."""
    env = np.zeros((H, W), np.float32)
    env[10:30, 40:90] = 1                                   # a building near the TOP of the file = high y
    pq = str(tmp_path / "map.parquet")
    pd.DataFrame({"building_mask": [env.ravel()]}).to_parquet(pq)
    loaded = load_building_mask(pq)
    assert loaded.shape == (H, W) and loaded.dtype == np.float32
    assert loaded[H - 30:H - 10, 40:90].all() and loaded.sum() == env.sum()   # high grid rows = high y
    assert not loaded[10:30, 40:90].any()
    assert np.array_equal(dataset_to_grid(loaded), env)     # the same call converts back
    npy = str(tmp_path / "mask.npy")
    np.save(npy, loaded)
    assert np.array_equal(load_building_mask(npy), loaded)  # .npy is taken as already in the grid layout


def test_model_gets_dataset_rows_and_output_returns_in_grid_rows():
    """A sample taken at a high y must reach the model near the top of its input (dataset layout),
    and what the model predicts there must come back at that same high y."""
    seen = {}

    class FakeBackend:
        name = "fake"

        def __call__(self, x):
            assert x.shape == (1, 3, H, W) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
            seen["x"] = x.copy()
            out = np.ones((1, 1, H, W), np.float32)
            r, c = np.argwhere(x[0, 2] > 0)[0]
            out[0, 0, r, c] = 0.0                            # "emitter" exactly at the sampled pixel
            return out

    env_file = np.zeros((H, W), np.float32)
    env_file[0:20, 0:20] = 1                                # building at the top-left of the file
    grid = SampleGrid(origin_x=-450.0, origin_y=-447.9, m_per_px=3.52)
    x_m, y_m = -450.0 + 100.5 * 3.52, -447.9 + 200.5 * 3.52  # column 100, 200 pixels ABOVE the bottom edge
    assert grid.add(x_m, y_m, 75.0) and grid.to_pixel(x_m, y_m) == (200, 100)
    pl_db, (row, col), (ex, ey), _ = Reconstructor(FakeBackend(), dataset_to_grid(env_file), grid).run()

    x = seen["x"][0]
    assert np.array_equal(x[0], env_file)                    # model sees the mask exactly as the file stores it
    assert np.argwhere(x[2] > 0).tolist() == [[H - 1 - 200, 100]]   # high y -> near the top of the model input
    assert x[2, H - 1 - 200, 100] == pytest.approx((75.0 - MIN_DB) / (MAX_DB - MIN_DB))
    assert (row, col) == (200, 100) and pl_db[200, 100] == pytest.approx(MIN_DB)   # back in the grid layout
    assert (ex, ey) == pytest.approx((x_m, y_m))            # the estimate lands where the sample was taken


def fill(grid, gt, env, seed=0, pct=5):
    """Samples at map-frame positions; gt and env are in the grid layout."""
    rng = np.random.default_rng(seed)
    free = np.argwhere(env < 0.5)
    for r, c in free[rng.choice(len(free), int(pct / 100 * H * W), replace=False)]:
        grid.add(*grid.to_xy(r, c), float(gt[r, c]))


@need
def test_matches_offline_reconstruction():
    import torch
    from gnn_reconstruction.model import load_torch_model
    row = pd.read_parquet(MAP).iloc[0]
    gt_file = np.asarray(row["path_loss"], np.float32).reshape(H, W)       # dataset layout, as stored
    env_file = np.asarray(row["building_mask"], np.float32).reshape(H, W)
    gt, env = dataset_to_grid(gt_file), load_building_mask(MAP)             # grid layout, as the node sees them
    grid = SampleGrid(origin_x=-450.0, origin_y=-447.9, m_per_px=3.52)
    fill(grid, gt, env)
    pl_db, (r, c), _, _ = Reconstructor(make_backend("torch", WEIGHTS, fp16=False), env, grid).run()

    # Offline path, written out independently (as in benchmark_reconstruction.py): everything in the
    # dataset layout, straight from the file, with the same pixels sampled.
    mask_file = dataset_to_grid(grid.mask)
    sparse = np.zeros((H, W), np.float32)
    sparse[mask_file] = np.clip((gt_file[mask_file] - 30) / 110, 0, 1)
    x = torch.from_numpy(np.stack([env_file, np.zeros_like(env_file), sparse])[None])
    with torch.no_grad():
        ref_file = load_torch_model(WEIGHTS, "cpu")(x).numpy().reshape(H, W) * 110 + 30
    ref = dataset_to_grid(ref_file)
    assert np.abs(pl_db - ref).max() < 1e-3
    score = ~grid.mask & (env < 0.5)
    assert np.sqrt(np.mean((pl_db[score] - gt[score]) ** 2)) < 20  # sane map, not noise


@need
@pytest.mark.skipif(not os.path.exists(ONNX), reason="set GNN_ONNX")
def test_onnx_backend_matches_torch():
    env = load_building_mask(MAP)
    gt = np.asarray(pd.read_parquet(MAP).iloc[0]["path_loss"], np.float32).reshape(H, W)
    grid = SampleGrid()
    fill(grid, gt, env, seed=1)
    a = Reconstructor(make_backend("torch", WEIGHTS, fp16=False), env, grid).run()[0]
    b = Reconstructor(make_backend("onnx", ONNX), env, grid).run()[0]
    assert np.abs(a - b).mean() < 0.1
