"""Checks the ROS-free core against the offline benchmark's model on a real map (no ROS needed).
Run from the package folder:  GNN_WEIGHTS=... GNN_MAP=... python3 -m pytest test/ -q"""
import os

import numpy as np
import pandas as pd
import pytest

from gnn_reconstruction.core import H, W, Reconstructor, SampleGrid, load_building_mask, make_backend

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


def fill(grid, gt, env, seed=0, pct=5):
    rng = np.random.default_rng(seed)
    free = np.argwhere(env < 0.5)
    for r, c in free[rng.choice(len(free), int(pct / 100 * H * W), replace=False)]:
        grid.add(*grid.to_xy(r, c), float(gt[r, c]))


@need
def test_matches_offline_reconstruction():
    import torch
    from gnn_reconstruction.model import load_torch_model
    row = pd.read_parquet(MAP).iloc[0]
    gt = np.asarray(row["path_loss"], np.float32).reshape(H, W)
    env = load_building_mask(MAP)
    grid = SampleGrid()
    fill(grid, gt, env)
    pl_db, (r, c), _, _ = Reconstructor(make_backend("torch", WEIGHTS, fp16=False), env, grid).run()

    # Offline path, written out independently (as in benchmark_reconstruction.py).
    sparse = np.zeros((H, W), np.float32)
    sparse[grid.mask] = np.clip((gt[grid.mask] - 30) / 110, 0, 1)
    x = torch.from_numpy(np.stack([env, np.zeros_like(env), sparse])[None])
    with torch.no_grad():
        ref = load_torch_model(WEIGHTS, "cpu")(x).numpy().reshape(H, W) * 110 + 30
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
