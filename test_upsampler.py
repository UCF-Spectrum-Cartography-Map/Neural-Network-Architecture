"""
Tests for the upsampling decoder (SCRUM-93) and path-mask training input (SCRUM-104).

Run from the repo root:
    python -m pytest test_upsampler.py -q
The trained-checkpoint tests use benchmarks/GNN_upsampler.pth when it is in the repo, or:
    UPSAMPLER_CKPT=/path/to/best.pth python -m pytest test_upsampler.py -q

Tests that need the repo's GNN weights, test maps, ROS package or scene meshes skip themselves
when those are missing; the skip reason says what was not found.
"""
import copy
import glob
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import benchmark_reconstruction as br  # noqa: E402
import map_scale  # noqa: E402
import train_upsampler as tu  # noqa: E402
from vig_upsampler import SPARSE_CH, UpsampledViG, blockiness  # noqa: E402

WEIGHTS = os.path.join(ROOT, "benchmarks", "GNN_no_tx.pth")
UPSAMPLER = os.path.join(ROOT, "benchmarks", "GNN_upsampler.pth")
TEST_DIR = os.path.join(ROOT, "data", "gnn_inference_test")
MESH_DIR = os.path.join(ROOT, "data", "mitsuba_scenes", "ucf_campus", "meshes")
ROS_PKG = os.path.join(ROOT, "ros2", "gnn_reconstruction")
CPU = torch.device("cpu")
BARGS = SimpleNamespace(mask_kind="uniform", seed=0, avoid_buildings=False, exclude_buildings=True,
                        m_per_px=3.52, hit_radius_m=10.56)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def gnn():
    """The team's GNN with its real weights when present, random weights otherwise."""
    return br.load_model(WEIGHTS if os.path.exists(WEIGHTS) else "", CPU)


@pytest.fixture(scope="module")
def real_gnn():
    if not os.path.exists(WEIGHTS):
        pytest.skip("benchmarks/GNN_no_tx.pth not found")
    return br.load_model(WEIGHTS, CPU)


@pytest.fixture(scope="module")
def maps():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.parquet")))
    if not files:
        pytest.skip("data/gnn_inference_test/*.parquet not found")
    return [(os.path.basename(f),) + br.load_map(f) for f in files]


@pytest.fixture(scope="module")
def trained(real_gnn):
    """The trained decoder checkpoint, loaded with strict key matching."""
    ckpt = os.environ.get("UPSAMPLER_CKPT") or (UPSAMPLER if os.path.exists(UPSAMPLER) else None)
    if not ckpt:
        pytest.skip("no trained decoder: add benchmarks/GNN_upsampler.pth or set UPSAMPLER_CKPT")
    model = wrap(real_gnn)
    model.load_state_dict(tu.load_state(ckpt, CPU), strict=True)
    return model


def synthetic_input(seed=0, density=0.02):
    rng = np.random.default_rng(seed)
    env = np.zeros((256, 256), np.float32); env[40:90, 100:180] = 1
    sparse = np.zeros((256, 256), np.float32)
    idx = rng.choice(256 * 256, int(density * 256 * 256), replace=False)
    sparse.flat[idx] = rng.uniform(0.2, 0.9, idx.size).astype(np.float32)
    return torch.from_numpy(np.stack([env, np.zeros_like(env), sparse]))[None]


def wrap(gnn, **kw):
    return UpsampledViG(copy.deepcopy(gnn), feat_ch=gnn.patch_embed.proj.out_channels, vig_in_ch=3, **kw).eval()


# ------------------------------------------------------------------ model (SCRUM-93)
def test_output_shape_and_finite(gnn):
    with torch.no_grad():
        y = wrap(gnn)(synthetic_input())
    assert tuple(y.shape) == (1, 1, 256, 256)
    assert torch.isfinite(y).all()


def test_untrained_decoder_equals_gnn(gnn):
    """The decoder head starts at zero, so before training the output is the GNN's own map."""
    x = synthetic_input()
    with torch.no_grad():
        assert torch.allclose(wrap(gnn, dc_mode="none")(x), gnn(x), atol=1e-5)


def test_state_dict_layout(gnn):
    """Checkpoint format: every key is the GNN (vig.*), the decoder (decoder.*) or dc_logit."""
    sd = wrap(gnn).state_dict()
    assert len(sd) == 216
    assert set(k.split(".")[0] for k in sd) == {"vig", "decoder", "dc_logit"}
    assert {k[4:] for k in sd if k.startswith("vig.")} == set(gnn.state_dict())
    assert tuple(sd["decoder.skips.e256.0.weight"].shape) == (16, 4, 3, 3)  # 3 input channels + mask


def test_soft_data_consistency(gnn):
    """out = w*measured + (1-w)*raw on measured pixels, and raw everywhere else."""
    m = wrap(gnn); x = synthetic_input()
    with torch.no_grad():
        out, raw = m(x, return_raw=True)
    meas = x[:, SPARSE_CH:SPARSE_CH + 1]; mask = meas > 0
    w = torch.sigmoid(m.dc_logit)
    assert torch.allclose(out[mask], w * meas[mask] + (1 - w) * raw[mask], atol=1e-6)
    assert torch.equal(out[~mask], raw[~mask])


def test_hard_and_no_data_consistency(gnn):
    x = synthetic_input(); meas = x[:, SPARSE_CH:SPARSE_CH + 1]; mask = meas > 0
    with torch.no_grad():
        hard = wrap(gnn, dc_mode="hard")(x)
        out, raw = wrap(gnn, dc_mode="none")(x, return_raw=True)
    assert torch.allclose(hard[mask], meas[mask], atol=1e-6)
    assert torch.equal(out, raw)


@pytest.mark.parametrize("r,c", [(128, 128), (37, 201), (200, 60)])
def test_decoder_is_spatially_aligned(gnn, r, c):
    """Changing one sample must change the decoder output around THAT pixel (catches any shift)."""
    torch.manual_seed(0)
    m = wrap(gnn, dc_mode="none")
    torch.nn.init.normal_(m.decoder.head.weight, std=0.5)
    x1 = synthetic_input(); x1[0, SPARSE_CH, r, c] = 0.0
    x2 = x1.clone(); x2[0, SPARSE_CH, r, c] = 0.9
    feat = torch.zeros(1, 64, 32, 32)  # same node features for both, so only the skip path differs
    with torch.no_grad():
        d = (m.decoder(feat, torch.cat([x2, (x2[:, 2:3] > 0).float()], 1))
             - m.decoder(feat, torch.cat([x1, (x1[:, 2:3] > 0).float()], 1)))[0, 0].abs().numpy()
    rows, cols = np.nonzero(d > 0.25 * d.max())   # the strong response (GroupNorm leaks a faint global one)
    assert abs((rows.min() + rows.max()) / 2 - r) <= 2 and abs((cols.min() + cols.max()) / 2 - c) <= 2
    pr, pc = np.unravel_index(np.argmax(d), d.shape)
    assert math.hypot(pr - r, pc - c) <= 3


# ------------------------------------------------------------------ map scale
def test_known_dataset_scales():
    assert map_scale.resolve_scale(dataset="ucf_test") == 3.52
    assert map_scale.resolve_scale(dataset="diverse") == 1.0


def test_scale_from_path():
    assert map_scale.resolve_scale(path="data/gnn_inference_test") == 3.52
    assert map_scale.resolve_scale(path="/content/Neural-Network-Architecture/data/gnn_inference_test/") == 3.52
    assert map_scale.resolve_scale(path="/content/dataset_256_256_diverse/val") == 1.0


@pytest.mark.parametrize("kw", [dict(dataset="mars"), dict(path="/content/test300"), dict(path="/content/dataset_256/val"),
                                dict(path="/content/diverse_runs/maps"), dict()])
def test_unknown_scale_raises(kw):
    with pytest.raises(ValueError):
        map_scale.resolve_scale(**kw)


def test_ambiguous_path_raises():
    with pytest.raises(ValueError):
        map_scale.resolve_scale(path="/content/dataset_256_256_diverse/gnn_inference_test")


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf")])
def test_bad_scale_raises(bad):
    with pytest.raises(ValueError):
        map_scale.resolve_scale(m_per_px=bad)
    with pytest.raises(ValueError):
        map_scale.pose_to_pixel(1.0, 1.0, bad)


def test_explicit_scale_overrides_dataset():
    assert map_scale.resolve_scale(dataset="diverse", path="data/gnn_inference_test", m_per_px=2.0) == 2.0


def test_hit_radius_matches_team_default():
    """3 px at 3.52 m/px is the benchmark's 10.56 m, and a 3 px error counts as a hit."""
    r = map_scale.hit_radius_m(3.52)
    assert r == pytest.approx(10.56)
    assert 3.0 * 3.52 <= r
    assert not (math.hypot(3, 1) * 3.52 <= r)


def test_pose_pixel_round_trip():
    rng = np.random.default_rng(0)
    for res, ox, oy in ((3.52, 0.0, 0.0), (1.0, -128.0, 40.0), (3.52, -450.0, -447.9)):
        for _ in range(3000):
            x, y = ox + rng.uniform(0, 256 * res), oy + rng.uniform(0, 256 * res)
            px = map_scale.pose_to_pixel(x, y, res, ox, oy)
            assert px is not None
            xc, yc = map_scale.pixel_to_pose(*px, res, ox, oy)
            assert abs(xc - x) <= res / 2 + 1e-9 and abs(yc - y) <= res / 2 + 1e-9
            assert map_scale.pose_to_pixel(xc, yc, res, ox, oy) == px


def test_pose_edges():
    assert map_scale.pose_to_pixel(0.0, 0.0, 3.52) == (0, 0)
    assert map_scale.pose_to_pixel(3.52 * 256 - 1e-6, 0.0, 3.52) == (0, 255)
    assert map_scale.pose_to_pixel(3.52 * 256, 10.0, 3.52) is None       # exactly on the far edge = outside
    assert map_scale.pose_to_pixel(-0.01, 10.0, 3.52) is None
    assert map_scale.pose_to_pixel(10.0, 5.0, 3.52) == (1, 2)            # x -> col, y -> row
    with pytest.raises(ValueError):
        map_scale.pose_to_pixel(float("nan"), 1.0, 3.52)


def test_matches_ros_grid():
    """Same pixel as the team's ROS SampleGrid for every pose, inside and outside the map."""
    if not os.path.isdir(ROS_PKG):
        pytest.skip("ros2/gnn_reconstruction not found")
    sys.path.insert(0, ROS_PKG)
    from gnn_reconstruction.core import SampleGrid
    rng = np.random.default_rng(1)
    for res, ox, oy in ((3.52, 0.0, 0.0), (3.52, -450.0, -447.9), (1.0, 12.5, -7.25)):
        g = SampleGrid(ox, oy, res)
        for _ in range(4000):
            x, y = ox + rng.uniform(-50, 256 * res + 50), oy + rng.uniform(-50, 256 * res + 50)
            assert map_scale.pose_to_pixel(x, y, res, ox, oy) == g.to_pixel(x, y)
        assert map_scale.pixel_to_pose(17, 201, res, ox, oy) == pytest.approx(g.to_xy(17, 201))


def test_dataset_rows_are_top_down(maps):
    """World points on real buildings land on building pixels only after dataset_to_grid():
    the dataset stores row 0 at the highest y, the grid has row 0 at the lowest y."""
    trimesh = pytest.importorskip("trimesh")
    geom = pytest.importorskip("shapely.geometry")
    files = sorted(glob.glob(os.path.join(MESH_DIR, "*building*.ply")))
    floor = os.path.join(MESH_DIR, "floor.ply")
    if not files or not os.path.exists(floor):
        pytest.skip("UCF scene meshes not found")
    b = trimesh.load(floor).bounds
    cx, cy = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
    res = 900.0 / 256
    ox, oy = cx - 450.0, cy - 450.0
    env_dataset = maps[0][1] > 0.5
    pixels = []
    for part in trimesh.load(files[0]).split(only_watertight=False):
        hull = geom.MultiPoint(part.vertices[:, :2]).convex_hull
        if hull.area < 400:
            continue
        p = hull.representative_point()
        px = map_scale.pose_to_pixel(p.x, p.y, res, ox, oy)
        if px is not None:
            pixels.append(px)
    n = len(pixels)
    assert n >= 30
    e = env_dataset
    candidates = {"as stored": e, "rows flipped": map_scale.dataset_to_grid(e), "columns flipped": np.fliplr(e),
                  "rows and columns flipped": np.flipud(np.fliplr(e)), "x/y swapped": e.T,
                  "swapped, rows flipped": np.flipud(e.T), "swapped, columns flipped": np.fliplr(e.T),
                  "swapped, both flipped": np.flipud(np.fliplr(e.T))}
    share = {k: sum(bool(v[px]) for px in pixels) / n for k, v in candidates.items()}
    assert share["rows flipped"] >= 0.85, share            # x -> column is right, only the row axis is reversed
    assert all(v <= 0.5 for k, v in share.items() if k != "rows flipped"), share
    assert np.array_equal(map_scale.dataset_to_grid(map_scale.dataset_to_grid(e)), e)


def test_model_input_holds_no_hidden_ground_truth():
    """The model input depends only on the building mask and the sampled values: scrambling every
    unsampled ground-truth pixel and moving the true emitter marker must not change one number."""
    rng = np.random.default_rng(0)
    env = np.zeros((256, 256), np.float32); env[60:120, 40:90] = 1
    gt = rng.uniform(45, 135, (256, 256)).astype(np.float32)
    tx = np.zeros((256, 256), np.float32); tx[30, 200] = 1
    mask = br.make_mask(2, env, np.random.default_rng(1), "uniform", False)
    x = br.build_input(env, tx, gt, mask, False)
    gt2 = gt.copy(); gt2[~mask] = rng.uniform(45, 135, int((~mask).sum()))
    x2 = br.build_input(env, np.roll(tx, (50, -70), (0, 1)), gt2, mask, False)
    assert torch.equal(x, x2)
    assert not x[0, 1].any()                                             # emitter channel is never filled
    assert np.array_equal((x[0, SPARSE_CH] > 0).numpy(), mask)           # "measured" = exactly the sampled pixels


def test_ros_node_pipeline_matches_direct_inference(gnn, maps):
    """Samples sent as map-frame positions through the team's SampleGrid + Reconstructor give the same
    map as calling the model directly, and the emitter estimate converts back to the same pixel.
    Works with both versions of the ROS package: the one that used dataset rows as grid rows, and
    the one with the row-convention fix (mask flipped on load, rows flipped around the model)."""
    if not os.path.isdir(ROS_PKG):
        pytest.skip("ros2/gnn_reconstruction not found")
    sys.path.insert(0, ROS_PKG)
    from gnn_reconstruction import core
    model = wrap(gnn)

    class Backend:
        name = "test"

        def __call__(self, x):
            assert x.shape == (1, 3, 256, 256) and x.dtype == np.float32   # the node's model contract
            with torch.no_grad():
                return model(torch.from_numpy(x)).numpy()

    fixed = hasattr(core, "dataset_to_grid")
    to_grid = map_scale.dataset_to_grid if fixed else (lambda a: a)
    name, env, tx, gt = maps[0]                                             # dataset layout, as stored
    mask = br.make_mask(2, env, np.random.default_rng(0), "uniform", False)
    ox, oy, res = -450.0, -447.9, 3.52
    grid = core.SampleGrid(ox, oy, res)
    for r, c in zip(*np.nonzero(mask)):
        grid_row = 255 - int(r) if fixed else int(r)                       # where that file pixel really is
        x_m, y_m = map_scale.pixel_to_pose(grid_row, int(c), res, ox, oy)
        assert grid.add(x_m, y_m, float(gt[r, c]))
    assert np.array_equal(grid.mask, to_grid(mask))                        # every sample landed on its own pixel
    pl_db, (row, col), (ex, ey), _ = core.Reconstructor(Backend(), to_grid(env).astype(np.float32), grid).run()
    with torch.no_grad():
        direct = model(br.build_input(env, tx, gt, mask, False))[0, 0].numpy() * (br.MAX_DB - br.MIN_DB) + br.MIN_DB
    expect = to_grid(direct)
    assert pl_db.shape == (256, 256)
    assert float(np.abs(pl_db - expect).mean()) < 0.01 and float(np.abs(pl_db - expect).max()) < 1.0
    assert map_scale.pose_to_pixel(ex, ey, res, ox, oy) == (int(row), int(col))


# ------------------------------------------------------------------ block-artifact metric
def test_blockiness_synthetic():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:256, :256]
    smooth = np.sin(xx / 23.0) + np.cos(yy / 17.0) + 0.05 * rng.standard_normal((256, 256))
    assert 0.8 < blockiness(smooth) < 1.25
    blocks = np.kron(rng.standard_normal((32, 32)), np.ones((8, 8))) + 0.01 * rng.standard_normal((256, 256))
    assert blockiness(blocks) > 20
    assert blockiness(np.roll(blocks, 4, axis=1), patch=8) < blockiness(blocks)   # grid must line up to count
    valid = np.ones((256, 256), bool); valid[:, 100:140] = False
    assert np.isfinite(blockiness(smooth, valid))


def test_blockiness_on_real_maps(real_gnn, maps):
    """Ground truth has no 8-px grid (about 1.0); the raw GNN output is strongly blocky."""
    gt_scores, gnn_scores = [], []
    for name, env, tx, gt in maps[:4]:
        mask = br.make_mask(2, env, np.random.default_rng(0), "uniform", False)
        with torch.no_grad():
            pred = real_gnn(br.build_input(env, tx, gt, mask, False))[0, 0].numpy()
        gt_scores.append(blockiness(gt, env < 0.5)); gnn_scores.append(blockiness(pred, env < 0.5))
    assert 0.9 < np.mean(gt_scores) < 1.1
    assert np.mean(gnn_scores) > 8


# ------------------------------------------------------------------ path-shaped training input (SCRUM-104)
def _neighbour_share(mask):
    """Share of sampled pixels with another sample directly above, below, left or right."""
    p = np.pad(mask, 1)
    nb = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return float((mask & nb).sum() / max(mask.sum(), 1))


def _toy_map(seed=0):
    rng = np.random.default_rng(seed)
    env = np.zeros((256, 256), np.uint8); env[30:80, 30:120] = 1; env[150:220, 140:230] = 1
    pl = rng.uniform(0.05, 0.95, (256, 256)).astype(np.float16)
    return env, pl


@pytest.mark.parametrize("density", [0.5, 2.0, 10.0])
def test_walk_training_input(density):
    env, pl = _toy_map()
    x = tu.train_input(env, pl, density, np.random.default_rng(3), False, kind="walk")
    assert x.shape == (3, 256, 256) and x.dtype == np.float32
    mask = x[2] > 0
    assert mask.sum() == max(1, int(round(density / 100 * 256 * 256)))
    assert not (mask & (env > 0)).any()                       # a rover path never enters a building
    assert np.array_equal(x[0], env.astype(np.float32)) and not x[1].any()
    assert np.allclose(x[2][mask], np.maximum(pl.astype(np.float32)[mask], 1e-4))   # samples sit on their own pixel
    assert _neighbour_share(mask) > 0.9                        # a path, not scattered points


def test_uniform_training_input_unchanged():
    """kind='uniform' draws exactly what the 1 November training run drew for the same seed."""
    env, pl = _toy_map()
    for avoid in (False, True):
        x = tu.train_input(env, pl, 2.0, np.random.default_rng(5), avoid)
        rng = np.random.default_rng(5)
        pool = np.flatnonzero(env.ravel() == 0) if avoid else np.arange(256 * 256)
        idx = rng.choice(pool, size=int(round(0.02 * 256 * 256)), replace=False)
        assert np.array_equal(np.flatnonzero(x[2].ravel() > 0), np.sort(idx))
        assert _neighbour_share(x[2] > 0) < 0.3


def test_mix_draws_both_kinds_and_is_deterministic():
    env, pl = _toy_map()
    shares = [_neighbour_share(tu.train_input(env, pl, 2.0, np.random.default_rng(s), False, kind="mix")[2] > 0)
              for s in range(24)]
    assert any(s > 0.9 for s in shares) and any(s < 0.3 for s in shares)
    a = tu.train_input(env, pl, 2.0, np.random.default_rng(11), False, kind="mix")
    b = tu.train_input(env, pl, 2.0, np.random.default_rng(11), False, kind="mix")
    assert np.array_equal(a, b)


def test_validation_scores_walk_and_uniform(gnn):
    """val_rmse runs on walk and mix masks and mix is the mean of the two."""
    env, pl = _toy_map()
    cache = SimpleNamespace(env=[env, env], pl=[pl, pl]); cache_len = 2
    class C:  # minimal stand-in for tu.Cache
        env, pl = cache.env, cache.pl
        def __len__(self): return cache_len
    u = tu.val_rmse(gnn, C(), CPU, False, kind="uniform")
    w = tu.val_rmse(gnn, C(), CPU, False, kind="walk")
    mix = tu.val_rmse(gnn, C(), CPU, False, kind="mix")
    assert all(np.isfinite(v) and v > 0 for v in (u, w, mix))
    assert mix == pytest.approx((u + w) / 2, rel=1e-6)


# ------------------------------------------------------------------ regression against the team benchmark
def _bench(model, maps, density, tmpdir, kind="uniform"):
    a = SimpleNamespace(**{**vars(BARGS), "mask_kind": kind})
    out = []
    for name, env, tx, gt in maps:
        mask = br.get_mask(str(tmpdir), name, density, env, a)
        with torch.no_grad():
            pred = model(br.build_input(env, tx, gt, mask, False))[0, 0].numpy()
        pred_db = pred * (br.MAX_DB - br.MIN_DB) + br.MIN_DB
        out.append({**br.score(pred_db, gt, env, tx, mask, a), "blockiness": blockiness(pred_db, env < 0.5)})
    return {k: float(np.mean([o[k] for o in out])) for k in ("rmse_db", "blockiness")}


def test_baseline_reproduces_team_number(real_gnn, maps, tmp_path):
    """GNN_no_tx on the 10 repo maps at 0.5 % uniform samples is 18.85 dB, as in the team's table."""
    if len(maps) != 10:
        pytest.skip("needs the 10 repo test maps")
    assert _bench(real_gnn, maps, 0.5, tmp_path)["rmse_db"] == pytest.approx(18.85, abs=0.05)


def test_trained_decoder_removes_block_artifacts(real_gnn, trained, maps, tmp_path):
    """The actual SCRUM-93 fix: a trained checkpoint loads strictly, is far less blocky and more accurate."""
    for density in (0.5, 2):
        base = _bench(real_gnn, maps, density, tmp_path); dec = _bench(trained, maps, density, tmp_path)
        assert dec["blockiness"] < 3.0 < base["blockiness"]
        assert dec["rmse_db"] < base["rmse_db"] - 3.0


def test_trained_decoder_output_is_the_right_way_up(trained, maps):
    """Aligned, not just smooth: on every map the output is closer to the ground truth than to the
    ground truth mirrored top-bottom, mirrored left-right, or with x and y swapped, and on average
    by a wide margin (a map whose emitter sits near a mirror axis looks similar mirrored)."""
    flips = {"rows flipped": np.flipud, "columns flipped": np.fliplr, "x/y swapped": np.transpose}
    right, wrong = [], {k: [] for k in flips}
    for name, env, tx, gt in maps:
        mask = br.make_mask(2, env, np.random.default_rng(0), "uniform", False)
        with torch.no_grad():
            pred = trained(br.build_input(env, tx, gt, mask, False))[0, 0].numpy() * (br.MAX_DB - br.MIN_DB) + br.MIN_DB
        rmse = lambda ref: math.sqrt(float(np.mean((pred - ref) ** 2)))
        right.append(rmse(gt))
        for k, f in flips.items():
            wrong[k].append(rmse(f(gt)))
            assert right[-1] < wrong[k][-1], (name, k)
    for k in flips:
        assert np.mean(right) < 0.6 * np.mean(wrong[k]), k
