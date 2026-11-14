"""Checks the ROS-free tracker against the notebook's simulation (no ROS or GPU needed)."""
import numpy as np

from gnn_reconstruction.tracker import EmitterTracker

M_PER_PX = 3.52


def simulate(speed=1.0, dt=0.5, noise_px=5.0, outlier_p=0.1, seed=0, m_per_px=M_PER_PX):
    rng = np.random.default_rng(seed)
    wp = np.array([[60, 60], [180, 70], [200, 180], [90, 200], [70, 120]], float) * m_per_px
    seg = np.diff(wp, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    t = np.arange(0, seg_len.sum() / speed, dt)
    s = t * speed
    idx = np.minimum(np.searchsorted(np.cumsum(seg_len), s, side="right"), len(seg) - 1)
    start = np.concatenate([[0], np.cumsum(seg_len)])[idx]
    truth = wp[idx] + seg[idx] / seg_len[idx, None] * (s - start)[:, None]
    meas = truth + rng.normal(0, noise_px * m_per_px, truth.shape)
    jump = rng.random(len(t)) < outlier_p
    ang = rng.uniform(0, 2 * np.pi, jump.sum())
    r = rng.uniform(10, 25, jump.sum()) * m_per_px
    meas[jump] = truth[jump] + np.c_[np.cos(ang), np.sin(ang)] * r[:, None]
    meas = (np.clip(np.floor(meas / m_per_px), 0, 255) + 0.5) * m_per_px
    return t, truth, meas


def rmse(est, truth):
    return float(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))))


def test_matches_notebook_result():
    t, truth, meas = simulate()
    trk = EmitterTracker(accel_std=0.1, meas_std=6 * M_PER_PX)
    est = np.array([trk.update(z, ti)[0] for ti, z in zip(t, meas)])
    assert abs(rmse(meas, truth) - 30.8) < 0.2   # raw GNN-like estimates
    assert abs(rmse(est, truth) - 6.4) < 0.2     # notebook: hand-written Kalman 6.4 m


def test_static_emitter_converges():
    rng = np.random.default_rng(1)
    trk = EmitterTracker(accel_std=0.1, meas_std=21.0)
    for i in range(200):
        pos, _ = trk.update(np.array([500.0, 300.0]) + rng.normal(0, 18, 2), 0.5 * i)
    assert np.hypot(pos[0] - 500, pos[1] - 300) < 8
    assert np.hypot(*trk.velocity) < 0.5
